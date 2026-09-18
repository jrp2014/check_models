# Model Output Gallery

Generated on: 2026-09-18 23:43:35 BST

- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 44.7 MB

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image-57f7cdeb1b88952a.jpg)

## Current-run Chooser

Mechanical observations and captured resource facts for this run only. No concerns detected does not mean the response fulfilled an arbitrary prompt or described the image accurately. Consult the assessment scope above. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is the measured time to first token (input preparation, prefill and the first decode step) when captured, else upstream's model-loop first-token time; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                     | Mechanical checks      | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------|------------------------|---------|---------------------|-----------------|---------|------------|---------|-----------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)   | `no concerns detected` | 11.48s  | 29.6 tok/s          | 3.59            | 23      | 2,393      | 151     | none                                                                                          |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)              | `no concerns detected` | 16.34s  | 77.8 tok/s          | 1.31            | 19      | 1,634      | 997     | none                                                                                          |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                            | `no concerns detected` | 9.76s   | 74.1 tok/s          | 5.84            | 8.8     | 6,450      | 140     | none                                                                                          |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                   | `no concerns detected` | 8.50s   | 35.5 tok/s          | 1.79            | 18      | 2,612      | 138     | none                                                                                          |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                             | `no concerns detected` | 6.02s   | 57.4 tok/s          | 2.02            | 10      | 2,113      | 126     | none                                                                                          |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                               | `no concerns detected` | 5.78s   | 37.7 tok/s          | 1.43            | 17      | 2,113      | 77      | none                                                                                          |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                    | `no concerns detected` | 3.36s   | 214 tok/s           | 1.12            | 4.0     | 2,103      | 81      | none                                                                                          |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                              | `no concerns detected` | 3.23s   | 105 tok/s           | 0.72            | 7.0     | 391        | 101     | none                                                                                          |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)           | `no concerns detected` | 6.77s   | 67.2 tok/s          | 3.09            | 13      | 2,926      | 109     | none                                                                                          |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)           | `no concerns detected` | 7.77s   | 62.3 tok/s          | 2.66            | 13      | 2,926      | 187     | none                                                                                          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)               | `no concerns detected` | 3.75s   | 191 tok/s           | 1.53            | 7.8     | 2,925      | 102     | none                                                                                          |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                 | `no concerns detected` | 4.87s   | 154 tok/s           | 2.62            | 3.9     | 4,083      | 84      | none                                                                                          |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                        | `no concerns detected` | 6.73s   | 71.8 tok/s          | 1.28            | 24      | 1,289      | 155     | none                                                                                          |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                          | `no concerns detected` | 4.21s   | 54.9 tok/s          | 0.78            | 9.3     | 1,133      | 107     | none                                                                                          |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                               | `no concerns detected` | 27.70s  | 86.3 tok/s          | 15.37           | 8.4     | 16,549     | 912     | none                                                                                          |
| [`mlx-community/Qwen3-VL-32B-Instruct-4bit`](#model-mlx-community-qwen3-vl-32b-instruct-4bit)                             | `no concerns detected` | 77.47s  | 20.4 tok/s          | 65.80           | 26      | 16,547     | 177     | none                                                                                          |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                               | `no concerns detected` | 39.47s  | 69.7 tok/s          | 35.72           | 11      | 16,547     | 93      | none                                                                                          |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                          | `no concerns detected` | 39.73s  | 72.4 tok/s          | 34.63           | 25      | 16,563     | 97      | none                                                                                          |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                            | `no concerns detected` | 44.16s  | 91.4 tok/s          | 40.63           | 11      | 16,563     | 86      | none                                                                                          |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                            | `no concerns detected` | 36.64s  | 47.4 tok/s          | 20.49           | 92      | 3,491      | 121     | none                                                                                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                             | `no concerns detected` | 18.44s  | 57.6 tok/s          | 13.94           | 14      | 16,558     | 136     | none                                                                                          |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                             | `no concerns detected` | 4.62s   | 100 tok/s           | 1.56            | 6.5     | 2,090      | 96      | none                                                                                          |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                           | `no concerns detected` | 5.10s   | 60.4 tok/s          | 0.93            | 7.6     | 595        | 101     | none                                                                                          |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                   | `no concerns detected` | 5.04s   | 108 tok/s           | 0.99            | 16      | 595        | 96      | none                                                                                          |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                           | `no concerns detected` | 8.45s   | 26.3 tok/s          | 1.69            | 20      | 595        | 89      | none                                                                                          |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                           | `no concerns detected` | 3.74s   | 125 tok/s           | 0.77            | 6.0     | 591        | 73      | none                                                                                          |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                              | `no concerns detected` | 2.91s   | 176 tok/s           | 1.14            | 4.7     | 1,379      | 65      | none                                                                                          |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                      | `concerns detected`    | 23.56s  | 41.6 tok/s          | 13.67           | 78      | 6,450      | 103     | prompt hint repeated                                                                          |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                     | `concerns detected`    | 6.16s   | 72.5 tok/s          | 1.35            | 8.6     | 1,524      | 216     | duplicate keywords                                                                            |
| [`mlx-community/Qwen2-VL-7B-Instruct-4bit`](#model-mlx-community-qwen2-vl-7b-instruct-4bit)                               | `concerns detected`    | 41.22s  | 91.3 tok/s          | 38.22           | 9.3     | 16,558     | 117     | prompt hint repeated                                                                          |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                  | `concerns detected`    | 65.24s  | 29.8 tok/s          | 57.99           | 21      | 16,563     | 124     | unsupplied place name                                                                         |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                              | `concerns detected`    | 3.33s   | 131 tok/s           | 1.07            | 5.6     | 1,426      | 98      | prompt hint repeated                                                                          |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                   | `concerns detected`    | 6.81s   | 46.1 tok/s          | 2.85            | 28      | 591        | 80      | duplicate keywords; prompt hint repeated                                                      |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                   | `concerns detected`    | 9.47s   | 30.3 tok/s          | 1.47            | 17      | 590        | 152     | unsupplied place name                                                                         |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                 | `concerns detected`    | 7.12s   | 40.2 tok/s          | 2.17            | 16      | 3,116      | 102     | prompt hint repeated                                                                          |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                              | `major concerns`       | 2.26s   | 476 tok/s           | 0.60            | 1.9     | 2,112      | 200     | repeated text; stopped early: repeating; duplicate keywords; prompt hint repeated             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                     | `major concerns`       | 20.21s  | 63.0 tok/s          | 1.38            | 20      | 1,324      | 1,000   | cut off at token limit                                                                        |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)              | `major concerns`       | 14.66s  | 25.7 tok/s          | 2.00            | 9.8     | 301        | 275     | repeated text; stopped early: repeating; duplicate keywords                                   |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)              | `major concerns`       | 57.79s  | 18.8 tok/s          | 2.20            | 15      | 301        | 1,000   | repeated text; cut off at token limit                                                         |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                           | `major concerns`       | 53.04s  | 24.3 tok/s          | 7.92            | 25      | 4,403      | 1,000   | cut off at token limit                                                                        |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                     | `major concerns`       | 41.31s  | 74.7 tok/s          | 35.71           | 23      | 16,547     | 200     | stopped early: repeating; duplicate keywords                                                  |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                             | `major concerns`       | 2.21s   | 320 tok/s           | 0.92            | 1.1     | 1,205      | 33      | labelled fields not detected                                                                  |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                         | `major concerns`       | 4.43s   | 71.0 tok/s          | 0.83            | 7.2     | 589        | 88      | labelled fields not detected                                                                  |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                   | `major concerns`       | 5.26s   | 102 tok/s           | 1.51            | 6.7     | 2,195      | 200     | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                              | `major concerns`       | 2.34s   | 193 tok/s           | 0.66            | 1.8     | 330        | 92      | labelled fields not detected                                                                  |
| [`mlx-community/paligemma2-10b-mix-448-4bit`](#model-mlx-community-paligemma2-10b-mix-448-4bit)                           | `major concerns`       | 4.02s   | insufficient sample | 1.50            | 9.7     | 1,344      | 7       | labelled fields not detected                                                                  |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                            | `not assessed`         | 0.16s   | -                   | -               | -       | -          | -       | none                                                                                          |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                             | `not assessed`         | 2.13s   | -                   | -               | -       | -          | -       | none                                                                                          |
| [`mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit`](#model-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) | `not assessed`         | 0.25s   | -                   | -               | -       | -          | -       | none                                                                                          |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `mlx-community/granite-4.0-3b-vision-4bit` at 2.91s

Lowest peak memory among completions without detected concerns: `mlx-community/North-Micro-Vision-Instruct-4bit` at 3.9 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                     | Mechanical checks | Observations                                                                                  |
|---------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                              | `major concerns`  | repeated text; stopped early: repeating; duplicate keywords; prompt hint repeated             |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                     | `major concerns`  | cut off at token limit                                                                        |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)              | `major concerns`  | repeated text; stopped early: repeating; duplicate keywords                                   |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)              | `major concerns`  | repeated text; cut off at token limit                                                         |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                           | `major concerns`  | cut off at token limit                                                                        |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                     | `major concerns`  | stopped early: repeating; duplicate keywords                                                  |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                             | `major concerns`  | labelled fields not detected                                                                  |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                         | `major concerns`  | labelled fields not detected                                                                  |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                   | `major concerns`  | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                              | `major concerns`  | labelled fields not detected                                                                  |
| [`mlx-community/paligemma2-10b-mix-448-4bit`](#model-mlx-community-paligemma2-10b-mix-448-4bit)                           | `major concerns`  | labelled fields not detected                                                                  |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                            | `not assessed`    | none                                                                                          |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                             | `not assessed`    | none                                                                                          |
| [`mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit`](#model-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) | `not assessed`    | none                                                                                          |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

A compact preview of each model's final answer (or failure evidence for crashes), in chooser order. Where the requested catalogue fields were detected, the preview shows a little of each: the title, the start of the description, and the first keywords with their count, so the weakest field is not hidden behind a long description. Otherwise it is the first 280 characters. A closed reasoning trace is left out of the preview and reported as an omitted-character count; the complete output, trace included, is in the model's evidence block below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                     | Mechanical checks      | Output preview                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)   | `no concerns detected` | Title: Swan on a calm river with moored boats \| Description: A solitary white swan swims gracefully on a calm river, with leisure boats and cruisers moored alongside r... \| Keywords (20): Swan, river, boats, mooring, residential buildings, greenery, water reflection, aquatic bird, marina, ...                                                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)              | `no concerns detected` | Title: Solitary White Swan Glides on Calm River with Boats and Lush Foliage \| Description: On 12 September 2026 at 17:41 UTC+01 (52.393850°N, 0.270830°E), a solitary wh... \| Keywords (17): White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, ...[2,490 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                            | `no concerns detected` | Title: Swan at River Marina \| Description: A white swan glides on calm river waters, with leisure boats moored alongside riverside residential buildings in the... \| Keywords (13): Swan, River, Marina, Leisure boats, Residential buildings, Calm waters, White swan, Riverbank, Boats moored, ...                                                                                |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                   | `no concerns detected` | Title: Solitary Swan on Calm River Waters with Moored Boats and Riverside Buildings. \| Description: A solitary white swan glides gracefully across the calm river wate... \| Keywords (20): swan, river, boats, moored, residential buildings, foliage, calm waters, leisure, cruisers, riverside, ...                                                                               |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                             | `no concerns detected` | Title: Swan on Canal with Moored Boats \| Description: A white swan glides on a tranquil canal, surrounded by moored boats and riverside buildings, with lush greenery... \| Keywords (19): Swan, Canal, Boats, Mooring, Residential Buildings, Greenery, Trees, Water Reflection, Motorboat, Pier, ...                                                                               |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                               | `no concerns detected` | Title: Swan on Calm River \| Description: A white swan glides on tranquil river waters near a marina with moored boats and residential buildings, captured at dusk. \| Keywords (15): swan, river, marina, boats, residential buildings, foliage, trees, water reflection, architecture, motorboat, ...                                                                               |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                    | `no concerns detected` | Title: Swan gliding past moored boats on tranquil riverbank \| Description: A white swan swims gracefully on calm river waters, framed by lush greenery, with leisur... \| Keywords (14): swan, river, boats, marina, greenery, water, reflection, architecture, leisure, waterfront, aquatic bird, ...                                                                               |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                              | `no concerns detected` | Title: Swan Glides on Calm River Near Moored Boats \| Description: A white swan swims peacefully in a tranquil waterway, framed by greenery and moored leisure bo... \| Keywords (16): swan, river, waterway, mooring, motorboat, boats, greenery, trees, vegetation, waterfront, water reflection, ...                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)           | `no concerns detected` | Title: **Swan Gliding Past Moored Boats on a Serene River** \| Description: A solitary white swan swims calmly on a tranquil river, surrounded by reflections of... \| Keywords (16): swan, river, moored boats, marina, residential waterfront, greenery, calm water, reflections, leisure boats, ...                                                                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)           | `no concerns detected` | Title: *Elegant Swan Glides Past Moored Boats on Serene River* \| Description: A solitary white swan gracefully swims across calm river waters on **September 12, 2026,... \| Keywords (20): swan, river, leisure boats, moored boats, riverside residential buildings, greenery, golden sunlight, ...                                                                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)               | `no concerns detected` | Title: Swans by Riverside Marina \| Description: A white swan swims serenely in calm waters at dusk near a marina moored with leisure boats, framed by lush greenery... \| Keywords (17): swan, canal, marina, mooring, motorboat, pier, waterfront, waterway, greenery, residential architecture, ...                                                                                |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                 | `no concerns detected` | Title: Swan on the River \| Description: A solitary white swan glides gracefully across calm river waters, framed by lush greenery. In the background, leisure boa... \| Keywords (13): Swan, River, Greenery, Marina, Boat, Waterfront, Water, Aquatic Bird, Architecture, Leisure Boats, Foliage, ...                                                                               |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                        | `no concerns detected` | Title: White Swan Glides Past Moored Boats On Calm River \| Description: A solitary white swan glades across a calm river in the foreground, its reflection ripplin... \| Keywords (19): white swan, river, moored boats, motorboats, cruisers, marina, pontoon, water reflection, swan, waterfowl, ...                                                                               |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                          | `no concerns detected` | Title: Serene Swan on the River \| Description: A white swan is seen swimming peacefully on a calm river, with boats and houses in the background, captured on S... \| Keywords (18): swan, river, calm, boats, houses, foliage, waterfront, mooring, leisure, tranquility, architecture, waterway, ...                                                                               |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                               | `no concerns detected` | Title: Solitary Swan on Riverfront Marina \| Description: A solitary white swan glides across calm river waters in a canal with moored leisure boats and reside... \| Keywords (15): Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, ...[2,859 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/Qwen3-VL-32B-Instruct-4bit`](#model-mlx-community-qwen3-vl-32b-instruct-4bit)                             | `no concerns detected` | Title: Swan Gliding on River Near Moored Boats \| Description: A solitary white swan glides across calm river waters, framed by overhanging green foliage, with se... \| Keywords (18): swan, river, moored boats, leisure craft, residential buildings, greenery, water reflection, foliage, pier, ...                                                                               |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                               | `no concerns detected` | Title: Swan on River with Moored Boats \| Description: A solitary white swan glides calmly on a river, framed by green foliage, with moored leisure boats and resid... \| Keywords (17): swan, river, boats, mooring, canal, greenery, trees, water reflection, waterfront, residential, motorboat, ...                                                                               |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                          | `no concerns detected` | Title: Swan Swimming Past Moored Boats \| Description: A solitary white swan glides across calm river waters near a riverside mooring of leisure cruisers and resid... \| Keywords (18): swan, river, moored boats, riverbank, vegetation, water reflection, cruiser, residential buildings, trees, ...                                                                               |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                            | `no concerns detected` | Title: Swan Glides Past Moored Boats on Calm Riverbank \| Description: A solitary white swan swims peacefully on a reflective river, with leisure boats moored along t... \| Keywords (15): swan, river, boats, mooring, residential buildings, water reflection, foliage, trees, motorboat, canal, ...                                                                               |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                            | `no concerns detected` | Title: Swan glides past moored boats on a calm river \| Description: A solitary white swan swims across calm river waters, framed by lush foreground foliage and... \| Keywords (18): Swan, waterfowl, river, marina, mooring, motorboat, pier, waterfront, waterway, canal, bird, swimming, trees, ...                                                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                             | `no concerns detected` | Title: Swan Gliding on Calm River with Moored Boats \| Description: A serene scene captures a solitary white swan gracefully swimming across calm river waters, frame... \| Keywords (23): Swan, River, Calm, Moored Boats, Residential Buildings, Greenery, Waterfront, Waterway, Motorboat, Pier, ...                                                                               |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                             | `no concerns detected` | Title: Solitary Swan Glides on Tranquil River \| Description: A serene scene captures a graceful white swan effortlessly gliding across the calm waters of a river,... \| Keywords (12): Swan, River, Greenery, Boats, Cruisers, Buildings, Lighting, Afternoon, Tranquil, Waterfowl, Architecture, ...                                                                               |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                           | `no concerns detected` | Title: White Swan Gliding on a River Near Moored Boats \| Description: A solitary white swan glides across a calm river surface reflecting leisure boats and residential... \| Keywords (19): Swan, Waterfowl, River, Canal, Boat, Motorboat, Marina, Mooring, Architecture, Residential, Waterway, ...                                                                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                   | `no concerns detected` | Title: White swan swimming on calm river with moored boats \| Description: A solitary white swan glides across calm river waters near a wooden pier, with several... \| Keywords (15): white swan, river, motorboat, waterfowl, waterfront, greenery, reflection, residential building, boat, pier, ...                                                                               |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                           | `no concerns detected` | Title: White Swan Swimming Past Moored River Boats \| Description: A single white swan swims across calm river waters in front of leisure boats moored beside rivers... \| Keywords (18): swan, river, boats, marina, moored, waterfront, waterway, residential, architecture, greenery, waterfowl, ...                                                                               |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                           | `no concerns detected` | Title: Swan glides on calm water beneath residential buildings \| Description: A solitary white swan gracefully floats on the calm river water, framed by dense greenery... \| Keywords (16): swan, waterfowl, river, canal, marina, boats, trees, greenery, quiet, architecture, moored, peaceful, ...                                                                               |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                              | `no concerns detected` | Title: "Swan on the Canal" \| Description: A white swan glides gracefully across a calm river, framed by lush foliage and moored boats, with a serene residential backdrop. \| Keywords (10): swan, canal, water, foliage, boats, mooring, residential, reflection, serene, waterway                                                                                                  |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                      | `concerns detected`    | Title: Solitary swan glides on calm river waters \| Description: A solitary white swan glides gracefully across calm river waters, framed by lush foliage, with leisu... \| Keywords (17): swan, river, water, foliage, boats, cruisers, residential buildings, moored, overcast, water reflection, ...                                                                               |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                     | `concerns detected`    | Title: Serene River Scene with White Swan and Moored Boats \| Description: A tranquil riverside view features a solitary white swan gliding across calm water, surro... \| Keywords (42): swan, river, boats, mooring, canal, greenery, trees, vegetation, water reflection, waterfowl, waterfront, ...                                                                               |
| [`mlx-community/Qwen2-VL-7B-Instruct-4bit`](#model-mlx-community-qwen2-vl-7b-instruct-4bit)                               | `concerns detected`    | Title: Graceful Swan on the River \| Description: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruis... \| Keywords (26): swan, river, calm water, lush foliage, leisure boats, cruisers, riverside buildings, residential area, ...                                                                               |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                  | `concerns detected`    | Title: Swan swimming near moored boats on the River Thames \| Description: A solitary white swan swims on the calm water of the River Thames at Lambeth, with its re... \| Keywords (15): Swan, River Thames, Boat, Waterfowl, Water, Reflection, Mooring, Residential building, Foliage, Greenery, ...                                                                               |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                              | `concerns detected`    | Title: Swan on River \| Description: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers mo... \| Keywords (20): Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, ...                                                                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                   | `concerns detected`    | Title: White Swan Gliding on River Near Moored Boats \| Description: A solitary white swan glides across calm river waters, with leisure boats and cruisers moored al... \| Keywords (16): swan, waterfowl, river, boat, marina, mooring, motorboat, cruiser, pier, waterfront, waterway, greenery, ...                                                                               |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                   | `concerns detected`    | Title: Swan on River Cam, Cambridge – September 2026 \| Description: A white swan swims on the River Cam at 52.393850°N, 0.270830°E, captured on 12 September... \| Keywords (18): Swan, River Cam, Cambridge, Waterfowl, Boat, Mooring, Marina, Riverbank, Waterway, Residential building, Trees, ...                                                                                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                 | `concerns detected`    | Title: Swan in River with Moored Boats \| Description: A solitary white swan gracefully swims across calm river waters, framed by lush foliage, with leisure boats... \| Keywords (18): Swan, River, Boats, Marina, Water, Foliage, Residential, Mooring, Waterfront, Waterway, Trees, Vegetation, ...                                                                                |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                              | `major concerns`       | Title: Swans glide across calm river waters in a serene canal setting. \| Description: A solitary white swan glides gracefully across calm river waters framed by l... \| Keywords (63): Swans, canal, water, serene, leisure, boats, water, architecture, greenery, water, architecture, waterway, ...                                                                               |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                     | `major concerns`       | Title: Swan on canal with moored boats at marina \| Description: A white swan glides across a calm canal, surrounded by moored motorboats and residential buildings with lush greenery, at 2026-09-12 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E. \| Keywords (3): Swans, Canal, Motorboat[3,708 characters of reasoning omitted; complete output in the evidence block]         |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)              | `major concerns`       | Title: Serene River Scene with a Majestic White Swan \| Description: A tranquil river scene unfolds with a solitary white swan gliding effortlessly across the... \| Keywords (48): riverside residential area, leisure boats, calm waters, serene atmosphere, lush foliage, white swan, riverside, ...                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)              | `major concerns`       | Title: Serene River Scene with a Majestic White Swan \| Description: A tranquil river scene unfolds on September 12, 2026, at 17:41:02 UTC+01:00, near 52.393850°N, 0.27... \| Keywords (120): River, White Swan, Lush Greenery, Moored Boats, Riversid Residential Buildings, Tranquil Atmosphere, ...                                                                               |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                           | `major concerns`       | Title: Mute Swan on River with Moored Boats \| Description: Captured on 2026-09-12 17:41:02 BST at 52.393850°N, 0.270830°E, a mute swan swims on calm water that reflects moored motor cruisers and riverside apartments. The scene shows boats including ‘Samar I... \| Keywords (2): mute swan, river                                                                               |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                     | `major concerns`       | Title: Solitary swan on a calm river with moored boats \| Description: A white swan glides across the calm waters of a river, framed by green foliage in the foregrou... \| Keywords (57): swan, river, boats, mooring, motorboat, residential, building, balcony, trees, foliage, greenery, water, ...                                                                               |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                             | `major concerns`       | A solitary white swan gliding across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.                                                                                                                                                                                                    |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                         | `major concerns`       | A serene scene unfolds on a calm river, where a graceful swan glides across the water. Several boats, including motorboats and sailboats, are moored along the riverbank, nestled beneath a row of residential buildings. Lush greenery frames the scene, with trees and bushes linin...                                                                                              |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                   | `major concerns`       | <\|channel\|> analysis<\|message\|> The image shows a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a...                                                                                          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                              | `major concerns`       | "Amidst the Riverbank, a solitary white swan glides gracefully across calm waters, framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background. The serene setting is complemented by the presence of greenery, archit...                                                                                              |
| [`mlx-community/paligemma2-10b-mix-448-4bit`](#model-mlx-community-paligemma2-10b-mix-448-4bit)                           | `major concerns`       | - title, description, keywords                                                                                                                                                                                                                                                                                                                                                        |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                            | `not assessed`         | Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'                                                                                                                                                                                                                                                               |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                             | `not assessed`         | Model generation failed for mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]                                                                                                                                                                                                                                              |
| [`mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit`](#model-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) | `not assessed`         | Model loading failed: Received 729 parameters not in model:<br>backbone.embeddings.biases,<br>backbone.embeddings.scales,<br>backbone.embeddings.weight,<br>backbone.layers.0.mixer.A_log,<br>backbone.layers.0.mixer.D,<br>backbone.layers.0.mixer.conv1d.bias,<br>backbone.layers.0.mixer.conv1d.weight,...                                                                         |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.1`
- `mlx`: `0.32.3.dev20260912+229f5b430`
- `transformers`: `5.17.0`
- `tokenizers`: `0.23.2`
- `huggingface-hub`: `1.32.0`
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

- *Description:* A solitary white swan glides gracefully across calm river
  waters framed by lush foliage, with leisure boats and cruisers moored
  alongside riverside residential buildings in the background.
- *Keywords:* Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring,
  Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection,
  Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat, boating,
  boats, cabin cruiser, calm water, cruiser, daytime, dock, environment,
  foliage, harbor, houses, landscape, leisure, mute swan, natural, nature,
  outdoor, peaceful, recreation, reflection, residential, river, riverside,
  scenic, serene, swan, tranquility, travel, watercraft, white swan, wildlife
- *Date:* 2026-09-12 17:41:02 UTC+01:00
- *Time:* 17:41:02
- *GPS:* 52.393850°N, 0.270830°E

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
> &#45; Capture date/time: 2026-09-12 17:41:02 UTC+01:00
> &#45; GPS: 52.393850°N, 0.270830°E
>
> &#8203;Descriptive hints:
> &#45; Description hint: A solitary white swan glides gracefully across calm
> river waters framed by lush foliage, with leisure boats and cruisers moored
> alongside riverside residential buildings in the background.
> &#45; Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina,
> Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water
> reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture,
> boat
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
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.25s
- *Generation time:* 8.70s
- *Total time:* 11.48s
- *Prompt preparation time:* 0.52
- *First-token latency:* 3.04
- *Cleanup time:* 0.111
- *Prompt tokens:* 2,393
- *Generation tokens:* 151
- *Total tokens:* 2,544
- *Prompt throughput (raw):* 788 tok/s
- *Generation throughput (raw):* 29.6 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 18
- *Post-cleanup active memory (GB):* 0.000396
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
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
Swan on a calm river with moored boats

Description:
A solitary white swan swims gracefully on a calm river, with leisure boats and cruisers moored alongside riverside residential buildings in the background, captured on 12 September 2026 at 17:41 UTC+01:00 near 52.393850°N, 0.270830°E.

Keywords:
Swan, river, boats, mooring, residential buildings, greenery, water reflection, aquatic bird, marina, waterfront, waterway, trees, vegetation, calm water, leisure boats, cruisers, riverside, architecture, waterfowl, swimming
</pre>

</details>

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 1.62s
- *Generation time:* 14.12s
- *Total time:* 16.34s
- *Prompt preparation time:* 0.602
- *First-token latency:* 0.731
- *Cleanup time:* 0.112
- *Prompt tokens:* 1,634
- *Generation tokens:* 997
- *Total tokens:* 2,631
- *Prompt throughput (raw):* 2,234 tok/s
- *Generation throughput (raw):* 77.8 tok/s
- *Peak memory (GB):* 19
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.18
- *Model-load active memory (GB):* 17.4
- *Post-cleanup active memory (GB):* 0.000609
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1522
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
Let's tackle this step by step. First, the title needs to be concrete and 5-10 words. The main elements are the swan, river, boats, greenery. Maybe "Solitary White Swan Gliding on Calm River with Boats and Greenery" but need to be concise. Wait, the hint says solitary white swan, calm river, lush foliage, leisure boats. So maybe "Solitary White Swan on Calm River with Boats and Lush Foliage" – that's 9 words, good.

Next, the description. Need to combine context (date/time: 2026-09-12 17:41 UTC+01, GPS 52.393850°N, 0.270830°E), main subject (white swan), setting (river, boats, residential buildings, greenery), action (glides), lighting (calm, late afternoon? Wait, 17:41 is 5:41 PM, so golden hour maybe). Let's put it together: "On 12 September 2026 at 17:41 UTC+01, a solitary white swan glides gracefully across calm river waters at 52.393850°N, 0.270830°E, framed by lush foliage, leisure boats, and riverside residential buildings." That covers date, time, GPS, subject, action, setting, lighting (calm), distinctive details.

Keywords: Need 10-18 unique, comma-separated, covering context and visible details. Let's list from hint: Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming (wait, but the swan is swimming? Wait, the action is glide, but swimming is related. Wait, the hint has "Swimming" but maybe better to use "Waterfowl" instead? Wait, no, the hint has "Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat". Let's pick the most relevant: White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, Residential Buildings, Calm Water, Golden Hour, September 12, 2026, UTC+01, 52.393850°N, 0.270830°E. Wait, but need to be concise and relevant. Let's check the hint again: the descriptive hint mentions solitary white swan, calm river, lush foliage, leisure boats, residential buildings. Keyword hints include Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat. So let's pick the best ones: White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, Residential Buildings, Calm Water, September 12, 2026, UTC+01, 52.393850°N, 0.270830°E. Wait, but maybe more concise. Let's see: White Swan
&lt;/think&gt;

Title: Solitary White Swan Glides on Calm River with Boats and Lush Foliage

Description: On 12 September 2026 at 17:41 UTC+01 (52.393850°N, 0.270830°E), a solitary white swan glides gracefully across calm river waters, framed by lush foliage, leisure boats, and riverside residential buildings.

Keywords: White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, Residential Buildings, Calm Water, September 12, 2026, UTC+01, 52.393850°N, 0.270830°E
</pre>

<details>
<summary>Exact raw output</summary>

```text
Let's tackle this step by step. First, the title needs to be concrete and 5-10 words. The main elements are the swan, river, boats, greenery. Maybe "Solitary White Swan Gliding on Calm River with Boats and Greenery" but need to be concise. Wait, the hint says solitary white swan, calm river, lush foliage, leisure boats. So maybe "Solitary White Swan on Calm River with Boats and Lush Foliage" – that's 9 words, good.

Next, the description. Need to combine context (date/time: 2026-09-12 17:41 UTC+01, GPS 52.393850°N, 0.270830°E), main subject (white swan), setting (river, boats, residential buildings, greenery), action (glides), lighting (calm, late afternoon? Wait, 17:41 is 5:41 PM, so golden hour maybe). Let's put it together: "On 12 September 2026 at 17:41 UTC+01, a solitary white swan glides gracefully across calm river waters at 52.393850°N, 0.270830°E, framed by lush foliage, leisure boats, and riverside residential buildings." That covers date, time, GPS, subject, action, setting, lighting (calm), distinctive details.

Keywords: Need 10-18 unique, comma-separated, covering context and visible details. Let's list from hint: Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming (wait, but the swan is swimming? Wait, the action is glide, but swimming is related. Wait, the hint has "Swimming" but maybe better to use "Waterfowl" instead? Wait, no, the hint has "Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat". Let's pick the most relevant: White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, Residential Buildings, Calm Water, Golden Hour, September 12, 2026, UTC+01, 52.393850°N, 0.270830°E. Wait, but need to be concise and relevant. Let's check the hint again: the descriptive hint mentions solitary white swan, calm river, lush foliage, leisure boats, residential buildings. Keyword hints include Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat. So let's pick the best ones: White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, Residential Buildings, Calm Water, September 12, 2026, UTC+01, 52.393850°N, 0.270830°E. Wait, but maybe more concise. Let's see: White Swan
</think>

Title: Solitary White Swan Glides on Calm River with Boats and Lush Foliage

Description: On 12 September 2026 at 17:41 UTC+01 (52.393850°N, 0.270830°E), a solitary white swan glides gracefully across calm river waters, framed by lush foliage, leisure boats, and riverside residential buildings.

Keywords: White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, Residential Buildings, Calm Water, September 12, 2026, UTC+01, 52.393850°N, 0.270830°E
```

</details>

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
- *Model load time:* 1.49s
- *Generation time:* 7.73s
- *Total time:* 9.76s
- *Prompt preparation time:* 0.532
- *First-token latency:* 5.34
- *Cleanup time:* 0.0933
- *Prompt tokens:* 6,450
- *Generation tokens:* 140
- *Total tokens:* 6,590
- *Prompt throughput (raw):* 1,208 tok/s
- *Generation throughput (raw):* 74.1 tok/s
- *Peak memory (GB):* 8.8
- *Active memory (GB):* 7.1
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 7.07
- *Post-cleanup active memory (GB):* 0.00092
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1413
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
Title: Swan at River Marina
Description: A white swan glides on calm river waters, with leisure boats moored alongside riverside residential buildings in the background, captured on 2026-09-12 at 17:41:02 UTC+01:00 at 52.393850°N, 0.270830°E.
Keywords: Swan, River, Marina, Leisure boats, Residential buildings, Calm waters, White swan, Riverbank, Boats moored, Riverside, 2026-09-12 17:41:02, 52.393850°N, 0.270830°E
</pre>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 2.09s
- *Generation time:* 5.68s
- *Total time:* 8.50s
- *Prompt preparation time:* 0.725
- *First-token latency:* 1.08
- *Cleanup time:* 0.106
- *Prompt tokens:* 2,612
- *Generation tokens:* 138
- *Total tokens:* 2,750
- *Prompt throughput (raw):* 2,411 tok/s
- *Generation throughput (raw):* 35.5 tok/s
- *Peak memory (GB):* 18
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.43
- *Model-load active memory (GB):* 16.9
- *Post-cleanup active memory (GB):* 0.00149
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1369
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
 Title: Solitary Swan on Calm River Waters with Moored Boats and Riverside Buildings.

Description: A solitary white swan glides gracefully across the calm river waters, framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background. The serene and peaceful atmosphere is enhanced by the soft play of light on the water, creating a picturesque scene of natural beauty and human habitation.

Keywords: swan, river, boats, moored, residential buildings, foliage, calm waters, leisure, cruisers, riverside, solitary, white, graceful, serene, peaceful, picturesque, light, water, natural beauty, human habitation.
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
- *Model load time:* 1.15s
- *Generation time:* 4.22s
- *Total time:* 6.02s
- *Prompt preparation time:* 0.64
- *First-token latency:* 1.4
- *Cleanup time:* 0.0997
- *Prompt tokens:* 2,113
- *Generation tokens:* 126
- *Total tokens:* 2,239
- *Prompt throughput (raw):* 1,510 tok/s
- *Generation throughput (raw):* 57.4 tok/s
- *Peak memory (GB):* 10
- *Active memory (GB):* 8.9
- *Cache memory (GB):* 0.51
- *Model-load active memory (GB):* 8.94
- *Post-cleanup active memory (GB):* 0.0018
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1337
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
Title: Swan on Canal with Moored Boats

Description:
A white swan glides on a tranquil canal, surrounded by moored boats and riverside buildings, with lush greenery framing the scene.

Keywords:
Swan, Canal, Boats, Mooring, Residential Buildings, Greenery, Trees, Water Reflection, Motorboat, Pier, Riverbank, Waterway, Aquatic Bird, Architecture, Vegetation, Waterfront, 2026-09-12, 52.393850°N, 0.270830°E
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
- *Model load time:* 1.66s
- *Generation time:* 3.48s
- *Total time:* 5.78s
- *Prompt preparation time:* 0.637
- *First-token latency:* 0.816
- *Cleanup time:* 0.102
- *Prompt tokens:* 2,113
- *Generation tokens:* 77
- *Total tokens:* 2,190
- *Prompt throughput (raw):* 2,590 tok/s
- *Generation throughput (raw):* 37.7 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 15.9
- *Post-cleanup active memory (GB):* 0.00212
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1337
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
Title: Swan on Calm River
Description: A white swan glides on tranquil river waters near a marina with moored boats and residential buildings, captured at dusk.
Keywords: swan, river, marina, boats, residential buildings, foliage, trees, water reflection, architecture, motorboat, mooring, waterfront, aquatic bird, vegetation, calm waters
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Swan on Calm River
Description: A white swan glides on tranquil river waters near a marina with moored boats and residential buildings, captured at dusk. 
Keywords: swan, river, marina, boats, residential buildings, foliage, trees, water reflection, architecture, motorboat, mooring, waterfront, aquatic bird, vegetation, calm waters
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
- *Model load time:* 1.17s
- *Generation time:* 1.51s
- *Total time:* 3.36s
- *Prompt preparation time:* 0.684
- *First-token latency:* 0.468
- *Cleanup time:* 0.0924
- *Prompt tokens:* 2,103
- *Generation tokens:* 81
- *Total tokens:* 2,184
- *Prompt throughput (raw):* 4,492 tok/s
- *Generation throughput (raw):* 214 tok/s
- *Peak memory (GB):* 4.0
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 2.81
- *Post-cleanup active memory (GB):* 0.00303
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1383
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
Title: Swan gliding past moored boats on tranquil riverbank
Description: A white swan swims gracefully on calm river waters, framed by lush greenery, with leisure boats and residential buildings in the background.
Keywords: swan, river, boats, marina, greenery, water, reflection, architecture, leisure, waterfront, aquatic bird, trees, vegetation, canal
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
- *Model load time:* 1.04s
- *Generation time:* 1.68s
- *Total time:* 3.23s
- *Prompt preparation time:* 0.498
- *First-token latency:* 0.239
- *Cleanup time:* 0.0979
- *Prompt tokens:* 391
- *Generation tokens:* 101
- *Total tokens:* 492
- *Prompt throughput (raw):* 1,634 tok/s
- *Generation throughput (raw):* 105 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 6.1
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 6.15
- *Post-cleanup active memory (GB):* 0.00439
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1387
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
Title: Swan Glides on Calm River Near Moored Boats

Description: A white swan swims peacefully in a tranquil waterway, framed by greenery and moored leisure boats, under soft daylight near riverside homes, capturing a serene riverside scene in late summer.

Keywords: swan, river, waterway, mooring, motorboat, boats, greenery, trees, vegetation, waterfront, water reflection, waterfowl, marina, architecture, leisure, aquatic bird
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
- *Model load time:* 1.52s
- *Generation time:* 4.71s
- *Total time:* 6.77s
- *Prompt preparation time:* 0.53
- *First-token latency:* 2.58
- *Cleanup time:* 0.0973
- *Prompt tokens:* 2,926
- *Generation tokens:* 109
- *Total tokens:* 3,035
- *Prompt throughput (raw):* 1,133 tok/s
- *Generation throughput (raw):* 67.2 tok/s
- *Peak memory (GB):* 13
- *Active memory (GB):* 8.0
- *Cache memory (GB):* 0.56
- *Model-load active memory (GB):* 8.01
- *Post-cleanup active memory (GB):* 0.00466
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3770
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
**Swan Gliding Past Moored Boats on a Serene River**

Description:
A solitary white swan swims calmly on a tranquil river, surrounded by reflections of lush greenery and moored leisure boats, with residential buildings and marina infrastructure visible in the background under soft evening light.

Keywords:
swan, river, moored boats, marina, residential waterfront, greenery, calm water, reflections, leisure boats, pier, waterfowl, aquatic bird, trees, vegetation, waterway, evening light
</pre>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-nvfp4</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.57s
- *Generation time:* 5.67s
- *Total time:* 7.77s
- *Prompt preparation time:* 0.523
- *First-token latency:* 2.16
- *Cleanup time:* 0.106
- *Prompt tokens:* 2,926
- *Generation tokens:* 187
- *Total tokens:* 3,113
- *Prompt throughput (raw):* 1,356 tok/s
- *Generation throughput (raw):* 62.3 tok/s
- *Peak memory (GB):* 13
- *Active memory (GB):* 8.4
- *Cache memory (GB):* 0.60
- *Model-load active memory (GB):* 8.43
- *Post-cleanup active memory (GB):* 0.00492
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3770
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 28777b889d841a86369c736175cb77258c8134b2
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ministral-3-14B-Instruct-2512-nvfp4/snapshots/28777b889d841a86369c736175cb77258c8134b2
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
**Title:**
*Elegant Swan Glides Past Moored Boats on Serene River*

**Description:**
A solitary white swan gracefully swims across calm river waters on **September 12, 2026, near 52.393850°N, 0.270830°E (UTC+01:00)**, framed by lush greenery and leisure boats moored alongside riverside residential buildings. Golden late-afternoon sunlight enhances reflections, while distinctive mooring posts and varied boat designs add character to the tranquil waterfront scene.

**Keywords:**
swan, river, leisure boats, moored boats, riverside residential buildings, greenery, golden sunlight, reflections, mooring posts, calm waters, waterfront, aquatic bird, waterway, canal, marina, trees, vegetation, waterfowl, late afternoon, serene river
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
- *Model load time:* 1.14s
- *Generation time:* 2.06s
- *Total time:* 3.75s
- *Prompt preparation time:* 0.539
- *First-token latency:* 1.02
- *Cleanup time:* 0.0855
- *Prompt tokens:* 2,925
- *Generation tokens:* 102
- *Total tokens:* 3,027
- *Prompt throughput (raw):* 2,868 tok/s
- *Generation throughput (raw):* 191 tok/s
- *Peak memory (GB):* 7.8
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.38
- *Model-load active memory (GB):* 2.75
- *Post-cleanup active memory (GB):* 0.00518
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3769
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
Swans by Riverside Marina

Description:
A white swan swims serenely in calm waters at dusk near a marina moored with leisure boats, framed by lush greenery and riverside residential brick buildings under soft, natural lighting.

Keywords:
swan, canal, marina, mooring, motorboat, pier, waterfront, waterway, greenery, residential architecture, boat moorings, aquatic bird, water reflection, trees, vegetation, swan grace, evening lighting
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
- *Model load time:* 0.95s
- *Generation time:* 3.17s
- *Total time:* 4.87s
- *Prompt preparation time:* 0.744
- *First-token latency:* 1.9
- *Cleanup time:* 0.0955
- *Prompt tokens:* 4,083
- *Generation tokens:* 84
- *Total tokens:* 4,167
- *Prompt throughput (raw):* 2,149 tok/s
- *Generation throughput (raw):* 154 tok/s
- *Peak memory (GB):* 3.9
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.65
- *Model-load active memory (GB):* 2.18
- *Post-cleanup active memory (GB):* 0.00643
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1463
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
Title: Swan on the River

Description: A solitary white swan glides gracefully across calm river waters, framed by lush greenery. In the background, leisure boats and cruisers are moored alongside a riverside residential building, reflecting the tranquil waterfront setting.

Keywords: Swan, River, Greenery, Marina, Boat, Waterfront, Water, Aquatic Bird, Architecture, Leisure Boats, Foliage, Reflection, Swan on the River
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
- *Model load time:* 2.61s
- *Generation time:* 3.45s
- *Total time:* 6.73s
- *Prompt preparation time:* 0.664
- *First-token latency:* 0.66
- *Cleanup time:* 0.139
- *Prompt tokens:* 1,289
- *Generation tokens:* 155
- *Total tokens:* 1,444
- *Prompt throughput (raw):* 1,953 tok/s
- *Generation throughput (raw):* 71.8 tok/s
- *Peak memory (GB):* 24
- *Active memory (GB):* 23
- *Cache memory (GB):* 0.14
- *Model-load active memory (GB):* 23.1
- *Post-cleanup active memory (GB):* 0.00693
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1423
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
Title: White Swan Glides Past Moored Boats On Calm River

Description: A solitary white swan glades across a calm river in the foreground, its reflection rippling in the water, while a collection of motorboats and cruisers—some with red and blue canvas covers—are moored along a wooden pontoon. Behind the marina, brick residential buildings with white-framed windows and balconies line the riverbank, framed overhanging tree branches and lush green foliage.

Keywords: white swan, river, moored boats, motorboats, cruisers, marina, pontoon, water reflection, swan, waterfowl, riverbank, residential buildings, trees, greenery, vegetation, mooring, architecture, waterfront, aquatic bird
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
- *Model load time:* 0.97s
- *Generation time:* 2.73s
- *Total time:* 4.21s
- *Prompt preparation time:* 0.502
- *First-token latency:* 0.3
- *Cleanup time:* 0.0931
- *Prompt tokens:* 1,133
- *Generation tokens:* 107
- *Total tokens:* 1,240
- *Prompt throughput (raw):* 3,774 tok/s
- *Generation throughput (raw):* 54.9 tok/s
- *Peak memory (GB):* 9.3
- *Active memory (GB):* 8.3
- *Cache memory (GB):* 0.52
- *Model-load active memory (GB):* 8.3
- *Post-cleanup active memory (GB):* 0.007
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1353
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
Title: Serene Swan on the River

Description: A white swan is seen swimming peacefully on a calm river, with boats and houses in the background, captured on September 12, 2026.

Keywords: swan, river, calm, boats, houses, foliage, waterfront, mooring, leisure, tranquility, architecture, waterway, aquatic bird, greenery, water reflection, motorboat, pier, vegetation
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
- *Model load time:* 0.79s
- *Generation time:* 25.95s
- *Total time:* 27.70s
- *Prompt preparation time:* 0.957
- *First-token latency:* 14.5
- *Cleanup time:* 0.097
- *Prompt tokens:* 16,549
- *Generation tokens:* 912
- *Total tokens:* 17,461
- *Prompt throughput (raw):* 1,140 tok/s
- *Generation throughput (raw):* 86.3 tok/s
- *Peak memory (GB):* 8.4
- *Active memory (GB):* 4.3
- *Cache memory (GB):* 2.7
- *Model-load active memory (GB):* 4.26
- *Post-cleanup active memory (GB):* 0.00762
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1412
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
Got it, let's tackle this. First, the title needs to be 5-10 words. The main elements are the swan, river, boats, residential buildings. So maybe "Solitary Swan on Riverfront Marina" but let's check word count. "Solitary Swan Gliding on Riverfront Marina" – that's 10 words. Good.

Next, the description. Must combine context (capture date, GPS) and the main visible subject, setting, action, lighting, details. Wait, context says capture date/time is 2026-09-12 17:41:02 UTC+01:00, but we can't claim it's visible. The image is from that time. The main subject is a white swan on the water, background has boats, residential buildings, trees. Lighting: maybe overcast since it's a calm day, not too bright. Let's see. The description should include: "Solitary white swan glides across calm river waters in a canal-like waterway, framed by lush greenery and a riverside marina with moored leisure boats and residential buildings in the background. Captured on 2026-09-12 17:41 UTC+01:00, the scene features reflections on the water, a red inflatable boat, and tree branches framing the view."

Wait, but we can't claim the capture date/time is visible, just include it as context. The description should say "Captured on 2026-09-12 17:41 UTC+01:00" as part of the factual description. Let's structure it: "A solitary white swan glides across calm water in a canal with moored boats and residential buildings, framed by lush trees; captured at 2026-09-12 17:41 UTC+01:00."

Wait, the context says "Capture date/time: 2026-09-12 17:41:02 UTC+01:00" so we need to include that. The main visible details: swan, calm water, boats (white, blue, maroon covers), residential buildings (brown, balconies), trees. Also, the water reflection is visible, and the red inflatable boat. Let's make sure the description is factual. "A solitary white swan glides across calm river waters bordered by lush greenery and a marina with leisure boats moored alongside riverside residential buildings. Captured at 2026-09-12 17:41 UTC+01:00, the scene shows water reflections, a red inflatable boat, and tree branches framing the tranquil setting."

Now, keywords. Need 10-18 unique, comma-separated. From context: Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture. Also, the specific boats (but keywords are general). Let's pick: Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway. Wait, need to make sure they're unique. Let's list:

1. Canal
2. Riverbank
3. Marina
4. Mooring
5. Motorboat
6. Pier
7. Boat
8. Architecture
9. Waterway
10. Swimming
11. Trees
12. Vegetation
13. Water reflection
14. Waterfowl
15. Waterfront

&lt;/think&gt;

Title: Solitary Swan on Riverfront Marina
Description: A solitary white swan glides across calm river waters in a canal with moored leisure boats and residential buildings, framed by lush trees; captured at 2026-09-12 17:41 UTC+01:00.
Keywords: Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this. First, the title needs to be 5-10 words. The main elements are the swan, river, boats, residential buildings. So maybe "Solitary Swan on Riverfront Marina" but let's check word count. "Solitary Swan Gliding on Riverfront Marina" – that's 10 words. Good.

Next, the description. Must combine context (capture date, GPS) and the main visible subject, setting, action, lighting, details. Wait, context says capture date/time is 2026-09-12 17:41:02 UTC+01:00, but we can't claim it's visible. The image is from that time. The main subject is a white swan on the water, background has boats, residential buildings, trees. Lighting: maybe overcast since it's a calm day, not too bright. Let's see. The description should include: "Solitary white swan glides across calm river waters in a canal-like waterway, framed by lush greenery and a riverside marina with moored leisure boats and residential buildings in the background. Captured on 2026-09-12 17:41 UTC+01:00, the scene features reflections on the water, a red inflatable boat, and tree branches framing the view."

Wait, but we can't claim the capture date/time is visible, just include it as context. The description should say "Captured on 2026-09-12 17:41 UTC+01:00" as part of the factual description. Let's structure it: "A solitary white swan glides across calm water in a canal with moored boats and residential buildings, framed by lush trees; captured at 2026-09-12 17:41 UTC+01:00."

Wait, the context says "Capture date/time: 2026-09-12 17:41:02 UTC+01:00" so we need to include that. The main visible details: swan, calm water, boats (white, blue, maroon covers), residential buildings (brown, balconies), trees. Also, the water reflection is visible, and the red inflatable boat. Let's make sure the description is factual. "A solitary white swan glides across calm river waters bordered by lush greenery and a marina with leisure boats moored alongside riverside residential buildings. Captured at 2026-09-12 17:41 UTC+01:00, the scene shows water reflections, a red inflatable boat, and tree branches framing the tranquil setting."

Now, keywords. Need 10-18 unique, comma-separated. From context: Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture. Also, the specific boats (but keywords are general). Let's pick: Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway. Wait, need to make sure they're unique. Let's list:

1. Canal
2. Riverbank
3. Marina
4. Mooring
5. Motorboat
6. Pier
7. Boat
8. Architecture
9. Waterway
10. Swimming
11. Trees
12. Vegetation
13. Water reflection
14. Waterfowl
15. Waterfront

</think>

Title: Solitary Swan on Riverfront Marina
Description: A solitary white swan glides across calm river waters in a canal with moored leisure boats and residential buildings, framed by lush trees; captured at 2026-09-12 17:41 UTC+01:00.
Keywords: Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront
```

</details>

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
- *Model load time:* 1.98s
- *Generation time:* 74.51s
- *Total time:* 77.47s
- *Prompt preparation time:* 0.97
- *First-token latency:* 64.9
- *Cleanup time:* 0.138
- *Prompt tokens:* 16,547
- *Generation tokens:* 177
- *Total tokens:* 16,724
- *Prompt throughput (raw):* 255 tok/s
- *Generation throughput (raw):* 20.4 tok/s
- *Peak memory (GB):* 26
- *Active memory (GB):* 20
- *Cache memory (GB):* 5.5
- *Model-load active memory (GB):* 19.6
- *Post-cleanup active memory (GB):* 0.00824
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1404
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
Title: Swan Gliding on River Near Moored Boats

Description: A solitary white swan glides across calm river waters, framed by overhanging green foliage, with several leisure boats and cruisers moored at a riverside pier in front of residential buildings. The scene, captured on 2026-09-12 at 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E, reflects a tranquil riverside setting with visible water reflections and lush vegetation.

Keywords: swan, river, moored boats, leisure craft, residential buildings, greenery, water reflection, foliage, pier, canal, waterfront, waterway, aquatic bird, motorboat, marina, trees, vegetation, architecture
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
- *Model load time:* 1.45s
- *Generation time:* 37.06s
- *Total time:* 39.47s
- *Prompt preparation time:* 0.96
- *First-token latency:* 34.9
- *Cleanup time:* 0.106
- *Prompt tokens:* 16,547
- *Generation tokens:* 93
- *Total tokens:* 16,640
- *Prompt throughput (raw):* 475 tok/s
- *Generation throughput (raw):* 69.7 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 3.4
- *Model-load active memory (GB):* 5.77
- *Post-cleanup active memory (GB):* 0.00856
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1404
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
Title: Swan on River with Moored Boats

Description: A solitary white swan glides calmly on a river, framed by green foliage, with moored leisure boats and residential buildings visible in the background under soft, natural light.

Keywords: swan, river, boats, mooring, canal, greenery, trees, water reflection, waterfront, residential, motorboat, pier, waterway, vegetation, architecture, aquatic bird, waterfowl
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
- *Generation time:* 35.97s
- *Total time:* 39.73s
- *Prompt preparation time:* 0.986
- *First-token latency:* 33.7
- *Cleanup time:* 0.145
- *Prompt tokens:* 16,563
- *Generation tokens:* 97
- *Total tokens:* 16,660
- *Prompt throughput (raw):* 491 tok/s
- *Generation throughput (raw):* 72.4 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.89
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.00906
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1423
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
Title: Swan Swimming Past Moored Boats

Description: A solitary white swan glides across calm river waters near a riverside mooring of leisure cruisers and residential buildings, framed by lush foliage and dappled natural light.

Keywords: swan, river, moored boats, riverbank, vegetation, water reflection, cruiser, residential buildings, trees, leisure boats, waterfront, marina, wildlife, calm water, UK landscape, mooring, architecture, nature
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
- *Model load time:* 1.63s
- *Generation time:* 41.57s
- *Total time:* 44.16s
- *Prompt preparation time:* 0.946
- *First-token latency:* 39.8
- *Cleanup time:* 0.112
- *Prompt tokens:* 16,563
- *Generation tokens:* 86
- *Total tokens:* 16,649
- *Prompt throughput (raw):* 416 tok/s
- *Generation throughput (raw):* 91.4 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 6.0
- *Cache memory (GB):* 1.2
- *Model-load active memory (GB):* 5.96
- *Post-cleanup active memory (GB):* 0.00959
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1423
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
Title: Swan Glides Past Moored Boats on Calm Riverbank

Description: A solitary white swan swims peacefully on a reflective river, with leisure boats moored along the bank beside residential buildings under soft daylight.

Keywords: swan, river, boats, mooring, residential buildings, water reflection, foliage, trees, motorboat, canal, waterfront, waterway, aquatic bird, architecture, greenery
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
- *Model load time:* 12.73s
- *Generation time:* 23.05s
- *Total time:* 36.64s
- *Prompt preparation time:* 0.847
- *First-token latency:* 19
- *Cleanup time:* 0.313
- *Prompt tokens:* 3,491
- *Generation tokens:* 121
- *Total tokens:* 3,612
- *Prompt throughput (raw):* 183 tok/s
- *Generation throughput (raw):* 47.4 tok/s
- *Peak memory (GB):* 92
- *Active memory (GB):* 85
- *Cache memory (GB):* 0.41
- *Model-load active memory (GB):* 85
- *Post-cleanup active memory (GB):* 0.0106
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1410
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
Swan glides past moored boats on a calm river
Description:
A solitary white swan swims across calm river waters, framed by lush foreground foliage and a background of moored leisure boats alongside brick residential buildings with balconies. The scene is lit by soft, overcast daylight, with reflections of the boats and trees visible on the water's surface.
Keywords:
Swan, waterfowl, river, marina, mooring, motorboat, pier, waterfront, waterway, canal, bird, swimming, trees, vegetation, greenery, architecture, boat, water reflection
</pre>

</details>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### mlx-community/X-Reasoner-7B-8bit

<details>
<summary>Complete evidence: mlx-community/X-Reasoner-7B-8bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.21s
- *Generation time:* 16.31s
- *Total time:* 18.44s
- *Prompt preparation time:* 0.921
- *First-token latency:* 13.1
- *Cleanup time:* 0.106
- *Prompt tokens:* 16,558
- *Generation tokens:* 136
- *Total tokens:* 16,694
- *Prompt throughput (raw):* 1,263 tok/s
- *Generation throughput (raw):* 57.6 tok/s
- *Peak memory (GB):* 14
- *Active memory (GB):* 9.5
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 9.46
- *Post-cleanup active memory (GB):* 0.0109
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1462
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
Swan Gliding on Calm River with Moored Boats

Description:
A serene scene captures a solitary white swan gracefully swimming across calm river waters, framed by lush greenery and leisure boats moored alongside riverside residential buildings. The setting is illuminated by soft, natural light, highlighting the tranquil atmosphere of the waterfront.

Keywords:
Swan, River, Calm, Moored Boats, Residential Buildings, Greenery, Waterfront, Waterway, Motorboat, Pier, Marina, Reflection, Architecture, Canals, Waterfowl, Trees, Vegetation, Riverbank, Swimming, Leisure, Reflections, Soft Lighting, Natural Setting.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Swan Gliding on Calm River with Moored Boats

Description:
A serene scene captures a solitary white swan gracefully swimming across calm river waters, framed by lush greenery and leisure boats moored alongside riverside residential buildings. The setting is illuminated by soft, natural light, highlighting the tranquil atmosphere of the waterfront.

Keywords: 
Swan, River, Calm, Moored Boats, Residential Buildings, Greenery, Waterfront, Waterway, Motorboat, Pier, Marina, Reflection, Architecture, Canals, Waterfowl, Trees, Vegetation, Riverbank, Swimming, Leisure, Reflections, Soft Lighting, Natural Setting.
```

</details>

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
- *Model load time:* 1.16s
- *Generation time:* 2.52s
- *Total time:* 4.62s
- *Prompt preparation time:* 0.928
- *First-token latency:* 0.735
- *Cleanup time:* 0.113
- *Prompt tokens:* 2,090
- *Generation tokens:* 96
- *Total tokens:* 2,186
- *Prompt throughput (raw):* 2,842 tok/s
- *Generation throughput (raw):* 100 tok/s
- *Peak memory (GB):* 6.5
- *Active memory (GB):* 5.5
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 5.46
- *Post-cleanup active memory (GB):* 0.0114
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 4270
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
Title: Solitary Swan Glides on Tranquil River
Description: A serene scene captures a graceful white swan effortlessly gliding across the calm waters of a river, framed by lush greenery. Leisure boats and cruisers are moored alongside riverside residential buildings, creating a picturesque setting under the soft lighting of the late afternoon.
Keywords: Swan, River, Greenery, Boats, Cruisers, Buildings, Lighting, Afternoon, Tranquil, Waterfowl, Architecture, Waterway
</pre>

</details>

---

<a id="model-mlx-community-gemma-4-12b-it-4bit"></a>

### mlx-community/gemma-4-12B-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-4-12B-it-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4_unified)
- *Model load time:* 1.86s
- *Generation time:* 2.61s
- *Total time:* 5.10s
- *Prompt preparation time:* 0.62
- *First-token latency:* 0.359
- *Cleanup time:* 0.119
- *Prompt tokens:* 595
- *Generation tokens:* 101
- *Total tokens:* 696
- *Prompt throughput (raw):* 1,659 tok/s
- *Generation throughput (raw):* 60.4 tok/s
- *Peak memory (GB):* 7.6
- *Active memory (GB):* 6.8
- *Cache memory (GB):* 0.32
- *Model-load active memory (GB):* 6.76
- *Post-cleanup active memory (GB):* 0.013
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1386
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
Title:
White Swan Gliding on a River Near Moored Boats

Description:
A solitary white swan glides across a calm river surface reflecting leisure boats and residential buildings. The scene is framed by lush greenery and features multiple motorboats moored along a waterfront pier.

Keywords:
Swan, Waterfowl, River, Canal, Boat, Motorboat, Marina, Mooring, Architecture, Residential, Waterway, Reflection, Nature, Bird, Swimming, Waterfront, Greenery, Trees, Vegetation
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
- *Model load time:* 2.55s
- *Generation time:* 1.88s
- *Total time:* 5.04s
- *Prompt preparation time:* 0.599
- *First-token latency:* 0.416
- *Cleanup time:* 0.144
- *Prompt tokens:* 595
- *Generation tokens:* 96
- *Total tokens:* 691
- *Prompt throughput (raw):* 1,431 tok/s
- *Generation throughput (raw):* 108 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 15.4
- *Post-cleanup active memory (GB):* 0.0136
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1386
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
Title: White swan swimming on calm river with moored boats
Description: A solitary white swan glides across calm river waters near a wooden pier, with several leisure boats and residential buildings visible in the background. The scene is framed by lush green foliage reflecting in the still water under natural daylight.
Keywords: white swan, river, motorboat, waterfowl, waterfront, greenery, reflection, residential building, boat, pier, aquatic bird, leisure boat, vegetation, waterway, mooring
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
- *Model load time:* 2.74s
- *Generation time:* 5.08s
- *Total time:* 8.45s
- *Prompt preparation time:* 0.622
- *First-token latency:* 1.12
- *Cleanup time:* 0.154
- *Prompt tokens:* 595
- *Generation tokens:* 89
- *Total tokens:* 684
- *Prompt throughput (raw):* 529 tok/s
- *Generation throughput (raw):* 26.3 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.79
- *Model-load active memory (GB):* 18.4
- *Post-cleanup active memory (GB):* 0.0141
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1386
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
Title: White Swan Swimming Past Moored River Boats
Description: A single white swan swims across calm river waters in front of leisure boats moored beside riverside residential buildings. The scene is framed by lush green foliage under soft, late afternoon light.
Keywords: swan, river, boats, marina, moored, waterfront, waterway, residential, architecture, greenery, waterfowl, motorboat, reflection, riverbank, England, nature, swimming, leisure
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
- *Model load time:* 1.76s
- *Generation time:* 1.35s
- *Total time:* 3.74s
- *Prompt preparation time:* 0.615
- *First-token latency:* 0.189
- *Cleanup time:* 0.117
- *Prompt tokens:* 591
- *Generation tokens:* 73
- *Total tokens:* 664
- *Prompt throughput (raw):* 3,128 tok/s
- *Generation throughput (raw):* 125 tok/s
- *Peak memory (GB):* 6.0
- *Active memory (GB):* 5.2
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 5.16
- *Post-cleanup active memory (GB):* 0.0146
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1358
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
Title: Swan glides on calm water beneath residential buildings
Description: A solitary white swan gracefully floats on the calm river water, framed by dense greenery and moored leisure boats in the background.
Keywords: swan, waterfowl, river, canal, marina, boats, trees, greenery, quiet, architecture, moored, peaceful, waterfront, vegetation, gliding, pastime
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
- *Model load time:* 0.61s
- *Generation time:* 1.51s
- *Total time:* 2.91s
- *Prompt preparation time:* 0.792
- *First-token latency:* 0.392
- *Cleanup time:* 0.097
- *Prompt tokens:* 1,379
- *Generation tokens:* 65
- *Total tokens:* 1,444
- *Prompt throughput (raw):* 3,519 tok/s
- *Generation throughput (raw):* 176 tok/s
- *Peak memory (GB):* 4.7
- *Active memory (GB):* 3.0
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 3.03
- *Post-cleanup active memory (GB):* 0.0148
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1556
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
Title: "Swan on the Canal"
Description: A white swan glides gracefully across a calm river, framed by lush foliage and moored boats, with a serene residential backdrop.
Keywords: swan, canal, water, foliage, boats, mooring, residential, reflection, serene, waterway
</pre>

</details>

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

<details>
<summary>Complete evidence: mlx-community/GLM-4.6V-nvfp4</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Repeats the prompt's hint instead of describing the image:
  description
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 6.82s
- *Generation time:* 16.15s
- *Total time:* 23.56s
- *Prompt preparation time:* 0.59
- *First-token latency:* 12.4
- *Cleanup time:* 0.268
- *Prompt tokens:* 6,450
- *Generation tokens:* 103
- *Total tokens:* 6,553
- *Prompt throughput (raw):* 520 tok/s
- *Generation throughput (raw):* 41.6 tok/s
- *Peak memory (GB):* 78
- *Active memory (GB):* 62
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 61.9
- *Post-cleanup active memory (GB):* 0.00123
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1413
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
Title: Solitary swan glides on calm river waters
Description: A solitary white swan glides gracefully across calm river waters, framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background under overcast lighting.
Keywords: swan, river, water, foliage, boats, cruisers, residential buildings, moored, overcast, water reflection, waterfowl, waterfront, waterway, aquatic bird, greenery, trees, vegetation
</pre>

</details>

---

<a id="model-mlx-community-molmo2-8b-4bit"></a>

### mlx-community/Molmo2-8B-4bit

<details>
<summary>Complete evidence: mlx-community/Molmo2-8B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Duplicate keywords: london eye souvenir stall, london eye
  souvenir booth, london eye souvenir kiosk
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Model load time:* 1.17s
- *Generation time:* 4.33s
- *Total time:* 6.16s
- *Prompt preparation time:* 0.651
- *First-token latency:* 0.731
- *Cleanup time:* 0.104
- *Prompt tokens:* 1,524
- *Generation tokens:* 216
- *Total tokens:* 1,740
- *Prompt throughput (raw):* 2,084 tok/s
- *Generation throughput (raw):* 72.5 tok/s
- *Peak memory (GB):* 8.6
- *Active memory (GB):* 6.5
- *Cache memory (GB):* 0.31
- *Model-load active memory (GB):* 6.45
- *Post-cleanup active memory (GB):* 0.00549
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1370
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
Title: Serene River Scene with White Swan and Moored Boats

Description: A tranquil riverside view features a solitary white swan gliding across calm water, surrounded by lush greenery and trees. In the background, various leisure boats and cruisers are moored along the riverbank, with residential buildings visible beyond the foliage.

Keywords: swan, river, boats, mooring, canal, greenery, trees, vegetation, water reflection, waterfowl, waterfront, waterway, aquatic bird, architecture, boat, marina, leisure, residential, calm, serene, UK, London, Thames, London Eye, London Bridge, London Underground, London Tower, London Tower Bridge, London Eye Tower, London Eye Pier, London Eye Walkway, London Eye Observation Deck, London Eye Restaurant, London Eye Cafe, London Eye Gift Shop, London Eye Souvenir Shop, London Eye Souvenir Stall, London Eye Souvenir Booth, London Eye Souvenir Kiosk, London Eye Souvenir Stall, London Eye Souvenir Booth, London Eye Souvenir Kiosk
</pre>

</details>

---

<a id="model-mlx-community-qwen2-vl-7b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-7B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-7B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Repeats the prompt's hint instead of describing the image:
  description
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.81s
- *Generation time:* 39.51s
- *Total time:* 41.22s
- *Prompt preparation time:* 0.897
- *First-token latency:* 37.4
- *Cleanup time:* 0.109
- *Prompt tokens:* 16,558
- *Generation tokens:* 117
- *Total tokens:* 16,675
- *Prompt throughput (raw):* 443 tok/s
- *Generation throughput (raw):* 91.3 tok/s
- *Peak memory (GB):* 9.3
- *Active memory (GB):* 4.7
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 4.68
- *Post-cleanup active memory (GB):* 0.00731
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1462
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
Title: Graceful Swan on the River

Description: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.

Keywords: swan, river, calm water, lush foliage, leisure boats, cruisers, riverside buildings, residential area, greenery, water reflection, aquatic bird, nature, tranquility, serene, peaceful, waterfowl, waterway, scenic, picturesque, boating, mooring, waterfront, trees, vegetation, architecture, boat.
</pre>

</details>

---

<a id="model-mlx-community-qwen38-27b-4bit"></a>

### mlx-community/Qwen3.8-27B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.8-27B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Names a place the prompt did not supply: River Thames,
  Lambeth
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.12s
- *Generation time:* 62.16s
- *Total time:* 65.24s
- *Prompt preparation time:* 0.954
- *First-token latency:* 57.1
- *Cleanup time:* 0.133
- *Prompt tokens:* 16,563
- *Generation tokens:* 124
- *Total tokens:* 16,687
- *Prompt throughput (raw):* 290 tok/s
- *Generation throughput (raw):* 29.8 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 16
- *Cache memory (GB):* 1.9
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.0101
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1423
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 10c35caafbb80f7dc6a7a432cdd11af10a6d4818
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Swan swimming near moored boats on the River Thames

Description:
A solitary white swan swims on the calm water of the River Thames at Lambeth, with its reflection visible on the surface. In the background, several leisure boats and cruisers are moored along the riverbank in front of brown residential buildings, framed by overhanging green foliage.

Keywords:
Swan, River Thames, Boat, Waterfowl, Water, Reflection, Mooring, Residential building, Foliage, Greenery, Calm water, Leisure boat, Canopy, Urban river, Nature
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Repeats the prompt's hint instead of describing the image:
  description
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.78s
- *Generation time:* 1.82s
- *Total time:* 3.33s
- *Prompt preparation time:* 0.728
- *First-token latency:* 0.374
- *Cleanup time:* 0.081
- *Prompt tokens:* 1,426
- *Generation tokens:* 98
- *Total tokens:* 1,524
- *Prompt throughput (raw):* 3,817 tok/s
- *Generation throughput (raw):* 131 tok/s
- *Peak memory (GB):* 5.6
- *Active memory (GB):* 4.5
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 4.5
- *Post-cleanup active memory (GB):* 0.0103
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1364
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
 Title: Swan on River
Description: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
Keywords: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat
</pre>

</details>

---

<a id="model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

<details>
<summary>Complete evidence: mlx-community/diffusiongemma-26B-A4B-it-mxfp8</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Duplicate keywords: reflection; Repeats the prompt's hint
  instead of describing the image: description
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.32s
- *Generation time:* 2.85s
- *Total time:* 6.81s
- *Prompt preparation time:* 0.627
- *First-token latency:* 0.333
- *Cleanup time:* 0.151
- *Prompt tokens:* 591
- *Generation tokens:* 80
- *Total tokens:* 671
- *Prompt throughput (raw):* 1,775 tok/s
- *Generation throughput (raw):* 46.1 tok/s
- *Peak memory (GB):* 28
- *Active memory (GB):* 27
- *Cache memory (GB):* 0.01
- *Model-load active memory (GB):* 27.2
- *Post-cleanup active memory (GB):* 0.0114
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1358
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
Title: White Swan Gliding on River Near Moored Boats
Description: A solitary white swan glides across calm river waters, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
Keywords: swan, waterfowl, river, boat, marina, mooring, motorboat, cruiser, pier, waterfront, waterway, greenery, vegetation, architecture, reflection, reflection
</pre>

</details>

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### mlx-community/gemma-3-27b-it-qat-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3-27b-it-qat-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Names a place the prompt did not supply: River Cam
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3)
- *Model load time:* 2.39s
- *Generation time:* 6.49s
- *Total time:* 9.47s
- *Prompt preparation time:* 0.571
- *First-token latency:* 0.933
- *Cleanup time:* 0.151
- *Prompt tokens:* 590
- *Generation tokens:* 152
- *Total tokens:* 742
- *Prompt throughput (raw):* 632 tok/s
- *Generation throughput (raw):* 30.3 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.44
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.012
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1387
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
Title: Swan on River Cam, Cambridge – September 2026
Description: A white swan swims on the River Cam at 52.393850°N, 0.270830°E, captured on 12 September 2026 at 17:41 UTC+01:00; moored leisure boats and residential buildings line the riverbank under overcast conditions. The calm water reflects the boats and surrounding greenery.
Keywords: Swan, River Cam, Cambridge, Waterfowl, Boat, Mooring, Marina, Riverbank, Waterway, Residential building, Trees, Vegetation, Water reflection, Leisure boat, Cruiser, September, Aquatic bird, Overcast sky
</pre>

</details>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### mlx-community/pixtral-12b-8bit

<details>
<summary>Complete evidence: mlx-community/pixtral-12b-8bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Repeats the prompt's hint instead of describing the image:
  description
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.84s
- *Generation time:* 4.71s
- *Total time:* 7.12s
- *Prompt preparation time:* 0.566
- *First-token latency:* 1.65
- *Cleanup time:* 0.126
- *Prompt tokens:* 3,116
- *Generation tokens:* 102
- *Total tokens:* 3,218
- *Prompt throughput (raw):* 1,883 tok/s
- *Generation throughput (raw):* 40.2 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 13
- *Cache memory (GB):* 0.59
- *Model-load active memory (GB):* 13.5
- *Post-cleanup active memory (GB):* 0.0171
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1332
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
### Title:
Swan in River with Moored Boats

### Description:
A solitary white swan gracefully swims across calm river waters, framed by lush foliage, with leisure boats and residential buildings in the background.

### Keywords:
Swan, River, Boats, Marina, Water, Foliage, Residential, Mooring, Waterfront, Waterway, Trees, Vegetation, Reflection, Waterfowl, Architecture, Calm, Leisure, Riverbank
</pre>

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Duplicate keywords: canal, water,
  architecture, waterway, aquatic bird, waterfront; Repeats the prompt's hint
  instead of describing the image: description
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.41s
- *Generation time:* 1.02s
- *Total time:* 2.26s
- *Prompt preparation time:* 0.496
- *First-token latency:* 0.116
- *Cleanup time:* 0.0727
- *Prompt tokens:* 2,112
- *Generation tokens:* 200
- *Total tokens:* 2,312
- *Prompt throughput (raw):* 18,197 tok/s
- *Generation throughput (raw):* 476 tok/s
- *Peak memory (GB):* 1.9
- *Active memory (GB):* 0.90
- *Cache memory (GB):* 0.18
- *Model-load active memory (GB):* 0.897
- *Post-cleanup active memory (GB):* 0.000134
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1383
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
Swans glide across calm river waters in a serene canal setting.

Description:
A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.

Keywords:
Swans, canal, water, serene, leisure, boats, water, architecture, greenery, water, architecture, waterway, aquatic bird, waterfront, waterfront, waterway, aquatic bird, canal, water, architecture, water, waterway, waterway, water, aquatic bird, waterfront, water, waterway, waterway, water, water, waterway, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water
</pre>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-8bit"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-8bit

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-8bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 2.37s
- *Generation time:* 17.25s
- *Total time:* 20.21s
- *Prompt preparation time:* 0.587
- *First-token latency:* 0.727
- *Cleanup time:* 0.119
- *Prompt tokens:* 1,324
- *Generation tokens:* 1,000
- *Total tokens:* 2,324
- *Prompt throughput (raw):* 1,822 tok/s
- *Generation throughput (raw):* 63.0 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 17.8
- *Post-cleanup active memory (GB):* 0.00277
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1519
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
◁think▷Okay, let's tackle this task step by step. First, I need to create a British-English catalogue metadata based on the provided context and image. The key elements from the context are the capture date/time (2026-09-12 17:41:02 UTC+01:00) and GPS coordinates (52.393850°N, 0.270830°E). The descriptive hints mention a white swan, calm river, lush foliage, leisure boats, cruisers, residential buildings, and a pier. The keywords include Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat.

The image shows a white swan swimming in a canal with boats docked nearby, residential buildings in the background, greenery, and foliage framing the scene. The lighting suggests it's daytime with clear conditions. The swan is the main subject, and the boats are moored along the riverbank. The time is in the late afternoon or early evening (UTC+01:00), so the lighting might be softer, but the image doesn't show the sun's position directly, so the description should focus on what's visible.

Now, the title needs to be a concrete 5-10 words. Maybe something like "Swan on canal with moored boats at marina". That's concise and captures the main elements.

For the description, combine the context (date/time, GPS) with the main subject (swan), setting (canal, marina), and details (leisure boats, residential buildings, greenery). So: "A white swan glides across a calm canal, surrounded by moored motorboats and residential buildings with lush greenery, at 2026-09-12 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E."

For keywords, list 10-18 unique terms. Let's include the main elements: Swans, Canal, Motorboat, Marina, Residential buildings, Greenery, Water reflection, Leisure boats, Pier, UTC+01:00, UTC-1:00 (since the timestamp is UTC+01:00, but maybe the location's time is different? Wait, the GPS coordinates are in UTC+01:00, but the location's local time is not specified. The keywords should be descriptive of what's visible. So maybe include "Water reflection", "Leisure boats", "Residental buildings", "Greenery", "Motorboat", "Pier", "Swimming swan", "UTC+01:00", "UTC-1:00" (since the capture time is UTC+01:00, but the location's time zone? Maybe not. Alternatively, just use the time as provided. Wait, the user said to treat capture date/time and GPS as authoritative but not claim they're visible. So maybe the keywords shouldn't include the time zone unless visible. The image might not show the time, so perhaps omit the time zones. Wait, the keywords provided in the context include "UTC+01:00" and "UTC-1:00" as hints. But the image might not show the time, so maybe exclude those. Let me check the hints again. The keyword hints include "UTC+01:00" and "UTC-1:00" as hints. But the task says to retain details supported by the image. If the image doesn't show the time, then maybe don't include the time zones. The context says to treat capture date/time and GPS as authoritative but not claim they're visible. So the time zones might not be visible. So maybe exclude UTC+01:00 and UTC-1:00 from keywords. Instead, focus on visible elements: Swans, Canal, Motorboat, Marina, Residential buildings, Greenery, Water reflection, Leisure boats, Pier, Trees, Vegetation. Wait, the hints include "Trees" and "Vegetation". So maybe include those. Also, "Waterfowl" is another keyword hint. So maybe: Swans, Canal, Motorboat, Marina, Residential buildings, Greenery, Water reflection, Leisure boats, Pier, Trees, Vegetation, Waterfowl. That's 10 keywords. Let me check if they are unique and cover the scene. Yes, that works.◁/think▷Title: Swan on canal with moored boats at marina
Description: A white swan glides across a calm canal, surrounded by moored motorboats and residential buildings with lush greenery, at 2026-09-12 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E.
Keywords: Swans, Canal, Motorboat,
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this task step by step. First, I need to create a British-English catalogue metadata based on the provided context and image. The key elements from the context are the capture date/time (2026-09-12 17:41:02 UTC+01:00) and GPS coordinates (52.393850°N, 0.270830°E). The descriptive hints mention a white swan, calm river, lush foliage, leisure boats, cruisers, residential buildings, and a pier. The keywords include Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat.

The image shows a white swan swimming in a canal with boats docked nearby, residential buildings in the background, greenery, and foliage framing the scene. The lighting suggests it's daytime with clear conditions. The swan is the main subject, and the boats are moored along the riverbank. The time is in the late afternoon or early evening (UTC+01:00), so the lighting might be softer, but the image doesn't show the sun's position directly, so the description should focus on what's visible.

Now, the title needs to be a concrete 5-10 words. Maybe something like "Swan on canal with moored boats at marina". That's concise and captures the main elements.

For the description, combine the context (date/time, GPS) with the main subject (swan), setting (canal, marina), and details (leisure boats, residential buildings, greenery). So: "A white swan glides across a calm canal, surrounded by moored motorboats and residential buildings with lush greenery, at 2026-09-12 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E."

For keywords, list 10-18 unique terms. Let's include the main elements: Swans, Canal, Motorboat, Marina, Residential buildings, Greenery, Water reflection, Leisure boats, Pier, UTC+01:00, UTC-1:00 (since the timestamp is UTC+01:00, but maybe the location's time is different? Wait, the GPS coordinates are in UTC+01:00, but the location's local time is not specified. The keywords should be descriptive of what's visible. So maybe include "Water reflection", "Leisure boats", "Residental buildings", "Greenery", "Motorboat", "Pier", "Swimming swan", "UTC+01:00", "UTC-1:00" (since the capture time is UTC+01:00, but the location's time zone? Maybe not. Alternatively, just use the time as provided. Wait, the user said to treat capture date/time and GPS as authoritative but not claim they're visible. So maybe the keywords shouldn't include the time zone unless visible. The image might not show the time, so perhaps omit the time zones. Wait, the keywords provided in the context include "UTC+01:00" and "UTC-1:00" as hints. But the image might not show the time, so maybe exclude those. Let me check the hints again. The keyword hints include "UTC+01:00" and "UTC-1:00" as hints. But the task says to retain details supported by the image. If the image doesn't show the time, then maybe don't include the time zones. The context says to treat capture date/time and GPS as authoritative but not claim they're visible. So the time zones might not be visible. So maybe exclude UTC+01:00 and UTC-1:00 from keywords. Instead, focus on visible elements: Swans, Canal, Motorboat, Marina, Residential buildings, Greenery, Water reflection, Leisure boats, Pier, Trees, Vegetation. Wait, the hints include "Trees" and "Vegetation". So maybe include those. Also, "Waterfowl" is another keyword hint. So maybe: Swans, Canal, Motorboat, Marina, Residential buildings, Greenery, Water reflection, Leisure boats, Pier, Trees, Vegetation, Waterfowl. That's 10 keywords. Let me check if they are unique and cover the scene. Yes, that works.◁/think▷Title: Swan on canal with moored boats at marina  
Description: A white swan glides across a calm canal, surrounded by moored motorboats and residential buildings with lush greenery, at 2026-09-12 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E.  
Keywords: Swans, Canal, Motorboat,
```

</details>

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-4bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Duplicate keywords: riverside, riverside
  scenery, riverside beauty, riverside charm, riverside tranquility, riverside
  serenity, riverside calmness, riverside peacefulness
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.36s
- *Generation time:* 12.72s
- *Total time:* 14.66s
- *Prompt preparation time:* 0.567
- *First-token latency:* 1.47
- *Cleanup time:* 0.111
- *Prompt tokens:* 301
- *Generation tokens:* 275
- *Total tokens:* 576
- *Prompt throughput (raw):* 205 tok/s
- *Generation throughput (raw):* 25.7 tok/s
- *Peak memory (GB):* 9.8
- *Active memory (GB):* 6.0
- *Cache memory (GB):* 0.95
- *Model-load active memory (GB):* 6.02
- *Post-cleanup active memory (GB):* 0.00356
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1436
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 82f31be9840fa0d4c7e99257fe2e28b59a46df97
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-11B-Vision-Instruct-4bit/snapshots/82f31be9840fa0d4c7e99257fe2e28b59a46df97
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|eot_id\|>

*Readable output:*

<pre class="model-output-readable">
Title: Serene River Scene with a Majestic White Swan

Description: A tranquil river scene unfolds with a solitary white swan gliding effortlessly across the calm waters, surrounded by lush foliage and the serene atmosphere of a riverside residential area, where leisure boats and cruisers are moored, creating a picturesque scene with a sense of peace and harmony.

Keywords: riverside residential area, leisure boats, calm waters, serene atmosphere, lush foliage, white swan, riverside, moored boats, peaceful scene, picturesque scene, tranquil scene, riverside scene, riverside life, riverside living, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside
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
  the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.83s
- *Generation time:* 55.34s
- *Total time:* 57.79s
- *Prompt preparation time:* 0.608
- *First-token latency:* 1.63
- *Cleanup time:* 0.107
- *Prompt tokens:* 301
- *Generation tokens:* 1,000
- *Total tokens:* 1,301
- *Prompt throughput (raw):* 185 tok/s
- *Generation throughput (raw):* 18.8 tok/s
- *Peak memory (GB):* 15
- *Active memory (GB):* 11
- *Cache memory (GB):* 0.40
- *Model-load active memory (GB):* 11.4
- *Post-cleanup active memory (GB):* 0.00408
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1436
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
Title: Serene River Scene with a Majestic White Swan

Description: A tranquil river scene unfolds on September 12, 2026, at 17:41:02 UTC+01:00, near 52.393850°N, 0.270830°E, featuring a solitary white swan gliding across the calm waters, surrounded by lush greenery and a backdrop of moored boats and riverside residential buildings.

Keywords: River, White Swan, Lush Greenery, Moored Boats, Riversid Residential Buildings, Tranquil Atmosphere, September Day, 2026, 17:41:02 UTC+01:00, 52.393850°N, 0.270830°E, Water Reflection, Waterfront, Waterway, Aquatic Bird, Architecture, Boat, Nature, Wildlife, Scenic View, Peaceful Scene, Calm Waters, Serene Environment, Outdoor Activity, Water Sports, Bird Watching, Wildlife Photography, Nature Photography, Scenic Photography, Landscape Photography, Riverbank, Riverbank Scene, Riverbank View, Riverbank Photography, Riverbank Landscape, Riverbank Nature, Riverbank Wildlife, Riverbank Scenic, Riverbank Peaceful, Riverbank Calm, Riverbank Serene, Riverbank Outdoor, Riverbank Water Sports, Riverbank Bird Watching, Riverbank Wildlife Photography, Riverbank Nature Photography, Riverbank Scenic Photography, Riverbank Landscape Photography, Riverbank Riverbank Scene, Riverbank Riverbank View, Riverbank Riverbank Photography, Riverbank Riverbank Landscape, Riverbank Riverbank Nature, Riverbank Riverbank Wildlife, Riverbank Riverbank Scenic, Riverbank Riverbank Peaceful, Riverbank Riverbank Calm, Riverbank Riverbank Serene, Riverbank Riverbank Outdoor, Riverbank Riverbank Water Sports, Riverbank Riverbank Bird Watching, Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Nature Photography, Riverbank Riverbank Scenic Photography, Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Landscape, Riverbank Riverbank Riverbank Nature, Riverbank Riverbank Riverbank Wildlife, Riverbank Riverbank Riverbank Scenic, Riverbank Riverbank Riverbank Peaceful, Riverbank Riverbank Riverbank Calm, Riverbank Riverbank Riverbank Serene, Riverbank Riverbank Riverbank Outdoor, Riverbank Riverbank Riverbank Water Sports, Riverbank Riverbank Riverbank Bird Watching, Riverbank Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Riverbank Nature Photography, Riverbank Riverbank Riverbank Scenic Photography, Riverbank Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Riverbank Landscape, Riverbank Riverbank Riverbank Riverbank Nature, Riverbank Riverbank Riverbank Riverbank Wildlife, Riverbank Riverbank Riverbank Riverbank Scenic, Riverbank Riverbank Riverbank Riverbank Peaceful, Riverbank Riverbank Riverbank Riverbank Calm, Riverbank Riverbank Riverbank Riverbank Serene, Riverbank Riverbank Riverbank Riverbank Outdoor, Riverbank Riverbank Riverbank Riverbank Water Sports, Riverbank Riverbank Riverbank Riverbank Bird Watching, Riverbank Riverbank Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Riverbank Riverbank Nature Photography, Riverbank Riverbank Riverbank Riverbank Scenic Photography, Riverbank Riverbank Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Landscape, Riverbank Riverbank Riverbank Riverbank Riverbank Nature, Riverbank Riverbank Riverbank Riverbank Riverbank Wildlife, Riverbank Riverbank Riverbank Riverbank Riverbank Scenic, Riverbank Riverbank Riverbank Riverbank Riverbank Peaceful, Riverbank Riverbank Riverbank Riverbank Riverbank Calm, Riverbank Riverbank Riverbank Riverbank Riverbank Serene, Riverbank Riverbank Riverbank Riverbank Riverbank Outdoor, Riverbank Riverbank Riverbank Riverbank Riverbank Water Sports, Riverbank Riverbank Riverbank Riverbank Riverbank Bird Watching, Riverbank Riverbank Riverbank Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Nature Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Scenic Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank Landscape
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
- *Maintainer status:* none
- *Observations:* Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Model load time:* 3.25s
- *Generation time:* 49.12s
- *Total time:* 53.04s
- *Prompt preparation time:* 0.664
- *First-token latency:* 7.3
- *Cleanup time:* 0.134
- *Prompt tokens:* 4,403
- *Generation tokens:* 1,000
- *Total tokens:* 5,403
- *Prompt throughput (raw):* 603 tok/s
- *Generation throughput (raw):* 24.3 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.23
- *Model-load active memory (GB):* 23.9
- *Post-cleanup active memory (GB):* 0.0059
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1570
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
Title: Mute Swan on River with Moored Boats
Description:
Captured on 2026-09-12 17:41:02 BST at 52.393850°N, 0.270830°E, a mute swan swims on calm water that reflects moored motor cruisers and riverside apartments. The scene shows boats including ‘Samar II’ and ‘Magna Carta’ with red and blue covers at a wooden pier beneath a brown-brick apartment block with balconies, framed by overhanging tree branches and foreground foliage in soft daylight.
Keywords: mute swan, river
</pre>

</details>

---

<a id="model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit"></a>

### mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Generation was stopped early after sustained repeated
  output; Duplicate keywords: wildlife, animal, bird
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl_moe)
- *Model load time:* 1.95s
- *Generation time:* 38.39s
- *Total time:* 41.31s
- *Prompt preparation time:* 0.969
- *First-token latency:* 34.9
- *Cleanup time:* 0.137
- *Prompt tokens:* 16,547
- *Generation tokens:* 200
- *Total tokens:* 16,747
- *Prompt throughput (raw):* 475 tok/s
- *Generation throughput (raw):* 74.7 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 4.0
- *Model-load active memory (GB):* 18.3
- *Post-cleanup active memory (GB):* 0.00793
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1404
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
Title: Solitary swan on a calm river with moored boats
Description: A white swan glides across the calm waters of a river, framed by green foliage in the foreground. In the background, a row of moored motorboats and cruisers are docked beside a residential building with balconies, under a soft, overcast sky.
Keywords: swan, river, boats, mooring, motorboat, residential, building, balcony, trees, foliage, greenery, water, reflection, calm, wildlife, waterfowl, aquatic bird, architecture, canal, waterfront, marina, pier, waterway, leisure, nature, outdoor, serene, tranquil, peaceful, dusk, evening, overcast, cloudy, landscape, scene, view, perspective, foreground, background, composition, natural, wildlife, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal,
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
- *Model load time:* 0.34s
- *Generation time:* 1.02s
- *Total time:* 2.21s
- *Prompt preparation time:* 0.84
- *First-token latency:* 0.122
- *Cleanup time:* 0.0747
- *Prompt tokens:* 1,205
- *Generation tokens:* 33
- *Total tokens:* 1,238
- *Prompt throughput (raw):* 9,864 tok/s
- *Generation throughput (raw):* 320 tok/s
- *Peak memory (GB):* 1.1
- *Active memory (GB):* 0.16
- *Cache memory (GB):* 0.10
- *Model-load active memory (GB):* 0.16
- *Post-cleanup active memory (GB):* 0.0102
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1364
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
 A solitary white swan gliding across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
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
- *Model load time:* 1.78s
- *Generation time:* 2.07s
- *Total time:* 4.43s
- *Prompt preparation time:* 0.551
- *First-token latency:* 0.302
- *Cleanup time:* 0.126
- *Prompt tokens:* 589
- *Generation tokens:* 88
- *Total tokens:* 677
- *Prompt throughput (raw):* 1,952 tok/s
- *Generation throughput (raw):* 71.0 tok/s
- *Peak memory (GB):* 7.2
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 5.83
- *Post-cleanup active memory (GB):* 0.0125
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1389
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
A serene scene unfolds on a calm river, where a graceful swan glides across the water. Several boats, including motorboats and sailboats, are moored along the riverbank, nestled beneath a row of residential buildings. Lush greenery frames the scene, with trees and bushes lining the riverbanks, adding a touch of nature to the urban setting. The overall atmosphere is peaceful and tranquil, capturing a moment of calm in a waterside community.
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
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Unrecognised model control tokens remain
  visible; Required labelled fields not detected: title, description, keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Model load time:* 1.12s
- *Generation time:* 3.47s
- *Total time:* 5.26s
- *Prompt preparation time:* 0.665
- *First-token latency:* 0.877
- *Cleanup time:* 0.169
- *Prompt tokens:* 2,195
- *Generation tokens:* 200
- *Total tokens:* 2,395
- *Prompt throughput (raw):* 2,504 tok/s
- *Generation throughput (raw):* 102 tok/s
- *Peak memory (GB):* 6.7
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.70
- *Model-load active memory (GB):* 5.7
- *Post-cleanup active memory (GB):* 0.0152
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1591
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
&lt;|channel|&gt; analysis&lt;|message|&gt; The image shows a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel|> analysis<|message|> The image shows a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white
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
- *Model load time:* 0.59s
- *Generation time:* 1.14s
- *Total time:* 2.34s
- *Prompt preparation time:* 0.607
- *First-token latency:* 0.0877
- *Cleanup time:* 0.106
- *Prompt tokens:* 330
- *Generation tokens:* 92
- *Total tokens:* 422
- *Prompt throughput (raw):* 3,762 tok/s
- *Generation throughput (raw):* 193 tok/s
- *Peak memory (GB):* 1.8
- *Active memory (GB):* 0.61
- *Cache memory (GB):* 0.28
- *Model-load active memory (GB):* 0.614
- *Post-cleanup active memory (GB):* 0.0158
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1418
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
"Amidst the Riverbank, a solitary white swan glides gracefully across calm waters, framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background. The serene setting is complemented by the presence of greenery, architecture, and waterfowl, all under the watchful gaze of the canal. The image captures the essence of a peaceful riverside community, where nature and human life intersect."
</pre>

</details>

---

<a id="model-mlx-community-paligemma2-10b-mix-448-4bit"></a>

### mlx-community/paligemma2-10b-mix-448-4bit

<details>
<summary>Complete evidence: mlx-community/paligemma2-10b-mix-448-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: title, description,
  keywords
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Model load time:* 1.62s
- *Generation time:* 1.70s
- *Total time:* 4.02s
- *Prompt preparation time:* 0.688
- *First-token latency:* 0.911
- *Cleanup time:* 0.137
- *Prompt tokens:* 1,344
- *Generation tokens:* 7
- *Total tokens:* 1,351
- *Prompt throughput (raw):* 1,475 tok/s
- *Generation throughput (raw):* 36.5 tok/s
- *Peak memory (GB):* 9.7
- *Active memory (GB):* 6.0
- *Cache memory (GB):* 1.8
- *Model-load active memory (GB):* 6.05
- *Post-cleanup active memory (GB):* 0.0169
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1318
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c5f440f2179e48eb7c3266b874ec3625869c3620
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--paligemma2-10b-mix-448-4bit/snapshots/c5f440f2179e48eb7c3266b874ec3625869c3620
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
- title, description, keywords
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
- *Model load time:* 0.16s
- *Generation time:* -
- *Total time:* 0.16s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0649
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00212
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14457, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13369, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 826, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15585, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14472, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 14 files:   0%|          | 0/14 [00:00<?, ?it/s]
Fetching 14 files: 100%|##########| 14/14 [00:00<00:00, 4711.94it/s]
ERROR:root:Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
```

</details>

---

<a id="model-mlx-community-mage-vl-optiq-4bit"></a>

### mlx-community/Mage-VL-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Mage-VL-OptiQ-4bit</summary>

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* generation_before_first_token
- *Error stage:* Model Error
- *Error code:* MLX_VLM_GENERATION_BEFORE_FIRST_TOKEN_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model generation failed for
  mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600
  calculated=3800 grid=[(1, 50, 76)]
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* cu_seqlens mismatch: total_patches=7600
  calculated=3800 grid=[(1, 50, 76)]
- *Arch supported by installed mlx-vlm:* yes (model_type mage_vl)
- *Model load time:* 0.84s
- *Generation time:* 0.64s
- *Total time:* 2.13s
- *Prompt preparation time:* 0.646
- *First-token latency:* -
- *Cleanup time:* 0.0932
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00408
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1462
- *Processor:* mlx_vlm.models.mage_vl.processing_mage_vl.MageVLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* c98dad5f92f13334cc679c93fb9f185ab49ed626
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Mage-VL-OptiQ-4bit/snapshots/c98dad5f92f13334cc679c93fb9f185ab49ed626
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13703, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14380, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<5 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14275, in _generate_with_repetition_guard
    for chunk in stream_generate(
                 ~~~~~~~~~~~~~~~^
        model=model, processor=processor, prompt=prompt, image=image, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1083, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 434, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/mage_vl.py", line 66, in get_input_embeddings
    hidden_states = self.vision_tower(
        pixel_values.astype(inputs_embeds.dtype), patch_positions, grid_thw
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 381, in __call__
    cu = build_cu_seqlens(grid_thw, total_patches, self.config.frame_windows_size)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 309, in build_cu_seqlens
    raise ValueError(
        f"cu_seqlens mismatch: total_patches={total_patches} calculated={cu[-1]} grid={grid_thw}"
    )
ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15585, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14484, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14400, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13712, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 3074.10it/s]
[23:33:20] Generation error for mlx-community/Mage-VL-OptiQ-4bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13703, in
           _run_generation_guarded
               return generate_once()
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14380, in
           _generate_once
               return _generate_with_repetition_guard(
                   model=prepared.model,
               ...<5 lines>...
                   **prepared.generate_kwargs,
               )
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14275, in
           _generate_with_repetition_guard
               for chunk in stream_generate(
                            ~~~~~~~~~~~~~~~^
                   model=model, processor=processor, prompt=prompt, image=image, **kwargs
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               ):
               ^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1083, in
           stream_generate
               for n, (token, logprobs) in enumerate(gen):
                                           ~~~~~~~~~^^^^^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 434, in
           generate_step
               embedding_output = model.get_input_embeddings(
                   input_ids, pixel_values, mask=mask, **kwargs
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/mage_vl.py", line 66,
           in get_input_embeddings
               hidden_states = self.vision_tower(
                   pixel_values.astype(inputs_embeds.dtype), patch_positions, grid_thw
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 381,
           in __call__
               cu = build_cu_seqlens(grid_thw, total_patches, self.config.frame_windows_size)
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 309,
           in build_cu_seqlens
               raise ValueError(
                   f"cu_seqlens mismatch: total_patches={total_patches} calculated={cu[-1]}
           grid={grid_thw}"
               )
           ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]
```

</details>

---

<a id="model-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit"></a>

### mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit</summary>

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Model Error
- *Error code:* MLX_VLM_MODEL_LOAD_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model loading failed: Received 729 parameters not in model:
  <br>backbone.embeddings.biases,<br>backbone.embeddings.scales,<br>backbone.embeddings.weight,<br>backbone.layers.0.mixer.A_log,<br>backbone.layers.0.mixer.D,<br>backbone.layers.0.mixer.conv1d.bias,<br>backbone.layers.0.mixer.conv1d.weight,<br>backbone.layers.0.mixer.dt_bias,<br>backbone.layers.0.mixer.in_proj.biases,<br>backbone.layers.0.mixer.in_proj.scales,<br>backbone.layers.0.mixer.in_proj.weight,<br>backbone.layers.0.mixer.norm.weight,<br>backbone.layers.0.mixer.out_proj.biases,<br>backbone.layers.0.mixer.out_proj.scales,<br>backbone.layers.0.mixer.out_proj.weight,<br>backbone.layers.0.norm.weight,<br>backbone.layers.1.mixer.gate.e_score_correction_bias,<br>backbone.layers.1.mixer.gate.weight,<br>backbone.layers.1.mixer.shared_experts.down_proj.biases,<br>backbone.layers.1.mixer.shared_experts.down_proj.scales,<br>backbone.layers.1.mixer.shared_experts.down_proj.weight,<br>backbone.layers.1.mixer.shared_experts.up_proj.biases,<br>backbone.layers.1.mixer.shared_experts.up_proj.scales,<br>backbone.layers.1.mixer.shared_experts.up_proj.weight,<br>backbone.layers.1.mixer.switch_mlp.fc1.biases,<br>backbone.layers.1.mixer.switch_mlp.fc1.scales,<br>backbone.layers.1.mixer.switch_mlp.fc1.weight,<br>backbone.layers.1.mixer.switch_mlp.fc2.biases,<br>backbone.layers.1.mixer.switch_mlp.fc2.scales,<br>backbone.layers.1.mixer.switch_mlp.fc2.weight,<br>backbone.layers.1.norm.weight,<br>backbone.layers.10.mixer.gate.e_score_correction_bias,<br>backbone.layers.10.mixer.gate.weight,<br>backbone.layers.10.mixer.shared_experts.down_proj.biases,<br>backbone.layers.10.mixer.shared_experts.down_proj.scales,<br>backbone.layers.10.mixer.shared_experts.down_proj.weight,<br>backbone.layers.10.mixer.shared_experts.up_proj.biases,<br>backbone.layers.10.mixer.shared_experts.up_proj.scales,<br>backbone.layers.10.mixer.shared_experts.up_proj.weight,<br>backbone.layers.10.mixer.switch_mlp.fc1.biases,<br>backbone.layers.10.mixer.switch_mlp.fc1.scales,<br>backbone.layers.10.mixer.switch_mlp.fc1.weight,<br>backbone.layers.10.mixer.switch_mlp.fc2.biases,<br>backbone.layers.10.mixer.switch_mlp.fc2.scales,<br>backbone.layers.10.mixer.switch_mlp.fc2.weight,<br>backbone.layers.10.norm.weight,<br>backbone.layers.11.mixer.A_log,<br>backbone.layers.11.mixer.D,<br>backbone.layers.11.mixer.conv1d.bias,<br>backbone.layers.11.mixer.conv1d.weight,<br>backbone.layers.11.mixer.dt_bias,<br>backbone.layers.11.mixer.in_proj.biases,<br>backbone.layers.11.mixer.in_proj.scales,<br>backbone.layers.11.mixer.in_proj.weight,<br>backbone.layers.11.mixer.norm.weight,<br>backbone.layers.11.mixer.out_proj.biases,<br>backbone.layers.11.mixer.out_proj.scales,<br>backbone.layers.11.mixer.out_proj.weight,<br>backbone.layers.11.norm.weight,<br>backbone.layers.12.mixer.k_proj.biases,<br>backbone.layers.12.mixer.k_proj.scales,<br>backbone.layers.12.mixer.k_proj.weight,<br>backbone.layers.12.mixer.o_proj.biases,<br>backbone.layers.12.mixer.o_proj.scales,<br>backbone.layers.12.mixer.o_proj.weight,<br>backbone.layers.12.mixer.q_proj.biases,<br>backbone.layers.12.mixer.q_proj.scales,<br>backbone.layers.12.mixer.q_proj.weight,<br>backbone.layers.12.mixer.v_proj.biases,<br>backbone.layers.12.mixer.v_proj.scales,<br>backbone.layers.12.mixer.v_proj.weight,<br>backbone.layers.12.norm.weight,<br>backbone.layers.13.mixer.gate.e_score_correction_bias,<br>backbone.layers.13.mixer.gate.weight,<br>backbone.layers.13.mixer.shared_experts.down_proj.biases,<br>backbone.layers.13.mixer.shared_experts.down_proj.scales,<br>backbone.layers.13.mixer.shared_experts.down_proj.weight,<br>backbone.layers.13.mixer.shared_experts.up_proj.biases,<br>backbone.layers.13.mixer.shared_experts.up_proj.scales,<br>backbone.layers.13.mixer.shared_experts.up_proj.weight,<br>backbone.layers.13.mixer.switch_mlp.fc1.biases,<br>backbone.layers.13.mixer.switch_mlp.fc1.scales,<br>backbone.layers.13.mixer.switch_mlp.fc1.weight,<br>backbone.layers.13.mixer.switch_mlp.fc2.biases,<br>backbone.layers.13.mixer.switch_mlp.fc2.scales,<br>backbone.layers.13.mixer.switch_mlp.fc2.weight,<br>backbone.layers.13.norm.weight,<br>backbone.layers.14.mixer.A_log,<br>backbone.layers.14.mixer.D,<br>backbone.layers.14.mixer.conv1d.bias,<br>backbone.layers.14.mixer.conv1d.weight,<br>backbone.layers.14.mixer.dt_bias,<br>backbone.layers.14.mixer.in_proj.biases,<br>backbone.layers.14.mixer.in_proj.scales,<br>backbone.layers.14.mixer.in_proj.weight,<br>backbone.layers.14.mixer.norm.weight,<br>backbone.layers.14.mixer.out_proj.biases,<br>backbone.layers.14.mixer.out_proj.scales,<br>backbone.layers.14.mixer.out_proj.weight,<br>backbone.layers.14.norm.weight,<br>backbone.layers.15.mixer.gate.e_score_correction_bias,<br>backbone.layers.15.mixer.gate.weight,<br>backbone.layers.15.mixer.shared_experts.down_proj.biases,<br>backbone.layers.15.mixer.shared_experts.down_proj.scales,<br>backbone.layers.15.mixer.shared_experts.down_proj.weight,<br>backbone.layers.15.mixer.shared_experts.up_proj.biases,<br>backbone.layers.15.mixer.shared_experts.up_proj.scales,<br>backbone.layers.15.mixer.shared_experts.up_proj.weight,<br>backbone.layers.15.mixer.switch_mlp.fc1.biases,<br>backbone.layers.15.mixer.switch_mlp.fc1.scales,<br>backbone.layers.15.mixer.switch_mlp.fc1.weight,<br>backbone.layers.15.mixer.switch_mlp.fc2.biases,<br>backbone.layers.15.mixer.switch_mlp.fc2.scales,<br>backbone.layers.15.mixer.switch_mlp.fc2.weight,<br>backbone.layers.15.norm.weight,<br>backbone.layers.16.mixer.A_log,<br>backbone.layers.16.mixer.D,<br>backbone.layers.16.mixer.conv1d.bias,<br>backbone.layers.16.mixer.conv1d.weight,<br>backbone.layers.16.mixer.dt_bias,<br>backbone.layers.16.mixer.in_proj.biases,<br>backbone.layers.16.mixer.in_proj.scales,<br>backbone.layers.16.mixer.in_proj.weight,<br>backbone.layers.16.mixer.norm.weight,<br>backbone.layers.16.mixer.out_proj.biases,<br>backbone.layers.16.mixer.out_proj.scales,<br>backbone.layers.16.mixer.out_proj.weight,<br>backbone.layers.16.norm.weight,<br>backbone.layers.17.mixer.gate.e_score_correction_bias,<br>backbone.layers.17.mixer.gate.weight,<br>backbone.layers.17.mixer.shared_experts.down_proj.biases,<br>backbone.layers.17.mixer.shared_experts.down_proj.scales,<br>backbone.layers.17.mixer.shared_experts.down_proj.weight,<br>backbone.layers.17.mixer.shared_experts.up_proj.biases,<br>backbone.layers.17.mixer.shared_experts.up_proj.scales,<br>backbone.layers.17.mixer.shared_experts.up_proj.weight,<br>backbone.layers.17.mixer.switch_mlp.fc1.biases,<br>backbone.layers.17.mixer.switch_mlp.fc1.scales,<br>backbone.layers.17.mixer.switch_mlp.fc1.weight,<br>backbone.layers.17.mixer.switch_mlp.fc2.biases,<br>backbone.layers.17.mixer.switch_mlp.fc2.scales,<br>backbone.layers.17.mixer.switch_mlp.fc2.weight,<br>backbone.layers.17.norm.weight,<br>backbone.layers.18.mixer.A_log,<br>backbone.layers.18.mixer.D,<br>backbone.layers.18.mixer.conv1d.bias,<br>backbone.layers.18.mixer.conv1d.weight,<br>backbone.layers.18.mixer.dt_bias,<br>backbone.layers.18.mixer.in_proj.biases,<br>backbone.layers.18.mixer.in_proj.scales,<br>backbone.layers.18.mixer.in_proj.weight,<br>backbone.layers.18.mixer.norm.weight,<br>backbone.layers.18.mixer.out_proj.biases,<br>backbone.layers.18.mixer.out_proj.scales,<br>backbone.layers.18.mixer.out_proj.weight,<br>backbone.layers.18.norm.weight,<br>backbone.layers.19.mixer.k_proj.biases,<br>backbone.layers.19.mixer.k_proj.scales,<br>backbone.layers.19.mixer.k_proj.weight,<br>backbone.layers.19.mixer.o_proj.biases,<br>backbone.layers.19.mixer.o_proj.scales,<br>backbone.layers.19.mixer.o_proj.weight,<br>backbone.layers.19.mixer.q_proj.biases,<br>backbone.layers.19.mixer.q_proj.scales,<br>backbone.layers.19.mixer.q_proj.weight,<br>backbone.layers.19.mixer.v_proj.biases,<br>backbone.layers.19.mixer.v_proj.scales,<br>backbone.layers.19.mixer.v_proj.weight,<br>backbone.layers.19.norm.weight,<br>backbone.layers.2.mixer.A_log,<br>backbone.layers.2.mixer.D,<br>backbone.layers.2.mixer.conv1d.bias,<br>backbone.layers.2.mixer.conv1d.weight,<br>backbone.layers.2.mixer.dt_bias,<br>backbone.layers.2.mixer.in_proj.biases,<br>backbone.layers.2.mixer.in_proj.scales,<br>backbone.layers.2.mixer.in_proj.weight,<br>backbone.layers.2.mixer.norm.weight,<br>backbone.layers.2.mixer.out_proj.biases,<br>backbone.layers.2.mixer.out_proj.scales,<br>backbone.layers.2.mixer.out_proj.weight,<br>backbone.layers.2.norm.weight,<br>backbone.layers.20.mixer.gate.e_score_correction_bias,<br>backbone.layers.20.mixer.gate.weight,<br>backbone.layers.20.mixer.shared_experts.down_proj.biases,<br>backbone.layers.20.mixer.shared_experts.down_proj.scales,<br>backbone.layers.20.mixer.shared_experts.down_proj.weight,<br>backbone.layers.20.mixer.shared_experts.up_proj.biases,<br>backbone.layers.20.mixer.shared_experts.up_proj.scales,<br>backbone.layers.20.mixer.shared_experts.up_proj.weight,<br>backbone.layers.20.mixer.switch_mlp.fc1.biases,<br>backbone.layers.20.mixer.switch_mlp.fc1.scales,<br>backbone.layers.20.mixer.switch_mlp.fc1.weight,<br>backbone.layers.20.mixer.switch_mlp.fc2.biases,<br>backbone.layers.20.mixer.switch_mlp.fc2.scales,<br>backbone.layers.20.mixer.switch_mlp.fc2.weight,<br>backbone.layers.20.norm.weight,<br>backbone.layers.21.mixer.A_log,<br>backbone.layers.21.mixer.D,<br>backbone.layers.21.mixer.conv1d.bias,<br>backbone.layers.21.mixer.conv1d.weight,<br>backbone.layers.21.mixer.dt_bias,<br>backbone.layers.21.mixer.in_proj.biases,<br>backbone.layers.21.mixer.in_proj.scales,<br>backbone.layers.21.mixer.in_proj.weight,<br>backbone.layers.21.mixer.norm.weight,<br>backbone.layers.21.mixer.out_proj.biases,<br>backbone.layers.21.mixer.out_proj.scales,<br>backbone.layers.21.mixer.out_proj.weight,<br>backbone.layers.21.norm.weight,<br>backbone.layers.22.mixer.gate.e_score_correction_bias,<br>backbone.layers.22.mixer.gate.weight,<br>backbone.layers.22.mixer.shared_experts.down_proj.biases,<br>backbone.layers.22.mixer.shared_experts.down_proj.scales,<br>backbone.layers.22.mixer.shared_experts.down_proj.weight,<br>backbone.layers.22.mixer.shared_experts.up_proj.biases,<br>backbone.layers.22.mixer.shared_experts.up_proj.scales,<br>backbone.layers.22.mixer.shared_experts.up_proj.weight,<br>backbone.layers.22.mixer.switch_mlp.fc1.biases,<br>backbone.layers.22.mixer.switch_mlp.fc1.scales,<br>backbone.layers.22.mixer.switch_mlp.fc1.weight,<br>backbone.layers.22.mixer.switch_mlp.fc2.biases,<br>backbone.layers.22.mixer.switch_mlp.fc2.scales,<br>backbone.layers.22.mixer.switch_mlp.fc2.weight,<br>backbone.layers.22.norm.weight,<br>backbone.layers.23.mixer.A_log,<br>backbone.layers.23.mixer.D,<br>backbone.layers.23.mixer.conv1d.bias,<br>backbone.layers.23.mixer.conv1d.weight,<br>backbone.layers.23.mixer.dt_bias,<br>backbone.layers.23.mixer.in_proj.biases,<br>backbone.layers.23.mixer.in_proj.scales,<br>backbone.layers.23.mixer.in_proj.weight,<br>backbone.layers.23.mixer.norm.weight,<br>backbone.layers.23.mixer.out_proj.biases,<br>backbone.layers.23.mixer.out_proj.scales,<br>backbone.layers.23.mixer.out_proj.weight,<br>backbone.layers.23.norm.weight,<br>backbone.layers.24.mixer.gate.e_score_correction_bias,<br>backbone.layers.24.mixer.gate.weight,<br>backbone.layers.24.mixer.shared_experts.down_proj.biases,<br>backbone.layers.24.mixer.shared_experts.down_proj.scales,<br>backbone.layers.24.mixer.shared_experts.down_proj.weight,<br>backbone.layers.24.mixer.shared_experts.up_proj.biases,<br>backbone.layers.24.mixer.shared_experts.up_proj.scales,<br>backbone.layers.24.mixer.shared_experts.up_proj.weight,<br>backbone.layers.24.mixer.switch_mlp.fc1.biases,<br>backbone.layers.24.mixer.switch_mlp.fc1.scales,<br>backbone.layers.24.mixer.switch_mlp.fc1.weight,<br>backbone.layers.24.mixer.switch_mlp.fc2.biases,<br>backbone.layers.24.mixer.switch_mlp.fc2.scales,<br>backbone.layers.24.mixer.switch_mlp.fc2.weight,<br>backbone.layers.24.norm.weight,<br>backbone.layers.25.mixer.A_log,<br>backbone.layers.25.mixer.D,<br>backbone.layers.25.mixer.conv1d.bias,<br>backbone.layers.25.mixer.conv1d.weight,<br>backbone.layers.25.mixer.dt_bias,<br>backbone.layers.25.mixer.in_proj.biases,<br>backbone.layers.25.mixer.in_proj.scales,<br>backbone.layers.25.mixer.in_proj.weight,<br>backbone.layers.25.mixer.norm.weight,<br>backbone.layers.25.mixer.out_proj.biases,<br>backbone.layers.25.mixer.out_proj.scales,<br>backbone.layers.25.mixer.out_proj.weight,<br>backbone.layers.25.norm.weight,<br>backbone.layers.26.mixer.k_proj.biases,<br>backbone.layers.26.mixer.k_proj.scales,<br>backbone.layers.26.mixer.k_proj.weight,<br>backbone.layers.26.mixer.o_proj.biases,<br>backbone.layers.26.mixer.o_proj.scales,<br>backbone.layers.26.mixer.o_proj.weight,<br>backbone.layers.26.mixer.q_proj.biases,<br>backbone.layers.26.mixer.q_proj.scales,<br>backbone.layers.26.mixer.q_proj.weight,<br>backbone.layers.26.mixer.v_proj.biases,<br>backbone.layers.26.mixer.v_proj.scales,<br>backbone.layers.26.mixer.v_proj.weight,<br>backbone.layers.26.norm.weight,<br>backbone.layers.27.mixer.gate.e_score_correction_bias,<br>backbone.layers.27.mixer.gate.weight,<br>backbone.layers.27.mixer.shared_experts.down_proj.biases,<br>backbone.layers.27.mixer.shared_experts.down_proj.scales,<br>backbone.layers.27.mixer.shared_experts.down_proj.weight,<br>backbone.layers.27.mixer.shared_experts.up_proj.biases,<br>backbone.layers.27.mixer.shared_experts.up_proj.scales,<br>backbone.layers.27.mixer.shared_experts.up_proj.weight,<br>backbone.layers.27.mixer.switch_mlp.fc1.biases,<br>backbone.layers.27.mixer.switch_mlp.fc1.scales,<br>backbone.layers.27.mixer.switch_mlp.fc1.weight,<br>backbone.layers.27.mixer.switch_mlp.fc2.biases,<br>backbone.layers.27.mixer.switch_mlp.fc2.scales,<br>backbone.layers.27.mixer.switch_mlp.fc2.weight,<br>backbone.layers.27.norm.weight,<br>backbone.layers.28.mixer.A_log,<br>backbone.layers.28.mixer.D,<br>backbone.layers.28.mixer.conv1d.bias,<br>backbone.layers.28.mixer.conv1d.weight,<br>backbone.layers.28.mixer.dt_bias,<br>backbone.layers.28.mixer.in_proj.biases,<br>backbone.layers.28.mixer.in_proj.scales,<br>backbone.layers.28.mixer.in_proj.weight,<br>backbone.layers.28.mixer.norm.weight,<br>backbone.layers.28.mixer.out_proj.biases,<br>backbone.layers.28.mixer.out_proj.scales,<br>backbone.layers.28.mixer.out_proj.weight,<br>backbone.layers.28.norm.weight,<br>backbone.layers.29.mixer.gate.e_score_correction_bias,<br>backbone.layers.29.mixer.gate.weight,<br>backbone.layers.29.mixer.shared_experts.down_proj.biases,<br>backbone.layers.29.mixer.shared_experts.down_proj.scales,<br>backbone.layers.29.mixer.shared_experts.down_proj.weight,<br>backbone.layers.29.mixer.shared_experts.up_proj.biases,<br>backbone.layers.29.mixer.shared_experts.up_proj.scales,<br>backbone.layers.29.mixer.shared_experts.up_proj.weight,<br>backbone.layers.29.mixer.switch_mlp.fc1.biases,<br>backbone.layers.29.mixer.switch_mlp.fc1.scales,<br>backbone.layers.29.mixer.switch_mlp.fc1.weight,<br>backbone.layers.29.mixer.switch_mlp.fc2.biases,<br>backbone.layers.29.mixer.switch_mlp.fc2.scales,<br>backbone.layers.29.mixer.switch_mlp.fc2.weight,<br>backbone.layers.29.norm.weight,<br>backbone.layers.3.mixer.gate.e_score_correction_bias,<br>backbone.layers.3.mixer.gate.weight,<br>backbone.layers.3.mixer.shared_experts.down_proj.biases,<br>backbone.layers.3.mixer.shared_experts.down_proj.scales,<br>backbone.layers.3.mixer.shared_experts.down_proj.weight,<br>backbone.layers.3.mixer.shared_experts.up_proj.biases,<br>backbone.layers.3.mixer.shared_experts.up_proj.scales,<br>backbone.layers.3.mixer.shared_experts.up_proj.weight,<br>backbone.layers.3.mixer.switch_mlp.fc1.biases,<br>backbone.layers.3.mixer.switch_mlp.fc1.scales,<br>backbone.layers.3.mixer.switch_mlp.fc1.weight,<br>backbone.layers.3.mixer.switch_mlp.fc2.biases,<br>backbone.layers.3.mixer.switch_mlp.fc2.scales,<br>backbone.layers.3.mixer.switch_mlp.fc2.weight,<br>backbone.layers.3.norm.weight,<br>backbone.layers.30.mixer.A_log,<br>backbone.layers.30.mixer.D,<br>backbone.layers.30.mixer.conv1d.bias,<br>backbone.layers.30.mixer.conv1d.weight,<br>backbone.layers.30.mixer.dt_bias,<br>backbone.layers.30.mixer.in_proj.biases,<br>backbone.layers.30.mixer.in_proj.scales,<br>backbone.layers.30.mixer.in_proj.weight,<br>backbone.layers.30.mixer.norm.weight,<br>backbone.layers.30.mixer.out_proj.biases,<br>backbone.layers.30.mixer.out_proj.scales,<br>backbone.layers.30.mixer.out_proj.weight,<br>backbone.layers.30.norm.weight,<br>backbone.layers.31.mixer.gate.e_score_correction_bias,<br>backbone.layers.31.mixer.gate.weight,<br>backbone.layers.31.mixer.shared_experts.down_proj.biases,<br>backbone.layers.31.mixer.shared_experts.down_proj.scales,<br>backbone.layers.31.mixer.shared_experts.down_proj.weight,<br>backbone.layers.31.mixer.shared_experts.up_proj.biases,<br>backbone.layers.31.mixer.shared_experts.up_proj.scales,<br>backbone.layers.31.mixer.shared_experts.up_proj.weight,<br>backbone.layers.31.mixer.switch_mlp.fc1.biases,<br>backbone.layers.31.mixer.switch_mlp.fc1.scales,<br>backbone.layers.31.mixer.switch_mlp.fc1.weight,<br>backbone.layers.31.mixer.switch_mlp.fc2.biases,<br>backbone.layers.31.mixer.switch_mlp.fc2.scales,<br>backbone.layers.31.mixer.switch_mlp.fc2.weight,<br>backbone.layers.31.norm.weight,<br>backbone.layers.32.mixer.A_log,<br>backbone.layers.32.mixer.D,<br>backbone.layers.32.mixer.conv1d.bias,<br>backbone.layers.32.mixer.conv1d.weight,<br>backbone.layers.32.mixer.dt_bias,<br>backbone.layers.32.mixer.in_proj.biases,<br>backbone.layers.32.mixer.in_proj.scales,<br>backbone.layers.32.mixer.in_proj.weight,<br>backbone.layers.32.mixer.norm.weight,<br>backbone.layers.32.mixer.out_proj.biases,<br>backbone.layers.32.mixer.out_proj.scales,<br>backbone.layers.32.mixer.out_proj.weight,<br>backbone.layers.32.norm.weight,<br>backbone.layers.33.mixer.k_proj.biases,<br>backbone.layers.33.mixer.k_proj.scales,<br>backbone.layers.33.mixer.k_proj.weight,<br>backbone.layers.33.mixer.o_proj.biases,<br>backbone.layers.33.mixer.o_proj.scales,<br>backbone.layers.33.mixer.o_proj.weight,<br>backbone.layers.33.mixer.q_proj.biases,<br>backbone.layers.33.mixer.q_proj.scales,<br>backbone.layers.33.mixer.q_proj.weight,<br>backbone.layers.33.mixer.v_proj.biases,<br>backbone.layers.33.mixer.v_proj.scales,<br>backbone.layers.33.mixer.v_proj.weight,<br>backbone.layers.33.norm.weight,<br>backbone.layers.34.mixer.gate.e_score_correction_bias,<br>backbone.layers.34.mixer.gate.weight,<br>backbone.layers.34.mixer.shared_experts.down_proj.biases,<br>backbone.layers.34.mixer.shared_experts.down_proj.scales,<br>backbone.layers.34.mixer.shared_experts.down_proj.weight,<br>backbone.layers.34.mixer.shared_experts.up_proj.biases,<br>backbone.layers.34.mixer.shared_experts.up_proj.scales,<br>backbone.layers.34.mixer.shared_experts.up_proj.weight,<br>backbone.layers.34.mixer.switch_mlp.fc1.biases,<br>backbone.layers.34.mixer.switch_mlp.fc1.scales,<br>backbone.layers.34.mixer.switch_mlp.fc1.weight,<br>backbone.layers.34.mixer.switch_mlp.fc2.biases,<br>backbone.layers.34.mixer.switch_mlp.fc2.scales,<br>backbone.layers.34.mixer.switch_mlp.fc2.weight,<br>backbone.layers.34.norm.weight,<br>backbone.layers.35.mixer.A_log,<br>backbone.layers.35.mixer.D,<br>backbone.layers.35.mixer.conv1d.bias,<br>backbone.layers.35.mixer.conv1d.weight,<br>backbone.layers.35.mixer.dt_bias,<br>backbone.layers.35.mixer.in_proj.biases,<br>backbone.layers.35.mixer.in_proj.scales,<br>backbone.layers.35.mixer.in_proj.weight,<br>backbone.layers.35.mixer.norm.weight,<br>backbone.layers.35.mixer.out_proj.biases,<br>backbone.layers.35.mixer.out_proj.scales,<br>backbone.layers.35.mixer.out_proj.weight,<br>backbone.layers.35.norm.weight,<br>backbone.layers.36.mixer.gate.e_score_correction_bias,<br>backbone.layers.36.mixer.gate.weight,<br>backbone.layers.36.mixer.shared_experts.down_proj.biases,<br>backbone.layers.36.mixer.shared_experts.down_proj.scales,<br>backbone.layers.36.mixer.shared_experts.down_proj.weight,<br>backbone.layers.36.mixer.shared_experts.up_proj.biases,<br>backbone.layers.36.mixer.shared_experts.up_proj.scales,<br>backbone.layers.36.mixer.shared_experts.up_proj.weight,<br>backbone.layers.36.mixer.switch_mlp.fc1.biases,<br>backbone.layers.36.mixer.switch_mlp.fc1.scales,<br>backbone.layers.36.mixer.switch_mlp.fc1.weight,<br>backbone.layers.36.mixer.switch_mlp.fc2.biases,<br>backbone.layers.36.mixer.switch_mlp.fc2.scales,<br>backbone.layers.36.mixer.switch_mlp.fc2.weight,<br>backbone.layers.36.norm.weight,<br>backbone.layers.37.mixer.A_log,<br>backbone.layers.37.mixer.D,<br>backbone.layers.37.mixer.conv1d.bias,<br>backbone.layers.37.mixer.conv1d.weight,<br>backbone.layers.37.mixer.dt_bias,<br>backbone.layers.37.mixer.in_proj.biases,<br>backbone.layers.37.mixer.in_proj.scales,<br>backbone.layers.37.mixer.in_proj.weight,<br>backbone.layers.37.mixer.norm.weight,<br>backbone.layers.37.mixer.out_proj.biases,<br>backbone.layers.37.mixer.out_proj.scales,<br>backbone.layers.37.mixer.out_proj.weight,<br>backbone.layers.37.norm.weight,<br>backbone.layers.38.mixer.gate.e_score_correction_bias,<br>backbone.layers.38.mixer.gate.weight,<br>backbone.layers.38.mixer.shared_experts.down_proj.biases,<br>backbone.layers.38.mixer.shared_experts.down_proj.scales,<br>backbone.layers.38.mixer.shared_experts.down_proj.weight,<br>backbone.layers.38.mixer.shared_experts.up_proj.biases,<br>backbone.layers.38.mixer.shared_experts.up_proj.scales,<br>backbone.layers.38.mixer.shared_experts.up_proj.weight,<br>backbone.layers.38.mixer.switch_mlp.fc1.biases,<br>backbone.layers.38.mixer.switch_mlp.fc1.scales,<br>backbone.layers.38.mixer.switch_mlp.fc1.weight,<br>backbone.layers.38.mixer.switch_mlp.fc2.biases,<br>backbone.layers.38.mixer.switch_mlp.fc2.scales,<br>backbone.layers.38.mixer.switch_mlp.fc2.weight,<br>backbone.layers.38.norm.weight,<br>backbone.layers.39.mixer.A_log,<br>backbone.layers.39.mixer.D,<br>backbone.layers.39.mixer.conv1d.bias,<br>backbone.layers.39.mixer.conv1d.weight,<br>backbone.layers.39.mixer.dt_bias,<br>backbone.layers.39.mixer.in_proj.biases,<br>backbone.layers.39.mixer.in_proj.scales,<br>backbone.layers.39.mixer.in_proj.weight,<br>backbone.layers.39.mixer.norm.weight,<br>backbone.layers.39.mixer.out_proj.biases,<br>backbone.layers.39.mixer.out_proj.scales,<br>backbone.layers.39.mixer.out_proj.weight,<br>backbone.layers.39.norm.weight,<br>backbone.layers.4.mixer.A_log,<br>backbone.layers.4.mixer.D,<br>backbone.layers.4.mixer.conv1d.bias,<br>backbone.layers.4.mixer.conv1d.weight,<br>backbone.layers.4.mixer.dt_bias,<br>backbone.layers.4.mixer.in_proj.biases,<br>backbone.layers.4.mixer.in_proj.scales,<br>backbone.layers.4.mixer.in_proj.weight,<br>backbone.layers.4.mixer.norm.weight,<br>backbone.layers.4.mixer.out_proj.biases,<br>backbone.layers.4.mixer.out_proj.scales,<br>backbone.layers.4.mixer.out_proj.weight,<br>backbone.layers.4.norm.weight,<br>backbone.layers.40.mixer.gate.e_score_correction_bias,<br>backbone.layers.40.mixer.gate.weight,<br>backbone.layers.40.mixer.shared_experts.down_proj.biases,<br>backbone.layers.40.mixer.shared_experts.down_proj.scales,<br>backbone.layers.40.mixer.shared_experts.down_proj.weight,<br>backbone.layers.40.mixer.shared_experts.up_proj.biases,<br>backbone.layers.40.mixer.shared_experts.up_proj.scales,<br>backbone.layers.40.mixer.shared_experts.up_proj.weight,<br>backbone.layers.40.mixer.switch_mlp.fc1.biases,<br>backbone.layers.40.mixer.switch_mlp.fc1.scales,<br>backbone.layers.40.mixer.switch_mlp.fc1.weight,<br>backbone.layers.40.mixer.switch_mlp.fc2.biases,<br>backbone.layers.40.mixer.switch_mlp.fc2.scales,<br>backbone.layers.40.mixer.switch_mlp.fc2.weight,<br>backbone.layers.40.norm.weight,<br>backbone.layers.41.mixer.A_log,<br>backbone.layers.41.mixer.D,<br>backbone.layers.41.mixer.conv1d.bias,<br>backbone.layers.41.mixer.conv1d.weight,<br>backbone.layers.41.mixer.dt_bias,<br>backbone.layers.41.mixer.in_proj.biases,<br>backbone.layers.41.mixer.in_proj.scales,<br>backbone.layers.41.mixer.in_proj.weight,<br>backbone.layers.41.mixer.norm.weight,<br>backbone.layers.41.mixer.out_proj.biases,<br>backbone.layers.41.mixer.out_proj.scales,<br>backbone.layers.41.mixer.out_proj.weight,<br>backbone.layers.41.norm.weight,<br>backbone.layers.42.mixer.k_proj.biases,<br>backbone.layers.42.mixer.k_proj.scales,<br>backbone.layers.42.mixer.k_proj.weight,<br>backbone.layers.42.mixer.o_proj.biases,<br>backbone.layers.42.mixer.o_proj.scales,<br>backbone.layers.42.mixer.o_proj.weight,<br>backbone.layers.42.mixer.q_proj.biases,<br>backbone.layers.42.mixer.q_proj.scales,<br>backbone.layers.42.mixer.q_proj.weight,<br>backbone.layers.42.mixer.v_proj.biases,<br>backbone.layers.42.mixer.v_proj.scales,<br>backbone.layers.42.mixer.v_proj.weight,<br>backbone.layers.42.norm.weight,<br>backbone.layers.43.mixer.gate.e_score_correction_bias,<br>backbone.layers.43.mixer.gate.weight,<br>backbone.layers.43.mixer.shared_experts.down_proj.biases,<br>backbone.layers.43.mixer.shared_experts.down_proj.scales,<br>backbone.layers.43.mixer.shared_experts.down_proj.weight,<br>backbone.layers.43.mixer.shared_experts.up_proj.biases,<br>backbone.layers.43.mixer.shared_experts.up_proj.scales,<br>backbone.layers.43.mixer.shared_experts.up_proj.weight,<br>backbone.layers.43.mixer.switch_mlp.fc1.biases,<br>backbone.layers.43.mixer.switch_mlp.fc1.scales,<br>backbone.layers.43.mixer.switch_mlp.fc1.weight,<br>backbone.layers.43.mixer.switch_mlp.fc2.biases,<br>backbone.layers.43.mixer.switch_mlp.fc2.scales,<br>backbone.layers.43.mixer.switch_mlp.fc2.weight,<br>backbone.layers.43.norm.weight,<br>backbone.layers.44.mixer.A_log,<br>backbone.layers.44.mixer.D,<br>backbone.layers.44.mixer.conv1d.bias,<br>backbone.layers.44.mixer.conv1d.weight,<br>backbone.layers.44.mixer.dt_bias,<br>backbone.layers.44.mixer.in_proj.biases,<br>backbone.layers.44.mixer.in_proj.scales,<br>backbone.layers.44.mixer.in_proj.weight,<br>backbone.layers.44.mixer.norm.weight,<br>backbone.layers.44.mixer.out_proj.biases,<br>backbone.layers.44.mixer.out_proj.scales,<br>backbone.layers.44.mixer.out_proj.weight,<br>backbone.layers.44.norm.weight,<br>backbone.layers.45.mixer.gate.e_score_correction_bias,<br>backbone.layers.45.mixer.gate.weight,<br>backbone.layers.45.mixer.shared_experts.down_proj.biases,<br>backbone.layers.45.mixer.shared_experts.down_proj.scales,<br>backbone.layers.45.mixer.shared_experts.down_proj.weight,<br>backbone.layers.45.mixer.shared_experts.up_proj.biases,<br>backbone.layers.45.mixer.shared_experts.up_proj.scales,<br>backbone.layers.45.mixer.shared_experts.up_proj.weight,<br>backbone.layers.45.mixer.switch_mlp.fc1.biases,<br>backbone.layers.45.mixer.switch_mlp.fc1.scales,<br>backbone.layers.45.mixer.switch_mlp.fc1.weight,<br>backbone.layers.45.mixer.switch_mlp.fc2.biases,<br>backbone.layers.45.mixer.switch_mlp.fc2.scales,<br>backbone.layers.45.mixer.switch_mlp.fc2.weight,<br>backbone.layers.45.norm.weight,<br>backbone.layers.46.mixer.A_log,<br>backbone.layers.46.mixer.D,<br>backbone.layers.46.mixer.conv1d.bias,<br>backbone.layers.46.mixer.conv1d.weight,<br>backbone.layers.46.mixer.dt_bias,<br>backbone.layers.46.mixer.in_proj.biases,<br>backbone.layers.46.mixer.in_proj.scales,<br>backbone.layers.46.mixer.in_proj.weight,<br>backbone.layers.46.mixer.norm.weight,<br>backbone.layers.46.mixer.out_proj.biases,<br>backbone.layers.46.mixer.out_proj.scales,<br>backbone.layers.46.mixer.out_proj.weight,<br>backbone.layers.46.norm.weight,<br>backbone.layers.47.mixer.gate.e_score_correction_bias,<br>backbone.layers.47.mixer.gate.weight,<br>backbone.layers.47.mixer.shared_experts.down_proj.biases,<br>backbone.layers.47.mixer.shared_experts.down_proj.scales,<br>backbone.layers.47.mixer.shared_experts.down_proj.weight,<br>backbone.layers.47.mixer.shared_experts.up_proj.biases,<br>backbone.layers.47.mixer.shared_experts.up_proj.scales,<br>backbone.layers.47.mixer.shared_experts.up_proj.weight,<br>backbone.layers.47.mixer.switch_mlp.fc1.biases,<br>backbone.layers.47.mixer.switch_mlp.fc1.scales,<br>backbone.layers.47.mixer.switch_mlp.fc1.weight,<br>backbone.layers.47.mixer.switch_mlp.fc2.biases,<br>backbone.layers.47.mixer.switch_mlp.fc2.scales,<br>backbone.layers.47.mixer.switch_mlp.fc2.weight,<br>backbone.layers.47.norm.weight,<br>backbone.layers.48.mixer.A_log,<br>backbone.layers.48.mixer.D,<br>backbone.layers.48.mixer.conv1d.bias,<br>backbone.layers.48.mixer.conv1d.weight,<br>backbone.layers.48.mixer.dt_bias,<br>backbone.layers.48.mixer.in_proj.biases,<br>backbone.layers.48.mixer.in_proj.scales,<br>backbone.layers.48.mixer.in_proj.weight,<br>backbone.layers.48.mixer.norm.weight,<br>backbone.layers.48.mixer.out_proj.biases,<br>backbone.layers.48.mixer.out_proj.scales,<br>backbone.layers.48.mixer.out_proj.weight,<br>backbone.layers.48.norm.weight,<br>backbone.layers.49.mixer.gate.e_score_correction_bias,<br>backbone.layers.49.mixer.gate.weight,<br>backbone.layers.49.mixer.shared_experts.down_proj.biases,<br>backbone.layers.49.mixer.shared_experts.down_proj.scales,<br>backbone.layers.49.mixer.shared_experts.down_proj.weight,<br>backbone.layers.49.mixer.shared_experts.up_proj.biases,<br>backbone.layers.49.mixer.shared_experts.up_proj.scales,<br>backbone.layers.49.mixer.shared_experts.up_proj.weight,<br>backbone.layers.49.mixer.switch_mlp.fc1.biases,<br>backbone.layers.49.mixer.switch_mlp.fc1.scales,<br>backbone.layers.49.mixer.switch_mlp.fc1.weight,<br>backbone.layers.49.mixer.switch_mlp.fc2.biases,<br>backbone.layers.49.mixer.switch_mlp.fc2.scales,<br>backbone.layers.49.mixer.switch_mlp.fc2.weight,<br>backbone.layers.49.norm.weight,<br>backbone.layers.5.mixer.k_proj.biases,<br>backbone.layers.5.mixer.k_proj.scales,<br>backbone.layers.5.mixer.k_proj.weight,<br>backbone.layers.5.mixer.o_proj.biases,<br>backbone.layers.5.mixer.o_proj.scales,<br>backbone.layers.5.mixer.o_proj.weight,<br>backbone.layers.5.mixer.q_proj.biases,<br>backbone.layers.5.mixer.q_proj.scales,<br>backbone.layers.5.mixer.q_proj.weight,<br>backbone.layers.5.mixer.v_proj.biases,<br>backbone.layers.5.mixer.v_proj.scales,<br>backbone.layers.5.mixer.v_proj.weight,<br>backbone.layers.5.norm.weight,<br>backbone.layers.50.mixer.A_log,<br>backbone.layers.50.mixer.D,<br>backbone.layers.50.mixer.conv1d.bias,<br>backbone.layers.50.mixer.conv1d.weight,<br>backbone.layers.50.mixer.dt_bias,<br>backbone.layers.50.mixer.in_proj.biases,<br>backbone.layers.50.mixer.in_proj.scales,<br>backbone.layers.50.mixer.in_proj.weight,<br>backbone.layers.50.mixer.norm.weight,<br>backbone.layers.50.mixer.out_proj.biases,<br>backbone.layers.50.mixer.out_proj.scales,<br>backbone.layers.50.mixer.out_proj.weight,<br>backbone.layers.50.norm.weight,<br>backbone.layers.51.mixer.gate.e_score_correction_bias,<br>backbone.layers.51.mixer.gate.weight,<br>backbone.layers.51.mixer.shared_experts.down_proj.biases,<br>backbone.layers.51.mixer.shared_experts.down_proj.scales,<br>backbone.layers.51.mixer.shared_experts.down_proj.weight,<br>backbone.layers.51.mixer.shared_experts.up_proj.biases,<br>backbone.layers.51.mixer.shared_experts.up_proj.scales,<br>backbone.layers.51.mixer.shared_experts.up_proj.weight,<br>backbone.layers.51.mixer.switch_mlp.fc1.biases,<br>backbone.layers.51.mixer.switch_mlp.fc1.scales,<br>backbone.layers.51.mixer.switch_mlp.fc1.weight,<br>backbone.layers.51.mixer.switch_mlp.fc2.biases,<br>backbone.layers.51.mixer.switch_mlp.fc2.scales,<br>backbone.layers.51.mixer.switch_mlp.fc2.weight,<br>backbone.layers.51.norm.weight,<br>backbone.layers.6.mixer.gate.e_score_correction_bias,<br>backbone.layers.6.mixer.gate.weight,<br>backbone.layers.6.mixer.shared_experts.down_proj.biases,<br>backbone.layers.6.mixer.shared_experts.down_proj.scales,<br>backbone.layers.6.mixer.shared_experts.down_proj.weight,<br>backbone.layers.6.mixer.shared_experts.up_proj.biases,<br>backbone.layers.6.mixer.shared_experts.up_proj.scales,<br>backbone.layers.6.mixer.shared_experts.up_proj.weight,<br>backbone.layers.6.mixer.switch_mlp.fc1.biases,<br>backbone.layers.6.mixer.switch_mlp.fc1.scales,<br>backbone.layers.6.mixer.switch_mlp.fc1.weight,<br>backbone.layers.6.mixer.switch_mlp.fc2.biases,<br>backbone.layers.6.mixer.switch_mlp.fc2.scales,<br>backbone.layers.6.mixer.switch_mlp.fc2.weight,<br>backbone.layers.6.norm.weight,<br>backbone.layers.7.mixer.A_log,<br>backbone.layers.7.mixer.D,<br>backbone.layers.7.mixer.conv1d.bias,<br>backbone.layers.7.mixer.conv1d.weight,<br>backbone.layers.7.mixer.dt_bias,<br>backbone.layers.7.mixer.in_proj.biases,<br>backbone.layers.7.mixer.in_proj.scales,<br>backbone.layers.7.mixer.in_proj.weight,<br>backbone.layers.7.mixer.norm.weight,<br>backbone.layers.7.mixer.out_proj.biases,<br>backbone.layers.7.mixer.out_proj.scales,<br>backbone.layers.7.mixer.out_proj.weight,<br>backbone.layers.7.norm.weight,<br>backbone.layers.8.mixer.gate.e_score_correction_bias,<br>backbone.layers.8.mixer.gate.weight,<br>backbone.layers.8.mixer.shared_experts.down_proj.biases,<br>backbone.layers.8.mixer.shared_experts.down_proj.scales,<br>backbone.layers.8.mixer.shared_experts.down_proj.weight,<br>backbone.layers.8.mixer.shared_experts.up_proj.biases,<br>backbone.layers.8.mixer.shared_experts.up_proj.scales,<br>backbone.layers.8.mixer.shared_experts.up_proj.weight,<br>backbone.layers.8.mixer.switch_mlp.fc1.biases,<br>backbone.layers.8.mixer.switch_mlp.fc1.scales,<br>backbone.layers.8.mixer.switch_mlp.fc1.weight,<br>backbone.layers.8.mixer.switch_mlp.fc2.biases,<br>backbone.layers.8.mixer.switch_mlp.fc2.scales,<br>backbone.layers.8.mixer.switch_mlp.fc2.weight,<br>backbone.layers.8.norm.weight,<br>backbone.layers.9.mixer.A_log,<br>backbone.layers.9.mixer.D,<br>backbone.layers.9.mixer.conv1d.bias,<br>backbone.layers.9.mixer.conv1d.weight,<br>backbone.layers.9.mixer.dt_bias,<br>backbone.layers.9.mixer.in_proj.biases,<br>backbone.layers.9.mixer.in_proj.scales,<br>backbone.layers.9.mixer.in_proj.weight,<br>backbone.layers.9.mixer.norm.weight,<br>backbone.layers.9.mixer.out_proj.biases,<br>backbone.layers.9.mixer.out_proj.scales,<br>backbone.layers.9.mixer.out_proj.weight,<br>backbone.layers.9.norm.weight,<br>backbone.norm_f.weight,<br>lm_head.biases,<br>lm_head.scales,<br>lm_head.weight.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Received 729 parameters not in model:
  <br>backbone.embeddings.biases,<br>backbone.embeddings.scales,<br>backbone.embeddings.weight,<br>backbone.layers.0.mixer.A_log,<br>backbone.layers.0.mixer.D,<br>backbone.layers.0.mixer.conv1d.bias,<br>backbone.layers.0.mixer.conv1d.weight,<br>backbone.layers.0.mixer.dt_bias,<br>backbone.layers.0.mixer.in_proj.biases,<br>backbone.layers.0.mixer.in_proj.scales,<br>backbone.layers.0.mixer.in_proj.weight,<br>backbone.layers.0.mixer.norm.weight,<br>backbone.layers.0.mixer.out_proj.biases,<br>backbone.layers.0.mixer.out_proj.scales,<br>backbone.layers.0.mixer.out_proj.weight,<br>backbone.layers.0.norm.weight,<br>backbone.layers.1.mixer.gate.e_score_correction_bias,<br>backbone.layers.1.mixer.gate.weight,<br>backbone.layers.1.mixer.shared_experts.down_proj.biases,<br>backbone.layers.1.mixer.shared_experts.down_proj.scales,<br>backbone.layers.1.mixer.shared_experts.down_proj.weight,<br>backbone.layers.1.mixer.shared_experts.up_proj.biases,<br>backbone.layers.1.mixer.shared_experts.up_proj.scales,<br>backbone.layers.1.mixer.shared_experts.up_proj.weight,<br>backbone.layers.1.mixer.switch_mlp.fc1.biases,<br>backbone.layers.1.mixer.switch_mlp.fc1.scales,<br>backbone.layers.1.mixer.switch_mlp.fc1.weight,<br>backbone.layers.1.mixer.switch_mlp.fc2.biases,<br>backbone.layers.1.mixer.switch_mlp.fc2.scales,<br>backbone.layers.1.mixer.switch_mlp.fc2.weight,<br>backbone.layers.1.norm.weight,<br>backbone.layers.10.mixer.gate.e_score_correction_bias,<br>backbone.layers.10.mixer.gate.weight,<br>backbone.layers.10.mixer.shared_experts.down_proj.biases,<br>backbone.layers.10.mixer.shared_experts.down_proj.scales,<br>backbone.layers.10.mixer.shared_experts.down_proj.weight,<br>backbone.layers.10.mixer.shared_experts.up_proj.biases,<br>backbone.layers.10.mixer.shared_experts.up_proj.scales,<br>backbone.layers.10.mixer.shared_experts.up_proj.weight,<br>backbone.layers.10.mixer.switch_mlp.fc1.biases,<br>backbone.layers.10.mixer.switch_mlp.fc1.scales,<br>backbone.layers.10.mixer.switch_mlp.fc1.weight,<br>backbone.layers.10.mixer.switch_mlp.fc2.biases,<br>backbone.layers.10.mixer.switch_mlp.fc2.scales,<br>backbone.layers.10.mixer.switch_mlp.fc2.weight,<br>backbone.layers.10.norm.weight,<br>backbone.layers.11.mixer.A_log,<br>backbone.layers.11.mixer.D,<br>backbone.layers.11.mixer.conv1d.bias,<br>backbone.layers.11.mixer.conv1d.weight,<br>backbone.layers.11.mixer.dt_bias,<br>backbone.layers.11.mixer.in_proj.biases,<br>backbone.layers.11.mixer.in_proj.scales,<br>backbone.layers.11.mixer.in_proj.weight,<br>backbone.layers.11.mixer.norm.weight,<br>backbone.layers.11.mixer.out_proj.biases,<br>backbone.layers.11.mixer.out_proj.scales,<br>backbone.layers.11.mixer.out_proj.weight,<br>backbone.layers.11.norm.weight,<br>backbone.layers.12.mixer.k_proj.biases,<br>backbone.layers.12.mixer.k_proj.scales,<br>backbone.layers.12.mixer.k_proj.weight,<br>backbone.layers.12.mixer.o_proj.biases,<br>backbone.layers.12.mixer.o_proj.scales,<br>backbone.layers.12.mixer.o_proj.weight,<br>backbone.layers.12.mixer.q_proj.biases,<br>backbone.layers.12.mixer.q_proj.scales,<br>backbone.layers.12.mixer.q_proj.weight,<br>backbone.layers.12.mixer.v_proj.biases,<br>backbone.layers.12.mixer.v_proj.scales,<br>backbone.layers.12.mixer.v_proj.weight,<br>backbone.layers.12.norm.weight,<br>backbone.layers.13.mixer.gate.e_score_correction_bias,<br>backbone.layers.13.mixer.gate.weight,<br>backbone.layers.13.mixer.shared_experts.down_proj.biases,<br>backbone.layers.13.mixer.shared_experts.down_proj.scales,<br>backbone.layers.13.mixer.shared_experts.down_proj.weight,<br>backbone.layers.13.mixer.shared_experts.up_proj.biases,<br>backbone.layers.13.mixer.shared_experts.up_proj.scales,<br>backbone.layers.13.mixer.shared_experts.up_proj.weight,<br>backbone.layers.13.mixer.switch_mlp.fc1.biases,<br>backbone.layers.13.mixer.switch_mlp.fc1.scales,<br>backbone.layers.13.mixer.switch_mlp.fc1.weight,<br>backbone.layers.13.mixer.switch_mlp.fc2.biases,<br>backbone.layers.13.mixer.switch_mlp.fc2.scales,<br>backbone.layers.13.mixer.switch_mlp.fc2.weight,<br>backbone.layers.13.norm.weight,<br>backbone.layers.14.mixer.A_log,<br>backbone.layers.14.mixer.D,<br>backbone.layers.14.mixer.conv1d.bias,<br>backbone.layers.14.mixer.conv1d.weight,<br>backbone.layers.14.mixer.dt_bias,<br>backbone.layers.14.mixer.in_proj.biases,<br>backbone.layers.14.mixer.in_proj.scales,<br>backbone.layers.14.mixer.in_proj.weight,<br>backbone.layers.14.mixer.norm.weight,<br>backbone.layers.14.mixer.out_proj.biases,<br>backbone.layers.14.mixer.out_proj.scales,<br>backbone.layers.14.mixer.out_proj.weight,<br>backbone.layers.14.norm.weight,<br>backbone.layers.15.mixer.gate.e_score_correction_bias,<br>backbone.layers.15.mixer.gate.weight,<br>backbone.layers.15.mixer.shared_experts.down_proj.biases,<br>backbone.layers.15.mixer.shared_experts.down_proj.scales,<br>backbone.layers.15.mixer.shared_experts.down_proj.weight,<br>backbone.layers.15.mixer.shared_experts.up_proj.biases,<br>backbone.layers.15.mixer.shared_experts.up_proj.scales,<br>backbone.layers.15.mixer.shared_experts.up_proj.weight,<br>backbone.layers.15.mixer.switch_mlp.fc1.biases,<br>backbone.layers.15.mixer.switch_mlp.fc1.scales,<br>backbone.layers.15.mixer.switch_mlp.fc1.weight,<br>backbone.layers.15.mixer.switch_mlp.fc2.biases,<br>backbone.layers.15.mixer.switch_mlp.fc2.scales,<br>backbone.layers.15.mixer.switch_mlp.fc2.weight,<br>backbone.layers.15.norm.weight,<br>backbone.layers.16.mixer.A_log,<br>backbone.layers.16.mixer.D,<br>backbone.layers.16.mixer.conv1d.bias,<br>backbone.layers.16.mixer.conv1d.weight,<br>backbone.layers.16.mixer.dt_bias,<br>backbone.layers.16.mixer.in_proj.biases,<br>backbone.layers.16.mixer.in_proj.scales,<br>backbone.layers.16.mixer.in_proj.weight,<br>backbone.layers.16.mixer.norm.weight,<br>backbone.layers.16.mixer.out_proj.biases,<br>backbone.layers.16.mixer.out_proj.scales,<br>backbone.layers.16.mixer.out_proj.weight,<br>backbone.layers.16.norm.weight,<br>backbone.layers.17.mixer.gate.e_score_correction_bias,<br>backbone.layers.17.mixer.gate.weight,<br>backbone.layers.17.mixer.shared_experts.down_proj.biases,<br>backbone.layers.17.mixer.shared_experts.down_proj.scales,<br>backbone.layers.17.mixer.shared_experts.down_proj.weight,<br>backbone.layers.17.mixer.shared_experts.up_proj.biases,<br>backbone.layers.17.mixer.shared_experts.up_proj.scales,<br>backbone.layers.17.mixer.shared_experts.up_proj.weight,<br>backbone.layers.17.mixer.switch_mlp.fc1.biases,<br>backbone.layers.17.mixer.switch_mlp.fc1.scales,<br>backbone.layers.17.mixer.switch_mlp.fc1.weight,<br>backbone.layers.17.mixer.switch_mlp.fc2.biases,<br>backbone.layers.17.mixer.switch_mlp.fc2.scales,<br>backbone.layers.17.mixer.switch_mlp.fc2.weight,<br>backbone.layers.17.norm.weight,<br>backbone.layers.18.mixer.A_log,<br>backbone.layers.18.mixer.D,<br>backbone.layers.18.mixer.conv1d.bias,<br>backbone.layers.18.mixer.conv1d.weight,<br>backbone.layers.18.mixer.dt_bias,<br>backbone.layers.18.mixer.in_proj.biases,<br>backbone.layers.18.mixer.in_proj.scales,<br>backbone.layers.18.mixer.in_proj.weight,<br>backbone.layers.18.mixer.norm.weight,<br>backbone.layers.18.mixer.out_proj.biases,<br>backbone.layers.18.mixer.out_proj.scales,<br>backbone.layers.18.mixer.out_proj.weight,<br>backbone.layers.18.norm.weight,<br>backbone.layers.19.mixer.k_proj.biases,<br>backbone.layers.19.mixer.k_proj.scales,<br>backbone.layers.19.mixer.k_proj.weight,<br>backbone.layers.19.mixer.o_proj.biases,<br>backbone.layers.19.mixer.o_proj.scales,<br>backbone.layers.19.mixer.o_proj.weight,<br>backbone.layers.19.mixer.q_proj.biases,<br>backbone.layers.19.mixer.q_proj.scales,<br>backbone.layers.19.mixer.q_proj.weight,<br>backbone.layers.19.mixer.v_proj.biases,<br>backbone.layers.19.mixer.v_proj.scales,<br>backbone.layers.19.mixer.v_proj.weight,<br>backbone.layers.19.norm.weight,<br>backbone.layers.2.mixer.A_log,<br>backbone.layers.2.mixer.D,<br>backbone.layers.2.mixer.conv1d.bias,<br>backbone.layers.2.mixer.conv1d.weight,<br>backbone.layers.2.mixer.dt_bias,<br>backbone.layers.2.mixer.in_proj.biases,<br>backbone.layers.2.mixer.in_proj.scales,<br>backbone.layers.2.mixer.in_proj.weight,<br>backbone.layers.2.mixer.norm.weight,<br>backbone.layers.2.mixer.out_proj.biases,<br>backbone.layers.2.mixer.out_proj.scales,<br>backbone.layers.2.mixer.out_proj.weight,<br>backbone.layers.2.norm.weight,<br>backbone.layers.20.mixer.gate.e_score_correction_bias,<br>backbone.layers.20.mixer.gate.weight,<br>backbone.layers.20.mixer.shared_experts.down_proj.biases,<br>backbone.layers.20.mixer.shared_experts.down_proj.scales,<br>backbone.layers.20.mixer.shared_experts.down_proj.weight,<br>backbone.layers.20.mixer.shared_experts.up_proj.biases,<br>backbone.layers.20.mixer.shared_experts.up_proj.scales,<br>backbone.layers.20.mixer.shared_experts.up_proj.weight,<br>backbone.layers.20.mixer.switch_mlp.fc1.biases,<br>backbone.layers.20.mixer.switch_mlp.fc1.scales,<br>backbone.layers.20.mixer.switch_mlp.fc1.weight,<br>backbone.layers.20.mixer.switch_mlp.fc2.biases,<br>backbone.layers.20.mixer.switch_mlp.fc2.scales,<br>backbone.layers.20.mixer.switch_mlp.fc2.weight,<br>backbone.layers.20.norm.weight,<br>backbone.layers.21.mixer.A_log,<br>backbone.layers.21.mixer.D,<br>backbone.layers.21.mixer.conv1d.bias,<br>backbone.layers.21.mixer.conv1d.weight,<br>backbone.layers.21.mixer.dt_bias,<br>backbone.layers.21.mixer.in_proj.biases,<br>backbone.layers.21.mixer.in_proj.scales,<br>backbone.layers.21.mixer.in_proj.weight,<br>backbone.layers.21.mixer.norm.weight,<br>backbone.layers.21.mixer.out_proj.biases,<br>backbone.layers.21.mixer.out_proj.scales,<br>backbone.layers.21.mixer.out_proj.weight,<br>backbone.layers.21.norm.weight,<br>backbone.layers.22.mixer.gate.e_score_correction_bias,<br>backbone.layers.22.mixer.gate.weight,<br>backbone.layers.22.mixer.shared_experts.down_proj.biases,<br>backbone.layers.22.mixer.shared_experts.down_proj.scales,<br>backbone.layers.22.mixer.shared_experts.down_proj.weight,<br>backbone.layers.22.mixer.shared_experts.up_proj.biases,<br>backbone.layers.22.mixer.shared_experts.up_proj.scales,<br>backbone.layers.22.mixer.shared_experts.up_proj.weight,<br>backbone.layers.22.mixer.switch_mlp.fc1.biases,<br>backbone.layers.22.mixer.switch_mlp.fc1.scales,<br>backbone.layers.22.mixer.switch_mlp.fc1.weight,<br>backbone.layers.22.mixer.switch_mlp.fc2.biases,<br>backbone.layers.22.mixer.switch_mlp.fc2.scales,<br>backbone.layers.22.mixer.switch_mlp.fc2.weight,<br>backbone.layers.22.norm.weight,<br>backbone.layers.23.mixer.A_log,<br>backbone.layers.23.mixer.D,<br>backbone.layers.23.mixer.conv1d.bias,<br>backbone.layers.23.mixer.conv1d.weight,<br>backbone.layers.23.mixer.dt_bias,<br>backbone.layers.23.mixer.in_proj.biases,<br>backbone.layers.23.mixer.in_proj.scales,<br>backbone.layers.23.mixer.in_proj.weight,<br>backbone.layers.23.mixer.norm.weight,<br>backbone.layers.23.mixer.out_proj.biases,<br>backbone.layers.23.mixer.out_proj.scales,<br>backbone.layers.23.mixer.out_proj.weight,<br>backbone.layers.23.norm.weight,<br>backbone.layers.24.mixer.gate.e_score_correction_bias,<br>backbone.layers.24.mixer.gate.weight,<br>backbone.layers.24.mixer.shared_experts.down_proj.biases,<br>backbone.layers.24.mixer.shared_experts.down_proj.scales,<br>backbone.layers.24.mixer.shared_experts.down_proj.weight,<br>backbone.layers.24.mixer.shared_experts.up_proj.biases,<br>backbone.layers.24.mixer.shared_experts.up_proj.scales,<br>backbone.layers.24.mixer.shared_experts.up_proj.weight,<br>backbone.layers.24.mixer.switch_mlp.fc1.biases,<br>backbone.layers.24.mixer.switch_mlp.fc1.scales,<br>backbone.layers.24.mixer.switch_mlp.fc1.weight,<br>backbone.layers.24.mixer.switch_mlp.fc2.biases,<br>backbone.layers.24.mixer.switch_mlp.fc2.scales,<br>backbone.layers.24.mixer.switch_mlp.fc2.weight,<br>backbone.layers.24.norm.weight,<br>backbone.layers.25.mixer.A_log,<br>backbone.layers.25.mixer.D,<br>backbone.layers.25.mixer.conv1d.bias,<br>backbone.layers.25.mixer.conv1d.weight,<br>backbone.layers.25.mixer.dt_bias,<br>backbone.layers.25.mixer.in_proj.biases,<br>backbone.layers.25.mixer.in_proj.scales,<br>backbone.layers.25.mixer.in_proj.weight,<br>backbone.layers.25.mixer.norm.weight,<br>backbone.layers.25.mixer.out_proj.biases,<br>backbone.layers.25.mixer.out_proj.scales,<br>backbone.layers.25.mixer.out_proj.weight,<br>backbone.layers.25.norm.weight,<br>backbone.layers.26.mixer.k_proj.biases,<br>backbone.layers.26.mixer.k_proj.scales,<br>backbone.layers.26.mixer.k_proj.weight,<br>backbone.layers.26.mixer.o_proj.biases,<br>backbone.layers.26.mixer.o_proj.scales,<br>backbone.layers.26.mixer.o_proj.weight,<br>backbone.layers.26.mixer.q_proj.biases,<br>backbone.layers.26.mixer.q_proj.scales,<br>backbone.layers.26.mixer.q_proj.weight,<br>backbone.layers.26.mixer.v_proj.biases,<br>backbone.layers.26.mixer.v_proj.scales,<br>backbone.layers.26.mixer.v_proj.weight,<br>backbone.layers.26.norm.weight,<br>backbone.layers.27.mixer.gate.e_score_correction_bias,<br>backbone.layers.27.mixer.gate.weight,<br>backbone.layers.27.mixer.shared_experts.down_proj.biases,<br>backbone.layers.27.mixer.shared_experts.down_proj.scales,<br>backbone.layers.27.mixer.shared_experts.down_proj.weight,<br>backbone.layers.27.mixer.shared_experts.up_proj.biases,<br>backbone.layers.27.mixer.shared_experts.up_proj.scales,<br>backbone.layers.27.mixer.shared_experts.up_proj.weight,<br>backbone.layers.27.mixer.switch_mlp.fc1.biases,<br>backbone.layers.27.mixer.switch_mlp.fc1.scales,<br>backbone.layers.27.mixer.switch_mlp.fc1.weight,<br>backbone.layers.27.mixer.switch_mlp.fc2.biases,<br>backbone.layers.27.mixer.switch_mlp.fc2.scales,<br>backbone.layers.27.mixer.switch_mlp.fc2.weight,<br>backbone.layers.27.norm.weight,<br>backbone.layers.28.mixer.A_log,<br>backbone.layers.28.mixer.D,<br>backbone.layers.28.mixer.conv1d.bias,<br>backbone.layers.28.mixer.conv1d.weight,<br>backbone.layers.28.mixer.dt_bias,<br>backbone.layers.28.mixer.in_proj.biases,<br>backbone.layers.28.mixer.in_proj.scales,<br>backbone.layers.28.mixer.in_proj.weight,<br>backbone.layers.28.mixer.norm.weight,<br>backbone.layers.28.mixer.out_proj.biases,<br>backbone.layers.28.mixer.out_proj.scales,<br>backbone.layers.28.mixer.out_proj.weight,<br>backbone.layers.28.norm.weight,<br>backbone.layers.29.mixer.gate.e_score_correction_bias,<br>backbone.layers.29.mixer.gate.weight,<br>backbone.layers.29.mixer.shared_experts.down_proj.biases,<br>backbone.layers.29.mixer.shared_experts.down_proj.scales,<br>backbone.layers.29.mixer.shared_experts.down_proj.weight,<br>backbone.layers.29.mixer.shared_experts.up_proj.biases,<br>backbone.layers.29.mixer.shared_experts.up_proj.scales,<br>backbone.layers.29.mixer.shared_experts.up_proj.weight,<br>backbone.layers.29.mixer.switch_mlp.fc1.biases,<br>backbone.layers.29.mixer.switch_mlp.fc1.scales,<br>backbone.layers.29.mixer.switch_mlp.fc1.weight,<br>backbone.layers.29.mixer.switch_mlp.fc2.biases,<br>backbone.layers.29.mixer.switch_mlp.fc2.scales,<br>backbone.layers.29.mixer.switch_mlp.fc2.weight,<br>backbone.layers.29.norm.weight,<br>backbone.layers.3.mixer.gate.e_score_correction_bias,<br>backbone.layers.3.mixer.gate.weight,<br>backbone.layers.3.mixer.shared_experts.down_proj.biases,<br>backbone.layers.3.mixer.shared_experts.down_proj.scales,<br>backbone.layers.3.mixer.shared_experts.down_proj.weight,<br>backbone.layers.3.mixer.shared_experts.up_proj.biases,<br>backbone.layers.3.mixer.shared_experts.up_proj.scales,<br>backbone.layers.3.mixer.shared_experts.up_proj.weight,<br>backbone.layers.3.mixer.switch_mlp.fc1.biases,<br>backbone.layers.3.mixer.switch_mlp.fc1.scales,<br>backbone.layers.3.mixer.switch_mlp.fc1.weight,<br>backbone.layers.3.mixer.switch_mlp.fc2.biases,<br>backbone.layers.3.mixer.switch_mlp.fc2.scales,<br>backbone.layers.3.mixer.switch_mlp.fc2.weight,<br>backbone.layers.3.norm.weight,<br>backbone.layers.30.mixer.A_log,<br>backbone.layers.30.mixer.D,<br>backbone.layers.30.mixer.conv1d.bias,<br>backbone.layers.30.mixer.conv1d.weight,<br>backbone.layers.30.mixer.dt_bias,<br>backbone.layers.30.mixer.in_proj.biases,<br>backbone.layers.30.mixer.in_proj.scales,<br>backbone.layers.30.mixer.in_proj.weight,<br>backbone.layers.30.mixer.norm.weight,<br>backbone.layers.30.mixer.out_proj.biases,<br>backbone.layers.30.mixer.out_proj.scales,<br>backbone.layers.30.mixer.out_proj.weight,<br>backbone.layers.30.norm.weight,<br>backbone.layers.31.mixer.gate.e_score_correction_bias,<br>backbone.layers.31.mixer.gate.weight,<br>backbone.layers.31.mixer.shared_experts.down_proj.biases,<br>backbone.layers.31.mixer.shared_experts.down_proj.scales,<br>backbone.layers.31.mixer.shared_experts.down_proj.weight,<br>backbone.layers.31.mixer.shared_experts.up_proj.biases,<br>backbone.layers.31.mixer.shared_experts.up_proj.scales,<br>backbone.layers.31.mixer.shared_experts.up_proj.weight,<br>backbone.layers.31.mixer.switch_mlp.fc1.biases,<br>backbone.layers.31.mixer.switch_mlp.fc1.scales,<br>backbone.layers.31.mixer.switch_mlp.fc1.weight,<br>backbone.layers.31.mixer.switch_mlp.fc2.biases,<br>backbone.layers.31.mixer.switch_mlp.fc2.scales,<br>backbone.layers.31.mixer.switch_mlp.fc2.weight,<br>backbone.layers.31.norm.weight,<br>backbone.layers.32.mixer.A_log,<br>backbone.layers.32.mixer.D,<br>backbone.layers.32.mixer.conv1d.bias,<br>backbone.layers.32.mixer.conv1d.weight,<br>backbone.layers.32.mixer.dt_bias,<br>backbone.layers.32.mixer.in_proj.biases,<br>backbone.layers.32.mixer.in_proj.scales,<br>backbone.layers.32.mixer.in_proj.weight,<br>backbone.layers.32.mixer.norm.weight,<br>backbone.layers.32.mixer.out_proj.biases,<br>backbone.layers.32.mixer.out_proj.scales,<br>backbone.layers.32.mixer.out_proj.weight,<br>backbone.layers.32.norm.weight,<br>backbone.layers.33.mixer.k_proj.biases,<br>backbone.layers.33.mixer.k_proj.scales,<br>backbone.layers.33.mixer.k_proj.weight,<br>backbone.layers.33.mixer.o_proj.biases,<br>backbone.layers.33.mixer.o_proj.scales,<br>backbone.layers.33.mixer.o_proj.weight,<br>backbone.layers.33.mixer.q_proj.biases,<br>backbone.layers.33.mixer.q_proj.scales,<br>backbone.layers.33.mixer.q_proj.weight,<br>backbone.layers.33.mixer.v_proj.biases,<br>backbone.layers.33.mixer.v_proj.scales,<br>backbone.layers.33.mixer.v_proj.weight,<br>backbone.layers.33.norm.weight,<br>backbone.layers.34.mixer.gate.e_score_correction_bias,<br>backbone.layers.34.mixer.gate.weight,<br>backbone.layers.34.mixer.shared_experts.down_proj.biases,<br>backbone.layers.34.mixer.shared_experts.down_proj.scales,<br>backbone.layers.34.mixer.shared_experts.down_proj.weight,<br>backbone.layers.34.mixer.shared_experts.up_proj.biases,<br>backbone.layers.34.mixer.shared_experts.up_proj.scales,<br>backbone.layers.34.mixer.shared_experts.up_proj.weight,<br>backbone.layers.34.mixer.switch_mlp.fc1.biases,<br>backbone.layers.34.mixer.switch_mlp.fc1.scales,<br>backbone.layers.34.mixer.switch_mlp.fc1.weight,<br>backbone.layers.34.mixer.switch_mlp.fc2.biases,<br>backbone.layers.34.mixer.switch_mlp.fc2.scales,<br>backbone.layers.34.mixer.switch_mlp.fc2.weight,<br>backbone.layers.34.norm.weight,<br>backbone.layers.35.mixer.A_log,<br>backbone.layers.35.mixer.D,<br>backbone.layers.35.mixer.conv1d.bias,<br>backbone.layers.35.mixer.conv1d.weight,<br>backbone.layers.35.mixer.dt_bias,<br>backbone.layers.35.mixer.in_proj.biases,<br>backbone.layers.35.mixer.in_proj.scales,<br>backbone.layers.35.mixer.in_proj.weight,<br>backbone.layers.35.mixer.norm.weight,<br>backbone.layers.35.mixer.out_proj.biases,<br>backbone.layers.35.mixer.out_proj.scales,<br>backbone.layers.35.mixer.out_proj.weight,<br>backbone.layers.35.norm.weight,<br>backbone.layers.36.mixer.gate.e_score_correction_bias,<br>backbone.layers.36.mixer.gate.weight,<br>backbone.layers.36.mixer.shared_experts.down_proj.biases,<br>backbone.layers.36.mixer.shared_experts.down_proj.scales,<br>backbone.layers.36.mixer.shared_experts.down_proj.weight,<br>backbone.layers.36.mixer.shared_experts.up_proj.biases,<br>backbone.layers.36.mixer.shared_experts.up_proj.scales,<br>backbone.layers.36.mixer.shared_experts.up_proj.weight,<br>backbone.layers.36.mixer.switch_mlp.fc1.biases,<br>backbone.layers.36.mixer.switch_mlp.fc1.scales,<br>backbone.layers.36.mixer.switch_mlp.fc1.weight,<br>backbone.layers.36.mixer.switch_mlp.fc2.biases,<br>backbone.layers.36.mixer.switch_mlp.fc2.scales,<br>backbone.layers.36.mixer.switch_mlp.fc2.weight,<br>backbone.layers.36.norm.weight,<br>backbone.layers.37.mixer.A_log,<br>backbone.layers.37.mixer.D,<br>backbone.layers.37.mixer.conv1d.bias,<br>backbone.layers.37.mixer.conv1d.weight,<br>backbone.layers.37.mixer.dt_bias,<br>backbone.layers.37.mixer.in_proj.biases,<br>backbone.layers.37.mixer.in_proj.scales,<br>backbone.layers.37.mixer.in_proj.weight,<br>backbone.layers.37.mixer.norm.weight,<br>backbone.layers.37.mixer.out_proj.biases,<br>backbone.layers.37.mixer.out_proj.scales,<br>backbone.layers.37.mixer.out_proj.weight,<br>backbone.layers.37.norm.weight,<br>backbone.layers.38.mixer.gate.e_score_correction_bias,<br>backbone.layers.38.mixer.gate.weight,<br>backbone.layers.38.mixer.shared_experts.down_proj.biases,<br>backbone.layers.38.mixer.shared_experts.down_proj.scales,<br>backbone.layers.38.mixer.shared_experts.down_proj.weight,<br>backbone.layers.38.mixer.shared_experts.up_proj.biases,<br>backbone.layers.38.mixer.shared_experts.up_proj.scales,<br>backbone.layers.38.mixer.shared_experts.up_proj.weight,<br>backbone.layers.38.mixer.switch_mlp.fc1.biases,<br>backbone.layers.38.mixer.switch_mlp.fc1.scales,<br>backbone.layers.38.mixer.switch_mlp.fc1.weight,<br>backbone.layers.38.mixer.switch_mlp.fc2.biases,<br>backbone.layers.38.mixer.switch_mlp.fc2.scales,<br>backbone.layers.38.mixer.switch_mlp.fc2.weight,<br>backbone.layers.38.norm.weight,<br>backbone.layers.39.mixer.A_log,<br>backbone.layers.39.mixer.D,<br>backbone.layers.39.mixer.conv1d.bias,<br>backbone.layers.39.mixer.conv1d.weight,<br>backbone.layers.39.mixer.dt_bias,<br>backbone.layers.39.mixer.in_proj.biases,<br>backbone.layers.39.mixer.in_proj.scales,<br>backbone.layers.39.mixer.in_proj.weight,<br>backbone.layers.39.mixer.norm.weight,<br>backbone.layers.39.mixer.out_proj.biases,<br>backbone.layers.39.mixer.out_proj.scales,<br>backbone.layers.39.mixer.out_proj.weight,<br>backbone.layers.39.norm.weight,<br>backbone.layers.4.mixer.A_log,<br>backbone.layers.4.mixer.D,<br>backbone.layers.4.mixer.conv1d.bias,<br>backbone.layers.4.mixer.conv1d.weight,<br>backbone.layers.4.mixer.dt_bias,<br>backbone.layers.4.mixer.in_proj.biases,<br>backbone.layers.4.mixer.in_proj.scales,<br>backbone.layers.4.mixer.in_proj.weight,<br>backbone.layers.4.mixer.norm.weight,<br>backbone.layers.4.mixer.out_proj.biases,<br>backbone.layers.4.mixer.out_proj.scales,<br>backbone.layers.4.mixer.out_proj.weight,<br>backbone.layers.4.norm.weight,<br>backbone.layers.40.mixer.gate.e_score_correction_bias,<br>backbone.layers.40.mixer.gate.weight,<br>backbone.layers.40.mixer.shared_experts.down_proj.biases,<br>backbone.layers.40.mixer.shared_experts.down_proj.scales,<br>backbone.layers.40.mixer.shared_experts.down_proj.weight,<br>backbone.layers.40.mixer.shared_experts.up_proj.biases,<br>backbone.layers.40.mixer.shared_experts.up_proj.scales,<br>backbone.layers.40.mixer.shared_experts.up_proj.weight,<br>backbone.layers.40.mixer.switch_mlp.fc1.biases,<br>backbone.layers.40.mixer.switch_mlp.fc1.scales,<br>backbone.layers.40.mixer.switch_mlp.fc1.weight,<br>backbone.layers.40.mixer.switch_mlp.fc2.biases,<br>backbone.layers.40.mixer.switch_mlp.fc2.scales,<br>backbone.layers.40.mixer.switch_mlp.fc2.weight,<br>backbone.layers.40.norm.weight,<br>backbone.layers.41.mixer.A_log,<br>backbone.layers.41.mixer.D,<br>backbone.layers.41.mixer.conv1d.bias,<br>backbone.layers.41.mixer.conv1d.weight,<br>backbone.layers.41.mixer.dt_bias,<br>backbone.layers.41.mixer.in_proj.biases,<br>backbone.layers.41.mixer.in_proj.scales,<br>backbone.layers.41.mixer.in_proj.weight,<br>backbone.layers.41.mixer.norm.weight,<br>backbone.layers.41.mixer.out_proj.biases,<br>backbone.layers.41.mixer.out_proj.scales,<br>backbone.layers.41.mixer.out_proj.weight,<br>backbone.layers.41.norm.weight,<br>backbone.layers.42.mixer.k_proj.biases,<br>backbone.layers.42.mixer.k_proj.scales,<br>backbone.layers.42.mixer.k_proj.weight,<br>backbone.layers.42.mixer.o_proj.biases,<br>backbone.layers.42.mixer.o_proj.scales,<br>backbone.layers.42.mixer.o_proj.weight,<br>backbone.layers.42.mixer.q_proj.biases,<br>backbone.layers.42.mixer.q_proj.scales,<br>backbone.layers.42.mixer.q_proj.weight,<br>backbone.layers.42.mixer.v_proj.biases,<br>backbone.layers.42.mixer.v_proj.scales,<br>backbone.layers.42.mixer.v_proj.weight,<br>backbone.layers.42.norm.weight,<br>backbone.layers.43.mixer.gate.e_score_correction_bias,<br>backbone.layers.43.mixer.gate.weight,<br>backbone.layers.43.mixer.shared_experts.down_proj.biases,<br>backbone.layers.43.mixer.shared_experts.down_proj.scales,<br>backbone.layers.43.mixer.shared_experts.down_proj.weight,<br>backbone.layers.43.mixer.shared_experts.up_proj.biases,<br>backbone.layers.43.mixer.shared_experts.up_proj.scales,<br>backbone.layers.43.mixer.shared_experts.up_proj.weight,<br>backbone.layers.43.mixer.switch_mlp.fc1.biases,<br>backbone.layers.43.mixer.switch_mlp.fc1.scales,<br>backbone.layers.43.mixer.switch_mlp.fc1.weight,<br>backbone.layers.43.mixer.switch_mlp.fc2.biases,<br>backbone.layers.43.mixer.switch_mlp.fc2.scales,<br>backbone.layers.43.mixer.switch_mlp.fc2.weight,<br>backbone.layers.43.norm.weight,<br>backbone.layers.44.mixer.A_log,<br>backbone.layers.44.mixer.D,<br>backbone.layers.44.mixer.conv1d.bias,<br>backbone.layers.44.mixer.conv1d.weight,<br>backbone.layers.44.mixer.dt_bias,<br>backbone.layers.44.mixer.in_proj.biases,<br>backbone.layers.44.mixer.in_proj.scales,<br>backbone.layers.44.mixer.in_proj.weight,<br>backbone.layers.44.mixer.norm.weight,<br>backbone.layers.44.mixer.out_proj.biases,<br>backbone.layers.44.mixer.out_proj.scales,<br>backbone.layers.44.mixer.out_proj.weight,<br>backbone.layers.44.norm.weight,<br>backbone.layers.45.mixer.gate.e_score_correction_bias,<br>backbone.layers.45.mixer.gate.weight,<br>backbone.layers.45.mixer.shared_experts.down_proj.biases,<br>backbone.layers.45.mixer.shared_experts.down_proj.scales,<br>backbone.layers.45.mixer.shared_experts.down_proj.weight,<br>backbone.layers.45.mixer.shared_experts.up_proj.biases,<br>backbone.layers.45.mixer.shared_experts.up_proj.scales,<br>backbone.layers.45.mixer.shared_experts.up_proj.weight,<br>backbone.layers.45.mixer.switch_mlp.fc1.biases,<br>backbone.layers.45.mixer.switch_mlp.fc1.scales,<br>backbone.layers.45.mixer.switch_mlp.fc1.weight,<br>backbone.layers.45.mixer.switch_mlp.fc2.biases,<br>backbone.layers.45.mixer.switch_mlp.fc2.scales,<br>backbone.layers.45.mixer.switch_mlp.fc2.weight,<br>backbone.layers.45.norm.weight,<br>backbone.layers.46.mixer.A_log,<br>backbone.layers.46.mixer.D,<br>backbone.layers.46.mixer.conv1d.bias,<br>backbone.layers.46.mixer.conv1d.weight,<br>backbone.layers.46.mixer.dt_bias,<br>backbone.layers.46.mixer.in_proj.biases,<br>backbone.layers.46.mixer.in_proj.scales,<br>backbone.layers.46.mixer.in_proj.weight,<br>backbone.layers.46.mixer.norm.weight,<br>backbone.layers.46.mixer.out_proj.biases,<br>backbone.layers.46.mixer.out_proj.scales,<br>backbone.layers.46.mixer.out_proj.weight,<br>backbone.layers.46.norm.weight,<br>backbone.layers.47.mixer.gate.e_score_correction_bias,<br>backbone.layers.47.mixer.gate.weight,<br>backbone.layers.47.mixer.shared_experts.down_proj.biases,<br>backbone.layers.47.mixer.shared_experts.down_proj.scales,<br>backbone.layers.47.mixer.shared_experts.down_proj.weight,<br>backbone.layers.47.mixer.shared_experts.up_proj.biases,<br>backbone.layers.47.mixer.shared_experts.up_proj.scales,<br>backbone.layers.47.mixer.shared_experts.up_proj.weight,<br>backbone.layers.47.mixer.switch_mlp.fc1.biases,<br>backbone.layers.47.mixer.switch_mlp.fc1.scales,<br>backbone.layers.47.mixer.switch_mlp.fc1.weight,<br>backbone.layers.47.mixer.switch_mlp.fc2.biases,<br>backbone.layers.47.mixer.switch_mlp.fc2.scales,<br>backbone.layers.47.mixer.switch_mlp.fc2.weight,<br>backbone.layers.47.norm.weight,<br>backbone.layers.48.mixer.A_log,<br>backbone.layers.48.mixer.D,<br>backbone.layers.48.mixer.conv1d.bias,<br>backbone.layers.48.mixer.conv1d.weight,<br>backbone.layers.48.mixer.dt_bias,<br>backbone.layers.48.mixer.in_proj.biases,<br>backbone.layers.48.mixer.in_proj.scales,<br>backbone.layers.48.mixer.in_proj.weight,<br>backbone.layers.48.mixer.norm.weight,<br>backbone.layers.48.mixer.out_proj.biases,<br>backbone.layers.48.mixer.out_proj.scales,<br>backbone.layers.48.mixer.out_proj.weight,<br>backbone.layers.48.norm.weight,<br>backbone.layers.49.mixer.gate.e_score_correction_bias,<br>backbone.layers.49.mixer.gate.weight,<br>backbone.layers.49.mixer.shared_experts.down_proj.biases,<br>backbone.layers.49.mixer.shared_experts.down_proj.scales,<br>backbone.layers.49.mixer.shared_experts.down_proj.weight,<br>backbone.layers.49.mixer.shared_experts.up_proj.biases,<br>backbone.layers.49.mixer.shared_experts.up_proj.scales,<br>backbone.layers.49.mixer.shared_experts.up_proj.weight,<br>backbone.layers.49.mixer.switch_mlp.fc1.biases,<br>backbone.layers.49.mixer.switch_mlp.fc1.scales,<br>backbone.layers.49.mixer.switch_mlp.fc1.weight,<br>backbone.layers.49.mixer.switch_mlp.fc2.biases,<br>backbone.layers.49.mixer.switch_mlp.fc2.scales,<br>backbone.layers.49.mixer.switch_mlp.fc2.weight,<br>backbone.layers.49.norm.weight,<br>backbone.layers.5.mixer.k_proj.biases,<br>backbone.layers.5.mixer.k_proj.scales,<br>backbone.layers.5.mixer.k_proj.weight,<br>backbone.layers.5.mixer.o_proj.biases,<br>backbone.layers.5.mixer.o_proj.scales,<br>backbone.layers.5.mixer.o_proj.weight,<br>backbone.layers.5.mixer.q_proj.biases,<br>backbone.layers.5.mixer.q_proj.scales,<br>backbone.layers.5.mixer.q_proj.weight,<br>backbone.layers.5.mixer.v_proj.biases,<br>backbone.layers.5.mixer.v_proj.scales,<br>backbone.layers.5.mixer.v_proj.weight,<br>backbone.layers.5.norm.weight,<br>backbone.layers.50.mixer.A_log,<br>backbone.layers.50.mixer.D,<br>backbone.layers.50.mixer.conv1d.bias,<br>backbone.layers.50.mixer.conv1d.weight,<br>backbone.layers.50.mixer.dt_bias,<br>backbone.layers.50.mixer.in_proj.biases,<br>backbone.layers.50.mixer.in_proj.scales,<br>backbone.layers.50.mixer.in_proj.weight,<br>backbone.layers.50.mixer.norm.weight,<br>backbone.layers.50.mixer.out_proj.biases,<br>backbone.layers.50.mixer.out_proj.scales,<br>backbone.layers.50.mixer.out_proj.weight,<br>backbone.layers.50.norm.weight,<br>backbone.layers.51.mixer.gate.e_score_correction_bias,<br>backbone.layers.51.mixer.gate.weight,<br>backbone.layers.51.mixer.shared_experts.down_proj.biases,<br>backbone.layers.51.mixer.shared_experts.down_proj.scales,<br>backbone.layers.51.mixer.shared_experts.down_proj.weight,<br>backbone.layers.51.mixer.shared_experts.up_proj.biases,<br>backbone.layers.51.mixer.shared_experts.up_proj.scales,<br>backbone.layers.51.mixer.shared_experts.up_proj.weight,<br>backbone.layers.51.mixer.switch_mlp.fc1.biases,<br>backbone.layers.51.mixer.switch_mlp.fc1.scales,<br>backbone.layers.51.mixer.switch_mlp.fc1.weight,<br>backbone.layers.51.mixer.switch_mlp.fc2.biases,<br>backbone.layers.51.mixer.switch_mlp.fc2.scales,<br>backbone.layers.51.mixer.switch_mlp.fc2.weight,<br>backbone.layers.51.norm.weight,<br>backbone.layers.6.mixer.gate.e_score_correction_bias,<br>backbone.layers.6.mixer.gate.weight,<br>backbone.layers.6.mixer.shared_experts.down_proj.biases,<br>backbone.layers.6.mixer.shared_experts.down_proj.scales,<br>backbone.layers.6.mixer.shared_experts.down_proj.weight,<br>backbone.layers.6.mixer.shared_experts.up_proj.biases,<br>backbone.layers.6.mixer.shared_experts.up_proj.scales,<br>backbone.layers.6.mixer.shared_experts.up_proj.weight,<br>backbone.layers.6.mixer.switch_mlp.fc1.biases,<br>backbone.layers.6.mixer.switch_mlp.fc1.scales,<br>backbone.layers.6.mixer.switch_mlp.fc1.weight,<br>backbone.layers.6.mixer.switch_mlp.fc2.biases,<br>backbone.layers.6.mixer.switch_mlp.fc2.scales,<br>backbone.layers.6.mixer.switch_mlp.fc2.weight,<br>backbone.layers.6.norm.weight,<br>backbone.layers.7.mixer.A_log,<br>backbone.layers.7.mixer.D,<br>backbone.layers.7.mixer.conv1d.bias,<br>backbone.layers.7.mixer.conv1d.weight,<br>backbone.layers.7.mixer.dt_bias,<br>backbone.layers.7.mixer.in_proj.biases,<br>backbone.layers.7.mixer.in_proj.scales,<br>backbone.layers.7.mixer.in_proj.weight,<br>backbone.layers.7.mixer.norm.weight,<br>backbone.layers.7.mixer.out_proj.biases,<br>backbone.layers.7.mixer.out_proj.scales,<br>backbone.layers.7.mixer.out_proj.weight,<br>backbone.layers.7.norm.weight,<br>backbone.layers.8.mixer.gate.e_score_correction_bias,<br>backbone.layers.8.mixer.gate.weight,<br>backbone.layers.8.mixer.shared_experts.down_proj.biases,<br>backbone.layers.8.mixer.shared_experts.down_proj.scales,<br>backbone.layers.8.mixer.shared_experts.down_proj.weight,<br>backbone.layers.8.mixer.shared_experts.up_proj.biases,<br>backbone.layers.8.mixer.shared_experts.up_proj.scales,<br>backbone.layers.8.mixer.shared_experts.up_proj.weight,<br>backbone.layers.8.mixer.switch_mlp.fc1.biases,<br>backbone.layers.8.mixer.switch_mlp.fc1.scales,<br>backbone.layers.8.mixer.switch_mlp.fc1.weight,<br>backbone.layers.8.mixer.switch_mlp.fc2.biases,<br>backbone.layers.8.mixer.switch_mlp.fc2.scales,<br>backbone.layers.8.mixer.switch_mlp.fc2.weight,<br>backbone.layers.8.norm.weight,<br>backbone.layers.9.mixer.A_log,<br>backbone.layers.9.mixer.D,<br>backbone.layers.9.mixer.conv1d.bias,<br>backbone.layers.9.mixer.conv1d.weight,<br>backbone.layers.9.mixer.dt_bias,<br>backbone.layers.9.mixer.in_proj.biases,<br>backbone.layers.9.mixer.in_proj.scales,<br>backbone.layers.9.mixer.in_proj.weight,<br>backbone.layers.9.mixer.norm.weight,<br>backbone.layers.9.mixer.out_proj.biases,<br>backbone.layers.9.mixer.out_proj.scales,<br>backbone.layers.9.mixer.out_proj.weight,<br>backbone.layers.9.norm.weight,<br>backbone.norm_f.weight,<br>lm_head.biases,<br>lm_head.scales,<br>lm_head.weight.
- *Arch supported by installed mlx-vlm:* yes (model_type
  nemotronh_nano_omni_reasoning_v3 via nemotron_h_nano_omni)
- *Model load time:* 0.25s
- *Generation time:* -
- *Total time:* 0.25s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0671
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.0059
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* 8f86ad8f279ce1ec3b8b65970f32da1e89ab7a45
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit/snapshots/8f86ad8f279ce1ec3b8b65970f32da1e89ab7a45
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14457, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13369, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 826, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1172, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 729 parameters not in model: 
backbone.embeddings.biases,
backbone.embeddings.scales,
backbone.embeddings.weight,
backbone.layers.0.mixer.A_log,
backbone.layers.0.mixer.D,
backbone.layers.0.mixer.conv1d.bias,
backbone.layers.0.mixer.conv1d.weight,
backbone.layers.0.mixer.dt_bias,
backbone.layers.0.mixer.in_proj.biases,
backbone.layers.0.mixer.in_proj.scales,
backbone.layers.0.mixer.in_proj.weight,
backbone.layers.0.mixer.norm.weight,
backbone.layers.0.mixer.out_proj.biases,
backbone.layers.0.mixer.out_proj.scales,
backbone.layers.0.mixer.out_proj.weight,
backbone.layers.0.norm.weight,
backbone.layers.1.mixer.gate.e_score_correction_bias,
backbone.layers.1.mixer.gate.weight,
backbone.layers.1.mixer.shared_experts.down_proj.biases,
backbone.layers.1.mixer.shared_experts.down_proj.scales,
backbone.layers.1.mixer.shared_experts.down_proj.weight,
backbone.layers.1.mixer.shared_experts.up_proj.biases,
backbone.layers.1.mixer.shared_experts.up_proj.scales,
backbone.layers.1.mixer.shared_experts.up_proj.weight,
backbone.layers.1.mixer.switch_mlp.fc1.biases,
backbone.layers.1.mixer.switch_mlp.fc1.scales,
backbone.layers.1.mixer.switch_mlp.fc1.weight,
backbone.layers.1.mixer.switch_mlp.fc2.biases,
backbone.layers.1.mixer.switch_mlp.fc2.scales,
backbone.layers.1.mixer.switch_mlp.fc2.weight,
backbone.layers.1.norm.weight,
backbone.layers.10.mixer.gate.e_score_correction_bias,
backbone.layers.10.mixer.gate.weight,
backbone.layers.10.mixer.shared_experts.down_proj.biases,
backbone.layers.10.mixer.shared_experts.down_proj.scales,
backbone.layers.10.mixer.shared_experts.down_proj.weight,
backbone.layers.10.mixer.shared_experts.up_proj.biases,
backbone.layers.10.mixer.shared_experts.up_proj.scales,
backbone.layers.10.mixer.shared_experts.up_proj.weight,
backbone.layers.10.mixer.switch_mlp.fc1.biases,
backbone.layers.10.mixer.switch_mlp.fc1.scales,
backbone.layers.10.mixer.switch_mlp.fc1.weight,
backbone.layers.10.mixer.switch_mlp.fc2.biases,
backbone.layers.10.mixer.switch_mlp.fc2.scales,
backbone.layers.10.mixer.switch_mlp.fc2.weight,
backbone.layers.10.norm.weight,
backbone.layers.11.mixer.A_log,
backbone.layers.11.mixer.D,
backbone.layers.11.mixer.conv1d.bias,
backbone.layers.11.mixer.conv1d.weight,
backbone.layers.11.mixer.dt_bias,
backbone.layers.11.mixer.in_proj.biases,
backbone.layers.11.mixer.in_proj.scales,
backbone.layers.11.mixer.in_proj.weight,
backbone.layers.11.mixer.norm.weight,
backbone.layers.11.mixer.out_proj.biases,
backbone.layers.11.mixer.out_proj.scales,
backbone.layers.11.mixer.out_proj.weight,
backbone.layers.11.norm.weight,
backbone.layers.12.mixer.k_proj.biases,
backbone.layers.12.mixer.k_proj.scales,
backbone.layers.12.mixer.k_proj.weight,
backbone.layers.12.mixer.o_proj.biases,
backbone.layers.12.mixer.o_proj.scales,
backbone.layers.12.mixer.o_proj.weight,
backbone.layers.12.mixer.q_proj.biases,
backbone.layers.12.mixer.q_proj.scales,
backbone.layers.12.mixer.q_proj.weight,
backbone.layers.12.mixer.v_proj.biases,
backbone.layers.12.mixer.v_proj.scales,
backbone.layers.12.mixer.v_proj.weight,
backbone.layers.12.norm.weight,
backbone.layers.13.mixer.gate.e_score_correction_bias,
backbone.layers.13.mixer.gate.weight,
backbone.layers.13.mixer.shared_experts.down_proj.biases,
backbone.layers.13.mixer.shared_experts.down_proj.scales,
backbone.layers.13.mixer.shared_experts.down_proj.weight,
backbone.layers.13.mixer.shared_experts.up_proj.biases,
backbone.layers.13.mixer.shared_experts.up_proj.scales,
backbone.layers.13.mixer.shared_experts.up_proj.weight,
backbone.layers.13.mixer.switch_mlp.fc1.biases,
backbone.layers.13.mixer.switch_mlp.fc1.scales,
backbone.layers.13.mixer.switch_mlp.fc1.weight,
backbone.layers.13.mixer.switch_mlp.fc2.biases,
backbone.layers.13.mixer.switch_mlp.fc2.scales,
backbone.layers.13.mixer.switch_mlp.fc2.weight,
backbone.layers.13.norm.weight,
backbone.layers.14.mixer.A_log,
backbone.layers.14.mixer.D,
backbone.layers.14.mixer.conv1d.bias,
backbone.layers.14.mixer.conv1d.weight,
backbone.layers.14.mixer.dt_bias,
backbone.layers.14.mixer.in_proj.biases,
backbone.layers.14.mixer.in_proj.scales,
backbone.layers.14.mixer.in_proj.weight,
backbone.layers.14.mixer.norm.weight,
backbone.layers.14.mixer.out_proj.biases,
backbone.layers.14.mixer.out_proj.scales,
backbone.layers.14.mixer.out_proj.weight,
backbone.layers.14.norm.weight,
backbone.layers.15.mixer.gate.e_score_correction_bias,
backbone.layers.15.mixer.gate.weight,
backbone.layers.15.mixer.shared_experts.down_proj.biases,
backbone.layers.15.mixer.shared_experts.down_proj.scales,
backbone.layers.15.mixer.shared_experts.down_proj.weight,
backbone.layers.15.mixer.shared_experts.up_proj.biases,
backbone.layers.15.mixer.shared_experts.up_proj.scales,
backbone.layers.15.mixer.shared_experts.up_proj.weight,
backbone.layers.15.mixer.switch_mlp.fc1.biases,
backbone.layers.15.mixer.switch_mlp.fc1.scales,
backbone.layers.15.mixer.switch_mlp.fc1.weight,
backbone.layers.15.mixer.switch_mlp.fc2.biases,
backbone.layers.15.mixer.switch_mlp.fc2.scales,
backbone.layers.15.mixer.switch_mlp.fc2.weight,
backbone.layers.15.norm.weight,
backbone.layers.16.mixer.A_log,
backbone.layers.16.mixer.D,
backbone.layers.16.mixer.conv1d.bias,
backbone.layers.16.mixer.conv1d.weight,
backbone.layers.16.mixer.dt_bias,
backbone.layers.16.mixer.in_proj.biases,
backbone.layers.16.mixer.in_proj.scales,
backbone.layers.16.mixer.in_proj.weight,
backbone.layers.16.mixer.norm.weight,
backbone.layers.16.mixer.out_proj.biases,
backbone.layers.16.mixer.out_proj.scales,
backbone.layers.16.mixer.out_proj.weight,
backbone.layers.16.norm.weight,
backbone.layers.17.mixer.gate.e_score_correction_bias,
backbone.layers.17.mixer.gate.weight,
backbone.layers.17.mixer.shared_experts.down_proj.biases,
backbone.layers.17.mixer.shared_experts.down_proj.scales,
backbone.layers.17.mixer.shared_experts.down_proj.weight,
backbone.layers.17.mixer.shared_experts.up_proj.biases,
backbone.layers.17.mixer.shared_experts.up_proj.scales,
backbone.layers.17.mixer.shared_experts.up_proj.weight,
backbone.layers.17.mixer.switch_mlp.fc1.biases,
backbone.layers.17.mixer.switch_mlp.fc1.scales,
backbone.layers.17.mixer.switch_mlp.fc1.weight,
backbone.layers.17.mixer.switch_mlp.fc2.biases,
backbone.layers.17.mixer.switch_mlp.fc2.scales,
backbone.layers.17.mixer.switch_mlp.fc2.weight,
backbone.layers.17.norm.weight,
backbone.layers.18.mixer.A_log,
backbone.layers.18.mixer.D,
backbone.layers.18.mixer.conv1d.bias,
backbone.layers.18.mixer.conv1d.weight,
backbone.layers.18.mixer.dt_bias,
backbone.layers.18.mixer.in_proj.biases,
backbone.layers.18.mixer.in_proj.scales,
backbone.layers.18.mixer.in_proj.weight,
backbone.layers.18.mixer.norm.weight,
backbone.layers.18.mixer.out_proj.biases,
backbone.layers.18.mixer.out_proj.scales,
backbone.layers.18.mixer.out_proj.weight,
backbone.layers.18.norm.weight,
backbone.layers.19.mixer.k_proj.biases,
backbone.layers.19.mixer.k_proj.scales,
backbone.layers.19.mixer.k_proj.weight,
backbone.layers.19.mixer.o_proj.biases,
backbone.layers.19.mixer.o_proj.scales,
backbone.layers.19.mixer.o_proj.weight,
backbone.layers.19.mixer.q_proj.biases,
backbone.layers.19.mixer.q_proj.scales,
backbone.layers.19.mixer.q_proj.weight,
backbone.layers.19.mixer.v_proj.biases,
backbone.layers.19.mixer.v_proj.scales,
backbone.layers.19.mixer.v_proj.weight,
backbone.layers.19.norm.weight,
backbone.layers.2.mixer.A_log,
backbone.layers.2.mixer.D,
backbone.layers.2.mixer.conv1d.bias,
backbone.layers.2.mixer.conv1d.weight,
backbone.layers.2.mixer.dt_bias,
backbone.layers.2.mixer.in_proj.biases,
backbone.layers.2.mixer.in_proj.scales,
backbone.layers.2.mixer.in_proj.weight,
backbone.layers.2.mixer.norm.weight,
backbone.layers.2.mixer.out_proj.biases,
backbone.layers.2.mixer.out_proj.scales,
backbone.layers.2.mixer.out_proj.weight,
backbone.layers.2.norm.weight,
backbone.layers.20.mixer.gate.e_score_correction_bias,
backbone.layers.20.mixer.gate.weight,
backbone.layers.20.mixer.shared_experts.down_proj.biases,
backbone.layers.20.mixer.shared_experts.down_proj.scales,
backbone.layers.20.mixer.shared_experts.down_proj.weight,
backbone.layers.20.mixer.shared_experts.up_proj.biases,
backbone.layers.20.mixer.shared_experts.up_proj.scales,
backbone.layers.20.mixer.shared_experts.up_proj.weight,
backbone.layers.20.mixer.switch_mlp.fc1.biases,
backbone.layers.20.mixer.switch_mlp.fc1.scales,
backbone.layers.20.mixer.switch_mlp.fc1.weight,
backbone.layers.20.mixer.switch_mlp.fc2.biases,
backbone.layers.20.mixer.switch_mlp.fc2.scales,
backbone.layers.20.mixer.switch_mlp.fc2.weight,
backbone.layers.20.norm.weight,
backbone.layers.21.mixer.A_log,
backbone.layers.21.mixer.D,
backbone.layers.21.mixer.conv1d.bias,
backbone.layers.21.mixer.conv1d.weight,
backbone.layers.21.mixer.dt_bias,
backbone.layers.21.mixer.in_proj.biases,
backbone.layers.21.mixer.in_proj.scales,
backbone.layers.21.mixer.in_proj.weight,
backbone.layers.21.mixer.norm.weight,
backbone.layers.21.mixer.out_proj.biases,
backbone.layers.21.mixer.out_proj.scales,
backbone.layers.21.mixer.out_proj.weight,
backbone.layers.21.norm.weight,
backbone.layers.22.mixer.gate.e_score_correction_bias,
backbone.layers.22.mixer.gate.weight,
backbone.layers.22.mixer.shared_experts.down_proj.biases,
backbone.layers.22.mixer.shared_experts.down_proj.scales,
backbone.layers.22.mixer.shared_experts.down_proj.weight,
backbone.layers.22.mixer.shared_experts.up_proj.biases,
backbone.layers.22.mixer.shared_experts.up_proj.scales,
backbone.layers.22.mixer.shared_experts.up_proj.weight,
backbone.layers.22.mixer.switch_mlp.fc1.biases,
backbone.layers.22.mixer.switch_mlp.fc1.scales,
backbone.layers.22.mixer.switch_mlp.fc1.weight,
backbone.layers.22.mixer.switch_mlp.fc2.biases,
backbone.layers.22.mixer.switch_mlp.fc2.scales,
backbone.layers.22.mixer.switch_mlp.fc2.weight,
backbone.layers.22.norm.weight,
backbone.layers.23.mixer.A_log,
backbone.layers.23.mixer.D,
backbone.layers.23.mixer.conv1d.bias,
backbone.layers.23.mixer.conv1d.weight,
backbone.layers.23.mixer.dt_bias,
backbone.layers.23.mixer.in_proj.biases,
backbone.layers.23.mixer.in_proj.scales,
backbone.layers.23.mixer.in_proj.weight,
backbone.layers.23.mixer.norm.weight,
backbone.layers.23.mixer.out_proj.biases,
backbone.layers.23.mixer.out_proj.scales,
backbone.layers.23.mixer.out_proj.weight,
backbone.layers.23.norm.weight,
backbone.layers.24.mixer.gate.e_score_correction_bias,
backbone.layers.24.mixer.gate.weight,
backbone.layers.24.mixer.shared_experts.down_proj.biases,
backbone.layers.24.mixer.shared_experts.down_proj.scales,
backbone.layers.24.mixer.shared_experts.down_proj.weight,
backbone.layers.24.mixer.shared_experts.up_proj.biases,
backbone.layers.24.mixer.shared_experts.up_proj.scales,
backbone.layers.24.mixer.shared_experts.up_proj.weight,
backbone.layers.24.mixer.switch_mlp.fc1.biases,
backbone.layers.24.mixer.switch_mlp.fc1.scales,
backbone.layers.24.mixer.switch_mlp.fc1.weight,
backbone.layers.24.mixer.switch_mlp.fc2.biases,
backbone.layers.24.mixer.switch_mlp.fc2.scales,
backbone.layers.24.mixer.switch_mlp.fc2.weight,
backbone.layers.24.norm.weight,
backbone.layers.25.mixer.A_log,
backbone.layers.25.mixer.D,
backbone.layers.25.mixer.conv1d.bias,
backbone.layers.25.mixer.conv1d.weight,
backbone.layers.25.mixer.dt_bias,
backbone.layers.25.mixer.in_proj.biases,
backbone.layers.25.mixer.in_proj.scales,
backbone.layers.25.mixer.in_proj.weight,
backbone.layers.25.mixer.norm.weight,
backbone.layers.25.mixer.out_proj.biases,
backbone.layers.25.mixer.out_proj.scales,
backbone.layers.25.mixer.out_proj.weight,
backbone.layers.25.norm.weight,
backbone.layers.26.mixer.k_proj.biases,
backbone.layers.26.mixer.k_proj.scales,
backbone.layers.26.mixer.k_proj.weight,
backbone.layers.26.mixer.o_proj.biases,
backbone.layers.26.mixer.o_proj.scales,
backbone.layers.26.mixer.o_proj.weight,
backbone.layers.26.mixer.q_proj.biases,
backbone.layers.26.mixer.q_proj.scales,
backbone.layers.26.mixer.q_proj.weight,
backbone.layers.26.mixer.v_proj.biases,
backbone.layers.26.mixer.v_proj.scales,
backbone.layers.26.mixer.v_proj.weight,
backbone.layers.26.norm.weight,
backbone.layers.27.mixer.gate.e_score_correction_bias,
backbone.layers.27.mixer.gate.weight,
backbone.layers.27.mixer.shared_experts.down_proj.biases,
backbone.layers.27.mixer.shared_experts.down_proj.scales,
backbone.layers.27.mixer.shared_experts.down_proj.weight,
backbone.layers.27.mixer.shared_experts.up_proj.biases,
backbone.layers.27.mixer.shared_experts.up_proj.scales,
backbone.layers.27.mixer.shared_experts.up_proj.weight,
backbone.layers.27.mixer.switch_mlp.fc1.biases,
backbone.layers.27.mixer.switch_mlp.fc1.scales,
backbone.layers.27.mixer.switch_mlp.fc1.weight,
backbone.layers.27.mixer.switch_mlp.fc2.biases,
backbone.layers.27.mixer.switch_mlp.fc2.scales,
backbone.layers.27.mixer.switch_mlp.fc2.weight,
backbone.layers.27.norm.weight,
backbone.layers.28.mixer.A_log,
backbone.layers.28.mixer.D,
backbone.layers.28.mixer.conv1d.bias,
backbone.layers.28.mixer.conv1d.weight,
backbone.layers.28.mixer.dt_bias,
backbone.layers.28.mixer.in_proj.biases,
backbone.layers.28.mixer.in_proj.scales,
backbone.layers.28.mixer.in_proj.weight,
backbone.layers.28.mixer.norm.weight,
backbone.layers.28.mixer.out_proj.biases,
backbone.layers.28.mixer.out_proj.scales,
backbone.layers.28.mixer.out_proj.weight,
backbone.layers.28.norm.weight,
backbone.layers.29.mixer.gate.e_score_correction_bias,
backbone.layers.29.mixer.gate.weight,
backbone.layers.29.mixer.shared_experts.down_proj.biases,
backbone.layers.29.mixer.shared_experts.down_proj.scales,
backbone.layers.29.mixer.shared_experts.down_proj.weight,
backbone.layers.29.mixer.shared_experts.up_proj.biases,
backbone.layers.29.mixer.shared_experts.up_proj.scales,
backbone.layers.29.mixer.shared_experts.up_proj.weight,
backbone.layers.29.mixer.switch_mlp.fc1.biases,
backbone.layers.29.mixer.switch_mlp.fc1.scales,
backbone.layers.29.mixer.switch_mlp.fc1.weight,
backbone.layers.29.mixer.switch_mlp.fc2.biases,
backbone.layers.29.mixer.switch_mlp.fc2.scales,
backbone.layers.29.mixer.switch_mlp.fc2.weight,
backbone.layers.29.norm.weight,
backbone.layers.3.mixer.gate.e_score_correction_bias,
backbone.layers.3.mixer.gate.weight,
backbone.layers.3.mixer.shared_experts.down_proj.biases,
backbone.layers.3.mixer.shared_experts.down_proj.scales,
backbone.layers.3.mixer.shared_experts.down_proj.weight,
backbone.layers.3.mixer.shared_experts.up_proj.biases,
backbone.layers.3.mixer.shared_experts.up_proj.scales,
backbone.layers.3.mixer.shared_experts.up_proj.weight,
backbone.layers.3.mixer.switch_mlp.fc1.biases,
backbone.layers.3.mixer.switch_mlp.fc1.scales,
backbone.layers.3.mixer.switch_mlp.fc1.weight,
backbone.layers.3.mixer.switch_mlp.fc2.biases,
backbone.layers.3.mixer.switch_mlp.fc2.scales,
backbone.layers.3.mixer.switch_mlp.fc2.weight,
backbone.layers.3.norm.weight,
backbone.layers.30.mixer.A_log,
backbone.layers.30.mixer.D,
backbone.layers.30.mixer.conv1d.bias,
backbone.layers.30.mixer.conv1d.weight,
backbone.layers.30.mixer.dt_bias,
backbone.layers.30.mixer.in_proj.biases,
backbone.layers.30.mixer.in_proj.scales,
backbone.layers.30.mixer.in_proj.weight,
backbone.layers.30.mixer.norm.weight,
backbone.layers.30.mixer.out_proj.biases,
backbone.layers.30.mixer.out_proj.scales,
backbone.layers.30.mixer.out_proj.weight,
backbone.layers.30.norm.weight,
backbone.layers.31.mixer.gate.e_score_correction_bias,
backbone.layers.31.mixer.gate.weight,
backbone.layers.31.mixer.shared_experts.down_proj.biases,
backbone.layers.31.mixer.shared_experts.down_proj.scales,
backbone.layers.31.mixer.shared_experts.down_proj.weight,
backbone.layers.31.mixer.shared_experts.up_proj.biases,
backbone.layers.31.mixer.shared_experts.up_proj.scales,
backbone.layers.31.mixer.shared_experts.up_proj.weight,
backbone.layers.31.mixer.switch_mlp.fc1.biases,
backbone.layers.31.mixer.switch_mlp.fc1.scales,
backbone.layers.31.mixer.switch_mlp.fc1.weight,
backbone.layers.31.mixer.switch_mlp.fc2.biases,
backbone.layers.31.mixer.switch_mlp.fc2.scales,
backbone.layers.31.mixer.switch_mlp.fc2.weight,
backbone.layers.31.norm.weight,
backbone.layers.32.mixer.A_log,
backbone.layers.32.mixer.D,
backbone.layers.32.mixer.conv1d.bias,
backbone.layers.32.mixer.conv1d.weight,
backbone.layers.32.mixer.dt_bias,
backbone.layers.32.mixer.in_proj.biases,
backbone.layers.32.mixer.in_proj.scales,
backbone.layers.32.mixer.in_proj.weight,
backbone.layers.32.mixer.norm.weight,
backbone.layers.32.mixer.out_proj.biases,
backbone.layers.32.mixer.out_proj.scales,
backbone.layers.32.mixer.out_proj.weight,
backbone.layers.32.norm.weight,
backbone.layers.33.mixer.k_proj.biases,
backbone.layers.33.mixer.k_proj.scales,
backbone.layers.33.mixer.k_proj.weight,
backbone.layers.33.mixer.o_proj.biases,
backbone.layers.33.mixer.o_proj.scales,
backbone.layers.33.mixer.o_proj.weight,
backbone.layers.33.mixer.q_proj.biases,
backbone.layers.33.mixer.q_proj.scales,
backbone.layers.33.mixer.q_proj.weight,
backbone.layers.33.mixer.v_proj.biases,
backbone.layers.33.mixer.v_proj.scales,
backbone.layers.33.mixer.v_proj.weight,
backbone.layers.33.norm.weight,
backbone.layers.34.mixer.gate.e_score_correction_bias,
backbone.layers.34.mixer.gate.weight,
backbone.layers.34.mixer.shared_experts.down_proj.biases,
backbone.layers.34.mixer.shared_experts.down_proj.scales,
backbone.layers.34.mixer.shared_experts.down_proj.weight,
backbone.layers.34.mixer.shared_experts.up_proj.biases,
backbone.layers.34.mixer.shared_experts.up_proj.scales,
backbone.layers.34.mixer.shared_experts.up_proj.weight,
backbone.layers.34.mixer.switch_mlp.fc1.biases,
backbone.layers.34.mixer.switch_mlp.fc1.scales,
backbone.layers.34.mixer.switch_mlp.fc1.weight,
backbone.layers.34.mixer.switch_mlp.fc2.biases,
backbone.layers.34.mixer.switch_mlp.fc2.scales,
backbone.layers.34.mixer.switch_mlp.fc2.weight,
backbone.layers.34.norm.weight,
backbone.layers.35.mixer.A_log,
backbone.layers.35.mixer.D,
backbone.layers.35.mixer.conv1d.bias,
backbone.layers.35.mixer.conv1d.weight,
backbone.layers.35.mixer.dt_bias,
backbone.layers.35.mixer.in_proj.biases,
backbone.layers.35.mixer.in_proj.scales,
backbone.layers.35.mixer.in_proj.weight,
backbone.layers.35.mixer.norm.weight,
backbone.layers.35.mixer.out_proj.biases,
backbone.layers.35.mixer.out_proj.scales,
backbone.layers.35.mixer.out_proj.weight,
backbone.layers.35.norm.weight,
backbone.layers.36.mixer.gate.e_score_correction_bias,
backbone.layers.36.mixer.gate.weight,
backbone.layers.36.mixer.shared_experts.down_proj.biases,
backbone.layers.36.mixer.shared_experts.down_proj.scales,
backbone.layers.36.mixer.shared_experts.down_proj.weight,
backbone.layers.36.mixer.shared_experts.up_proj.biases,
backbone.layers.36.mixer.shared_experts.up_proj.scales,
backbone.layers.36.mixer.shared_experts.up_proj.weight,
backbone.layers.36.mixer.switch_mlp.fc1.biases,
backbone.layers.36.mixer.switch_mlp.fc1.scales,
backbone.layers.36.mixer.switch_mlp.fc1.weight,
backbone.layers.36.mixer.switch_mlp.fc2.biases,
backbone.layers.36.mixer.switch_mlp.fc2.scales,
backbone.layers.36.mixer.switch_mlp.fc2.weight,
backbone.layers.36.norm.weight,
backbone.layers.37.mixer.A_log,
backbone.layers.37.mixer.D,
backbone.layers.37.mixer.conv1d.bias,
backbone.layers.37.mixer.conv1d.weight,
backbone.layers.37.mixer.dt_bias,
backbone.layers.37.mixer.in_proj.biases,
backbone.layers.37.mixer.in_proj.scales,
backbone.layers.37.mixer.in_proj.weight,
backbone.layers.37.mixer.norm.weight,
backbone.layers.37.mixer.out_proj.biases,
backbone.layers.37.mixer.out_proj.scales,
backbone.layers.37.mixer.out_proj.weight,
backbone.layers.37.norm.weight,
backbone.layers.38.mixer.gate.e_score_correction_bias,
backbone.layers.38.mixer.gate.weight,
backbone.layers.38.mixer.shared_experts.down_proj.biases,
backbone.layers.38.mixer.shared_experts.down_proj.scales,
backbone.layers.38.mixer.shared_experts.down_proj.weight,
backbone.layers.38.mixer.shared_experts.up_proj.biases,
backbone.layers.38.mixer.shared_experts.up_proj.scales,
backbone.layers.38.mixer.shared_experts.up_proj.weight,
backbone.layers.38.mixer.switch_mlp.fc1.biases,
backbone.layers.38.mixer.switch_mlp.fc1.scales,
backbone.layers.38.mixer.switch_mlp.fc1.weight,
backbone.layers.38.mixer.switch_mlp.fc2.biases,
backbone.layers.38.mixer.switch_mlp.fc2.scales,
backbone.layers.38.mixer.switch_mlp.fc2.weight,
backbone.layers.38.norm.weight,
backbone.layers.39.mixer.A_log,
backbone.layers.39.mixer.D,
backbone.layers.39.mixer.conv1d.bias,
backbone.layers.39.mixer.conv1d.weight,
backbone.layers.39.mixer.dt_bias,
backbone.layers.39.mixer.in_proj.biases,
backbone.layers.39.mixer.in_proj.scales,
backbone.layers.39.mixer.in_proj.weight,
backbone.layers.39.mixer.norm.weight,
backbone.layers.39.mixer.out_proj.biases,
backbone.layers.39.mixer.out_proj.scales,
backbone.layers.39.mixer.out_proj.weight,
backbone.layers.39.norm.weight,
backbone.layers.4.mixer.A_log,
backbone.layers.4.mixer.D,
backbone.layers.4.mixer.conv1d.bias,
backbone.layers.4.mixer.conv1d.weight,
backbone.layers.4.mixer.dt_bias,
backbone.layers.4.mixer.in_proj.biases,
backbone.layers.4.mixer.in_proj.scales,
backbone.layers.4.mixer.in_proj.weight,
backbone.layers.4.mixer.norm.weight,
backbone.layers.4.mixer.out_proj.biases,
backbone.layers.4.mixer.out_proj.scales,
backbone.layers.4.mixer.out_proj.weight,
backbone.layers.4.norm.weight,
backbone.layers.40.mixer.gate.e_score_correction_bias,
backbone.layers.40.mixer.gate.weight,
backbone.layers.40.mixer.shared_experts.down_proj.biases,
backbone.layers.40.mixer.shared_experts.down_proj.scales,
backbone.layers.40.mixer.shared_experts.down_proj.weight,
backbone.layers.40.mixer.shared_experts.up_proj.biases,
backbone.layers.40.mixer.shared_experts.up_proj.scales,
backbone.layers.40.mixer.shared_experts.up_proj.weight,
backbone.layers.40.mixer.switch_mlp.fc1.biases,
backbone.layers.40.mixer.switch_mlp.fc1.scales,
backbone.layers.40.mixer.switch_mlp.fc1.weight,
backbone.layers.40.mixer.switch_mlp.fc2.biases,
backbone.layers.40.mixer.switch_mlp.fc2.scales,
backbone.layers.40.mixer.switch_mlp.fc2.weight,
backbone.layers.40.norm.weight,
backbone.layers.41.mixer.A_log,
backbone.layers.41.mixer.D,
backbone.layers.41.mixer.conv1d.bias,
backbone.layers.41.mixer.conv1d.weight,
backbone.layers.41.mixer.dt_bias,
backbone.layers.41.mixer.in_proj.biases,
backbone.layers.41.mixer.in_proj.scales,
backbone.layers.41.mixer.in_proj.weight,
backbone.layers.41.mixer.norm.weight,
backbone.layers.41.mixer.out_proj.biases,
backbone.layers.41.mixer.out_proj.scales,
backbone.layers.41.mixer.out_proj.weight,
backbone.layers.41.norm.weight,
backbone.layers.42.mixer.k_proj.biases,
backbone.layers.42.mixer.k_proj.scales,
backbone.layers.42.mixer.k_proj.weight,
backbone.layers.42.mixer.o_proj.biases,
backbone.layers.42.mixer.o_proj.scales,
backbone.layers.42.mixer.o_proj.weight,
backbone.layers.42.mixer.q_proj.biases,
backbone.layers.42.mixer.q_proj.scales,
backbone.layers.42.mixer.q_proj.weight,
backbone.layers.42.mixer.v_proj.biases,
backbone.layers.42.mixer.v_proj.scales,
backbone.layers.42.mixer.v_proj.weight,
backbone.layers.42.norm.weight,
backbone.layers.43.mixer.gate.e_score_correction_bias,
backbone.layers.43.mixer.gate.weight,
backbone.layers.43.mixer.shared_experts.down_proj.biases,
backbone.layers.43.mixer.shared_experts.down_proj.scales,
backbone.layers.43.mixer.shared_experts.down_proj.weight,
backbone.layers.43.mixer.shared_experts.up_proj.biases,
backbone.layers.43.mixer.shared_experts.up_proj.scales,
backbone.layers.43.mixer.shared_experts.up_proj.weight,
backbone.layers.43.mixer.switch_mlp.fc1.biases,
backbone.layers.43.mixer.switch_mlp.fc1.scales,
backbone.layers.43.mixer.switch_mlp.fc1.weight,
backbone.layers.43.mixer.switch_mlp.fc2.biases,
backbone.layers.43.mixer.switch_mlp.fc2.scales,
backbone.layers.43.mixer.switch_mlp.fc2.weight,
backbone.layers.43.norm.weight,
backbone.layers.44.mixer.A_log,
backbone.layers.44.mixer.D,
backbone.layers.44.mixer.conv1d.bias,
backbone.layers.44.mixer.conv1d.weight,
backbone.layers.44.mixer.dt_bias,
backbone.layers.44.mixer.in_proj.biases,
backbone.layers.44.mixer.in_proj.scales,
backbone.layers.44.mixer.in_proj.weight,
backbone.layers.44.mixer.norm.weight,
backbone.layers.44.mixer.out_proj.biases,
backbone.layers.44.mixer.out_proj.scales,
backbone.layers.44.mixer.out_proj.weight,
backbone.layers.44.norm.weight,
backbone.layers.45.mixer.gate.e_score_correction_bias,
backbone.layers.45.mixer.gate.weight,
backbone.layers.45.mixer.shared_experts.down_proj.biases,
backbone.layers.45.mixer.shared_experts.down_proj.scales,
backbone.layers.45.mixer.shared_experts.down_proj.weight,
backbone.layers.45.mixer.shared_experts.up_proj.biases,
backbone.layers.45.mixer.shared_experts.up_proj.scales,
backbone.layers.45.mixer.shared_experts.up_proj.weight,
backbone.layers.45.mixer.switch_mlp.fc1.biases,
backbone.layers.45.mixer.switch_mlp.fc1.scales,
backbone.layers.45.mixer.switch_mlp.fc1.weight,
backbone.layers.45.mixer.switch_mlp.fc2.biases,
backbone.layers.45.mixer.switch_mlp.fc2.scales,
backbone.layers.45.mixer.switch_mlp.fc2.weight,
backbone.layers.45.norm.weight,
backbone.layers.46.mixer.A_log,
backbone.layers.46.mixer.D,
backbone.layers.46.mixer.conv1d.bias,
backbone.layers.46.mixer.conv1d.weight,
backbone.layers.46.mixer.dt_bias,
backbone.layers.46.mixer.in_proj.biases,
backbone.layers.46.mixer.in_proj.scales,
backbone.layers.46.mixer.in_proj.weight,
backbone.layers.46.mixer.norm.weight,
backbone.layers.46.mixer.out_proj.biases,
backbone.layers.46.mixer.out_proj.scales,
backbone.layers.46.mixer.out_proj.weight,
backbone.layers.46.norm.weight,
backbone.layers.47.mixer.gate.e_score_correction_bias,
backbone.layers.47.mixer.gate.weight,
backbone.layers.47.mixer.shared_experts.down_proj.biases,
backbone.layers.47.mixer.shared_experts.down_proj.scales,
backbone.layers.47.mixer.shared_experts.down_proj.weight,
backbone.layers.47.mixer.shared_experts.up_proj.biases,
backbone.layers.47.mixer.shared_experts.up_proj.scales,
backbone.layers.47.mixer.shared_experts.up_proj.weight,
backbone.layers.47.mixer.switch_mlp.fc1.biases,
backbone.layers.47.mixer.switch_mlp.fc1.scales,
backbone.layers.47.mixer.switch_mlp.fc1.weight,
backbone.layers.47.mixer.switch_mlp.fc2.biases,
backbone.layers.47.mixer.switch_mlp.fc2.scales,
backbone.layers.47.mixer.switch_mlp.fc2.weight,
backbone.layers.47.norm.weight,
backbone.layers.48.mixer.A_log,
backbone.layers.48.mixer.D,
backbone.layers.48.mixer.conv1d.bias,
backbone.layers.48.mixer.conv1d.weight,
backbone.layers.48.mixer.dt_bias,
backbone.layers.48.mixer.in_proj.biases,
backbone.layers.48.mixer.in_proj.scales,
backbone.layers.48.mixer.in_proj.weight,
backbone.layers.48.mixer.norm.weight,
backbone.layers.48.mixer.out_proj.biases,
backbone.layers.48.mixer.out_proj.scales,
backbone.layers.48.mixer.out_proj.weight,
backbone.layers.48.norm.weight,
backbone.layers.49.mixer.gate.e_score_correction_bias,
backbone.layers.49.mixer.gate.weight,
backbone.layers.49.mixer.shared_experts.down_proj.biases,
backbone.layers.49.mixer.shared_experts.down_proj.scales,
backbone.layers.49.mixer.shared_experts.down_proj.weight,
backbone.layers.49.mixer.shared_experts.up_proj.biases,
backbone.layers.49.mixer.shared_experts.up_proj.scales,
backbone.layers.49.mixer.shared_experts.up_proj.weight,
backbone.layers.49.mixer.switch_mlp.fc1.biases,
backbone.layers.49.mixer.switch_mlp.fc1.scales,
backbone.layers.49.mixer.switch_mlp.fc1.weight,
backbone.layers.49.mixer.switch_mlp.fc2.biases,
backbone.layers.49.mixer.switch_mlp.fc2.scales,
backbone.layers.49.mixer.switch_mlp.fc2.weight,
backbone.layers.49.norm.weight,
backbone.layers.5.mixer.k_proj.biases,
backbone.layers.5.mixer.k_proj.scales,
backbone.layers.5.mixer.k_proj.weight,
backbone.layers.5.mixer.o_proj.biases,
backbone.layers.5.mixer.o_proj.scales,
backbone.layers.5.mixer.o_proj.weight,
backbone.layers.5.mixer.q_proj.biases,
backbone.layers.5.mixer.q_proj.scales,
backbone.layers.5.mixer.q_proj.weight,
backbone.layers.5.mixer.v_proj.biases,
backbone.layers.5.mixer.v_proj.scales,
backbone.layers.5.mixer.v_proj.weight,
backbone.layers.5.norm.weight,
backbone.layers.50.mixer.A_log,
backbone.layers.50.mixer.D,
backbone.layers.50.mixer.conv1d.bias,
backbone.layers.50.mixer.conv1d.weight,
backbone.layers.50.mixer.dt_bias,
backbone.layers.50.mixer.in_proj.biases,
backbone.layers.50.mixer.in_proj.scales,
backbone.layers.50.mixer.in_proj.weight,
backbone.layers.50.mixer.norm.weight,
backbone.layers.50.mixer.out_proj.biases,
backbone.layers.50.mixer.out_proj.scales,
backbone.layers.50.mixer.out_proj.weight,
backbone.layers.50.norm.weight,
backbone.layers.51.mixer.gate.e_score_correction_bias,
backbone.layers.51.mixer.gate.weight,
backbone.layers.51.mixer.shared_experts.down_proj.biases,
backbone.layers.51.mixer.shared_experts.down_proj.scales,
backbone.layers.51.mixer.shared_experts.down_proj.weight,
backbone.layers.51.mixer.shared_experts.up_proj.biases,
backbone.layers.51.mixer.shared_experts.up_proj.scales,
backbone.layers.51.mixer.shared_experts.up_proj.weight,
backbone.layers.51.mixer.switch_mlp.fc1.biases,
backbone.layers.51.mixer.switch_mlp.fc1.scales,
backbone.layers.51.mixer.switch_mlp.fc1.weight,
backbone.layers.51.mixer.switch_mlp.fc2.biases,
backbone.layers.51.mixer.switch_mlp.fc2.scales,
backbone.layers.51.mixer.switch_mlp.fc2.weight,
backbone.layers.51.norm.weight,
backbone.layers.6.mixer.gate.e_score_correction_bias,
backbone.layers.6.mixer.gate.weight,
backbone.layers.6.mixer.shared_experts.down_proj.biases,
backbone.layers.6.mixer.shared_experts.down_proj.scales,
backbone.layers.6.mixer.shared_experts.down_proj.weight,
backbone.layers.6.mixer.shared_experts.up_proj.biases,
backbone.layers.6.mixer.shared_experts.up_proj.scales,
backbone.layers.6.mixer.shared_experts.up_proj.weight,
backbone.layers.6.mixer.switch_mlp.fc1.biases,
backbone.layers.6.mixer.switch_mlp.fc1.scales,
backbone.layers.6.mixer.switch_mlp.fc1.weight,
backbone.layers.6.mixer.switch_mlp.fc2.biases,
backbone.layers.6.mixer.switch_mlp.fc2.scales,
backbone.layers.6.mixer.switch_mlp.fc2.weight,
backbone.layers.6.norm.weight,
backbone.layers.7.mixer.A_log,
backbone.layers.7.mixer.D,
backbone.layers.7.mixer.conv1d.bias,
backbone.layers.7.mixer.conv1d.weight,
backbone.layers.7.mixer.dt_bias,
backbone.layers.7.mixer.in_proj.biases,
backbone.layers.7.mixer.in_proj.scales,
backbone.layers.7.mixer.in_proj.weight,
backbone.layers.7.mixer.norm.weight,
backbone.layers.7.mixer.out_proj.biases,
backbone.layers.7.mixer.out_proj.scales,
backbone.layers.7.mixer.out_proj.weight,
backbone.layers.7.norm.weight,
backbone.layers.8.mixer.gate.e_score_correction_bias,
backbone.layers.8.mixer.gate.weight,
backbone.layers.8.mixer.shared_experts.down_proj.biases,
backbone.layers.8.mixer.shared_experts.down_proj.scales,
backbone.layers.8.mixer.shared_experts.down_proj.weight,
backbone.layers.8.mixer.shared_experts.up_proj.biases,
backbone.layers.8.mixer.shared_experts.up_proj.scales,
backbone.layers.8.mixer.shared_experts.up_proj.weight,
backbone.layers.8.mixer.switch_mlp.fc1.biases,
backbone.layers.8.mixer.switch_mlp.fc1.scales,
backbone.layers.8.mixer.switch_mlp.fc1.weight,
backbone.layers.8.mixer.switch_mlp.fc2.biases,
backbone.layers.8.mixer.switch_mlp.fc2.scales,
backbone.layers.8.mixer.switch_mlp.fc2.weight,
backbone.layers.8.norm.weight,
backbone.layers.9.mixer.A_log,
backbone.layers.9.mixer.D,
backbone.layers.9.mixer.conv1d.bias,
backbone.layers.9.mixer.conv1d.weight,
backbone.layers.9.mixer.dt_bias,
backbone.layers.9.mixer.in_proj.biases,
backbone.layers.9.mixer.in_proj.scales,
backbone.layers.9.mixer.in_proj.weight,
backbone.layers.9.mixer.norm.weight,
backbone.layers.9.mixer.out_proj.biases,
backbone.layers.9.mixer.out_proj.scales,
backbone.layers.9.mixer.out_proj.weight,
backbone.layers.9.norm.weight,
backbone.norm_f.weight,
lm_head.biases,
lm_head.scales,
lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15585, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14472, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 729 parameters not in model: 
backbone.embeddings.biases,
backbone.embeddings.scales,
backbone.embeddings.weight,
backbone.layers.0.mixer.A_log,
backbone.layers.0.mixer.D,
backbone.layers.0.mixer.conv1d.bias,
backbone.layers.0.mixer.conv1d.weight,
backbone.layers.0.mixer.dt_bias,
backbone.layers.0.mixer.in_proj.biases,
backbone.layers.0.mixer.in_proj.scales,
backbone.layers.0.mixer.in_proj.weight,
backbone.layers.0.mixer.norm.weight,
backbone.layers.0.mixer.out_proj.biases,
backbone.layers.0.mixer.out_proj.scales,
backbone.layers.0.mixer.out_proj.weight,
backbone.layers.0.norm.weight,
backbone.layers.1.mixer.gate.e_score_correction_bias,
backbone.layers.1.mixer.gate.weight,
backbone.layers.1.mixer.shared_experts.down_proj.biases,
backbone.layers.1.mixer.shared_experts.down_proj.scales,
backbone.layers.1.mixer.shared_experts.down_proj.weight,
backbone.layers.1.mixer.shared_experts.up_proj.biases,
backbone.layers.1.mixer.shared_experts.up_proj.scales,
backbone.layers.1.mixer.shared_experts.up_proj.weight,
backbone.layers.1.mixer.switch_mlp.fc1.biases,
backbone.layers.1.mixer.switch_mlp.fc1.scales,
backbone.layers.1.mixer.switch_mlp.fc1.weight,
backbone.layers.1.mixer.switch_mlp.fc2.biases,
backbone.layers.1.mixer.switch_mlp.fc2.scales,
backbone.layers.1.mixer.switch_mlp.fc2.weight,
backbone.layers.1.norm.weight,
backbone.layers.10.mixer.gate.e_score_correction_bias,
backbone.layers.10.mixer.gate.weight,
backbone.layers.10.mixer.shared_experts.down_proj.biases,
backbone.layers.10.mixer.shared_experts.down_proj.scales,
backbone.layers.10.mixer.shared_experts.down_proj.weight,
backbone.layers.10.mixer.shared_experts.up_proj.biases,
backbone.layers.10.mixer.shared_experts.up_proj.scales,
backbone.layers.10.mixer.shared_experts.up_proj.weight,
backbone.layers.10.mixer.switch_mlp.fc1.biases,
backbone.layers.10.mixer.switch_mlp.fc1.scales,
backbone.layers.10.mixer.switch_mlp.fc1.weight,
backbone.layers.10.mixer.switch_mlp.fc2.biases,
backbone.layers.10.mixer.switch_mlp.fc2.scales,
backbone.layers.10.mixer.switch_mlp.fc2.weight,
backbone.layers.10.norm.weight,
backbone.layers.11.mixer.A_log,
backbone.layers.11.mixer.D,
backbone.layers.11.mixer.conv1d.bias,
backbone.layers.11.mixer.conv1d.weight,
backbone.layers.11.mixer.dt_bias,
backbone.layers.11.mixer.in_proj.biases,
backbone.layers.11.mixer.in_proj.scales,
backbone.layers.11.mixer.in_proj.weight,
backbone.layers.11.mixer.norm.weight,
backbone.layers.11.mixer.out_proj.biases,
backbone.layers.11.mixer.out_proj.scales,
backbone.layers.11.mixer.out_proj.weight,
backbone.layers.11.norm.weight,
backbone.layers.12.mixer.k_proj.biases,
backbone.layers.12.mixer.k_proj.scales,
backbone.layers.12.mixer.k_proj.weight,
backbone.layers.12.mixer.o_proj.biases,
backbone.layers.12.mixer.o_proj.scales,
backbone.layers.12.mixer.o_proj.weight,
backbone.layers.12.mixer.q_proj.biases,
backbone.layers.12.mixer.q_proj.scales,
backbone.layers.12.mixer.q_proj.weight,
backbone.layers.12.mixer.v_proj.biases,
backbone.layers.12.mixer.v_proj.scales,
backbone.layers.12.mixer.v_proj.weight,
backbone.layers.12.norm.weight,
backbone.layers.13.mixer.gate.e_score_correction_bias,
backbone.layers.13.mixer.gate.weight,
backbone.layers.13.mixer.shared_experts.down_proj.biases,
backbone.layers.13.mixer.shared_experts.down_proj.scales,
backbone.layers.13.mixer.shared_experts.down_proj.weight,
backbone.layers.13.mixer.shared_experts.up_proj.biases,
backbone.layers.13.mixer.shared_experts.up_proj.scales,
backbone.layers.13.mixer.shared_experts.up_proj.weight,
backbone.layers.13.mixer.switch_mlp.fc1.biases,
backbone.layers.13.mixer.switch_mlp.fc1.scales,
backbone.layers.13.mixer.switch_mlp.fc1.weight,
backbone.layers.13.mixer.switch_mlp.fc2.biases,
backbone.layers.13.mixer.switch_mlp.fc2.scales,
backbone.layers.13.mixer.switch_mlp.fc2.weight,
backbone.layers.13.norm.weight,
backbone.layers.14.mixer.A_log,
backbone.layers.14.mixer.D,
backbone.layers.14.mixer.conv1d.bias,
backbone.layers.14.mixer.conv1d.weight,
backbone.layers.14.mixer.dt_bias,
backbone.layers.14.mixer.in_proj.biases,
backbone.layers.14.mixer.in_proj.scales,
backbone.layers.14.mixer.in_proj.weight,
backbone.layers.14.mixer.norm.weight,
backbone.layers.14.mixer.out_proj.biases,
backbone.layers.14.mixer.out_proj.scales,
backbone.layers.14.mixer.out_proj.weight,
backbone.layers.14.norm.weight,
backbone.layers.15.mixer.gate.e_score_correction_bias,
backbone.layers.15.mixer.gate.weight,
backbone.layers.15.mixer.shared_experts.down_proj.biases,
backbone.layers.15.mixer.shared_experts.down_proj.scales,
backbone.layers.15.mixer.shared_experts.down_proj.weight,
backbone.layers.15.mixer.shared_experts.up_proj.biases,
backbone.layers.15.mixer.shared_experts.up_proj.scales,
backbone.layers.15.mixer.shared_experts.up_proj.weight,
backbone.layers.15.mixer.switch_mlp.fc1.biases,
backbone.layers.15.mixer.switch_mlp.fc1.scales,
backbone.layers.15.mixer.switch_mlp.fc1.weight,
backbone.layers.15.mixer.switch_mlp.fc2.biases,
backbone.layers.15.mixer.switch_mlp.fc2.scales,
backbone.layers.15.mixer.switch_mlp.fc2.weight,
backbone.layers.15.norm.weight,
backbone.layers.16.mixer.A_log,
backbone.layers.16.mixer.D,
backbone.layers.16.mixer.conv1d.bias,
backbone.layers.16.mixer.conv1d.weight,
backbone.layers.16.mixer.dt_bias,
backbone.layers.16.mixer.in_proj.biases,
backbone.layers.16.mixer.in_proj.scales,
backbone.layers.16.mixer.in_proj.weight,
backbone.layers.16.mixer.norm.weight,
backbone.layers.16.mixer.out_proj.biases,
backbone.layers.16.mixer.out_proj.scales,
backbone.layers.16.mixer.out_proj.weight,
backbone.layers.16.norm.weight,
backbone.layers.17.mixer.gate.e_score_correction_bias,
backbone.layers.17.mixer.gate.weight,
backbone.layers.17.mixer.shared_experts.down_proj.biases,
backbone.layers.17.mixer.shared_experts.down_proj.scales,
backbone.layers.17.mixer.shared_experts.down_proj.weight,
backbone.layers.17.mixer.shared_experts.up_proj.biases,
backbone.layers.17.mixer.shared_experts.up_proj.scales,
backbone.layers.17.mixer.shared_experts.up_proj.weight,
backbone.layers.17.mixer.switch_mlp.fc1.biases,
backbone.layers.17.mixer.switch_mlp.fc1.scales,
backbone.layers.17.mixer.switch_mlp.fc1.weight,
backbone.layers.17.mixer.switch_mlp.fc2.biases,
backbone.layers.17.mixer.switch_mlp.fc2.scales,
backbone.layers.17.mixer.switch_mlp.fc2.weight,
backbone.layers.17.norm.weight,
backbone.layers.18.mixer.A_log,
backbone.layers.18.mixer.D,
backbone.layers.18.mixer.conv1d.bias,
backbone.layers.18.mixer.conv1d.weight,
backbone.layers.18.mixer.dt_bias,
backbone.layers.18.mixer.in_proj.biases,
backbone.layers.18.mixer.in_proj.scales,
backbone.layers.18.mixer.in_proj.weight,
backbone.layers.18.mixer.norm.weight,
backbone.layers.18.mixer.out_proj.biases,
backbone.layers.18.mixer.out_proj.scales,
backbone.layers.18.mixer.out_proj.weight,
backbone.layers.18.norm.weight,
backbone.layers.19.mixer.k_proj.biases,
backbone.layers.19.mixer.k_proj.scales,
backbone.layers.19.mixer.k_proj.weight,
backbone.layers.19.mixer.o_proj.biases,
backbone.layers.19.mixer.o_proj.scales,
backbone.layers.19.mixer.o_proj.weight,
backbone.layers.19.mixer.q_proj.biases,
backbone.layers.19.mixer.q_proj.scales,
backbone.layers.19.mixer.q_proj.weight,
backbone.layers.19.mixer.v_proj.biases,
backbone.layers.19.mixer.v_proj.scales,
backbone.layers.19.mixer.v_proj.weight,
backbone.layers.19.norm.weight,
backbone.layers.2.mixer.A_log,
backbone.layers.2.mixer.D,
backbone.layers.2.mixer.conv1d.bias,
backbone.layers.2.mixer.conv1d.weight,
backbone.layers.2.mixer.dt_bias,
backbone.layers.2.mixer.in_proj.biases,
backbone.layers.2.mixer.in_proj.scales,
backbone.layers.2.mixer.in_proj.weight,
backbone.layers.2.mixer.norm.weight,
backbone.layers.2.mixer.out_proj.biases,
backbone.layers.2.mixer.out_proj.scales,
backbone.layers.2.mixer.out_proj.weight,
backbone.layers.2.norm.weight,
backbone.layers.20.mixer.gate.e_score_correction_bias,
backbone.layers.20.mixer.gate.weight,
backbone.layers.20.mixer.shared_experts.down_proj.biases,
backbone.layers.20.mixer.shared_experts.down_proj.scales,
backbone.layers.20.mixer.shared_experts.down_proj.weight,
backbone.layers.20.mixer.shared_experts.up_proj.biases,
backbone.layers.20.mixer.shared_experts.up_proj.scales,
backbone.layers.20.mixer.shared_experts.up_proj.weight,
backbone.layers.20.mixer.switch_mlp.fc1.biases,
backbone.layers.20.mixer.switch_mlp.fc1.scales,
backbone.layers.20.mixer.switch_mlp.fc1.weight,
backbone.layers.20.mixer.switch_mlp.fc2.biases,
backbone.layers.20.mixer.switch_mlp.fc2.scales,
backbone.layers.20.mixer.switch_mlp.fc2.weight,
backbone.layers.20.norm.weight,
backbone.layers.21.mixer.A_log,
backbone.layers.21.mixer.D,
backbone.layers.21.mixer.conv1d.bias,
backbone.layers.21.mixer.conv1d.weight,
backbone.layers.21.mixer.dt_bias,
backbone.layers.21.mixer.in_proj.biases,
backbone.layers.21.mixer.in_proj.scales,
backbone.layers.21.mixer.in_proj.weight,
backbone.layers.21.mixer.norm.weight,
backbone.layers.21.mixer.out_proj.biases,
backbone.layers.21.mixer.out_proj.scales,
backbone.layers.21.mixer.out_proj.weight,
backbone.layers.21.norm.weight,
backbone.layers.22.mixer.gate.e_score_correction_bias,
backbone.layers.22.mixer.gate.weight,
backbone.layers.22.mixer.shared_experts.down_proj.biases,
backbone.layers.22.mixer.shared_experts.down_proj.scales,
backbone.layers.22.mixer.shared_experts.down_proj.weight,
backbone.layers.22.mixer.shared_experts.up_proj.biases,
backbone.layers.22.mixer.shared_experts.up_proj.scales,
backbone.layers.22.mixer.shared_experts.up_proj.weight,
backbone.layers.22.mixer.switch_mlp.fc1.biases,
backbone.layers.22.mixer.switch_mlp.fc1.scales,
backbone.layers.22.mixer.switch_mlp.fc1.weight,
backbone.layers.22.mixer.switch_mlp.fc2.biases,
backbone.layers.22.mixer.switch_mlp.fc2.scales,
backbone.layers.22.mixer.switch_mlp.fc2.weight,
backbone.layers.22.norm.weight,
backbone.layers.23.mixer.A_log,
backbone.layers.23.mixer.D,
backbone.layers.23.mixer.conv1d.bias,
backbone.layers.23.mixer.conv1d.weight,
backbone.layers.23.mixer.dt_bias,
backbone.layers.23.mixer.in_proj.biases,
backbone.layers.23.mixer.in_proj.scales,
backbone.layers.23.mixer.in_proj.weight,
backbone.layers.23.mixer.norm.weight,
backbone.layers.23.mixer.out_proj.biases,
backbone.layers.23.mixer.out_proj.scales,
backbone.layers.23.mixer.out_proj.weight,
backbone.layers.23.norm.weight,
backbone.layers.24.mixer.gate.e_score_correction_bias,
backbone.layers.24.mixer.gate.weight,
backbone.layers.24.mixer.shared_experts.down_proj.biases,
backbone.layers.24.mixer.shared_experts.down_proj.scales,
backbone.layers.24.mixer.shared_experts.down_proj.weight,
backbone.layers.24.mixer.shared_experts.up_proj.biases,
backbone.layers.24.mixer.shared_experts.up_proj.scales,
backbone.layers.24.mixer.shared_experts.up_proj.weight,
backbone.layers.24.mixer.switch_mlp.fc1.biases,
backbone.layers.24.mixer.switch_mlp.fc1.scales,
backbone.layers.24.mixer.switch_mlp.fc1.weight,
backbone.layers.24.mixer.switch_mlp.fc2.biases,
backbone.layers.24.mixer.switch_mlp.fc2.scales,
backbone.layers.24.mixer.switch_mlp.fc2.weight,
backbone.layers.24.norm.weight,
backbone.layers.25.mixer.A_log,
backbone.layers.25.mixer.D,
backbone.layers.25.mixer.conv1d.bias,
backbone.layers.25.mixer.conv1d.weight,
backbone.layers.25.mixer.dt_bias,
backbone.layers.25.mixer.in_proj.biases,
backbone.layers.25.mixer.in_proj.scales,
backbone.layers.25.mixer.in_proj.weight,
backbone.layers.25.mixer.norm.weight,
backbone.layers.25.mixer.out_proj.biases,
backbone.layers.25.mixer.out_proj.scales,
backbone.layers.25.mixer.out_proj.weight,
backbone.layers.25.norm.weight,
backbone.layers.26.mixer.k_proj.biases,
backbone.layers.26.mixer.k_proj.scales,
backbone.layers.26.mixer.k_proj.weight,
backbone.layers.26.mixer.o_proj.biases,
backbone.layers.26.mixer.o_proj.scales,
backbone.layers.26.mixer.o_proj.weight,
backbone.layers.26.mixer.q_proj.biases,
backbone.layers.26.mixer.q_proj.scales,
backbone.layers.26.mixer.q_proj.weight,
backbone.layers.26.mixer.v_proj.biases,
backbone.layers.26.mixer.v_proj.scales,
backbone.layers.26.mixer.v_proj.weight,
backbone.layers.26.norm.weight,
backbone.layers.27.mixer.gate.e_score_correction_bias,
backbone.layers.27.mixer.gate.weight,
backbone.layers.27.mixer.shared_experts.down_proj.biases,
backbone.layers.27.mixer.shared_experts.down_proj.scales,
backbone.layers.27.mixer.shared_experts.down_proj.weight,
backbone.layers.27.mixer.shared_experts.up_proj.biases,
backbone.layers.27.mixer.shared_experts.up_proj.scales,
backbone.layers.27.mixer.shared_experts.up_proj.weight,
backbone.layers.27.mixer.switch_mlp.fc1.biases,
backbone.layers.27.mixer.switch_mlp.fc1.scales,
backbone.layers.27.mixer.switch_mlp.fc1.weight,
backbone.layers.27.mixer.switch_mlp.fc2.biases,
backbone.layers.27.mixer.switch_mlp.fc2.scales,
backbone.layers.27.mixer.switch_mlp.fc2.weight,
backbone.layers.27.norm.weight,
backbone.layers.28.mixer.A_log,
backbone.layers.28.mixer.D,
backbone.layers.28.mixer.conv1d.bias,
backbone.layers.28.mixer.conv1d.weight,
backbone.layers.28.mixer.dt_bias,
backbone.layers.28.mixer.in_proj.biases,
backbone.layers.28.mixer.in_proj.scales,
backbone.layers.28.mixer.in_proj.weight,
backbone.layers.28.mixer.norm.weight,
backbone.layers.28.mixer.out_proj.biases,
backbone.layers.28.mixer.out_proj.scales,
backbone.layers.28.mixer.out_proj.weight,
backbone.layers.28.norm.weight,
backbone.layers.29.mixer.gate.e_score_correction_bias,
backbone.layers.29.mixer.gate.weight,
backbone.layers.29.mixer.shared_experts.down_proj.biases,
backbone.layers.29.mixer.shared_experts.down_proj.scales,
backbone.layers.29.mixer.shared_experts.down_proj.weight,
backbone.layers.29.mixer.shared_experts.up_proj.biases,
backbone.layers.29.mixer.shared_experts.up_proj.scales,
backbone.layers.29.mixer.shared_experts.up_proj.weight,
backbone.layers.29.mixer.switch_mlp.fc1.biases,
backbone.layers.29.mixer.switch_mlp.fc1.scales,
backbone.layers.29.mixer.switch_mlp.fc1.weight,
backbone.layers.29.mixer.switch_mlp.fc2.biases,
backbone.layers.29.mixer.switch_mlp.fc2.scales,
backbone.layers.29.mixer.switch_mlp.fc2.weight,
backbone.layers.29.norm.weight,
backbone.layers.3.mixer.gate.e_score_correction_bias,
backbone.layers.3.mixer.gate.weight,
backbone.layers.3.mixer.shared_experts.down_proj.biases,
backbone.layers.3.mixer.shared_experts.down_proj.scales,
backbone.layers.3.mixer.shared_experts.down_proj.weight,
backbone.layers.3.mixer.shared_experts.up_proj.biases,
backbone.layers.3.mixer.shared_experts.up_proj.scales,
backbone.layers.3.mixer.shared_experts.up_proj.weight,
backbone.layers.3.mixer.switch_mlp.fc1.biases,
backbone.layers.3.mixer.switch_mlp.fc1.scales,
backbone.layers.3.mixer.switch_mlp.fc1.weight,
backbone.layers.3.mixer.switch_mlp.fc2.biases,
backbone.layers.3.mixer.switch_mlp.fc2.scales,
backbone.layers.3.mixer.switch_mlp.fc2.weight,
backbone.layers.3.norm.weight,
backbone.layers.30.mixer.A_log,
backbone.layers.30.mixer.D,
backbone.layers.30.mixer.conv1d.bias,
backbone.layers.30.mixer.conv1d.weight,
backbone.layers.30.mixer.dt_bias,
backbone.layers.30.mixer.in_proj.biases,
backbone.layers.30.mixer.in_proj.scales,
backbone.layers.30.mixer.in_proj.weight,
backbone.layers.30.mixer.norm.weight,
backbone.layers.30.mixer.out_proj.biases,
backbone.layers.30.mixer.out_proj.scales,
backbone.layers.30.mixer.out_proj.weight,
backbone.layers.30.norm.weight,
backbone.layers.31.mixer.gate.e_score_correction_bias,
backbone.layers.31.mixer.gate.weight,
backbone.layers.31.mixer.shared_experts.down_proj.biases,
backbone.layers.31.mixer.shared_experts.down_proj.scales,
backbone.layers.31.mixer.shared_experts.down_proj.weight,
backbone.layers.31.mixer.shared_experts.up_proj.biases,
backbone.layers.31.mixer.shared_experts.up_proj.scales,
backbone.layers.31.mixer.shared_experts.up_proj.weight,
backbone.layers.31.mixer.switch_mlp.fc1.biases,
backbone.layers.31.mixer.switch_mlp.fc1.scales,
backbone.layers.31.mixer.switch_mlp.fc1.weight,
backbone.layers.31.mixer.switch_mlp.fc2.biases,
backbone.layers.31.mixer.switch_mlp.fc2.scales,
backbone.layers.31.mixer.switch_mlp.fc2.weight,
backbone.layers.31.norm.weight,
backbone.layers.32.mixer.A_log,
backbone.layers.32.mixer.D,
backbone.layers.32.mixer.conv1d.bias,
backbone.layers.32.mixer.conv1d.weight,
backbone.layers.32.mixer.dt_bias,
backbone.layers.32.mixer.in_proj.biases,
backbone.layers.32.mixer.in_proj.scales,
backbone.layers.32.mixer.in_proj.weight,
backbone.layers.32.mixer.norm.weight,
backbone.layers.32.mixer.out_proj.biases,
backbone.layers.32.mixer.out_proj.scales,
backbone.layers.32.mixer.out_proj.weight,
backbone.layers.32.norm.weight,
backbone.layers.33.mixer.k_proj.biases,
backbone.layers.33.mixer.k_proj.scales,
backbone.layers.33.mixer.k_proj.weight,
backbone.layers.33.mixer.o_proj.biases,
backbone.layers.33.mixer.o_proj.scales,
backbone.layers.33.mixer.o_proj.weight,
backbone.layers.33.mixer.q_proj.biases,
backbone.layers.33.mixer.q_proj.scales,
backbone.layers.33.mixer.q_proj.weight,
backbone.layers.33.mixer.v_proj.biases,
backbone.layers.33.mixer.v_proj.scales,
backbone.layers.33.mixer.v_proj.weight,
backbone.layers.33.norm.weight,
backbone.layers.34.mixer.gate.e_score_correction_bias,
backbone.layers.34.mixer.gate.weight,
backbone.layers.34.mixer.shared_experts.down_proj.biases,
backbone.layers.34.mixer.shared_experts.down_proj.scales,
backbone.layers.34.mixer.shared_experts.down_proj.weight,
backbone.layers.34.mixer.shared_experts.up_proj.biases,
backbone.layers.34.mixer.shared_experts.up_proj.scales,
backbone.layers.34.mixer.shared_experts.up_proj.weight,
backbone.layers.34.mixer.switch_mlp.fc1.biases,
backbone.layers.34.mixer.switch_mlp.fc1.scales,
backbone.layers.34.mixer.switch_mlp.fc1.weight,
backbone.layers.34.mixer.switch_mlp.fc2.biases,
backbone.layers.34.mixer.switch_mlp.fc2.scales,
backbone.layers.34.mixer.switch_mlp.fc2.weight,
backbone.layers.34.norm.weight,
backbone.layers.35.mixer.A_log,
backbone.layers.35.mixer.D,
backbone.layers.35.mixer.conv1d.bias,
backbone.layers.35.mixer.conv1d.weight,
backbone.layers.35.mixer.dt_bias,
backbone.layers.35.mixer.in_proj.biases,
backbone.layers.35.mixer.in_proj.scales,
backbone.layers.35.mixer.in_proj.weight,
backbone.layers.35.mixer.norm.weight,
backbone.layers.35.mixer.out_proj.biases,
backbone.layers.35.mixer.out_proj.scales,
backbone.layers.35.mixer.out_proj.weight,
backbone.layers.35.norm.weight,
backbone.layers.36.mixer.gate.e_score_correction_bias,
backbone.layers.36.mixer.gate.weight,
backbone.layers.36.mixer.shared_experts.down_proj.biases,
backbone.layers.36.mixer.shared_experts.down_proj.scales,
backbone.layers.36.mixer.shared_experts.down_proj.weight,
backbone.layers.36.mixer.shared_experts.up_proj.biases,
backbone.layers.36.mixer.shared_experts.up_proj.scales,
backbone.layers.36.mixer.shared_experts.up_proj.weight,
backbone.layers.36.mixer.switch_mlp.fc1.biases,
backbone.layers.36.mixer.switch_mlp.fc1.scales,
backbone.layers.36.mixer.switch_mlp.fc1.weight,
backbone.layers.36.mixer.switch_mlp.fc2.biases,
backbone.layers.36.mixer.switch_mlp.fc2.scales,
backbone.layers.36.mixer.switch_mlp.fc2.weight,
backbone.layers.36.norm.weight,
backbone.layers.37.mixer.A_log,
backbone.layers.37.mixer.D,
backbone.layers.37.mixer.conv1d.bias,
backbone.layers.37.mixer.conv1d.weight,
backbone.layers.37.mixer.dt_bias,
backbone.layers.37.mixer.in_proj.biases,
backbone.layers.37.mixer.in_proj.scales,
backbone.layers.37.mixer.in_proj.weight,
backbone.layers.37.mixer.norm.weight,
backbone.layers.37.mixer.out_proj.biases,
backbone.layers.37.mixer.out_proj.scales,
backbone.layers.37.mixer.out_proj.weight,
backbone.layers.37.norm.weight,
backbone.layers.38.mixer.gate.e_score_correction_bias,
backbone.layers.38.mixer.gate.weight,
backbone.layers.38.mixer.shared_experts.down_proj.biases,
backbone.layers.38.mixer.shared_experts.down_proj.scales,
backbone.layers.38.mixer.shared_experts.down_proj.weight,
backbone.layers.38.mixer.shared_experts.up_proj.biases,
backbone.layers.38.mixer.shared_experts.up_proj.scales,
backbone.layers.38.mixer.shared_experts.up_proj.weight,
backbone.layers.38.mixer.switch_mlp.fc1.biases,
backbone.layers.38.mixer.switch_mlp.fc1.scales,
backbone.layers.38.mixer.switch_mlp.fc1.weight,
backbone.layers.38.mixer.switch_mlp.fc2.biases,
backbone.layers.38.mixer.switch_mlp.fc2.scales,
backbone.layers.38.mixer.switch_mlp.fc2.weight,
backbone.layers.38.norm.weight,
backbone.layers.39.mixer.A_log,
backbone.layers.39.mixer.D,
backbone.layers.39.mixer.conv1d.bias,
backbone.layers.39.mixer.conv1d.weight,
backbone.layers.39.mixer.dt_bias,
backbone.layers.39.mixer.in_proj.biases,
backbone.layers.39.mixer.in_proj.scales,
backbone.layers.39.mixer.in_proj.weight,
backbone.layers.39.mixer.norm.weight,
backbone.layers.39.mixer.out_proj.biases,
backbone.layers.39.mixer.out_proj.scales,
backbone.layers.39.mixer.out_proj.weight,
backbone.layers.39.norm.weight,
backbone.layers.4.mixer.A_log,
backbone.layers.4.mixer.D,
backbone.layers.4.mixer.conv1d.bias,
backbone.layers.4.mixer.conv1d.weight,
backbone.layers.4.mixer.dt_bias,
backbone.layers.4.mixer.in_proj.biases,
backbone.layers.4.mixer.in_proj.scales,
backbone.layers.4.mixer.in_proj.weight,
backbone.layers.4.mixer.norm.weight,
backbone.layers.4.mixer.out_proj.biases,
backbone.layers.4.mixer.out_proj.scales,
backbone.layers.4.mixer.out_proj.weight,
backbone.layers.4.norm.weight,
backbone.layers.40.mixer.gate.e_score_correction_bias,
backbone.layers.40.mixer.gate.weight,
backbone.layers.40.mixer.shared_experts.down_proj.biases,
backbone.layers.40.mixer.shared_experts.down_proj.scales,
backbone.layers.40.mixer.shared_experts.down_proj.weight,
backbone.layers.40.mixer.shared_experts.up_proj.biases,
backbone.layers.40.mixer.shared_experts.up_proj.scales,
backbone.layers.40.mixer.shared_experts.up_proj.weight,
backbone.layers.40.mixer.switch_mlp.fc1.biases,
backbone.layers.40.mixer.switch_mlp.fc1.scales,
backbone.layers.40.mixer.switch_mlp.fc1.weight,
backbone.layers.40.mixer.switch_mlp.fc2.biases,
backbone.layers.40.mixer.switch_mlp.fc2.scales,
backbone.layers.40.mixer.switch_mlp.fc2.weight,
backbone.layers.40.norm.weight,
backbone.layers.41.mixer.A_log,
backbone.layers.41.mixer.D,
backbone.layers.41.mixer.conv1d.bias,
backbone.layers.41.mixer.conv1d.weight,
backbone.layers.41.mixer.dt_bias,
backbone.layers.41.mixer.in_proj.biases,
backbone.layers.41.mixer.in_proj.scales,
backbone.layers.41.mixer.in_proj.weight,
backbone.layers.41.mixer.norm.weight,
backbone.layers.41.mixer.out_proj.biases,
backbone.layers.41.mixer.out_proj.scales,
backbone.layers.41.mixer.out_proj.weight,
backbone.layers.41.norm.weight,
backbone.layers.42.mixer.k_proj.biases,
backbone.layers.42.mixer.k_proj.scales,
backbone.layers.42.mixer.k_proj.weight,
backbone.layers.42.mixer.o_proj.biases,
backbone.layers.42.mixer.o_proj.scales,
backbone.layers.42.mixer.o_proj.weight,
backbone.layers.42.mixer.q_proj.biases,
backbone.layers.42.mixer.q_proj.scales,
backbone.layers.42.mixer.q_proj.weight,
backbone.layers.42.mixer.v_proj.biases,
backbone.layers.42.mixer.v_proj.scales,
backbone.layers.42.mixer.v_proj.weight,
backbone.layers.42.norm.weight,
backbone.layers.43.mixer.gate.e_score_correction_bias,
backbone.layers.43.mixer.gate.weight,
backbone.layers.43.mixer.shared_experts.down_proj.biases,
backbone.layers.43.mixer.shared_experts.down_proj.scales,
backbone.layers.43.mixer.shared_experts.down_proj.weight,
backbone.layers.43.mixer.shared_experts.up_proj.biases,
backbone.layers.43.mixer.shared_experts.up_proj.scales,
backbone.layers.43.mixer.shared_experts.up_proj.weight,
backbone.layers.43.mixer.switch_mlp.fc1.biases,
backbone.layers.43.mixer.switch_mlp.fc1.scales,
backbone.layers.43.mixer.switch_mlp.fc1.weight,
backbone.layers.43.mixer.switch_mlp.fc2.biases,
backbone.layers.43.mixer.switch_mlp.fc2.scales,
backbone.layers.43.mixer.switch_mlp.fc2.weight,
backbone.layers.43.norm.weight,
backbone.layers.44.mixer.A_log,
backbone.layers.44.mixer.D,
backbone.layers.44.mixer.conv1d.bias,
backbone.layers.44.mixer.conv1d.weight,
backbone.layers.44.mixer.dt_bias,
backbone.layers.44.mixer.in_proj.biases,
backbone.layers.44.mixer.in_proj.scales,
backbone.layers.44.mixer.in_proj.weight,
backbone.layers.44.mixer.norm.weight,
backbone.layers.44.mixer.out_proj.biases,
backbone.layers.44.mixer.out_proj.scales,
backbone.layers.44.mixer.out_proj.weight,
backbone.layers.44.norm.weight,
backbone.layers.45.mixer.gate.e_score_correction_bias,
backbone.layers.45.mixer.gate.weight,
backbone.layers.45.mixer.shared_experts.down_proj.biases,
backbone.layers.45.mixer.shared_experts.down_proj.scales,
backbone.layers.45.mixer.shared_experts.down_proj.weight,
backbone.layers.45.mixer.shared_experts.up_proj.biases,
backbone.layers.45.mixer.shared_experts.up_proj.scales,
backbone.layers.45.mixer.shared_experts.up_proj.weight,
backbone.layers.45.mixer.switch_mlp.fc1.biases,
backbone.layers.45.mixer.switch_mlp.fc1.scales,
backbone.layers.45.mixer.switch_mlp.fc1.weight,
backbone.layers.45.mixer.switch_mlp.fc2.biases,
backbone.layers.45.mixer.switch_mlp.fc2.scales,
backbone.layers.45.mixer.switch_mlp.fc2.weight,
backbone.layers.45.norm.weight,
backbone.layers.46.mixer.A_log,
backbone.layers.46.mixer.D,
backbone.layers.46.mixer.conv1d.bias,
backbone.layers.46.mixer.conv1d.weight,
backbone.layers.46.mixer.dt_bias,
backbone.layers.46.mixer.in_proj.biases,
backbone.layers.46.mixer.in_proj.scales,
backbone.layers.46.mixer.in_proj.weight,
backbone.layers.46.mixer.norm.weight,
backbone.layers.46.mixer.out_proj.biases,
backbone.layers.46.mixer.out_proj.scales,
backbone.layers.46.mixer.out_proj.weight,
backbone.layers.46.norm.weight,
backbone.layers.47.mixer.gate.e_score_correction_bias,
backbone.layers.47.mixer.gate.weight,
backbone.layers.47.mixer.shared_experts.down_proj.biases,
backbone.layers.47.mixer.shared_experts.down_proj.scales,
backbone.layers.47.mixer.shared_experts.down_proj.weight,
backbone.layers.47.mixer.shared_experts.up_proj.biases,
backbone.layers.47.mixer.shared_experts.up_proj.scales,
backbone.layers.47.mixer.shared_experts.up_proj.weight,
backbone.layers.47.mixer.switch_mlp.fc1.biases,
backbone.layers.47.mixer.switch_mlp.fc1.scales,
backbone.layers.47.mixer.switch_mlp.fc1.weight,
backbone.layers.47.mixer.switch_mlp.fc2.biases,
backbone.layers.47.mixer.switch_mlp.fc2.scales,
backbone.layers.47.mixer.switch_mlp.fc2.weight,
backbone.layers.47.norm.weight,
backbone.layers.48.mixer.A_log,
backbone.layers.48.mixer.D,
backbone.layers.48.mixer.conv1d.bias,
backbone.layers.48.mixer.conv1d.weight,
backbone.layers.48.mixer.dt_bias,
backbone.layers.48.mixer.in_proj.biases,
backbone.layers.48.mixer.in_proj.scales,
backbone.layers.48.mixer.in_proj.weight,
backbone.layers.48.mixer.norm.weight,
backbone.layers.48.mixer.out_proj.biases,
backbone.layers.48.mixer.out_proj.scales,
backbone.layers.48.mixer.out_proj.weight,
backbone.layers.48.norm.weight,
backbone.layers.49.mixer.gate.e_score_correction_bias,
backbone.layers.49.mixer.gate.weight,
backbone.layers.49.mixer.shared_experts.down_proj.biases,
backbone.layers.49.mixer.shared_experts.down_proj.scales,
backbone.layers.49.mixer.shared_experts.down_proj.weight,
backbone.layers.49.mixer.shared_experts.up_proj.biases,
backbone.layers.49.mixer.shared_experts.up_proj.scales,
backbone.layers.49.mixer.shared_experts.up_proj.weight,
backbone.layers.49.mixer.switch_mlp.fc1.biases,
backbone.layers.49.mixer.switch_mlp.fc1.scales,
backbone.layers.49.mixer.switch_mlp.fc1.weight,
backbone.layers.49.mixer.switch_mlp.fc2.biases,
backbone.layers.49.mixer.switch_mlp.fc2.scales,
backbone.layers.49.mixer.switch_mlp.fc2.weight,
backbone.layers.49.norm.weight,
backbone.layers.5.mixer.k_proj.biases,
backbone.layers.5.mixer.k_proj.scales,
backbone.layers.5.mixer.k_proj.weight,
backbone.layers.5.mixer.o_proj.biases,
backbone.layers.5.mixer.o_proj.scales,
backbone.layers.5.mixer.o_proj.weight,
backbone.layers.5.mixer.q_proj.biases,
backbone.layers.5.mixer.q_proj.scales,
backbone.layers.5.mixer.q_proj.weight,
backbone.layers.5.mixer.v_proj.biases,
backbone.layers.5.mixer.v_proj.scales,
backbone.layers.5.mixer.v_proj.weight,
backbone.layers.5.norm.weight,
backbone.layers.50.mixer.A_log,
backbone.layers.50.mixer.D,
backbone.layers.50.mixer.conv1d.bias,
backbone.layers.50.mixer.conv1d.weight,
backbone.layers.50.mixer.dt_bias,
backbone.layers.50.mixer.in_proj.biases,
backbone.layers.50.mixer.in_proj.scales,
backbone.layers.50.mixer.in_proj.weight,
backbone.layers.50.mixer.norm.weight,
backbone.layers.50.mixer.out_proj.biases,
backbone.layers.50.mixer.out_proj.scales,
backbone.layers.50.mixer.out_proj.weight,
backbone.layers.50.norm.weight,
backbone.layers.51.mixer.gate.e_score_correction_bias,
backbone.layers.51.mixer.gate.weight,
backbone.layers.51.mixer.shared_experts.down_proj.biases,
backbone.layers.51.mixer.shared_experts.down_proj.scales,
backbone.layers.51.mixer.shared_experts.down_proj.weight,
backbone.layers.51.mixer.shared_experts.up_proj.biases,
backbone.layers.51.mixer.shared_experts.up_proj.scales,
backbone.layers.51.mixer.shared_experts.up_proj.weight,
backbone.layers.51.mixer.switch_mlp.fc1.biases,
backbone.layers.51.mixer.switch_mlp.fc1.scales,
backbone.layers.51.mixer.switch_mlp.fc1.weight,
backbone.layers.51.mixer.switch_mlp.fc2.biases,
backbone.layers.51.mixer.switch_mlp.fc2.scales,
backbone.layers.51.mixer.switch_mlp.fc2.weight,
backbone.layers.51.norm.weight,
backbone.layers.6.mixer.gate.e_score_correction_bias,
backbone.layers.6.mixer.gate.weight,
backbone.layers.6.mixer.shared_experts.down_proj.biases,
backbone.layers.6.mixer.shared_experts.down_proj.scales,
backbone.layers.6.mixer.shared_experts.down_proj.weight,
backbone.layers.6.mixer.shared_experts.up_proj.biases,
backbone.layers.6.mixer.shared_experts.up_proj.scales,
backbone.layers.6.mixer.shared_experts.up_proj.weight,
backbone.layers.6.mixer.switch_mlp.fc1.biases,
backbone.layers.6.mixer.switch_mlp.fc1.scales,
backbone.layers.6.mixer.switch_mlp.fc1.weight,
backbone.layers.6.mixer.switch_mlp.fc2.biases,
backbone.layers.6.mixer.switch_mlp.fc2.scales,
backbone.layers.6.mixer.switch_mlp.fc2.weight,
backbone.layers.6.norm.weight,
backbone.layers.7.mixer.A_log,
backbone.layers.7.mixer.D,
backbone.layers.7.mixer.conv1d.bias,
backbone.layers.7.mixer.conv1d.weight,
backbone.layers.7.mixer.dt_bias,
backbone.layers.7.mixer.in_proj.biases,
backbone.layers.7.mixer.in_proj.scales,
backbone.layers.7.mixer.in_proj.weight,
backbone.layers.7.mixer.norm.weight,
backbone.layers.7.mixer.out_proj.biases,
backbone.layers.7.mixer.out_proj.scales,
backbone.layers.7.mixer.out_proj.weight,
backbone.layers.7.norm.weight,
backbone.layers.8.mixer.gate.e_score_correction_bias,
backbone.layers.8.mixer.gate.weight,
backbone.layers.8.mixer.shared_experts.down_proj.biases,
backbone.layers.8.mixer.shared_experts.down_proj.scales,
backbone.layers.8.mixer.shared_experts.down_proj.weight,
backbone.layers.8.mixer.shared_experts.up_proj.biases,
backbone.layers.8.mixer.shared_experts.up_proj.scales,
backbone.layers.8.mixer.shared_experts.up_proj.weight,
backbone.layers.8.mixer.switch_mlp.fc1.biases,
backbone.layers.8.mixer.switch_mlp.fc1.scales,
backbone.layers.8.mixer.switch_mlp.fc1.weight,
backbone.layers.8.mixer.switch_mlp.fc2.biases,
backbone.layers.8.mixer.switch_mlp.fc2.scales,
backbone.layers.8.mixer.switch_mlp.fc2.weight,
backbone.layers.8.norm.weight,
backbone.layers.9.mixer.A_log,
backbone.layers.9.mixer.D,
backbone.layers.9.mixer.conv1d.bias,
backbone.layers.9.mixer.conv1d.weight,
backbone.layers.9.mixer.dt_bias,
backbone.layers.9.mixer.in_proj.biases,
backbone.layers.9.mixer.in_proj.scales,
backbone.layers.9.mixer.in_proj.weight,
backbone.layers.9.mixer.norm.weight,
backbone.layers.9.mixer.out_proj.biases,
backbone.layers.9.mixer.out_proj.scales,
backbone.layers.9.mixer.out_proj.weight,
backbone.layers.9.norm.weight,
backbone.norm_f.weight,
lm_head.biases,
lm_head.scales,
lm_head.weight.

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 25 files:   0%|          | 0/25 [00:00<?, ?it/s]
Fetching 25 files: 100%|##########| 25/25 [00:00<00:00, 4387.35it/s]
```

</details>

---
