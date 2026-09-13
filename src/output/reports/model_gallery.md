# Model Output Gallery

Generated on: 2026-09-13 02:35:21 BST

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

| Model                                                                                                                   | Mechanical checks      | Total s | Gen TPS    | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------|------------------------|---------|------------|-----------------|---------|------------|---------|---------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `no concerns detected` | 11.48s  | 29.7 tok/s | 3.59            | 23      | 2,393      | 151     | none                                                                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)            | `no concerns detected` | 17.04s  | 74.0 tok/s | 1.34            | 19      | 1,634      | 997     | none                                                                                              |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                          | `no concerns detected` | 9.73s   | 75.2 tok/s | 5.85            | 8.7     | 6,450      | 140     | none                                                                                              |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `no concerns detected` | 28.26s  | 40.9 tok/s | 16.07           | 78      | 6,450      | 103     | none                                                                                              |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `no concerns detected` | 8.96s   | 32.6 tok/s | 1.87            | 18      | 2,612      | 138     | none                                                                                              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `no concerns detected` | 6.00s   | 34.9 tok/s | 1.48            | 17      | 2,113      | 77      | none                                                                                              |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `no concerns detected` | 3.11s   | 205 tok/s  | 0.97            | 4.0     | 2,103      | 80      | none                                                                                              |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                            | `no concerns detected` | 3.27s   | 103 tok/s  | 0.73            | 7.0     | 391        | 101     | none                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `no concerns detected` | 6.31s   | 67.1 tok/s | 2.61            | 13      | 2,926      | 109     | none                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `no concerns detected` | 7.69s   | 64.1 tok/s | 2.63            | 13      | 2,926      | 187     | none                                                                                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `no concerns detected` | 3.78s   | 190 tok/s  | 1.55            | 7.8     | 2,925      | 102     | none                                                                                              |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `no concerns detected` | 6.02s   | 159 tok/s  | 3.21            | 3.9     | 4,083      | 84      | none                                                                                              |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                      | `no concerns detected` | 6.69s   | 73.0 tok/s | 1.26            | 24      | 1,289      | 155     | none                                                                                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `no concerns detected` | 5.24s   | 36.6 tok/s | 0.87            | 9.3     | 1,133      | 107     | none                                                                                              |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `no concerns detected` | 35.94s  | 69.8 tok/s | 21.01           | 8.4     | 16,549     | 912     | none                                                                                              |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                             | `no concerns detected` | 48.18s  | 59.4 tok/s | 44.72           | 11      | 16,547     | 93      | none                                                                                              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `no concerns detected` | 46.02s  | 74.4 tok/s | 40.76           | 25      | 16,563     | 97      | none                                                                                              |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `no concerns detected` | 46.37s  | 88.2 tok/s | 42.71           | 11      | 16,563     | 86      | none                                                                                              |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `no concerns detected` | 76.80s  | 14.4 tok/s | 64.91           | 21      | 16,563     | 124     | none                                                                                              |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `no concerns detected` | 3.74s   | 123 tok/s  | 1.09            | 5.6     | 1,426      | 98      | none                                                                                              |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                          | `no concerns detected` | 100.61s | 48.6 tok/s | 75.79           | 92      | 3,491      | 121     | none                                                                                              |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `no concerns detected` | 17.88s  | 57.2 tok/s | 12.87           | 14      | 16,558     | 136     | none                                                                                              |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `no concerns detected` | 9.80s   | 30.2 tok/s | 1.75            | 17      | 590        | 152     | none                                                                                              |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `no concerns detected` | 5.32s   | 105 tok/s  | 0.98            | 16      | 595        | 96      | none                                                                                              |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `no concerns detected` | 8.48s   | 26.0 tok/s | 1.65            | 20      | 595        | 89      | none                                                                                              |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                         | `no concerns detected` | 3.73s   | 122 tok/s  | 0.77            | 5.9     | 591        | 73      | none                                                                                              |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `no concerns detected` | 3.01s   | 172 tok/s  | 1.18            | 4.7     | 1,379      | 65      | none                                                                                              |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `no concerns detected` | 7.31s   | 37.7 tok/s | 2.21            | 16      | 3,116      | 102     | none                                                                                              |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `concerns detected`    | 2.17s   | 475 tok/s  | 0.63            | 1.9     | 2,112      | 120     | duplicate keywords                                                                                |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `concerns detected`    | 6.24s   | 72.1 tok/s | 1.39            | 8.1     | 1,524      | 216     | duplicate keywords                                                                                |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                           | `concerns detected`    | 5.14s   | 102 tok/s  | 2.07            | 6.5     | 2,090      | 96      | control tokens visible                                                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `concerns detected`    | 6.71s   | 53.0 tok/s | 2.61            | 28      | 591        | 80      | duplicate keywords                                                                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                   | `major concerns`       | 20.55s  | 61.5 tok/s | 1.36            | 20      | 1,324      | 1,000   | cut off at token limit                                                                            |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `major concerns`       | 59.35s  | 21.6 tok/s | 9.10            | 25      | 4,403      | 1,000   | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `major concerns`       | 46.20s  | 71.2 tok/s | 40.44           | 23      | 16,547     | 200     | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                       | `major concerns`       | 5.22s   | 60.3 tok/s | 1.53            | 7.1     | 589        | 88      | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                 | `major concerns`       | 5.24s   | 100 tok/s  | 1.48            | 6.7     | 2,195      | 200     | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected     |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `major concerns`       | 2.29s   | 204 tok/s  | 0.67            | 1.8     | 330        | 92      | labelled fields not detected                                                                      |
| [`mlx-community/InternVL3_5-30B-A3B-4bit`](#model-mlx-community-internvl35-30b-a3b-4bit)                                | `not assessed`         | 0.16s   | -          | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)            | `not assessed`         | 2.54s   | -          | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                           | `not assessed`         | 0.19s   | -          | -               | -       | -          | -       | none                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `mlx-community/granite-4.0-3b-vision-4bit` at 3.01s

Lowest peak memory among completions without detected concerns: `mlx-community/North-Micro-Vision-Instruct-4bit` at 3.9 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Mechanical checks | Observations                                                                                      |
|--------------------------------------------------------------------------------------------------------------|-------------------|---------------------------------------------------------------------------------------------------|
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)        | `major concerns`  | cut off at token limit                                                                            |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)              | `major concerns`  | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)        | `major concerns`  | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                            | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | `major concerns`  | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected     |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/InternVL3_5-30B-A3B-4bit`](#model-mlx-community-internvl35-30b-a3b-4bit)                     | `not assessed`    | none                                                                                              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit) | `not assessed`    | none                                                                                              |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                | `not assessed`    | none                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

A compact preview of each model's final answer (or failure evidence for crashes), in chooser order. Where the requested catalogue fields were detected, the preview shows a little of each: the title, the start of the description, and the first keywords with their count, so the weakest field is not hidden behind a long description. Otherwise it is the first 280 characters. A closed reasoning trace is left out of the preview and reported as an omitted-character count; the complete output, trace included, is in the model's evidence block below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Mechanical checks      | Output preview                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `no concerns detected` | Title: Swan on a calm river with moored boats \| Description: A solitary white swan swims gracefully on a calm river, with leisure boats and cruisers moored alongside r... \| Keywords (20): Swan, river, boats, mooring, residential buildings, greenery, water reflection, aquatic bird, marina, ...                                                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)            | `no concerns detected` | Title: Solitary White Swan Glides on Calm River with Boats and Lush Foliage \| Description: On 12 September 2026 at 17:41 UTC+01 (52.393850°N, 0.270830°E), a solitary wh... \| Keywords (17): White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, ...[2,490 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                          | `no concerns detected` | Title: Swan at River Marina \| Description: A white swan glides on calm river waters, with leisure boats moored alongside riverside residential buildings in the... \| Keywords (13): Swan, River, Marina, Leisure boats, Residential buildings, Calm waters, White swan, Riverbank, Boats moored, ...                                                                                |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `no concerns detected` | Title: Solitary swan glides on calm river waters \| Description: A solitary white swan glides gracefully across calm river waters, framed by lush foliage, with leisu... \| Keywords (17): swan, river, water, foliage, boats, cruisers, residential buildings, moored, overcast, water reflection, ...                                                                               |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `no concerns detected` | Title: Solitary Swan on Calm River Waters with Moored Boats and Riverside Buildings. \| Description: A solitary white swan glides gracefully across the calm river wate... \| Keywords (20): swan, river, boats, moored, residential buildings, foliage, calm waters, leisure, cruisers, riverside, ...                                                                               |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `no concerns detected` | Title: Swan on Calm River \| Description: A white swan glides on tranquil river waters near a marina with moored boats and residential buildings, captured at dusk. \| Keywords (15): swan, river, marina, boats, residential buildings, foliage, trees, water reflection, architecture, motorboat, ...                                                                               |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `no concerns detected` | Title: Swan gliding on river near moored boats and houses \| Description: A white swan swims gracefully on calm river waters framed by lush greenery, with leisure boats... \| Keywords (14): swan, river, boats, houses, greenery, water, reflection, leisure, architecture, waterfront, waterway, ...                                                                               |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                            | `no concerns detected` | Title: Swan Glides on Calm River Near Moored Boats \| Description: A white swan swims peacefully in a tranquil waterway, framed by greenery and moored leisure bo... \| Keywords (16): swan, river, waterway, mooring, motorboat, boats, greenery, trees, vegetation, waterfront, water reflection, ...                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `no concerns detected` | Title: **Swan Gliding Past Moored Boats on a Serene River** \| Description: A solitary white swan swims calmly on a tranquil river, surrounded by reflections of... \| Keywords (16): swan, river, moored boats, marina, residential waterfront, greenery, calm water, reflections, leisure boats, ...                                                                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `no concerns detected` | Title: *Elegant Swan Glides Past Moored Boats on Serene River* \| Description: A solitary white swan gracefully swims across calm river waters on **September 12, 2026,... \| Keywords (20): swan, river, leisure boats, moored boats, riverside residential buildings, greenery, golden sunlight, ...                                                                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `no concerns detected` | Title: Swans by Riverside Marina \| Description: A white swan swims serenely in calm waters at dusk near a marina moored with leisure boats, framed by lush greenery... \| Keywords (17): swan, canal, marina, mooring, motorboat, pier, waterfront, waterway, greenery, residential architecture, ...                                                                                |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `no concerns detected` | Title: Swan on the River \| Description: A solitary white swan glides gracefully across calm river waters, framed by lush greenery. In the background, leisure boa... \| Keywords (13): Swan, River, Greenery, Marina, Boat, Waterfront, Water, Aquatic Bird, Architecture, Leisure Boats, Foliage, ...                                                                               |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                      | `no concerns detected` | Title: White Swan Glides Past Moored Boats On Calm River \| Description: A solitary white swan glades across a calm river in the foreground, its reflection ripplin... \| Keywords (19): white swan, river, moored boats, motorboats, cruisers, marina, pontoon, water reflection, swan, waterfowl, ...                                                                               |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `no concerns detected` | Title: Serene Swan on the River \| Description: A white swan is seen swimming peacefully on a calm river, with boats and houses in the background, captured on S... \| Keywords (18): swan, river, calm, boats, houses, foliage, waterfront, mooring, leisure, tranquility, architecture, waterway, ...                                                                               |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `no concerns detected` | Title: Solitary Swan on Riverfront Marina \| Description: A solitary white swan glides across calm river waters in a canal with moored leisure boats and reside... \| Keywords (15): Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, ...[2,859 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                             | `no concerns detected` | Title: Swan on River with Moored Boats \| Description: A solitary white swan glides calmly on a river, framed by green foliage, with moored leisure boats and resid... \| Keywords (17): swan, river, boats, mooring, canal, greenery, trees, water reflection, waterfront, residential, motorboat, ...                                                                               |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `no concerns detected` | Title: Swan Swimming Past Moored Boats \| Description: A solitary white swan glides across calm river waters near a riverside mooring of leisure cruisers and resid... \| Keywords (18): swan, river, moored boats, riverbank, vegetation, water reflection, cruiser, residential buildings, trees, ...                                                                               |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `no concerns detected` | Title: Swan Glides Past Moored Boats on Calm Riverbank \| Description: A solitary white swan swims peacefully on a reflective river, with leisure boats moored along t... \| Keywords (15): swan, river, boats, mooring, residential buildings, water reflection, foliage, trees, motorboat, canal, ...                                                                               |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `no concerns detected` | Title: Swan swimming near moored boats on the River Thames \| Description: A solitary white swan swims on the calm water of the River Thames at Lambeth, with its re... \| Keywords (15): Swan, River Thames, Boat, Waterfowl, Water, Reflection, Mooring, Residential building, Foliage, Greenery, ...                                                                               |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `no concerns detected` | Title: Swan on River \| Description: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers mo... \| Keywords (20): Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, ...                                                                               |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                          | `no concerns detected` | Title: Swan glides past moored boats on a calm river \| Description: A solitary white swan swims across calm river waters, framed by lush foreground foliage and... \| Keywords (18): Swan, waterfowl, river, marina, mooring, motorboat, pier, waterfront, waterway, canal, bird, swimming, trees, ...                                                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `no concerns detected` | Title: Swan Gliding on Calm River with Moored Boats \| Description: A serene scene captures a solitary white swan gracefully swimming across calm river waters, frame... \| Keywords (23): Swan, River, Calm, Moored Boats, Residential Buildings, Greenery, Waterfront, Waterway, Motorboat, Pier, ...                                                                               |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `no concerns detected` | Title: Swan on River Cam, Cambridge – September 2026 \| Description: A white swan swims on the River Cam at 52.393850°N, 0.270830°E, captured on 12 September... \| Keywords (18): Swan, River Cam, Cambridge, Waterfowl, Boat, Mooring, Marina, Riverbank, Waterway, Residential building, Trees, ...                                                                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `no concerns detected` | Title: White swan swimming on calm river with moored boats \| Description: A solitary white swan glides across calm river waters near a wooden pier, with several... \| Keywords (15): white swan, river, motorboat, waterfowl, waterfront, greenery, reflection, residential building, boat, pier, ...                                                                               |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `no concerns detected` | Title: White Swan Swimming Past Moored River Boats \| Description: A single white swan swims across calm river waters in front of leisure boats moored beside rivers... \| Keywords (18): swan, river, boats, marina, moored, waterfront, waterway, residential, architecture, greenery, waterfowl, ...                                                                               |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                         | `no concerns detected` | Title: Swan glides on calm water beneath residential buildings \| Description: A solitary white swan gracefully floats on the calm river water, framed by dense greenery... \| Keywords (16): swan, waterfowl, river, canal, marina, boats, trees, greenery, quiet, architecture, moored, peaceful, ...                                                                               |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `no concerns detected` | Title: "Swan on the Canal" \| Description: A white swan glides gracefully across a calm river, framed by lush foliage and moored boats, with a serene residential backdrop. \| Keywords (10): swan, canal, water, foliage, boats, mooring, residential, reflection, serene, waterway                                                                                                  |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `no concerns detected` | Title: Swan in River with Moored Boats \| Description: A solitary white swan gracefully swims across calm river waters, framed by lush foliage, with leisure boats... \| Keywords (18): Swan, River, Boats, Marina, Water, Foliage, Residential, Mooring, Waterfront, Waterway, Trees, Vegetation, ...                                                                                |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `concerns detected`    | Title: Swans glide across calm river waters in a serene canal setting. \| Description: A solitary white swan glides gracefully across calm river waters framed by lu... \| Keywords (24): Swans, canal, water, serene, leisure, boats, water, architecture, greenery, water, reflection, waterfowl, ...                                                                               |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `concerns detected`    | Title: Serene River Scene with White Swan and Moored Boats \| Description: A tranquil riverside view features a solitary white swan gliding across calm water, surro... \| Keywords (42): swan, river, boats, mooring, canal, greenery, trees, vegetation, water reflection, waterfowl, waterfront, ...                                                                               |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                           | `concerns detected`    | Title: (not detected) \| Description: A serene scene captures a graceful white swan effortlessly gliding across the calm waters of a river, framed by lush greenery... \| Keywords (12): Swan, River, Greenery, Boats, Cruisers, Buildings, Lighting, Afternoon, Tranquil, Waterfowl, Architecture, ...                                                                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `concerns detected`    | Title: White Swan Gliding on River Near Moored Boats \| Description: A solitary white swan glides across calm river waters, with leisure boats and cruisers moored al... \| Keywords (16): swan, waterfowl, river, boat, marina, mooring, motorboat, cruiser, pier, waterfront, waterway, greenery, ...                                                                               |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                   | `major concerns`       | Title: Swan on canal with moored boats at marina \| Description: A white swan glides across a calm canal, surrounded by moored motorboats and residential buildings with lush greenery, at 2026-09-12 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E. \| Keywords (3): Swans, Canal, Motorboat[3,708 characters of reasoning omitted; complete output in the evidence block]         |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `major concerns`       | Title: (not detected) \| Description: (not detected) \| Keywords (1): We need title 5-10 words. Concrete. Maybe "Mute Swan on River with Moored Boats and Apartments". Count words. Mute(1) Swan(2) on(3) River(4) with(5) Moored(6) Boats(7) and(8) Apartments(9). Good 9 words.                                                                                                     |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `major concerns`       | Title: Solitary swan on a calm river with moored boats \| Description: A white swan glides across the calm waters of a river, framed by green foliage in the foregrou... \| Keywords (57): swan, river, boats, mooring, motorboat, residential, building, balcony, trees, foliage, greenery, water, ...                                                                               |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                       | `major concerns`       | A serene scene unfolds on a calm river, where a graceful swan glides across the water. Several boats, including motorboats and sailboats, are moored along the riverbank, nestled beneath a row of residential buildings. Lush greenery frames the scene, with trees and bushes linin...                                                                                              |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                 | `major concerns`       | <\|channel\|> analysis<\|message\|> The image shows a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a...                                                                                          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                            | `major concerns`       | "Amidst the Riverbank, a solitary white swan glides gracefully across calm waters, framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background. The serene setting is complemented by the presence of greenery, archit...                                                                                              |
| [`mlx-community/InternVL3_5-30B-A3B-4bit`](#model-mlx-community-internvl35-30b-a3b-4bit)                                | `not assessed`         | Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'                                                                                                                                                                                                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)            | `not assessed`         | Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.                                                                                                                                                                                                                       |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                           | `not assessed`         | Model loading failed: Received 904 parameters not in model:<br>model.embed_tokens.biases,<br>model.embed_tokens.scales,<br>model.embed_tokens.weight,<br>model.layers.0.input_layernorm.weight,<br>model.layers.0.mlp.down_proj.biases,<br>model.layers.0.mlp.down_proj.scales,<br>model.layers.0.mlp.down...                                                                         |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.0`
- `mlx`: `0.32.3.dev20260912+229f5b430`
- `transformers`: `5.17.0`
- `tokenizers`: `0.23.2`
- `huggingface-hub`: `1.31.0`
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
- *Model load time:* 2.28s
- *Generation time:* 8.68s
- *Total time:* 11.48s
- *Prompt preparation time:* 0.523
- *First-token latency:* 2.99
- *Cleanup time:* 0.12
- *Prompt tokens:* 2,393
- *Generation tokens:* 151
- *Total tokens:* 2,544
- *Prompt throughput (raw):* 799 tok/s
- *Generation throughput (raw):* 29.7 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 18
- *Post-cleanup active memory (GB):* 0.000411
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
- *Model load time:* 1.60s
- *Generation time:* 14.81s
- *Total time:* 17.04s
- *Prompt preparation time:* 0.622
- *First-token latency:* 0.742
- *Cleanup time:* 0.113
- *Prompt tokens:* 1,634
- *Generation tokens:* 997
- *Total tokens:* 2,631
- *Prompt throughput (raw):* 2,203 tok/s
- *Generation throughput (raw):* 74.0 tok/s
- *Peak memory (GB):* 19
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.18
- *Model-load active memory (GB):* 17.4
- *Post-cleanup active memory (GB):* 0.000624
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
- *Model load time:* 1.46s
- *Generation time:* 7.71s
- *Total time:* 9.73s
- *Prompt preparation time:* 0.546
- *First-token latency:* 5.35
- *Cleanup time:* 0.0988
- *Prompt tokens:* 6,450
- *Generation tokens:* 140
- *Total tokens:* 6,590
- *Prompt throughput (raw):* 1,206 tok/s
- *Generation throughput (raw):* 75.2 tok/s
- *Peak memory (GB):* 8.7
- *Active memory (GB):* 7.1
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 7.07
- *Post-cleanup active memory (GB):* 0.000935
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
- *Model load time:* 9.09s
- *Generation time:* 18.59s
- *Total time:* 28.26s
- *Prompt preparation time:* 0.57
- *First-token latency:* 14.4
- *Cleanup time:* 0.27
- *Prompt tokens:* 6,450
- *Generation tokens:* 103
- *Total tokens:* 6,553
- *Prompt throughput (raw):* 448 tok/s
- *Generation throughput (raw):* 40.9 tok/s
- *Peak memory (GB):* 78
- *Active memory (GB):* 62
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 61.9
- *Post-cleanup active memory (GB):* 0.00125
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
- *Model load time:* 2.10s
- *Generation time:* 6.10s
- *Total time:* 8.96s
- *Prompt preparation time:* 0.752
- *First-token latency:* 1.12
- *Cleanup time:* 0.108
- *Prompt tokens:* 2,612
- *Generation tokens:* 138
- *Total tokens:* 2,750
- *Prompt throughput (raw):* 2,324 tok/s
- *Generation throughput (raw):* 32.6 tok/s
- *Peak memory (GB):* 18
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.43
- *Model-load active memory (GB):* 16.9
- *Post-cleanup active memory (GB):* 0.00151
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
- *Model load time:* 1.65s
- *Generation time:* 3.69s
- *Total time:* 6.00s
- *Prompt preparation time:* 0.655
- *First-token latency:* 0.85
- *Cleanup time:* 0.107
- *Prompt tokens:* 2,113
- *Generation tokens:* 77
- *Total tokens:* 2,190
- *Prompt throughput (raw):* 2,487 tok/s
- *Generation throughput (raw):* 34.9 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 15.9
- *Post-cleanup active memory (GB):* 0.00182
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
- *Model load time:* 1.16s
- *Generation time:* 1.37s
- *Total time:* 3.11s
- *Prompt preparation time:* 0.581
- *First-token latency:* 0.434
- *Cleanup time:* 0.0978
- *Prompt tokens:* 2,103
- *Generation tokens:* 80
- *Total tokens:* 2,183
- *Prompt throughput (raw):* 4,841 tok/s
- *Generation throughput (raw):* 205 tok/s
- *Peak memory (GB):* 4.0
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 2.81
- *Post-cleanup active memory (GB):* 0.00274
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1383
- *Image placeholders:* 1
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 12c5ae49304158b0a133fcea9ba4486a6d6c8cad
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-3B-OptiQ-4bit/snapshots/12c5ae49304158b0a133fcea9ba4486a6d6c8cad
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Swan gliding on river near moored boats and houses
Description: A white swan swims gracefully on calm river waters framed by lush greenery, with leisure boats and residential buildings in the background.
Keywords: swan, river, boats, houses, greenery, water, reflection, leisure, architecture, waterfront, waterway, aquatic bird, trees, vegetation
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
- *Generation time:* 1.71s
- *Total time:* 3.27s
- *Prompt preparation time:* 0.508
- *First-token latency:* 0.239
- *Cleanup time:* 0.104
- *Prompt tokens:* 391
- *Generation tokens:* 101
- *Total tokens:* 492
- *Prompt throughput (raw):* 1,633 tok/s
- *Generation throughput (raw):* 103 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 6.1
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 6.15
- *Post-cleanup active memory (GB):* 0.00305
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
- *Generation time:* 4.24s
- *Total time:* 6.31s
- *Prompt preparation time:* 0.546
- *First-token latency:* 2.09
- *Cleanup time:* 0.0984
- *Prompt tokens:* 2,926
- *Generation tokens:* 109
- *Total tokens:* 3,035
- *Prompt throughput (raw):* 1,401 tok/s
- *Generation throughput (raw):* 67.1 tok/s
- *Peak memory (GB):* 13
- *Active memory (GB):* 8.0
- *Cache memory (GB):* 0.56
- *Model-load active memory (GB):* 8
- *Post-cleanup active memory (GB):* 0.00331
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
- *Model load time:* 1.59s
- *Generation time:* 5.55s
- *Total time:* 7.69s
- *Prompt preparation time:* 0.54
- *First-token latency:* 2.11
- *Cleanup time:* 0.104
- *Prompt tokens:* 2,926
- *Generation tokens:* 187
- *Total tokens:* 3,113
- *Prompt throughput (raw):* 1,387 tok/s
- *Generation throughput (raw):* 64.1 tok/s
- *Peak memory (GB):* 13
- *Active memory (GB):* 8.4
- *Cache memory (GB):* 0.60
- *Model-load active memory (GB):* 8.43
- *Post-cleanup active memory (GB):* 0.00357
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
- *Model load time:* 1.13s
- *Generation time:* 2.09s
- *Total time:* 3.78s
- *Prompt preparation time:* 0.547
- *First-token latency:* 1.03
- *Cleanup time:* 0.0877
- *Prompt tokens:* 2,925
- *Generation tokens:* 102
- *Total tokens:* 3,027
- *Prompt throughput (raw):* 2,834 tok/s
- *Generation throughput (raw):* 190 tok/s
- *Peak memory (GB):* 7.8
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.38
- *Model-load active memory (GB):* 2.75
- *Post-cleanup active memory (GB):* 0.00383
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
- *Model load time:* 1.18s
- *Generation time:* 3.74s
- *Total time:* 6.02s
- *Prompt preparation time:* 1.09
- *First-token latency:* 2.41
- *Cleanup time:* 0.1
- *Prompt tokens:* 4,083
- *Generation tokens:* 84
- *Total tokens:* 4,167
- *Prompt throughput (raw):* 1,692 tok/s
- *Generation throughput (raw):* 159 tok/s
- *Peak memory (GB):* 3.9
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.65
- *Model-load active memory (GB):* 2.18
- *Post-cleanup active memory (GB):* 0.00508
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
- *Model load time:* 2.63s
- *Generation time:* 3.38s
- *Total time:* 6.69s
- *Prompt preparation time:* 0.675
- *First-token latency:* 0.631
- *Cleanup time:* 0.135
- *Prompt tokens:* 1,289
- *Generation tokens:* 155
- *Total tokens:* 1,444
- *Prompt throughput (raw):* 2,042 tok/s
- *Generation throughput (raw):* 73.0 tok/s
- *Peak memory (GB):* 24
- *Active memory (GB):* 23
- *Cache memory (GB):* 0.14
- *Model-load active memory (GB):* 23.1
- *Post-cleanup active memory (GB):* 0.00559
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1423
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5f31fcd089ce6f47f1073d3ae43dcb3a0bd1869f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-35B-A3B-OptiQ-4bit/snapshots/5f31fcd089ce6f47f1073d3ae43dcb3a0bd1869f
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
- *Model load time:* 0.93s
- *Generation time:* 3.80s
- *Total time:* 5.24s
- *Prompt preparation time:* 0.519
- *First-token latency:* 0.36
- *Cleanup time:* 0.102
- *Prompt tokens:* 1,133
- *Generation tokens:* 107
- *Total tokens:* 1,240
- *Prompt throughput (raw):* 3,148 tok/s
- *Generation throughput (raw):* 36.6 tok/s
- *Peak memory (GB):* 9.3
- *Active memory (GB):* 8.3
- *Cache memory (GB):* 0.52
- *Model-load active memory (GB):* 8.3
- *Post-cleanup active memory (GB):* 0.00565
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
- *Model load time:* 0.81s
- *Generation time:* 34.08s
- *Total time:* 35.94s
- *Prompt preparation time:* 1.04
- *First-token latency:* 20.1
- *Cleanup time:* 0.207
- *Prompt tokens:* 16,549
- *Generation tokens:* 912
- *Total tokens:* 17,461
- *Prompt throughput (raw):* 823 tok/s
- *Generation throughput (raw):* 69.8 tok/s
- *Peak memory (GB):* 8.4
- *Active memory (GB):* 4.3
- *Cache memory (GB):* 2.7
- *Model-load active memory (GB):* 4.26
- *Post-cleanup active memory (GB):* 0.00597
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
- *Model load time:* 0.92s
- *Generation time:* 46.29s
- *Total time:* 48.18s
- *Prompt preparation time:* 0.967
- *First-token latency:* 43.8
- *Cleanup time:* 0.261
- *Prompt tokens:* 16,547
- *Generation tokens:* 93
- *Total tokens:* 16,640
- *Prompt throughput (raw):* 377 tok/s
- *Generation throughput (raw):* 59.4 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 3.4
- *Model-load active memory (GB):* 5.77
- *Post-cleanup active memory (GB):* 0.00659
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
- *Model load time:* 2.93s
- *Generation time:* 42.07s
- *Total time:* 46.02s
- *Prompt preparation time:* 1.02
- *First-token latency:* 39.9
- *Cleanup time:* 0.139
- *Prompt tokens:* 16,563
- *Generation tokens:* 97
- *Total tokens:* 16,660
- *Prompt throughput (raw):* 415 tok/s
- *Generation throughput (raw):* 74.4 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.89
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.0071
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
- *Model load time:* 1.68s
- *Generation time:* 43.69s
- *Total time:* 46.37s
- *Prompt preparation time:* 0.989
- *First-token latency:* 41.8
- *Cleanup time:* 0.131
- *Prompt tokens:* 16,563
- *Generation tokens:* 86
- *Total tokens:* 16,649
- *Prompt throughput (raw):* 396 tok/s
- *Generation throughput (raw):* 88.2 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 6.0
- *Cache memory (GB):* 1.2
- *Model-load active memory (GB):* 5.96
- *Post-cleanup active memory (GB):* 0.00762
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

<a id="model-mlx-community-qwen38-27b-4bit"></a>

### mlx-community/Qwen3.8-27B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.8-27B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.23s
- *Generation time:* 73.55s
- *Total time:* 76.80s
- *Prompt preparation time:* 0.993
- *First-token latency:* 64
- *Cleanup time:* 0.321
- *Prompt tokens:* 16,563
- *Generation tokens:* 124
- *Total tokens:* 16,687
- *Prompt throughput (raw):* 259 tok/s
- *Generation throughput (raw):* 14.4 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 16
- *Cache memory (GB):* 1.9
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.00813
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1423
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 3e6447f082e89cc7f0bc6e5441afd38dfce760ff
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/3e6447f082e89cc7f0bc6e5441afd38dfce760ff
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
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.89s
- *Generation time:* 1.89s
- *Total time:* 3.74s
- *Prompt preparation time:* 0.952
- *First-token latency:* 0.381
- *Cleanup time:* 0.0878
- *Prompt tokens:* 1,426
- *Generation tokens:* 98
- *Total tokens:* 1,524
- *Prompt throughput (raw):* 3,742 tok/s
- *Generation throughput (raw):* 123 tok/s
- *Peak memory (GB):* 5.6
- *Active memory (GB):* 4.5
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 4.5
- *Post-cleanup active memory (GB):* 0.00824
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
- *Model load time:* 17.50s
- *Generation time:* 78.28s
- *Total time:* 100.61s
- *Prompt preparation time:* 4.83
- *First-token latency:* 69.9
- *Cleanup time:* 0.391
- *Prompt tokens:* 3,491
- *Generation tokens:* 121
- *Total tokens:* 3,612
- *Prompt throughput (raw):* 50.0 tok/s
- *Generation throughput (raw):* 48.6 tok/s
- *Peak memory (GB):* 92
- *Active memory (GB):* 85
- *Cache memory (GB):* 0.41
- *Model-load active memory (GB):* 85
- *Post-cleanup active memory (GB):* 0.0085
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
- *Model load time:* 1.26s
- *Generation time:* 15.25s
- *Total time:* 17.88s
- *Prompt preparation time:* 1.37
- *First-token latency:* 12
- *Cleanup time:* 0.11
- *Prompt tokens:* 16,558
- *Generation tokens:* 136
- *Total tokens:* 16,694
- *Prompt throughput (raw):* 1,376 tok/s
- *Generation throughput (raw):* 57.2 tok/s
- *Peak memory (GB):* 14
- *Active memory (GB):* 9.5
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 9.46
- *Post-cleanup active memory (GB):* 0.00882
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
- *Model load time:* 2.39s
- *Generation time:* 6.80s
- *Total time:* 9.80s
- *Prompt preparation time:* 0.594
- *First-token latency:* 0.985
- *Cleanup time:* 0.132
- *Prompt tokens:* 590
- *Generation tokens:* 152
- *Total tokens:* 742
- *Prompt throughput (raw):* 599 tok/s
- *Generation throughput (raw):* 30.2 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.44
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.00988
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
- *Model load time:* 2.80s
- *Generation time:* 1.90s
- *Total time:* 5.32s
- *Prompt preparation time:* 0.619
- *First-token latency:* 0.413
- *Cleanup time:* 0.146
- *Prompt tokens:* 595
- *Generation tokens:* 96
- *Total tokens:* 691
- *Prompt throughput (raw):* 1,442 tok/s
- *Generation throughput (raw):* 105 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 15.4
- *Post-cleanup active memory (GB):* 0.0109
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
- *Model load time:* 2.79s
- *Generation time:* 5.08s
- *Total time:* 8.48s
- *Prompt preparation time:* 0.597
- *First-token latency:* 1.08
- *Cleanup time:* 0.151
- *Prompt tokens:* 595
- *Generation tokens:* 89
- *Total tokens:* 684
- *Prompt throughput (raw):* 551 tok/s
- *Generation throughput (raw):* 26.0 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.79
- *Model-load active memory (GB):* 18.4
- *Post-cleanup active memory (GB):* 0.0115
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
- *Model load time:* 1.75s
- *Generation time:* 1.37s
- *Total time:* 3.73s
- *Prompt preparation time:* 0.606
- *First-token latency:* 0.188
- *Cleanup time:* 0.12
- *Prompt tokens:* 591
- *Generation tokens:* 73
- *Total tokens:* 664
- *Prompt throughput (raw):* 3,146 tok/s
- *Generation throughput (raw):* 122 tok/s
- *Peak memory (GB):* 5.9
- *Active memory (GB):* 5.2
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 5.16
- *Post-cleanup active memory (GB):* 0.012
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
- *Model load time:* 0.62s
- *Generation time:* 1.57s
- *Total time:* 3.01s
- *Prompt preparation time:* 0.822
- *First-token latency:* 0.403
- *Cleanup time:* 0.104
- *Prompt tokens:* 1,379
- *Generation tokens:* 65
- *Total tokens:* 1,444
- *Prompt throughput (raw):* 3,424 tok/s
- *Generation throughput (raw):* 172 tok/s
- *Peak memory (GB):* 4.7
- *Active memory (GB):* 3.0
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 3.03
- *Post-cleanup active memory (GB):* 0.0122
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
- *Model load time:* 1.84s
- *Generation time:* 4.92s
- *Total time:* 7.31s
- *Prompt preparation time:* 0.539
- *First-token latency:* 1.69
- *Cleanup time:* 0.123
- *Prompt tokens:* 3,116
- *Generation tokens:* 102
- *Total tokens:* 3,218
- *Prompt throughput (raw):* 1,845 tok/s
- *Generation throughput (raw):* 37.7 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 13
- *Cache memory (GB):* 0.59
- *Model-load active memory (GB):* 13.5
- *Post-cleanup active memory (GB):* 0.0135
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
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Duplicate keywords: canal, water, architecture, waterway,
  aquatic bird, waterfront
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.43s
- *Generation time:* 0.88s
- *Total time:* 2.17s
- *Prompt preparation time:* 0.514
- *First-token latency:* 0.131
- *Cleanup time:* 0.0729
- *Prompt tokens:* 2,112
- *Generation tokens:* 120
- *Total tokens:* 2,232
- *Prompt throughput (raw):* 16,118 tok/s
- *Generation throughput (raw):* 475 tok/s
- *Peak memory (GB):* 1.9
- *Active memory (GB):* 0.90
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 0.897
- *Post-cleanup active memory (GB):* 0.000149
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
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
Swans, canal, water, serene, leisure, boats, water, architecture, greenery, water, reflection, waterfowl, architecture, riverbank, waterway, aquatic bird, canal, mooring, waterfront, waterway, aquatic bird, waterfront, architecture, boat
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
- *Generation time:* 4.39s
- *Total time:* 6.24s
- *Prompt preparation time:* 0.666
- *First-token latency:* 0.758
- *Cleanup time:* 0.109
- *Prompt tokens:* 1,524
- *Generation tokens:* 216
- *Total tokens:* 1,740
- *Prompt throughput (raw):* 2,011 tok/s
- *Generation throughput (raw):* 72.1 tok/s
- *Peak memory (GB):* 8.1
- *Active memory (GB):* 6.4
- *Cache memory (GB):* 0.31
- *Model-load active memory (GB):* 6.45
- *Post-cleanup active memory (GB):* 0.00415
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

<a id="model-mlx-community-aya-vision-8b-4bit"></a>

### mlx-community/aya-vision-8b-4bit

<details>
<summary>Complete evidence: mlx-community/aya-vision-8b-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type aya_vision)
- *Model load time:* 1.16s
- *Generation time:* 3.02s
- *Total time:* 5.14s
- *Prompt preparation time:* 0.951
- *First-token latency:* 1.21
- *Cleanup time:* 0.116
- *Prompt tokens:* 2,090
- *Generation tokens:* 96
- *Total tokens:* 2,186
- *Prompt throughput (raw):* 1,723 tok/s
- *Generation throughput (raw):* 102 tok/s
- *Peak memory (GB):* 6.5
- *Active memory (GB):* 5.5
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 5.46
- *Post-cleanup active memory (GB):* 0.00934
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 4270
- *Image placeholders:* 1
- *Processor:* transformers.models.aya_vision.processing_aya_vision.AyaVisionProcessor
- *Tokenizer:* transformers.models.cohere.tokenization_cohere.CohereTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 3e679b3e08f04d741c8965ebf0cb258295e8acf7
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--aya-vision-8b-4bit/snapshots/3e679b3e08f04d741c8965ebf0cb258295e8acf7
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|END_OF_TURN_TOKEN\|>

*Readable output:*

<pre class="model-output-readable">
&lt;|START_RESPONSE|&gt;Title: Solitary Swan Glides on Tranquil River
Description: A serene scene captures a graceful white swan effortlessly gliding across the calm waters of a river, framed by lush greenery. Leisure boats and cruisers are moored alongside riverside residential buildings, creating a picturesque setting under the soft lighting of the late afternoon.
Keywords: Swan, River, Greenery, Boats, Cruisers, Buildings, Lighting, Afternoon, Tranquil, Waterfowl, Architecture, Waterway&lt;|END_RESPONSE|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|START_RESPONSE|>Title: Solitary Swan Glides on Tranquil River
Description: A serene scene captures a graceful white swan effortlessly gliding across the calm waters of a river, framed by lush greenery. Leisure boats and cruisers are moored alongside riverside residential buildings, creating a picturesque setting under the soft lighting of the late afternoon.
Keywords: Swan, River, Greenery, Boats, Cruisers, Buildings, Lighting, Afternoon, Tranquil, Waterfowl, Architecture, Waterway<|END_RESPONSE|>
```

</details>

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
- *Observations:* Duplicate keywords: reflection
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.38s
- *Generation time:* 2.62s
- *Total time:* 6.71s
- *Prompt preparation time:* 0.709
- *First-token latency:* 0.329
- *Cleanup time:* 0.129
- *Prompt tokens:* 591
- *Generation tokens:* 80
- *Total tokens:* 671
- *Prompt throughput (raw):* 1,798 tok/s
- *Generation throughput (raw):* 53.0 tok/s
- *Peak memory (GB):* 28
- *Active memory (GB):* 27
- *Cache memory (GB):* 0.01
- *Model-load active memory (GB):* 27.2
- *Post-cleanup active memory (GB):* 0.00934
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
- *Model load time:* 2.34s
- *Generation time:* 17.61s
- *Total time:* 20.55s
- *Prompt preparation time:* 0.59
- *First-token latency:* 0.706
- *Cleanup time:* 0.127
- *Prompt tokens:* 1,324
- *Generation tokens:* 1,000
- *Total tokens:* 2,324
- *Prompt throughput (raw):* 1,874 tok/s
- *Generation throughput (raw):* 61.5 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 1.5
- *Model-load active memory (GB):* 17.8
- *Post-cleanup active memory (GB):* 0.00248
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
- *Model load time:* 3.24s
- *Generation time:* 55.43s
- *Total time:* 59.35s
- *Prompt preparation time:* 0.675
- *First-token latency:* 8.47
- *Cleanup time:* 0.196
- *Prompt tokens:* 4,403
- *Generation tokens:* 1,000
- *Total tokens:* 5,403
- *Prompt throughput (raw):* 520 tok/s
- *Generation throughput (raw):* 21.6 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.25
- *Model-load active memory (GB):* 23.9
- *Post-cleanup active memory (GB):* 0.00456
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1570
- *Processor:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b4a74fa6001f1eca3b23eeeb702ffad2773a218f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-OptiQ-4bit/snapshots/b4a74fa6001f1eca3b23eeeb702ffad2773a218f
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
 to=self&lt;|message|&gt;Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need title 5-10 words. Concrete. Maybe "Mute Swan on River with Moored Boats and Apartments". Count words. Mute(1) Swan(2) on(3) River(4) with(5) Moored(6) Boats(7) and(8) Apartments(9). Good 9 words.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Capture date/time and GPS as authoritative facts but do not claim they are visible. So we can mention captured on 2026-09-12... at GPS...? Maybe include in description? It says combining relevant context with main visible subject, setting, action, lighting, and distinctive details. Context includes capture date/time and GPS. Should we include them in description? It says treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. So we can mention them as metadata in description? Possibly "Captured on 2026-09-12..." That is factual. Might be okay. Typically catalogue description includes date and location. Could incorporate. Must not claim they are visible. So we can say "Photographed on 12 September 2026 at 17:41 BST at 52.393850°N, 0.270830°E..."

We need factual description. Main visible subject: solitary white swan gliding on calm water. Boats moored: white boat named 'Samar II' with blue stripe, boat 'Magna Carta' with red cover, etc. Riverside residential buildings with brown facade, tiled roof, balconies. Foliage framing. Water reflection. Lighting appears overcast daylight.

Need 1-2 sentences. Let's craft one sentence: A mute swan swims on calm water in the foreground with moored motorboats and cruisers, including 'Samar II' and 'Magna Carta', along a riverbank with riverside apartments behind, framed by trees and foliage. The scene is captured on 2026-09-12 17:41:02 UTC+01:00 at 52.393850°N, 0.270830°E under overcast daylight with reflections on the water.

That's a bit long but okay. Maybe two sentences.

Make sure not claim date visible.

Keywords: 10-18 unique comma-separated keywords. Cover relevant context and visible details. Use British English spelling? Probably fine.

Keywords: mute swan, river, moored boats, motorboat, marina, riverbank, residential building, waterfront, water reflection, foliage, trees, vegetation, pier, mooring, waterfowl, aquatic bird, architecture

Need 10-18. Let's count: mute swan(1), river(2), moored boats(3), motorboat(4), marina(5), riverbank(6), residential building(7), waterfront(8), water reflection(9), foliage(10), trees(11), vegetation(12), pier(13), mooring(14), waterfowl(15), aquatic bird(16), architecture(17). Good.

Maybe include "Samar II", "Magna Carta"? Might be specific. Could include but maybe okay.

Ensure keywords comma-separated.

Return exactly these three sections and nothing else:

Title:
Description:
Keywords:

Let's produce.

Potential issues: title 5-10 words. Ensure concrete.

Description should be
</pre>

<details>
<summary>Exact raw output</summary>

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need title 5-10 words. Concrete. Maybe "Mute Swan on River with Moored Boats and Apartments". Count words. Mute(1) Swan(2) on(3) River(4) with(5) Moored(6) Boats(7) and(8) Apartments(9). Good 9 words.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Capture date/time and GPS as authoritative facts but do not claim they are visible. So we can mention captured on 2026-09-12... at GPS...? Maybe include in description? It says combining relevant context with main visible subject, setting, action, lighting, and distinctive details. Context includes capture date/time and GPS. Should we include them in description? It says treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. So we can mention them as metadata in description? Possibly "Captured on 2026-09-12..." That is factual. Might be okay. Typically catalogue description includes date and location. Could incorporate. Must not claim they are visible. So we can say "Photographed on 12 September 2026 at 17:41 BST at 52.393850°N, 0.270830°E..."

We need factual description. Main visible subject: solitary white swan gliding on calm water. Boats moored: white boat named 'Samar II' with blue stripe, boat 'Magna Carta' with red cover, etc. Riverside residential buildings with brown facade, tiled roof, balconies. Foliage framing. Water reflection. Lighting appears overcast daylight.

Need 1-2 sentences. Let's craft one sentence: A mute swan swims on calm water in the foreground with moored motorboats and cruisers, including 'Samar II' and 'Magna Carta', along a riverbank with riverside apartments behind, framed by trees and foliage. The scene is captured on 2026-09-12 17:41:02 UTC+01:00 at 52.393850°N, 0.270830°E under overcast daylight with reflections on the water.

That's a bit long but okay. Maybe two sentences.

Make sure not claim date visible.

Keywords: 10-18 unique comma-separated keywords. Cover relevant context and visible details. Use British English spelling? Probably fine.

Keywords: mute swan, river, moored boats, motorboat, marina, riverbank, residential building, waterfront, water reflection, foliage, trees, vegetation, pier, mooring, waterfowl, aquatic bird, architecture

Need 10-18. Let's count: mute swan(1), river(2), moored boats(3), motorboat(4), marina(5), riverbank(6), residential building(7), waterfront(8), water reflection(9), foliage(10), trees(11), vegetation(12), pier(13), mooring(14), waterfowl(15), aquatic bird(16), architecture(17). Good.

Maybe include "Samar II", "Magna Carta"? Might be specific. Could include but maybe okay.

Ensure keywords comma-separated.

Return exactly these three sections and nothing else:

Title:
Description:
Keywords:

Let's produce.

Potential issues: title 5-10 words. Ensure concrete.

Description should be 
```

</details>

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
- *Model load time:* 1.93s
- *Generation time:* 43.25s
- *Total time:* 46.20s
- *Prompt preparation time:* 1
- *First-token latency:* 39.6
- *Cleanup time:* 0.141
- *Prompt tokens:* 16,547
- *Generation tokens:* 200
- *Total tokens:* 16,747
- *Prompt throughput (raw):* 418 tok/s
- *Generation throughput (raw):* 71.2 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 4.0
- *Model-load active memory (GB):* 18.3
- *Post-cleanup active memory (GB):* 0.00628
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
- *Model load time:* 1.66s
- *Generation time:* 2.99s
- *Total time:* 5.22s
- *Prompt preparation time:* 0.56
- *First-token latency:* 1.02
- *Cleanup time:* 0.131
- *Prompt tokens:* 589
- *Generation tokens:* 88
- *Total tokens:* 677
- *Prompt throughput (raw):* 578 tok/s
- *Generation throughput (raw):* 60.3 tok/s
- *Peak memory (GB):* 7.1
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 5.83
- *Post-cleanup active memory (GB):* 0.0104
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
- *Model load time:* 1.10s
- *Generation time:* 3.47s
- *Total time:* 5.24s
- *Prompt preparation time:* 0.666
- *First-token latency:* 0.837
- *Cleanup time:* 0.174
- *Prompt tokens:* 2,195
- *Generation tokens:* 200
- *Total tokens:* 2,395
- *Prompt throughput (raw):* 2,623 tok/s
- *Generation throughput (raw):* 100 tok/s
- *Peak memory (GB):* 6.7
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.70
- *Model-load active memory (GB):* 5.69
- *Post-cleanup active memory (GB):* 0.0126
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
- *Model load time:* 0.56s
- *Generation time:* 1.12s
- *Total time:* 2.29s
- *Prompt preparation time:* 0.609
- *First-token latency:* 0.0918
- *Cleanup time:* 0.107
- *Prompt tokens:* 330
- *Generation tokens:* 92
- *Total tokens:* 422
- *Prompt throughput (raw):* 3,593 tok/s
- *Generation throughput (raw):* 204 tok/s
- *Peak memory (GB):* 1.8
- *Active memory (GB):* 0.61
- *Cache memory (GB):* 0.28
- *Model-load active memory (GB):* 0.611
- *Post-cleanup active memory (GB):* 0.0132
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

<a id="model-mlx-community-internvl35-30b-a3b-4bit"></a>

### mlx-community/InternVL3_5-30B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/InternVL3_5-30B-A3B-4bit</summary>

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
- *Model load time:* 0.15s
- *Generation time:* -
- *Total time:* 0.16s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0741
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00182
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* ed2ce3381528db1c5b70a2aad78a6390997e9250
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3_5-30B-A3B-4bit/snapshots/ed2ce3381528db1c5b70a2aad78a6390997e9250
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14168, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13080, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 823, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1306, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 964, in load_model
    model_class, _ = get_model_and_args(config=config, model_path=model_path)
                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 784, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15198, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14183, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 17 files:   0%|          | 0/17 [00:00<?, ?it/s]
Fetching 17 files: 100%|##########| 17/17 [00:00<00:00, 3441.44it/s]
ERROR:root:Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
```

</details>

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-4bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-4bit</summary>

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
  mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes
  (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* [broadcast_shapes] Shapes (1,1,301,6404) and
  (1,32,300,6404) cannot be broadcast.
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.39s
- *Generation time:* 0.56s
- *Total time:* 2.54s
- *Prompt preparation time:* 0.583
- *First-token latency:* -
- *Cleanup time:* 0.133
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00274
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
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

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13414, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14091, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<5 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13986, in _generate_with_repetition_guard
    for chunk in stream_generate(
                 ~~~~~~~~~~~~~~~^
        model=model, processor=processor, prompt=prompt, image=image, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1081, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 497, in generate_step
    chunk_output = model.language_model(
        inputs=input_ids[:, :n_to_process],
    ...<3 lines>...
        **chunk_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 360, in __call__
    hidden_states = self.model(
        input_ids=inputs,
    ...<5 lines>...
        cache=cache,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 321, in __call__
    layer_outputs = decoder_layer(
        hidden_states,
    ...<3 lines>...
        cache=c,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 251, in __call__
    hidden_states = self.cross_attn(
        hidden_states=hidden_states,
    ...<2 lines>...
        cache=cache,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 80, in __call__
    attn_output = scaled_dot_product_attention(
        query_states,
    ...<4 lines>...
        mask=attention_mask,  # add a dim for batch processing
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 418, in scaled_dot_product_attention
    return mx.fast.scaled_dot_product_attention(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        queries,
        ^^^^^^^^
    ...<4 lines>...
        sinks=sinks,
        ^^^^^^^^^^^^
    )
    ^
ValueError: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15198, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14195, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14111, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13423, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 4797.15it/s]
[02:14:11] Generation error for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13414, in
           _run_generation_guarded
               return generate_once()
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14091, in
           _generate_once
               return _generate_with_repetition_guard(
                   model=prepared.model,
               ...<5 lines>...
                   **prepared.generate_kwargs,
               )
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13986, in
           _generate_with_repetition_guard
               for chunk in stream_generate(
                            ~~~~~~~~~~~~~~~^
                   model=model, processor=processor, prompt=prompt, image=image, **kwargs
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               ):
               ^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1081, in
           stream_generate
               for n, (token, logprobs) in enumerate(gen):
                                           ~~~~~~~~~^^^^^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 497, in
           generate_step
               chunk_output = model.language_model(
                   inputs=input_ids[:, :n_to_process],
               ...<3 lines>...
                   **chunk_kwargs,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 360,
           in __call__
               hidden_states = self.model(
                   input_ids=inputs,
               ...<5 lines>...
                   cache=cache,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 321,
           in __call__
               layer_outputs = decoder_layer(
                   hidden_states,
               ...<3 lines>...
                   cache=c,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 251,
           in __call__
               hidden_states = self.cross_attn(
                   hidden_states=hidden_states,
               ...<2 lines>...
                   cache=cache,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 80,
           in __call__
               attn_output = scaled_dot_product_attention(
                   query_states,
               ...<4 lines>...
                   mask=attention_mask,  # add a dim for batch processing
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 418, in
           scaled_dot_product_attention
               return mx.fast.scaled_dot_product_attention(
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
                   queries,
                   ^^^^^^^^
               ...<4 lines>...
                   sinks=sinks,
                   ^^^^^^^^^^^^
               )
               ^
           ValueError: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be
           broadcast.
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
- *Failure phase:* model_load
- *Error stage:* Model Error
- *Error code:* MLX_VLM_MODEL_LOAD_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model loading failed: Received 904 parameters not in model:
  <br>model.embed_tokens.biases,<br>model.embed_tokens.scales,<br>model.embed_tokens.weight,<br>model.layers.0.input_layernorm.weight,<br>model.layers.0.mlp.down_proj.biases,<br>model.layers.0.mlp.down_proj.scales,<br>model.layers.0.mlp.down_proj.weight,<br>model.layers.0.mlp.gate_proj.biases,<br>model.layers.0.mlp.gate_proj.scales,<br>model.layers.0.mlp.gate_proj.weight,<br>model.layers.0.mlp.up_proj.biases,<br>model.layers.0.mlp.up_proj.scales,<br>model.layers.0.mlp.up_proj.weight,<br>model.layers.0.post_attention_layernorm.weight,<br>model.layers.0.self_attn.k_norm.weight,<br>model.layers.0.self_attn.k_proj.biases,<br>model.layers.0.self_attn.k_proj.scales,<br>model.layers.0.self_attn.k_proj.weight,<br>model.layers.0.self_attn.o_proj.biases,<br>model.layers.0.self_attn.o_proj.scales,<br>model.layers.0.self_attn.o_proj.weight,<br>model.layers.0.self_attn.q_norm.weight,<br>model.layers.0.self_attn.q_proj.biases,<br>model.layers.0.self_attn.q_proj.scales,<br>model.layers.0.self_attn.q_proj.weight,<br>model.layers.0.self_attn.v_proj.biases,<br>model.layers.0.self_attn.v_proj.scales,<br>model.layers.0.self_attn.v_proj.weight,<br>model.layers.1.input_layernorm.weight,<br>model.layers.1.mlp.down_proj.biases,<br>model.layers.1.mlp.down_proj.scales,<br>model.layers.1.mlp.down_proj.weight,<br>model.layers.1.mlp.gate_proj.biases,<br>model.layers.1.mlp.gate_proj.scales,<br>model.layers.1.mlp.gate_proj.weight,<br>model.layers.1.mlp.up_proj.biases,<br>model.layers.1.mlp.up_proj.scales,<br>model.layers.1.mlp.up_proj.weight,<br>model.layers.1.post_attention_layernorm.weight,<br>model.layers.1.self_attn.k_norm.weight,<br>model.layers.1.self_attn.k_proj.biases,<br>model.layers.1.self_attn.k_proj.scales,<br>model.layers.1.self_attn.k_proj.weight,<br>model.layers.1.self_attn.o_proj.biases,<br>model.layers.1.self_attn.o_proj.scales,<br>model.layers.1.self_attn.o_proj.weight,<br>model.layers.1.self_attn.q_norm.weight,<br>model.layers.1.self_attn.q_proj.biases,<br>model.layers.1.self_attn.q_proj.scales,<br>model.layers.1.self_attn.q_proj.weight,<br>model.layers.1.self_attn.v_proj.biases,<br>model.layers.1.self_attn.v_proj.scales,<br>model.layers.1.self_attn.v_proj.weight,<br>model.layers.10.input_layernorm.weight,<br>model.layers.10.mlp.down_proj.biases,<br>model.layers.10.mlp.down_proj.scales,<br>model.layers.10.mlp.down_proj.weight,<br>model.layers.10.mlp.gate_proj.biases,<br>model.layers.10.mlp.gate_proj.scales,<br>model.layers.10.mlp.gate_proj.weight,<br>model.layers.10.mlp.up_proj.biases,<br>model.layers.10.mlp.up_proj.scales,<br>model.layers.10.mlp.up_proj.weight,<br>model.layers.10.post_attention_layernorm.weight,<br>model.layers.10.self_attn.k_norm.weight,<br>model.layers.10.self_attn.k_proj.biases,<br>model.layers.10.self_attn.k_proj.scales,<br>model.layers.10.self_attn.k_proj.weight,<br>model.layers.10.self_attn.o_proj.biases,<br>model.layers.10.self_attn.o_proj.scales,<br>model.layers.10.self_attn.o_proj.weight,<br>model.layers.10.self_attn.q_norm.weight,<br>model.layers.10.self_attn.q_proj.biases,<br>model.layers.10.self_attn.q_proj.scales,<br>model.layers.10.self_attn.q_proj.weight,<br>model.layers.10.self_attn.v_proj.biases,<br>model.layers.10.self_attn.v_proj.scales,<br>model.layers.10.self_attn.v_proj.weight,<br>model.layers.11.input_layernorm.weight,<br>model.layers.11.mlp.down_proj.biases,<br>model.layers.11.mlp.down_proj.scales,<br>model.layers.11.mlp.down_proj.weight,<br>model.layers.11.mlp.gate_proj.biases,<br>model.layers.11.mlp.gate_proj.scales,<br>model.layers.11.mlp.gate_proj.weight,<br>model.layers.11.mlp.up_proj.biases,<br>model.layers.11.mlp.up_proj.scales,<br>model.layers.11.mlp.up_proj.weight,<br>model.layers.11.post_attention_layernorm.weight,<br>model.layers.11.self_attn.k_norm.weight,<br>model.layers.11.self_attn.k_proj.biases,<br>model.layers.11.self_attn.k_proj.scales,<br>model.layers.11.self_attn.k_proj.weight,<br>model.layers.11.self_attn.o_proj.biases,<br>model.layers.11.self_attn.o_proj.scales,<br>model.layers.11.self_attn.o_proj.weight,<br>model.layers.11.self_attn.q_norm.weight,<br>model.layers.11.self_attn.q_proj.biases,<br>model.layers.11.self_attn.q_proj.scales,<br>model.layers.11.self_attn.q_proj.weight,<br>model.layers.11.self_attn.v_proj.biases,<br>model.layers.11.self_attn.v_proj.scales,<br>model.layers.11.self_attn.v_proj.weight,<br>model.layers.12.input_layernorm.weight,<br>model.layers.12.mlp.down_proj.biases,<br>model.layers.12.mlp.down_proj.scales,<br>model.layers.12.mlp.down_proj.weight,<br>model.layers.12.mlp.gate_proj.biases,<br>model.layers.12.mlp.gate_proj.scales,<br>model.layers.12.mlp.gate_proj.weight,<br>model.layers.12.mlp.up_proj.biases,<br>model.layers.12.mlp.up_proj.scales,<br>model.layers.12.mlp.up_proj.weight,<br>model.layers.12.post_attention_layernorm.weight,<br>model.layers.12.self_attn.k_norm.weight,<br>model.layers.12.self_attn.k_proj.biases,<br>model.layers.12.self_attn.k_proj.scales,<br>model.layers.12.self_attn.k_proj.weight,<br>model.layers.12.self_attn.o_proj.biases,<br>model.layers.12.self_attn.o_proj.scales,<br>model.layers.12.self_attn.o_proj.weight,<br>model.layers.12.self_attn.q_norm.weight,<br>model.layers.12.self_attn.q_proj.biases,<br>model.layers.12.self_attn.q_proj.scales,<br>model.layers.12.self_attn.q_proj.weight,<br>model.layers.12.self_attn.v_proj.biases,<br>model.layers.12.self_attn.v_proj.scales,<br>model.layers.12.self_attn.v_proj.weight,<br>model.layers.13.input_layernorm.weight,<br>model.layers.13.mlp.down_proj.biases,<br>model.layers.13.mlp.down_proj.scales,<br>model.layers.13.mlp.down_proj.weight,<br>model.layers.13.mlp.gate_proj.biases,<br>model.layers.13.mlp.gate_proj.scales,<br>model.layers.13.mlp.gate_proj.weight,<br>model.layers.13.mlp.up_proj.biases,<br>model.layers.13.mlp.up_proj.scales,<br>model.layers.13.mlp.up_proj.weight,<br>model.layers.13.post_attention_layernorm.weight,<br>model.layers.13.self_attn.k_norm.weight,<br>model.layers.13.self_attn.k_proj.biases,<br>model.layers.13.self_attn.k_proj.scales,<br>model.layers.13.self_attn.k_proj.weight,<br>model.layers.13.self_attn.o_proj.biases,<br>model.layers.13.self_attn.o_proj.scales,<br>model.layers.13.self_attn.o_proj.weight,<br>model.layers.13.self_attn.q_norm.weight,<br>model.layers.13.self_attn.q_proj.biases,<br>model.layers.13.self_attn.q_proj.scales,<br>model.layers.13.self_attn.q_proj.weight,<br>model.layers.13.self_attn.v_proj.biases,<br>model.layers.13.self_attn.v_proj.scales,<br>model.layers.13.self_attn.v_proj.weight,<br>model.layers.14.input_layernorm.weight,<br>model.layers.14.mlp.down_proj.biases,<br>model.layers.14.mlp.down_proj.scales,<br>model.layers.14.mlp.down_proj.weight,<br>model.layers.14.mlp.gate_proj.biases,<br>model.layers.14.mlp.gate_proj.scales,<br>model.layers.14.mlp.gate_proj.weight,<br>model.layers.14.mlp.up_proj.biases,<br>model.layers.14.mlp.up_proj.scales,<br>model.layers.14.mlp.up_proj.weight,<br>model.layers.14.post_attention_layernorm.weight,<br>model.layers.14.self_attn.k_norm.weight,<br>model.layers.14.self_attn.k_proj.biases,<br>model.layers.14.self_attn.k_proj.scales,<br>model.layers.14.self_attn.k_proj.weight,<br>model.layers.14.self_attn.o_proj.biases,<br>model.layers.14.self_attn.o_proj.scales,<br>model.layers.14.self_attn.o_proj.weight,<br>model.layers.14.self_attn.q_norm.weight,<br>model.layers.14.self_attn.q_proj.biases,<br>model.layers.14.self_attn.q_proj.scales,<br>model.layers.14.self_attn.q_proj.weight,<br>model.layers.14.self_attn.v_proj.biases,<br>model.layers.14.self_attn.v_proj.scales,<br>model.layers.14.self_attn.v_proj.weight,<br>model.layers.15.input_layernorm.weight,<br>model.layers.15.mlp.down_proj.biases,<br>model.layers.15.mlp.down_proj.scales,<br>model.layers.15.mlp.down_proj.weight,<br>model.layers.15.mlp.gate_proj.biases,<br>model.layers.15.mlp.gate_proj.scales,<br>model.layers.15.mlp.gate_proj.weight,<br>model.layers.15.mlp.up_proj.biases,<br>model.layers.15.mlp.up_proj.scales,<br>model.layers.15.mlp.up_proj.weight,<br>model.layers.15.post_attention_layernorm.weight,<br>model.layers.15.self_attn.k_norm.weight,<br>model.layers.15.self_attn.k_proj.biases,<br>model.layers.15.self_attn.k_proj.scales,<br>model.layers.15.self_attn.k_proj.weight,<br>model.layers.15.self_attn.o_proj.biases,<br>model.layers.15.self_attn.o_proj.scales,<br>model.layers.15.self_attn.o_proj.weight,<br>model.layers.15.self_attn.q_norm.weight,<br>model.layers.15.self_attn.q_proj.biases,<br>model.layers.15.self_attn.q_proj.scales,<br>model.layers.15.self_attn.q_proj.weight,<br>model.layers.15.self_attn.v_proj.biases,<br>model.layers.15.self_attn.v_proj.scales,<br>model.layers.15.self_attn.v_proj.weight,<br>model.layers.16.input_layernorm.weight,<br>model.layers.16.mlp.down_proj.biases,<br>model.layers.16.mlp.down_proj.scales,<br>model.layers.16.mlp.down_proj.weight,<br>model.layers.16.mlp.gate_proj.biases,<br>model.layers.16.mlp.gate_proj.scales,<br>model.layers.16.mlp.gate_proj.weight,<br>model.layers.16.mlp.up_proj.biases,<br>model.layers.16.mlp.up_proj.scales,<br>model.layers.16.mlp.up_proj.weight,<br>model.layers.16.post_attention_layernorm.weight,<br>model.layers.16.self_attn.k_norm.weight,<br>model.layers.16.self_attn.k_proj.biases,<br>model.layers.16.self_attn.k_proj.scales,<br>model.layers.16.self_attn.k_proj.weight,<br>model.layers.16.self_attn.o_proj.biases,<br>model.layers.16.self_attn.o_proj.scales,<br>model.layers.16.self_attn.o_proj.weight,<br>model.layers.16.self_attn.q_norm.weight,<br>model.layers.16.self_attn.q_proj.biases,<br>model.layers.16.self_attn.q_proj.scales,<br>model.layers.16.self_attn.q_proj.weight,<br>model.layers.16.self_attn.v_proj.biases,<br>model.layers.16.self_attn.v_proj.scales,<br>model.layers.16.self_attn.v_proj.weight,<br>model.layers.17.input_layernorm.weight,<br>model.layers.17.mlp.down_proj.biases,<br>model.layers.17.mlp.down_proj.scales,<br>model.layers.17.mlp.down_proj.weight,<br>model.layers.17.mlp.gate_proj.biases,<br>model.layers.17.mlp.gate_proj.scales,<br>model.layers.17.mlp.gate_proj.weight,<br>model.layers.17.mlp.up_proj.biases,<br>model.layers.17.mlp.up_proj.scales,<br>model.layers.17.mlp.up_proj.weight,<br>model.layers.17.post_attention_layernorm.weight,<br>model.layers.17.self_attn.k_norm.weight,<br>model.layers.17.self_attn.k_proj.biases,<br>model.layers.17.self_attn.k_proj.scales,<br>model.layers.17.self_attn.k_proj.weight,<br>model.layers.17.self_attn.o_proj.biases,<br>model.layers.17.self_attn.o_proj.scales,<br>model.layers.17.self_attn.o_proj.weight,<br>model.layers.17.self_attn.q_norm.weight,<br>model.layers.17.self_attn.q_proj.biases,<br>model.layers.17.self_attn.q_proj.scales,<br>model.layers.17.self_attn.q_proj.weight,<br>model.layers.17.self_attn.v_proj.biases,<br>model.layers.17.self_attn.v_proj.scales,<br>model.layers.17.self_attn.v_proj.weight,<br>model.layers.18.input_layernorm.weight,<br>model.layers.18.mlp.down_proj.biases,<br>model.layers.18.mlp.down_proj.scales,<br>model.layers.18.mlp.down_proj.weight,<br>model.layers.18.mlp.gate_proj.biases,<br>model.layers.18.mlp.gate_proj.scales,<br>model.layers.18.mlp.gate_proj.weight,<br>model.layers.18.mlp.up_proj.biases,<br>model.layers.18.mlp.up_proj.scales,<br>model.layers.18.mlp.up_proj.weight,<br>model.layers.18.post_attention_layernorm.weight,<br>model.layers.18.self_attn.k_norm.weight,<br>model.layers.18.self_attn.k_proj.biases,<br>model.layers.18.self_attn.k_proj.scales,<br>model.layers.18.self_attn.k_proj.weight,<br>model.layers.18.self_attn.o_proj.biases,<br>model.layers.18.self_attn.o_proj.scales,<br>model.layers.18.self_attn.o_proj.weight,<br>model.layers.18.self_attn.q_norm.weight,<br>model.layers.18.self_attn.q_proj.biases,<br>model.layers.18.self_attn.q_proj.scales,<br>model.layers.18.self_attn.q_proj.weight,<br>model.layers.18.self_attn.v_proj.biases,<br>model.layers.18.self_attn.v_proj.scales,<br>model.layers.18.self_attn.v_proj.weight,<br>model.layers.19.input_layernorm.weight,<br>model.layers.19.mlp.down_proj.biases,<br>model.layers.19.mlp.down_proj.scales,<br>model.layers.19.mlp.down_proj.weight,<br>model.layers.19.mlp.gate_proj.biases,<br>model.layers.19.mlp.gate_proj.scales,<br>model.layers.19.mlp.gate_proj.weight,<br>model.layers.19.mlp.up_proj.biases,<br>model.layers.19.mlp.up_proj.scales,<br>model.layers.19.mlp.up_proj.weight,<br>model.layers.19.post_attention_layernorm.weight,<br>model.layers.19.self_attn.k_norm.weight,<br>model.layers.19.self_attn.k_proj.biases,<br>model.layers.19.self_attn.k_proj.scales,<br>model.layers.19.self_attn.k_proj.weight,<br>model.layers.19.self_attn.o_proj.biases,<br>model.layers.19.self_attn.o_proj.scales,<br>model.layers.19.self_attn.o_proj.weight,<br>model.layers.19.self_attn.q_norm.weight,<br>model.layers.19.self_attn.q_proj.biases,<br>model.layers.19.self_attn.q_proj.scales,<br>model.layers.19.self_attn.q_proj.weight,<br>model.layers.19.self_attn.v_proj.biases,<br>model.layers.19.self_attn.v_proj.scales,<br>model.layers.19.self_attn.v_proj.weight,<br>model.layers.2.input_layernorm.weight,<br>model.layers.2.mlp.down_proj.biases,<br>model.layers.2.mlp.down_proj.scales,<br>model.layers.2.mlp.down_proj.weight,<br>model.layers.2.mlp.gate_proj.biases,<br>model.layers.2.mlp.gate_proj.scales,<br>model.layers.2.mlp.gate_proj.weight,<br>model.layers.2.mlp.up_proj.biases,<br>model.layers.2.mlp.up_proj.scales,<br>model.layers.2.mlp.up_proj.weight,<br>model.layers.2.post_attention_layernorm.weight,<br>model.layers.2.self_attn.k_norm.weight,<br>model.layers.2.self_attn.k_proj.biases,<br>model.layers.2.self_attn.k_proj.scales,<br>model.layers.2.self_attn.k_proj.weight,<br>model.layers.2.self_attn.o_proj.biases,<br>model.layers.2.self_attn.o_proj.scales,<br>model.layers.2.self_attn.o_proj.weight,<br>model.layers.2.self_attn.q_norm.weight,<br>model.layers.2.self_attn.q_proj.biases,<br>model.layers.2.self_attn.q_proj.scales,<br>model.layers.2.self_attn.q_proj.weight,<br>model.layers.2.self_attn.v_proj.biases,<br>model.layers.2.self_attn.v_proj.scales,<br>model.layers.2.self_attn.v_proj.weight,<br>model.layers.20.input_layernorm.weight,<br>model.layers.20.mlp.down_proj.biases,<br>model.layers.20.mlp.down_proj.scales,<br>model.layers.20.mlp.down_proj.weight,<br>model.layers.20.mlp.gate_proj.biases,<br>model.layers.20.mlp.gate_proj.scales,<br>model.layers.20.mlp.gate_proj.weight,<br>model.layers.20.mlp.up_proj.biases,<br>model.layers.20.mlp.up_proj.scales,<br>model.layers.20.mlp.up_proj.weight,<br>model.layers.20.post_attention_layernorm.weight,<br>model.layers.20.self_attn.k_norm.weight,<br>model.layers.20.self_attn.k_proj.biases,<br>model.layers.20.self_attn.k_proj.scales,<br>model.layers.20.self_attn.k_proj.weight,<br>model.layers.20.self_attn.o_proj.biases,<br>model.layers.20.self_attn.o_proj.scales,<br>model.layers.20.self_attn.o_proj.weight,<br>model.layers.20.self_attn.q_norm.weight,<br>model.layers.20.self_attn.q_proj.biases,<br>model.layers.20.self_attn.q_proj.scales,<br>model.layers.20.self_attn.q_proj.weight,<br>model.layers.20.self_attn.v_proj.biases,<br>model.layers.20.self_attn.v_proj.scales,<br>model.layers.20.self_attn.v_proj.weight,<br>model.layers.21.input_layernorm.weight,<br>model.layers.21.mlp.down_proj.biases,<br>model.layers.21.mlp.down_proj.scales,<br>model.layers.21.mlp.down_proj.weight,<br>model.layers.21.mlp.gate_proj.biases,<br>model.layers.21.mlp.gate_proj.scales,<br>model.layers.21.mlp.gate_proj.weight,<br>model.layers.21.mlp.up_proj.biases,<br>model.layers.21.mlp.up_proj.scales,<br>model.layers.21.mlp.up_proj.weight,<br>model.layers.21.post_attention_layernorm.weight,<br>model.layers.21.self_attn.k_norm.weight,<br>model.layers.21.self_attn.k_proj.biases,<br>model.layers.21.self_attn.k_proj.scales,<br>model.layers.21.self_attn.k_proj.weight,<br>model.layers.21.self_attn.o_proj.biases,<br>model.layers.21.self_attn.o_proj.scales,<br>model.layers.21.self_attn.o_proj.weight,<br>model.layers.21.self_attn.q_norm.weight,<br>model.layers.21.self_attn.q_proj.biases,<br>model.layers.21.self_attn.q_proj.scales,<br>model.layers.21.self_attn.q_proj.weight,<br>model.layers.21.self_attn.v_proj.biases,<br>model.layers.21.self_attn.v_proj.scales,<br>model.layers.21.self_attn.v_proj.weight,<br>model.layers.22.input_layernorm.weight,<br>model.layers.22.mlp.down_proj.biases,<br>model.layers.22.mlp.down_proj.scales,<br>model.layers.22.mlp.down_proj.weight,<br>model.layers.22.mlp.gate_proj.biases,<br>model.layers.22.mlp.gate_proj.scales,<br>model.layers.22.mlp.gate_proj.weight,<br>model.layers.22.mlp.up_proj.biases,<br>model.layers.22.mlp.up_proj.scales,<br>model.layers.22.mlp.up_proj.weight,<br>model.layers.22.post_attention_layernorm.weight,<br>model.layers.22.self_attn.k_norm.weight,<br>model.layers.22.self_attn.k_proj.biases,<br>model.layers.22.self_attn.k_proj.scales,<br>model.layers.22.self_attn.k_proj.weight,<br>model.layers.22.self_attn.o_proj.biases,<br>model.layers.22.self_attn.o_proj.scales,<br>model.layers.22.self_attn.o_proj.weight,<br>model.layers.22.self_attn.q_norm.weight,<br>model.layers.22.self_attn.q_proj.biases,<br>model.layers.22.self_attn.q_proj.scales,<br>model.layers.22.self_attn.q_proj.weight,<br>model.layers.22.self_attn.v_proj.biases,<br>model.layers.22.self_attn.v_proj.scales,<br>model.layers.22.self_attn.v_proj.weight,<br>model.layers.23.input_layernorm.weight,<br>model.layers.23.mlp.down_proj.biases,<br>model.layers.23.mlp.down_proj.scales,<br>model.layers.23.mlp.down_proj.weight,<br>model.layers.23.mlp.gate_proj.biases,<br>model.layers.23.mlp.gate_proj.scales,<br>model.layers.23.mlp.gate_proj.weight,<br>model.layers.23.mlp.up_proj.biases,<br>model.layers.23.mlp.up_proj.scales,<br>model.layers.23.mlp.up_proj.weight,<br>model.layers.23.post_attention_layernorm.weight,<br>model.layers.23.self_attn.k_norm.weight,<br>model.layers.23.self_attn.k_proj.biases,<br>model.layers.23.self_attn.k_proj.scales,<br>model.layers.23.self_attn.k_proj.weight,<br>model.layers.23.self_attn.o_proj.biases,<br>model.layers.23.self_attn.o_proj.scales,<br>model.layers.23.self_attn.o_proj.weight,<br>model.layers.23.self_attn.q_norm.weight,<br>model.layers.23.self_attn.q_proj.biases,<br>model.layers.23.self_attn.q_proj.scales,<br>model.layers.23.self_attn.q_proj.weight,<br>model.layers.23.self_attn.v_proj.biases,<br>model.layers.23.self_attn.v_proj.scales,<br>model.layers.23.self_attn.v_proj.weight,<br>model.layers.24.input_layernorm.weight,<br>model.layers.24.mlp.down_proj.biases,<br>model.layers.24.mlp.down_proj.scales,<br>model.layers.24.mlp.down_proj.weight,<br>model.layers.24.mlp.gate_proj.biases,<br>model.layers.24.mlp.gate_proj.scales,<br>model.layers.24.mlp.gate_proj.weight,<br>model.layers.24.mlp.up_proj.biases,<br>model.layers.24.mlp.up_proj.scales,<br>model.layers.24.mlp.up_proj.weight,<br>model.layers.24.post_attention_layernorm.weight,<br>model.layers.24.self_attn.k_norm.weight,<br>model.layers.24.self_attn.k_proj.biases,<br>model.layers.24.self_attn.k_proj.scales,<br>model.layers.24.self_attn.k_proj.weight,<br>model.layers.24.self_attn.o_proj.biases,<br>model.layers.24.self_attn.o_proj.scales,<br>model.layers.24.self_attn.o_proj.weight,<br>model.layers.24.self_attn.q_norm.weight,<br>model.layers.24.self_attn.q_proj.biases,<br>model.layers.24.self_attn.q_proj.scales,<br>model.layers.24.self_attn.q_proj.weight,<br>model.layers.24.self_attn.v_proj.biases,<br>model.layers.24.self_attn.v_proj.scales,<br>model.layers.24.self_attn.v_proj.weight,<br>model.layers.25.input_layernorm.weight,<br>model.layers.25.mlp.down_proj.biases,<br>model.layers.25.mlp.down_proj.scales,<br>model.layers.25.mlp.down_proj.weight,<br>model.layers.25.mlp.gate_proj.biases,<br>model.layers.25.mlp.gate_proj.scales,<br>model.layers.25.mlp.gate_proj.weight,<br>model.layers.25.mlp.up_proj.biases,<br>model.layers.25.mlp.up_proj.scales,<br>model.layers.25.mlp.up_proj.weight,<br>model.layers.25.post_attention_layernorm.weight,<br>model.layers.25.self_attn.k_norm.weight,<br>model.layers.25.self_attn.k_proj.biases,<br>model.layers.25.self_attn.k_proj.scales,<br>model.layers.25.self_attn.k_proj.weight,<br>model.layers.25.self_attn.o_proj.biases,<br>model.layers.25.self_attn.o_proj.scales,<br>model.layers.25.self_attn.o_proj.weight,<br>model.layers.25.self_attn.q_norm.weight,<br>model.layers.25.self_attn.q_proj.biases,<br>model.layers.25.self_attn.q_proj.scales,<br>model.layers.25.self_attn.q_proj.weight,<br>model.layers.25.self_attn.v_proj.biases,<br>model.layers.25.self_attn.v_proj.scales,<br>model.layers.25.self_attn.v_proj.weight,<br>model.layers.26.input_layernorm.weight,<br>model.layers.26.mlp.down_proj.biases,<br>model.layers.26.mlp.down_proj.scales,<br>model.layers.26.mlp.down_proj.weight,<br>model.layers.26.mlp.gate_proj.biases,<br>model.layers.26.mlp.gate_proj.scales,<br>model.layers.26.mlp.gate_proj.weight,<br>model.layers.26.mlp.up_proj.biases,<br>model.layers.26.mlp.up_proj.scales,<br>model.layers.26.mlp.up_proj.weight,<br>model.layers.26.post_attention_layernorm.weight,<br>model.layers.26.self_attn.k_norm.weight,<br>model.layers.26.self_attn.k_proj.biases,<br>model.layers.26.self_attn.k_proj.scales,<br>model.layers.26.self_attn.k_proj.weight,<br>model.layers.26.self_attn.o_proj.biases,<br>model.layers.26.self_attn.o_proj.scales,<br>model.layers.26.self_attn.o_proj.weight,<br>model.layers.26.self_attn.q_norm.weight,<br>model.layers.26.self_attn.q_proj.biases,<br>model.layers.26.self_attn.q_proj.scales,<br>model.layers.26.self_attn.q_proj.weight,<br>model.layers.26.self_attn.v_proj.biases,<br>model.layers.26.self_attn.v_proj.scales,<br>model.layers.26.self_attn.v_proj.weight,<br>model.layers.27.input_layernorm.weight,<br>model.layers.27.mlp.down_proj.biases,<br>model.layers.27.mlp.down_proj.scales,<br>model.layers.27.mlp.down_proj.weight,<br>model.layers.27.mlp.gate_proj.biases,<br>model.layers.27.mlp.gate_proj.scales,<br>model.layers.27.mlp.gate_proj.weight,<br>model.layers.27.mlp.up_proj.biases,<br>model.layers.27.mlp.up_proj.scales,<br>model.layers.27.mlp.up_proj.weight,<br>model.layers.27.post_attention_layernorm.weight,<br>model.layers.27.self_attn.k_norm.weight,<br>model.layers.27.self_attn.k_proj.biases,<br>model.layers.27.self_attn.k_proj.scales,<br>model.layers.27.self_attn.k_proj.weight,<br>model.layers.27.self_attn.o_proj.biases,<br>model.layers.27.self_attn.o_proj.scales,<br>model.layers.27.self_attn.o_proj.weight,<br>model.layers.27.self_attn.q_norm.weight,<br>model.layers.27.self_attn.q_proj.biases,<br>model.layers.27.self_attn.q_proj.scales,<br>model.layers.27.self_attn.q_proj.weight,<br>model.layers.27.self_attn.v_proj.biases,<br>model.layers.27.self_attn.v_proj.scales,<br>model.layers.27.self_attn.v_proj.weight,<br>model.layers.28.input_layernorm.weight,<br>model.layers.28.mlp.down_proj.biases,<br>model.layers.28.mlp.down_proj.scales,<br>model.layers.28.mlp.down_proj.weight,<br>model.layers.28.mlp.gate_proj.biases,<br>model.layers.28.mlp.gate_proj.scales,<br>model.layers.28.mlp.gate_proj.weight,<br>model.layers.28.mlp.up_proj.biases,<br>model.layers.28.mlp.up_proj.scales,<br>model.layers.28.mlp.up_proj.weight,<br>model.layers.28.post_attention_layernorm.weight,<br>model.layers.28.self_attn.k_norm.weight,<br>model.layers.28.self_attn.k_proj.biases,<br>model.layers.28.self_attn.k_proj.scales,<br>model.layers.28.self_attn.k_proj.weight,<br>model.layers.28.self_attn.o_proj.biases,<br>model.layers.28.self_attn.o_proj.scales,<br>model.layers.28.self_attn.o_proj.weight,<br>model.layers.28.self_attn.q_norm.weight,<br>model.layers.28.self_attn.q_proj.biases,<br>model.layers.28.self_attn.q_proj.scales,<br>model.layers.28.self_attn.q_proj.weight,<br>model.layers.28.self_attn.v_proj.biases,<br>model.layers.28.self_attn.v_proj.scales,<br>model.layers.28.self_attn.v_proj.weight,<br>model.layers.29.input_layernorm.weight,<br>model.layers.29.mlp.down_proj.biases,<br>model.layers.29.mlp.down_proj.scales,<br>model.layers.29.mlp.down_proj.weight,<br>model.layers.29.mlp.gate_proj.biases,<br>model.layers.29.mlp.gate_proj.scales,<br>model.layers.29.mlp.gate_proj.weight,<br>model.layers.29.mlp.up_proj.biases,<br>model.layers.29.mlp.up_proj.scales,<br>model.layers.29.mlp.up_proj.weight,<br>model.layers.29.post_attention_layernorm.weight,<br>model.layers.29.self_attn.k_norm.weight,<br>model.layers.29.self_attn.k_proj.biases,<br>model.layers.29.self_attn.k_proj.scales,<br>model.layers.29.self_attn.k_proj.weight,<br>model.layers.29.self_attn.o_proj.biases,<br>model.layers.29.self_attn.o_proj.scales,<br>model.layers.29.self_attn.o_proj.weight,<br>model.layers.29.self_attn.q_norm.weight,<br>model.layers.29.self_attn.q_proj.biases,<br>model.layers.29.self_attn.q_proj.scales,<br>model.layers.29.self_attn.q_proj.weight,<br>model.layers.29.self_attn.v_proj.biases,<br>model.layers.29.self_attn.v_proj.scales,<br>model.layers.29.self_attn.v_proj.weight,<br>model.layers.3.input_layernorm.weight,<br>model.layers.3.mlp.down_proj.biases,<br>model.layers.3.mlp.down_proj.scales,<br>model.layers.3.mlp.down_proj.weight,<br>model.layers.3.mlp.gate_proj.biases,<br>model.layers.3.mlp.gate_proj.scales,<br>model.layers.3.mlp.gate_proj.weight,<br>model.layers.3.mlp.up_proj.biases,<br>model.layers.3.mlp.up_proj.scales,<br>model.layers.3.mlp.up_proj.weight,<br>model.layers.3.post_attention_layernorm.weight,<br>model.layers.3.self_attn.k_norm.weight,<br>model.layers.3.self_attn.k_proj.biases,<br>model.layers.3.self_attn.k_proj.scales,<br>model.layers.3.self_attn.k_proj.weight,<br>model.layers.3.self_attn.o_proj.biases,<br>model.layers.3.self_attn.o_proj.scales,<br>model.layers.3.self_attn.o_proj.weight,<br>model.layers.3.self_attn.q_norm.weight,<br>model.layers.3.self_attn.q_proj.biases,<br>model.layers.3.self_attn.q_proj.scales,<br>model.layers.3.self_attn.q_proj.weight,<br>model.layers.3.self_attn.v_proj.biases,<br>model.layers.3.self_attn.v_proj.scales,<br>model.layers.3.self_attn.v_proj.weight,<br>model.layers.30.input_layernorm.weight,<br>model.layers.30.mlp.down_proj.biases,<br>model.layers.30.mlp.down_proj.scales,<br>model.layers.30.mlp.down_proj.weight,<br>model.layers.30.mlp.gate_proj.biases,<br>model.layers.30.mlp.gate_proj.scales,<br>model.layers.30.mlp.gate_proj.weight,<br>model.layers.30.mlp.up_proj.biases,<br>model.layers.30.mlp.up_proj.scales,<br>model.layers.30.mlp.up_proj.weight,<br>model.layers.30.post_attention_layernorm.weight,<br>model.layers.30.self_attn.k_norm.weight,<br>model.layers.30.self_attn.k_proj.biases,<br>model.layers.30.self_attn.k_proj.scales,<br>model.layers.30.self_attn.k_proj.weight,<br>model.layers.30.self_attn.o_proj.biases,<br>model.layers.30.self_attn.o_proj.scales,<br>model.layers.30.self_attn.o_proj.weight,<br>model.layers.30.self_attn.q_norm.weight,<br>model.layers.30.self_attn.q_proj.biases,<br>model.layers.30.self_attn.q_proj.scales,<br>model.layers.30.self_attn.q_proj.weight,<br>model.layers.30.self_attn.v_proj.biases,<br>model.layers.30.self_attn.v_proj.scales,<br>model.layers.30.self_attn.v_proj.weight,<br>model.layers.31.input_layernorm.weight,<br>model.layers.31.mlp.down_proj.biases,<br>model.layers.31.mlp.down_proj.scales,<br>model.layers.31.mlp.down_proj.weight,<br>model.layers.31.mlp.gate_proj.biases,<br>model.layers.31.mlp.gate_proj.scales,<br>model.layers.31.mlp.gate_proj.weight,<br>model.layers.31.mlp.up_proj.biases,<br>model.layers.31.mlp.up_proj.scales,<br>model.layers.31.mlp.up_proj.weight,<br>model.layers.31.post_attention_layernorm.weight,<br>model.layers.31.self_attn.k_norm.weight,<br>model.layers.31.self_attn.k_proj.biases,<br>model.layers.31.self_attn.k_proj.scales,<br>model.layers.31.self_attn.k_proj.weight,<br>model.layers.31.self_attn.o_proj.biases,<br>model.layers.31.self_attn.o_proj.scales,<br>model.layers.31.self_attn.o_proj.weight,<br>model.layers.31.self_attn.q_norm.weight,<br>model.layers.31.self_attn.q_proj.biases,<br>model.layers.31.self_attn.q_proj.scales,<br>model.layers.31.self_attn.q_proj.weight,<br>model.layers.31.self_attn.v_proj.biases,<br>model.layers.31.self_attn.v_proj.scales,<br>model.layers.31.self_attn.v_proj.weight,<br>model.layers.32.input_layernorm.weight,<br>model.layers.32.mlp.down_proj.biases,<br>model.layers.32.mlp.down_proj.scales,<br>model.layers.32.mlp.down_proj.weight,<br>model.layers.32.mlp.gate_proj.biases,<br>model.layers.32.mlp.gate_proj.scales,<br>model.layers.32.mlp.gate_proj.weight,<br>model.layers.32.mlp.up_proj.biases,<br>model.layers.32.mlp.up_proj.scales,<br>model.layers.32.mlp.up_proj.weight,<br>model.layers.32.post_attention_layernorm.weight,<br>model.layers.32.self_attn.k_norm.weight,<br>model.layers.32.self_attn.k_proj.biases,<br>model.layers.32.self_attn.k_proj.scales,<br>model.layers.32.self_attn.k_proj.weight,<br>model.layers.32.self_attn.o_proj.biases,<br>model.layers.32.self_attn.o_proj.scales,<br>model.layers.32.self_attn.o_proj.weight,<br>model.layers.32.self_attn.q_norm.weight,<br>model.layers.32.self_attn.q_proj.biases,<br>model.layers.32.self_attn.q_proj.scales,<br>model.layers.32.self_attn.q_proj.weight,<br>model.layers.32.self_attn.v_proj.biases,<br>model.layers.32.self_attn.v_proj.scales,<br>model.layers.32.self_attn.v_proj.weight,<br>model.layers.33.input_layernorm.weight,<br>model.layers.33.mlp.down_proj.biases,<br>model.layers.33.mlp.down_proj.scales,<br>model.layers.33.mlp.down_proj.weight,<br>model.layers.33.mlp.gate_proj.biases,<br>model.layers.33.mlp.gate_proj.scales,<br>model.layers.33.mlp.gate_proj.weight,<br>model.layers.33.mlp.up_proj.biases,<br>model.layers.33.mlp.up_proj.scales,<br>model.layers.33.mlp.up_proj.weight,<br>model.layers.33.post_attention_layernorm.weight,<br>model.layers.33.self_attn.k_norm.weight,<br>model.layers.33.self_attn.k_proj.biases,<br>model.layers.33.self_attn.k_proj.scales,<br>model.layers.33.self_attn.k_proj.weight,<br>model.layers.33.self_attn.o_proj.biases,<br>model.layers.33.self_attn.o_proj.scales,<br>model.layers.33.self_attn.o_proj.weight,<br>model.layers.33.self_attn.q_norm.weight,<br>model.layers.33.self_attn.q_proj.biases,<br>model.layers.33.self_attn.q_proj.scales,<br>model.layers.33.self_attn.q_proj.weight,<br>model.layers.33.self_attn.v_proj.biases,<br>model.layers.33.self_attn.v_proj.scales,<br>model.layers.33.self_attn.v_proj.weight,<br>model.layers.34.input_layernorm.weight,<br>model.layers.34.mlp.down_proj.biases,<br>model.layers.34.mlp.down_proj.scales,<br>model.layers.34.mlp.down_proj.weight,<br>model.layers.34.mlp.gate_proj.biases,<br>model.layers.34.mlp.gate_proj.scales,<br>model.layers.34.mlp.gate_proj.weight,<br>model.layers.34.mlp.up_proj.biases,<br>model.layers.34.mlp.up_proj.scales,<br>model.layers.34.mlp.up_proj.weight,<br>model.layers.34.post_attention_layernorm.weight,<br>model.layers.34.self_attn.k_norm.weight,<br>model.layers.34.self_attn.k_proj.biases,<br>model.layers.34.self_attn.k_proj.scales,<br>model.layers.34.self_attn.k_proj.weight,<br>model.layers.34.self_attn.o_proj.biases,<br>model.layers.34.self_attn.o_proj.scales,<br>model.layers.34.self_attn.o_proj.weight,<br>model.layers.34.self_attn.q_norm.weight,<br>model.layers.34.self_attn.q_proj.biases,<br>model.layers.34.self_attn.q_proj.scales,<br>model.layers.34.self_attn.q_proj.weight,<br>model.layers.34.self_attn.v_proj.biases,<br>model.layers.34.self_attn.v_proj.scales,<br>model.layers.34.self_attn.v_proj.weight,<br>model.layers.35.input_layernorm.weight,<br>model.layers.35.mlp.down_proj.biases,<br>model.layers.35.mlp.down_proj.scales,<br>model.layers.35.mlp.down_proj.weight,<br>model.layers.35.mlp.gate_proj.biases,<br>model.layers.35.mlp.gate_proj.scales,<br>model.layers.35.mlp.gate_proj.weight,<br>model.layers.35.mlp.up_proj.biases,<br>model.layers.35.mlp.up_proj.scales,<br>model.layers.35.mlp.up_proj.weight,<br>model.layers.35.post_attention_layernorm.weight,<br>model.layers.35.self_attn.k_norm.weight,<br>model.layers.35.self_attn.k_proj.biases,<br>model.layers.35.self_attn.k_proj.scales,<br>model.layers.35.self_attn.k_proj.weight,<br>model.layers.35.self_attn.o_proj.biases,<br>model.layers.35.self_attn.o_proj.scales,<br>model.layers.35.self_attn.o_proj.weight,<br>model.layers.35.self_attn.q_norm.weight,<br>model.layers.35.self_attn.q_proj.biases,<br>model.layers.35.self_attn.q_proj.scales,<br>model.layers.35.self_attn.q_proj.weight,<br>model.layers.35.self_attn.v_proj.biases,<br>model.layers.35.self_attn.v_proj.scales,<br>model.layers.35.self_attn.v_proj.weight,<br>model.layers.4.input_layernorm.weight,<br>model.layers.4.mlp.down_proj.biases,<br>model.layers.4.mlp.down_proj.scales,<br>model.layers.4.mlp.down_proj.weight,<br>model.layers.4.mlp.gate_proj.biases,<br>model.layers.4.mlp.gate_proj.scales,<br>model.layers.4.mlp.gate_proj.weight,<br>model.layers.4.mlp.up_proj.biases,<br>model.layers.4.mlp.up_proj.scales,<br>model.layers.4.mlp.up_proj.weight,<br>model.layers.4.post_attention_layernorm.weight,<br>model.layers.4.self_attn.k_norm.weight,<br>model.layers.4.self_attn.k_proj.biases,<br>model.layers.4.self_attn.k_proj.scales,<br>model.layers.4.self_attn.k_proj.weight,<br>model.layers.4.self_attn.o_proj.biases,<br>model.layers.4.self_attn.o_proj.scales,<br>model.layers.4.self_attn.o_proj.weight,<br>model.layers.4.self_attn.q_norm.weight,<br>model.layers.4.self_attn.q_proj.biases,<br>model.layers.4.self_attn.q_proj.scales,<br>model.layers.4.self_attn.q_proj.weight,<br>model.layers.4.self_attn.v_proj.biases,<br>model.layers.4.self_attn.v_proj.scales,<br>model.layers.4.self_attn.v_proj.weight,<br>model.layers.5.input_layernorm.weight,<br>model.layers.5.mlp.down_proj.biases,<br>model.layers.5.mlp.down_proj.scales,<br>model.layers.5.mlp.down_proj.weight,<br>model.layers.5.mlp.gate_proj.biases,<br>model.layers.5.mlp.gate_proj.scales,<br>model.layers.5.mlp.gate_proj.weight,<br>model.layers.5.mlp.up_proj.biases,<br>model.layers.5.mlp.up_proj.scales,<br>model.layers.5.mlp.up_proj.weight,<br>model.layers.5.post_attention_layernorm.weight,<br>model.layers.5.self_attn.k_norm.weight,<br>model.layers.5.self_attn.k_proj.biases,<br>model.layers.5.self_attn.k_proj.scales,<br>model.layers.5.self_attn.k_proj.weight,<br>model.layers.5.self_attn.o_proj.biases,<br>model.layers.5.self_attn.o_proj.scales,<br>model.layers.5.self_attn.o_proj.weight,<br>model.layers.5.self_attn.q_norm.weight,<br>model.layers.5.self_attn.q_proj.biases,<br>model.layers.5.self_attn.q_proj.scales,<br>model.layers.5.self_attn.q_proj.weight,<br>model.layers.5.self_attn.v_proj.biases,<br>model.layers.5.self_attn.v_proj.scales,<br>model.layers.5.self_attn.v_proj.weight,<br>model.layers.6.input_layernorm.weight,<br>model.layers.6.mlp.down_proj.biases,<br>model.layers.6.mlp.down_proj.scales,<br>model.layers.6.mlp.down_proj.weight,<br>model.layers.6.mlp.gate_proj.biases,<br>model.layers.6.mlp.gate_proj.scales,<br>model.layers.6.mlp.gate_proj.weight,<br>model.layers.6.mlp.up_proj.biases,<br>model.layers.6.mlp.up_proj.scales,<br>model.layers.6.mlp.up_proj.weight,<br>model.layers.6.post_attention_layernorm.weight,<br>model.layers.6.self_attn.k_norm.weight,<br>model.layers.6.self_attn.k_proj.biases,<br>model.layers.6.self_attn.k_proj.scales,<br>model.layers.6.self_attn.k_proj.weight,<br>model.layers.6.self_attn.o_proj.biases,<br>model.layers.6.self_attn.o_proj.scales,<br>model.layers.6.self_attn.o_proj.weight,<br>model.layers.6.self_attn.q_norm.weight,<br>model.layers.6.self_attn.q_proj.biases,<br>model.layers.6.self_attn.q_proj.scales,<br>model.layers.6.self_attn.q_proj.weight,<br>model.layers.6.self_attn.v_proj.biases,<br>model.layers.6.self_attn.v_proj.scales,<br>model.layers.6.self_attn.v_proj.weight,<br>model.layers.7.input_layernorm.weight,<br>model.layers.7.mlp.down_proj.biases,<br>model.layers.7.mlp.down_proj.scales,<br>model.layers.7.mlp.down_proj.weight,<br>model.layers.7.mlp.gate_proj.biases,<br>model.layers.7.mlp.gate_proj.scales,<br>model.layers.7.mlp.gate_proj.weight,<br>model.layers.7.mlp.up_proj.biases,<br>model.layers.7.mlp.up_proj.scales,<br>model.layers.7.mlp.up_proj.weight,<br>model.layers.7.post_attention_layernorm.weight,<br>model.layers.7.self_attn.k_norm.weight,<br>model.layers.7.self_attn.k_proj.biases,<br>model.layers.7.self_attn.k_proj.scales,<br>model.layers.7.self_attn.k_proj.weight,<br>model.layers.7.self_attn.o_proj.biases,<br>model.layers.7.self_attn.o_proj.scales,<br>model.layers.7.self_attn.o_proj.weight,<br>model.layers.7.self_attn.q_norm.weight,<br>model.layers.7.self_attn.q_proj.biases,<br>model.layers.7.self_attn.q_proj.scales,<br>model.layers.7.self_attn.q_proj.weight,<br>model.layers.7.self_attn.v_proj.biases,<br>model.layers.7.self_attn.v_proj.scales,<br>model.layers.7.self_attn.v_proj.weight,<br>model.layers.8.input_layernorm.weight,<br>model.layers.8.mlp.down_proj.biases,<br>model.layers.8.mlp.down_proj.scales,<br>model.layers.8.mlp.down_proj.weight,<br>model.layers.8.mlp.gate_proj.biases,<br>model.layers.8.mlp.gate_proj.scales,<br>model.layers.8.mlp.gate_proj.weight,<br>model.layers.8.mlp.up_proj.biases,<br>model.layers.8.mlp.up_proj.scales,<br>model.layers.8.mlp.up_proj.weight,<br>model.layers.8.post_attention_layernorm.weight,<br>model.layers.8.self_attn.k_norm.weight,<br>model.layers.8.self_attn.k_proj.biases,<br>model.layers.8.self_attn.k_proj.scales,<br>model.layers.8.self_attn.k_proj.weight,<br>model.layers.8.self_attn.o_proj.biases,<br>model.layers.8.self_attn.o_proj.scales,<br>model.layers.8.self_attn.o_proj.weight,<br>model.layers.8.self_attn.q_norm.weight,<br>model.layers.8.self_attn.q_proj.biases,<br>model.layers.8.self_attn.q_proj.scales,<br>model.layers.8.self_attn.q_proj.weight,<br>model.layers.8.self_attn.v_proj.biases,<br>model.layers.8.self_attn.v_proj.scales,<br>model.layers.8.self_attn.v_proj.weight,<br>model.layers.9.input_layernorm.weight,<br>model.layers.9.mlp.down_proj.biases,<br>model.layers.9.mlp.down_proj.scales,<br>model.layers.9.mlp.down_proj.weight,<br>model.layers.9.mlp.gate_proj.biases,<br>model.layers.9.mlp.gate_proj.scales,<br>model.layers.9.mlp.gate_proj.weight,<br>model.layers.9.mlp.up_proj.biases,<br>model.layers.9.mlp.up_proj.scales,<br>model.layers.9.mlp.up_proj.weight,<br>model.layers.9.post_attention_layernorm.weight,<br>model.layers.9.self_attn.k_norm.weight,<br>model.layers.9.self_attn.k_proj.biases,<br>model.layers.9.self_attn.k_proj.scales,<br>model.layers.9.self_attn.k_proj.weight,<br>model.layers.9.self_attn.o_proj.biases,<br>model.layers.9.self_attn.o_proj.scales,<br>model.layers.9.self_attn.o_proj.weight,<br>model.layers.9.self_attn.q_norm.weight,<br>model.layers.9.self_attn.q_proj.biases,<br>model.layers.9.self_attn.q_proj.scales,<br>model.layers.9.self_attn.q_proj.weight,<br>model.layers.9.self_attn.v_proj.biases,<br>model.layers.9.self_attn.v_proj.scales,<br>model.layers.9.self_attn.v_proj.weight,<br>model.norm.weight.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Received 904 parameters not in model:
  <br>model.embed_tokens.biases,<br>model.embed_tokens.scales,<br>model.embed_tokens.weight,<br>model.layers.0.input_layernorm.weight,<br>model.layers.0.mlp.down_proj.biases,<br>model.layers.0.mlp.down_proj.scales,<br>model.layers.0.mlp.down_proj.weight,<br>model.layers.0.mlp.gate_proj.biases,<br>model.layers.0.mlp.gate_proj.scales,<br>model.layers.0.mlp.gate_proj.weight,<br>model.layers.0.mlp.up_proj.biases,<br>model.layers.0.mlp.up_proj.scales,<br>model.layers.0.mlp.up_proj.weight,<br>model.layers.0.post_attention_layernorm.weight,<br>model.layers.0.self_attn.k_norm.weight,<br>model.layers.0.self_attn.k_proj.biases,<br>model.layers.0.self_attn.k_proj.scales,<br>model.layers.0.self_attn.k_proj.weight,<br>model.layers.0.self_attn.o_proj.biases,<br>model.layers.0.self_attn.o_proj.scales,<br>model.layers.0.self_attn.o_proj.weight,<br>model.layers.0.self_attn.q_norm.weight,<br>model.layers.0.self_attn.q_proj.biases,<br>model.layers.0.self_attn.q_proj.scales,<br>model.layers.0.self_attn.q_proj.weight,<br>model.layers.0.self_attn.v_proj.biases,<br>model.layers.0.self_attn.v_proj.scales,<br>model.layers.0.self_attn.v_proj.weight,<br>model.layers.1.input_layernorm.weight,<br>model.layers.1.mlp.down_proj.biases,<br>model.layers.1.mlp.down_proj.scales,<br>model.layers.1.mlp.down_proj.weight,<br>model.layers.1.mlp.gate_proj.biases,<br>model.layers.1.mlp.gate_proj.scales,<br>model.layers.1.mlp.gate_proj.weight,<br>model.layers.1.mlp.up_proj.biases,<br>model.layers.1.mlp.up_proj.scales,<br>model.layers.1.mlp.up_proj.weight,<br>model.layers.1.post_attention_layernorm.weight,<br>model.layers.1.self_attn.k_norm.weight,<br>model.layers.1.self_attn.k_proj.biases,<br>model.layers.1.self_attn.k_proj.scales,<br>model.layers.1.self_attn.k_proj.weight,<br>model.layers.1.self_attn.o_proj.biases,<br>model.layers.1.self_attn.o_proj.scales,<br>model.layers.1.self_attn.o_proj.weight,<br>model.layers.1.self_attn.q_norm.weight,<br>model.layers.1.self_attn.q_proj.biases,<br>model.layers.1.self_attn.q_proj.scales,<br>model.layers.1.self_attn.q_proj.weight,<br>model.layers.1.self_attn.v_proj.biases,<br>model.layers.1.self_attn.v_proj.scales,<br>model.layers.1.self_attn.v_proj.weight,<br>model.layers.10.input_layernorm.weight,<br>model.layers.10.mlp.down_proj.biases,<br>model.layers.10.mlp.down_proj.scales,<br>model.layers.10.mlp.down_proj.weight,<br>model.layers.10.mlp.gate_proj.biases,<br>model.layers.10.mlp.gate_proj.scales,<br>model.layers.10.mlp.gate_proj.weight,<br>model.layers.10.mlp.up_proj.biases,<br>model.layers.10.mlp.up_proj.scales,<br>model.layers.10.mlp.up_proj.weight,<br>model.layers.10.post_attention_layernorm.weight,<br>model.layers.10.self_attn.k_norm.weight,<br>model.layers.10.self_attn.k_proj.biases,<br>model.layers.10.self_attn.k_proj.scales,<br>model.layers.10.self_attn.k_proj.weight,<br>model.layers.10.self_attn.o_proj.biases,<br>model.layers.10.self_attn.o_proj.scales,<br>model.layers.10.self_attn.o_proj.weight,<br>model.layers.10.self_attn.q_norm.weight,<br>model.layers.10.self_attn.q_proj.biases,<br>model.layers.10.self_attn.q_proj.scales,<br>model.layers.10.self_attn.q_proj.weight,<br>model.layers.10.self_attn.v_proj.biases,<br>model.layers.10.self_attn.v_proj.scales,<br>model.layers.10.self_attn.v_proj.weight,<br>model.layers.11.input_layernorm.weight,<br>model.layers.11.mlp.down_proj.biases,<br>model.layers.11.mlp.down_proj.scales,<br>model.layers.11.mlp.down_proj.weight,<br>model.layers.11.mlp.gate_proj.biases,<br>model.layers.11.mlp.gate_proj.scales,<br>model.layers.11.mlp.gate_proj.weight,<br>model.layers.11.mlp.up_proj.biases,<br>model.layers.11.mlp.up_proj.scales,<br>model.layers.11.mlp.up_proj.weight,<br>model.layers.11.post_attention_layernorm.weight,<br>model.layers.11.self_attn.k_norm.weight,<br>model.layers.11.self_attn.k_proj.biases,<br>model.layers.11.self_attn.k_proj.scales,<br>model.layers.11.self_attn.k_proj.weight,<br>model.layers.11.self_attn.o_proj.biases,<br>model.layers.11.self_attn.o_proj.scales,<br>model.layers.11.self_attn.o_proj.weight,<br>model.layers.11.self_attn.q_norm.weight,<br>model.layers.11.self_attn.q_proj.biases,<br>model.layers.11.self_attn.q_proj.scales,<br>model.layers.11.self_attn.q_proj.weight,<br>model.layers.11.self_attn.v_proj.biases,<br>model.layers.11.self_attn.v_proj.scales,<br>model.layers.11.self_attn.v_proj.weight,<br>model.layers.12.input_layernorm.weight,<br>model.layers.12.mlp.down_proj.biases,<br>model.layers.12.mlp.down_proj.scales,<br>model.layers.12.mlp.down_proj.weight,<br>model.layers.12.mlp.gate_proj.biases,<br>model.layers.12.mlp.gate_proj.scales,<br>model.layers.12.mlp.gate_proj.weight,<br>model.layers.12.mlp.up_proj.biases,<br>model.layers.12.mlp.up_proj.scales,<br>model.layers.12.mlp.up_proj.weight,<br>model.layers.12.post_attention_layernorm.weight,<br>model.layers.12.self_attn.k_norm.weight,<br>model.layers.12.self_attn.k_proj.biases,<br>model.layers.12.self_attn.k_proj.scales,<br>model.layers.12.self_attn.k_proj.weight,<br>model.layers.12.self_attn.o_proj.biases,<br>model.layers.12.self_attn.o_proj.scales,<br>model.layers.12.self_attn.o_proj.weight,<br>model.layers.12.self_attn.q_norm.weight,<br>model.layers.12.self_attn.q_proj.biases,<br>model.layers.12.self_attn.q_proj.scales,<br>model.layers.12.self_attn.q_proj.weight,<br>model.layers.12.self_attn.v_proj.biases,<br>model.layers.12.self_attn.v_proj.scales,<br>model.layers.12.self_attn.v_proj.weight,<br>model.layers.13.input_layernorm.weight,<br>model.layers.13.mlp.down_proj.biases,<br>model.layers.13.mlp.down_proj.scales,<br>model.layers.13.mlp.down_proj.weight,<br>model.layers.13.mlp.gate_proj.biases,<br>model.layers.13.mlp.gate_proj.scales,<br>model.layers.13.mlp.gate_proj.weight,<br>model.layers.13.mlp.up_proj.biases,<br>model.layers.13.mlp.up_proj.scales,<br>model.layers.13.mlp.up_proj.weight,<br>model.layers.13.post_attention_layernorm.weight,<br>model.layers.13.self_attn.k_norm.weight,<br>model.layers.13.self_attn.k_proj.biases,<br>model.layers.13.self_attn.k_proj.scales,<br>model.layers.13.self_attn.k_proj.weight,<br>model.layers.13.self_attn.o_proj.biases,<br>model.layers.13.self_attn.o_proj.scales,<br>model.layers.13.self_attn.o_proj.weight,<br>model.layers.13.self_attn.q_norm.weight,<br>model.layers.13.self_attn.q_proj.biases,<br>model.layers.13.self_attn.q_proj.scales,<br>model.layers.13.self_attn.q_proj.weight,<br>model.layers.13.self_attn.v_proj.biases,<br>model.layers.13.self_attn.v_proj.scales,<br>model.layers.13.self_attn.v_proj.weight,<br>model.layers.14.input_layernorm.weight,<br>model.layers.14.mlp.down_proj.biases,<br>model.layers.14.mlp.down_proj.scales,<br>model.layers.14.mlp.down_proj.weight,<br>model.layers.14.mlp.gate_proj.biases,<br>model.layers.14.mlp.gate_proj.scales,<br>model.layers.14.mlp.gate_proj.weight,<br>model.layers.14.mlp.up_proj.biases,<br>model.layers.14.mlp.up_proj.scales,<br>model.layers.14.mlp.up_proj.weight,<br>model.layers.14.post_attention_layernorm.weight,<br>model.layers.14.self_attn.k_norm.weight,<br>model.layers.14.self_attn.k_proj.biases,<br>model.layers.14.self_attn.k_proj.scales,<br>model.layers.14.self_attn.k_proj.weight,<br>model.layers.14.self_attn.o_proj.biases,<br>model.layers.14.self_attn.o_proj.scales,<br>model.layers.14.self_attn.o_proj.weight,<br>model.layers.14.self_attn.q_norm.weight,<br>model.layers.14.self_attn.q_proj.biases,<br>model.layers.14.self_attn.q_proj.scales,<br>model.layers.14.self_attn.q_proj.weight,<br>model.layers.14.self_attn.v_proj.biases,<br>model.layers.14.self_attn.v_proj.scales,<br>model.layers.14.self_attn.v_proj.weight,<br>model.layers.15.input_layernorm.weight,<br>model.layers.15.mlp.down_proj.biases,<br>model.layers.15.mlp.down_proj.scales,<br>model.layers.15.mlp.down_proj.weight,<br>model.layers.15.mlp.gate_proj.biases,<br>model.layers.15.mlp.gate_proj.scales,<br>model.layers.15.mlp.gate_proj.weight,<br>model.layers.15.mlp.up_proj.biases,<br>model.layers.15.mlp.up_proj.scales,<br>model.layers.15.mlp.up_proj.weight,<br>model.layers.15.post_attention_layernorm.weight,<br>model.layers.15.self_attn.k_norm.weight,<br>model.layers.15.self_attn.k_proj.biases,<br>model.layers.15.self_attn.k_proj.scales,<br>model.layers.15.self_attn.k_proj.weight,<br>model.layers.15.self_attn.o_proj.biases,<br>model.layers.15.self_attn.o_proj.scales,<br>model.layers.15.self_attn.o_proj.weight,<br>model.layers.15.self_attn.q_norm.weight,<br>model.layers.15.self_attn.q_proj.biases,<br>model.layers.15.self_attn.q_proj.scales,<br>model.layers.15.self_attn.q_proj.weight,<br>model.layers.15.self_attn.v_proj.biases,<br>model.layers.15.self_attn.v_proj.scales,<br>model.layers.15.self_attn.v_proj.weight,<br>model.layers.16.input_layernorm.weight,<br>model.layers.16.mlp.down_proj.biases,<br>model.layers.16.mlp.down_proj.scales,<br>model.layers.16.mlp.down_proj.weight,<br>model.layers.16.mlp.gate_proj.biases,<br>model.layers.16.mlp.gate_proj.scales,<br>model.layers.16.mlp.gate_proj.weight,<br>model.layers.16.mlp.up_proj.biases,<br>model.layers.16.mlp.up_proj.scales,<br>model.layers.16.mlp.up_proj.weight,<br>model.layers.16.post_attention_layernorm.weight,<br>model.layers.16.self_attn.k_norm.weight,<br>model.layers.16.self_attn.k_proj.biases,<br>model.layers.16.self_attn.k_proj.scales,<br>model.layers.16.self_attn.k_proj.weight,<br>model.layers.16.self_attn.o_proj.biases,<br>model.layers.16.self_attn.o_proj.scales,<br>model.layers.16.self_attn.o_proj.weight,<br>model.layers.16.self_attn.q_norm.weight,<br>model.layers.16.self_attn.q_proj.biases,<br>model.layers.16.self_attn.q_proj.scales,<br>model.layers.16.self_attn.q_proj.weight,<br>model.layers.16.self_attn.v_proj.biases,<br>model.layers.16.self_attn.v_proj.scales,<br>model.layers.16.self_attn.v_proj.weight,<br>model.layers.17.input_layernorm.weight,<br>model.layers.17.mlp.down_proj.biases,<br>model.layers.17.mlp.down_proj.scales,<br>model.layers.17.mlp.down_proj.weight,<br>model.layers.17.mlp.gate_proj.biases,<br>model.layers.17.mlp.gate_proj.scales,<br>model.layers.17.mlp.gate_proj.weight,<br>model.layers.17.mlp.up_proj.biases,<br>model.layers.17.mlp.up_proj.scales,<br>model.layers.17.mlp.up_proj.weight,<br>model.layers.17.post_attention_layernorm.weight,<br>model.layers.17.self_attn.k_norm.weight,<br>model.layers.17.self_attn.k_proj.biases,<br>model.layers.17.self_attn.k_proj.scales,<br>model.layers.17.self_attn.k_proj.weight,<br>model.layers.17.self_attn.o_proj.biases,<br>model.layers.17.self_attn.o_proj.scales,<br>model.layers.17.self_attn.o_proj.weight,<br>model.layers.17.self_attn.q_norm.weight,<br>model.layers.17.self_attn.q_proj.biases,<br>model.layers.17.self_attn.q_proj.scales,<br>model.layers.17.self_attn.q_proj.weight,<br>model.layers.17.self_attn.v_proj.biases,<br>model.layers.17.self_attn.v_proj.scales,<br>model.layers.17.self_attn.v_proj.weight,<br>model.layers.18.input_layernorm.weight,<br>model.layers.18.mlp.down_proj.biases,<br>model.layers.18.mlp.down_proj.scales,<br>model.layers.18.mlp.down_proj.weight,<br>model.layers.18.mlp.gate_proj.biases,<br>model.layers.18.mlp.gate_proj.scales,<br>model.layers.18.mlp.gate_proj.weight,<br>model.layers.18.mlp.up_proj.biases,<br>model.layers.18.mlp.up_proj.scales,<br>model.layers.18.mlp.up_proj.weight,<br>model.layers.18.post_attention_layernorm.weight,<br>model.layers.18.self_attn.k_norm.weight,<br>model.layers.18.self_attn.k_proj.biases,<br>model.layers.18.self_attn.k_proj.scales,<br>model.layers.18.self_attn.k_proj.weight,<br>model.layers.18.self_attn.o_proj.biases,<br>model.layers.18.self_attn.o_proj.scales,<br>model.layers.18.self_attn.o_proj.weight,<br>model.layers.18.self_attn.q_norm.weight,<br>model.layers.18.self_attn.q_proj.biases,<br>model.layers.18.self_attn.q_proj.scales,<br>model.layers.18.self_attn.q_proj.weight,<br>model.layers.18.self_attn.v_proj.biases,<br>model.layers.18.self_attn.v_proj.scales,<br>model.layers.18.self_attn.v_proj.weight,<br>model.layers.19.input_layernorm.weight,<br>model.layers.19.mlp.down_proj.biases,<br>model.layers.19.mlp.down_proj.scales,<br>model.layers.19.mlp.down_proj.weight,<br>model.layers.19.mlp.gate_proj.biases,<br>model.layers.19.mlp.gate_proj.scales,<br>model.layers.19.mlp.gate_proj.weight,<br>model.layers.19.mlp.up_proj.biases,<br>model.layers.19.mlp.up_proj.scales,<br>model.layers.19.mlp.up_proj.weight,<br>model.layers.19.post_attention_layernorm.weight,<br>model.layers.19.self_attn.k_norm.weight,<br>model.layers.19.self_attn.k_proj.biases,<br>model.layers.19.self_attn.k_proj.scales,<br>model.layers.19.self_attn.k_proj.weight,<br>model.layers.19.self_attn.o_proj.biases,<br>model.layers.19.self_attn.o_proj.scales,<br>model.layers.19.self_attn.o_proj.weight,<br>model.layers.19.self_attn.q_norm.weight,<br>model.layers.19.self_attn.q_proj.biases,<br>model.layers.19.self_attn.q_proj.scales,<br>model.layers.19.self_attn.q_proj.weight,<br>model.layers.19.self_attn.v_proj.biases,<br>model.layers.19.self_attn.v_proj.scales,<br>model.layers.19.self_attn.v_proj.weight,<br>model.layers.2.input_layernorm.weight,<br>model.layers.2.mlp.down_proj.biases,<br>model.layers.2.mlp.down_proj.scales,<br>model.layers.2.mlp.down_proj.weight,<br>model.layers.2.mlp.gate_proj.biases,<br>model.layers.2.mlp.gate_proj.scales,<br>model.layers.2.mlp.gate_proj.weight,<br>model.layers.2.mlp.up_proj.biases,<br>model.layers.2.mlp.up_proj.scales,<br>model.layers.2.mlp.up_proj.weight,<br>model.layers.2.post_attention_layernorm.weight,<br>model.layers.2.self_attn.k_norm.weight,<br>model.layers.2.self_attn.k_proj.biases,<br>model.layers.2.self_attn.k_proj.scales,<br>model.layers.2.self_attn.k_proj.weight,<br>model.layers.2.self_attn.o_proj.biases,<br>model.layers.2.self_attn.o_proj.scales,<br>model.layers.2.self_attn.o_proj.weight,<br>model.layers.2.self_attn.q_norm.weight,<br>model.layers.2.self_attn.q_proj.biases,<br>model.layers.2.self_attn.q_proj.scales,<br>model.layers.2.self_attn.q_proj.weight,<br>model.layers.2.self_attn.v_proj.biases,<br>model.layers.2.self_attn.v_proj.scales,<br>model.layers.2.self_attn.v_proj.weight,<br>model.layers.20.input_layernorm.weight,<br>model.layers.20.mlp.down_proj.biases,<br>model.layers.20.mlp.down_proj.scales,<br>model.layers.20.mlp.down_proj.weight,<br>model.layers.20.mlp.gate_proj.biases,<br>model.layers.20.mlp.gate_proj.scales,<br>model.layers.20.mlp.gate_proj.weight,<br>model.layers.20.mlp.up_proj.biases,<br>model.layers.20.mlp.up_proj.scales,<br>model.layers.20.mlp.up_proj.weight,<br>model.layers.20.post_attention_layernorm.weight,<br>model.layers.20.self_attn.k_norm.weight,<br>model.layers.20.self_attn.k_proj.biases,<br>model.layers.20.self_attn.k_proj.scales,<br>model.layers.20.self_attn.k_proj.weight,<br>model.layers.20.self_attn.o_proj.biases,<br>model.layers.20.self_attn.o_proj.scales,<br>model.layers.20.self_attn.o_proj.weight,<br>model.layers.20.self_attn.q_norm.weight,<br>model.layers.20.self_attn.q_proj.biases,<br>model.layers.20.self_attn.q_proj.scales,<br>model.layers.20.self_attn.q_proj.weight,<br>model.layers.20.self_attn.v_proj.biases,<br>model.layers.20.self_attn.v_proj.scales,<br>model.layers.20.self_attn.v_proj.weight,<br>model.layers.21.input_layernorm.weight,<br>model.layers.21.mlp.down_proj.biases,<br>model.layers.21.mlp.down_proj.scales,<br>model.layers.21.mlp.down_proj.weight,<br>model.layers.21.mlp.gate_proj.biases,<br>model.layers.21.mlp.gate_proj.scales,<br>model.layers.21.mlp.gate_proj.weight,<br>model.layers.21.mlp.up_proj.biases,<br>model.layers.21.mlp.up_proj.scales,<br>model.layers.21.mlp.up_proj.weight,<br>model.layers.21.post_attention_layernorm.weight,<br>model.layers.21.self_attn.k_norm.weight,<br>model.layers.21.self_attn.k_proj.biases,<br>model.layers.21.self_attn.k_proj.scales,<br>model.layers.21.self_attn.k_proj.weight,<br>model.layers.21.self_attn.o_proj.biases,<br>model.layers.21.self_attn.o_proj.scales,<br>model.layers.21.self_attn.o_proj.weight,<br>model.layers.21.self_attn.q_norm.weight,<br>model.layers.21.self_attn.q_proj.biases,<br>model.layers.21.self_attn.q_proj.scales,<br>model.layers.21.self_attn.q_proj.weight,<br>model.layers.21.self_attn.v_proj.biases,<br>model.layers.21.self_attn.v_proj.scales,<br>model.layers.21.self_attn.v_proj.weight,<br>model.layers.22.input_layernorm.weight,<br>model.layers.22.mlp.down_proj.biases,<br>model.layers.22.mlp.down_proj.scales,<br>model.layers.22.mlp.down_proj.weight,<br>model.layers.22.mlp.gate_proj.biases,<br>model.layers.22.mlp.gate_proj.scales,<br>model.layers.22.mlp.gate_proj.weight,<br>model.layers.22.mlp.up_proj.biases,<br>model.layers.22.mlp.up_proj.scales,<br>model.layers.22.mlp.up_proj.weight,<br>model.layers.22.post_attention_layernorm.weight,<br>model.layers.22.self_attn.k_norm.weight,<br>model.layers.22.self_attn.k_proj.biases,<br>model.layers.22.self_attn.k_proj.scales,<br>model.layers.22.self_attn.k_proj.weight,<br>model.layers.22.self_attn.o_proj.biases,<br>model.layers.22.self_attn.o_proj.scales,<br>model.layers.22.self_attn.o_proj.weight,<br>model.layers.22.self_attn.q_norm.weight,<br>model.layers.22.self_attn.q_proj.biases,<br>model.layers.22.self_attn.q_proj.scales,<br>model.layers.22.self_attn.q_proj.weight,<br>model.layers.22.self_attn.v_proj.biases,<br>model.layers.22.self_attn.v_proj.scales,<br>model.layers.22.self_attn.v_proj.weight,<br>model.layers.23.input_layernorm.weight,<br>model.layers.23.mlp.down_proj.biases,<br>model.layers.23.mlp.down_proj.scales,<br>model.layers.23.mlp.down_proj.weight,<br>model.layers.23.mlp.gate_proj.biases,<br>model.layers.23.mlp.gate_proj.scales,<br>model.layers.23.mlp.gate_proj.weight,<br>model.layers.23.mlp.up_proj.biases,<br>model.layers.23.mlp.up_proj.scales,<br>model.layers.23.mlp.up_proj.weight,<br>model.layers.23.post_attention_layernorm.weight,<br>model.layers.23.self_attn.k_norm.weight,<br>model.layers.23.self_attn.k_proj.biases,<br>model.layers.23.self_attn.k_proj.scales,<br>model.layers.23.self_attn.k_proj.weight,<br>model.layers.23.self_attn.o_proj.biases,<br>model.layers.23.self_attn.o_proj.scales,<br>model.layers.23.self_attn.o_proj.weight,<br>model.layers.23.self_attn.q_norm.weight,<br>model.layers.23.self_attn.q_proj.biases,<br>model.layers.23.self_attn.q_proj.scales,<br>model.layers.23.self_attn.q_proj.weight,<br>model.layers.23.self_attn.v_proj.biases,<br>model.layers.23.self_attn.v_proj.scales,<br>model.layers.23.self_attn.v_proj.weight,<br>model.layers.24.input_layernorm.weight,<br>model.layers.24.mlp.down_proj.biases,<br>model.layers.24.mlp.down_proj.scales,<br>model.layers.24.mlp.down_proj.weight,<br>model.layers.24.mlp.gate_proj.biases,<br>model.layers.24.mlp.gate_proj.scales,<br>model.layers.24.mlp.gate_proj.weight,<br>model.layers.24.mlp.up_proj.biases,<br>model.layers.24.mlp.up_proj.scales,<br>model.layers.24.mlp.up_proj.weight,<br>model.layers.24.post_attention_layernorm.weight,<br>model.layers.24.self_attn.k_norm.weight,<br>model.layers.24.self_attn.k_proj.biases,<br>model.layers.24.self_attn.k_proj.scales,<br>model.layers.24.self_attn.k_proj.weight,<br>model.layers.24.self_attn.o_proj.biases,<br>model.layers.24.self_attn.o_proj.scales,<br>model.layers.24.self_attn.o_proj.weight,<br>model.layers.24.self_attn.q_norm.weight,<br>model.layers.24.self_attn.q_proj.biases,<br>model.layers.24.self_attn.q_proj.scales,<br>model.layers.24.self_attn.q_proj.weight,<br>model.layers.24.self_attn.v_proj.biases,<br>model.layers.24.self_attn.v_proj.scales,<br>model.layers.24.self_attn.v_proj.weight,<br>model.layers.25.input_layernorm.weight,<br>model.layers.25.mlp.down_proj.biases,<br>model.layers.25.mlp.down_proj.scales,<br>model.layers.25.mlp.down_proj.weight,<br>model.layers.25.mlp.gate_proj.biases,<br>model.layers.25.mlp.gate_proj.scales,<br>model.layers.25.mlp.gate_proj.weight,<br>model.layers.25.mlp.up_proj.biases,<br>model.layers.25.mlp.up_proj.scales,<br>model.layers.25.mlp.up_proj.weight,<br>model.layers.25.post_attention_layernorm.weight,<br>model.layers.25.self_attn.k_norm.weight,<br>model.layers.25.self_attn.k_proj.biases,<br>model.layers.25.self_attn.k_proj.scales,<br>model.layers.25.self_attn.k_proj.weight,<br>model.layers.25.self_attn.o_proj.biases,<br>model.layers.25.self_attn.o_proj.scales,<br>model.layers.25.self_attn.o_proj.weight,<br>model.layers.25.self_attn.q_norm.weight,<br>model.layers.25.self_attn.q_proj.biases,<br>model.layers.25.self_attn.q_proj.scales,<br>model.layers.25.self_attn.q_proj.weight,<br>model.layers.25.self_attn.v_proj.biases,<br>model.layers.25.self_attn.v_proj.scales,<br>model.layers.25.self_attn.v_proj.weight,<br>model.layers.26.input_layernorm.weight,<br>model.layers.26.mlp.down_proj.biases,<br>model.layers.26.mlp.down_proj.scales,<br>model.layers.26.mlp.down_proj.weight,<br>model.layers.26.mlp.gate_proj.biases,<br>model.layers.26.mlp.gate_proj.scales,<br>model.layers.26.mlp.gate_proj.weight,<br>model.layers.26.mlp.up_proj.biases,<br>model.layers.26.mlp.up_proj.scales,<br>model.layers.26.mlp.up_proj.weight,<br>model.layers.26.post_attention_layernorm.weight,<br>model.layers.26.self_attn.k_norm.weight,<br>model.layers.26.self_attn.k_proj.biases,<br>model.layers.26.self_attn.k_proj.scales,<br>model.layers.26.self_attn.k_proj.weight,<br>model.layers.26.self_attn.o_proj.biases,<br>model.layers.26.self_attn.o_proj.scales,<br>model.layers.26.self_attn.o_proj.weight,<br>model.layers.26.self_attn.q_norm.weight,<br>model.layers.26.self_attn.q_proj.biases,<br>model.layers.26.self_attn.q_proj.scales,<br>model.layers.26.self_attn.q_proj.weight,<br>model.layers.26.self_attn.v_proj.biases,<br>model.layers.26.self_attn.v_proj.scales,<br>model.layers.26.self_attn.v_proj.weight,<br>model.layers.27.input_layernorm.weight,<br>model.layers.27.mlp.down_proj.biases,<br>model.layers.27.mlp.down_proj.scales,<br>model.layers.27.mlp.down_proj.weight,<br>model.layers.27.mlp.gate_proj.biases,<br>model.layers.27.mlp.gate_proj.scales,<br>model.layers.27.mlp.gate_proj.weight,<br>model.layers.27.mlp.up_proj.biases,<br>model.layers.27.mlp.up_proj.scales,<br>model.layers.27.mlp.up_proj.weight,<br>model.layers.27.post_attention_layernorm.weight,<br>model.layers.27.self_attn.k_norm.weight,<br>model.layers.27.self_attn.k_proj.biases,<br>model.layers.27.self_attn.k_proj.scales,<br>model.layers.27.self_attn.k_proj.weight,<br>model.layers.27.self_attn.o_proj.biases,<br>model.layers.27.self_attn.o_proj.scales,<br>model.layers.27.self_attn.o_proj.weight,<br>model.layers.27.self_attn.q_norm.weight,<br>model.layers.27.self_attn.q_proj.biases,<br>model.layers.27.self_attn.q_proj.scales,<br>model.layers.27.self_attn.q_proj.weight,<br>model.layers.27.self_attn.v_proj.biases,<br>model.layers.27.self_attn.v_proj.scales,<br>model.layers.27.self_attn.v_proj.weight,<br>model.layers.28.input_layernorm.weight,<br>model.layers.28.mlp.down_proj.biases,<br>model.layers.28.mlp.down_proj.scales,<br>model.layers.28.mlp.down_proj.weight,<br>model.layers.28.mlp.gate_proj.biases,<br>model.layers.28.mlp.gate_proj.scales,<br>model.layers.28.mlp.gate_proj.weight,<br>model.layers.28.mlp.up_proj.biases,<br>model.layers.28.mlp.up_proj.scales,<br>model.layers.28.mlp.up_proj.weight,<br>model.layers.28.post_attention_layernorm.weight,<br>model.layers.28.self_attn.k_norm.weight,<br>model.layers.28.self_attn.k_proj.biases,<br>model.layers.28.self_attn.k_proj.scales,<br>model.layers.28.self_attn.k_proj.weight,<br>model.layers.28.self_attn.o_proj.biases,<br>model.layers.28.self_attn.o_proj.scales,<br>model.layers.28.self_attn.o_proj.weight,<br>model.layers.28.self_attn.q_norm.weight,<br>model.layers.28.self_attn.q_proj.biases,<br>model.layers.28.self_attn.q_proj.scales,<br>model.layers.28.self_attn.q_proj.weight,<br>model.layers.28.self_attn.v_proj.biases,<br>model.layers.28.self_attn.v_proj.scales,<br>model.layers.28.self_attn.v_proj.weight,<br>model.layers.29.input_layernorm.weight,<br>model.layers.29.mlp.down_proj.biases,<br>model.layers.29.mlp.down_proj.scales,<br>model.layers.29.mlp.down_proj.weight,<br>model.layers.29.mlp.gate_proj.biases,<br>model.layers.29.mlp.gate_proj.scales,<br>model.layers.29.mlp.gate_proj.weight,<br>model.layers.29.mlp.up_proj.biases,<br>model.layers.29.mlp.up_proj.scales,<br>model.layers.29.mlp.up_proj.weight,<br>model.layers.29.post_attention_layernorm.weight,<br>model.layers.29.self_attn.k_norm.weight,<br>model.layers.29.self_attn.k_proj.biases,<br>model.layers.29.self_attn.k_proj.scales,<br>model.layers.29.self_attn.k_proj.weight,<br>model.layers.29.self_attn.o_proj.biases,<br>model.layers.29.self_attn.o_proj.scales,<br>model.layers.29.self_attn.o_proj.weight,<br>model.layers.29.self_attn.q_norm.weight,<br>model.layers.29.self_attn.q_proj.biases,<br>model.layers.29.self_attn.q_proj.scales,<br>model.layers.29.self_attn.q_proj.weight,<br>model.layers.29.self_attn.v_proj.biases,<br>model.layers.29.self_attn.v_proj.scales,<br>model.layers.29.self_attn.v_proj.weight,<br>model.layers.3.input_layernorm.weight,<br>model.layers.3.mlp.down_proj.biases,<br>model.layers.3.mlp.down_proj.scales,<br>model.layers.3.mlp.down_proj.weight,<br>model.layers.3.mlp.gate_proj.biases,<br>model.layers.3.mlp.gate_proj.scales,<br>model.layers.3.mlp.gate_proj.weight,<br>model.layers.3.mlp.up_proj.biases,<br>model.layers.3.mlp.up_proj.scales,<br>model.layers.3.mlp.up_proj.weight,<br>model.layers.3.post_attention_layernorm.weight,<br>model.layers.3.self_attn.k_norm.weight,<br>model.layers.3.self_attn.k_proj.biases,<br>model.layers.3.self_attn.k_proj.scales,<br>model.layers.3.self_attn.k_proj.weight,<br>model.layers.3.self_attn.o_proj.biases,<br>model.layers.3.self_attn.o_proj.scales,<br>model.layers.3.self_attn.o_proj.weight,<br>model.layers.3.self_attn.q_norm.weight,<br>model.layers.3.self_attn.q_proj.biases,<br>model.layers.3.self_attn.q_proj.scales,<br>model.layers.3.self_attn.q_proj.weight,<br>model.layers.3.self_attn.v_proj.biases,<br>model.layers.3.self_attn.v_proj.scales,<br>model.layers.3.self_attn.v_proj.weight,<br>model.layers.30.input_layernorm.weight,<br>model.layers.30.mlp.down_proj.biases,<br>model.layers.30.mlp.down_proj.scales,<br>model.layers.30.mlp.down_proj.weight,<br>model.layers.30.mlp.gate_proj.biases,<br>model.layers.30.mlp.gate_proj.scales,<br>model.layers.30.mlp.gate_proj.weight,<br>model.layers.30.mlp.up_proj.biases,<br>model.layers.30.mlp.up_proj.scales,<br>model.layers.30.mlp.up_proj.weight,<br>model.layers.30.post_attention_layernorm.weight,<br>model.layers.30.self_attn.k_norm.weight,<br>model.layers.30.self_attn.k_proj.biases,<br>model.layers.30.self_attn.k_proj.scales,<br>model.layers.30.self_attn.k_proj.weight,<br>model.layers.30.self_attn.o_proj.biases,<br>model.layers.30.self_attn.o_proj.scales,<br>model.layers.30.self_attn.o_proj.weight,<br>model.layers.30.self_attn.q_norm.weight,<br>model.layers.30.self_attn.q_proj.biases,<br>model.layers.30.self_attn.q_proj.scales,<br>model.layers.30.self_attn.q_proj.weight,<br>model.layers.30.self_attn.v_proj.biases,<br>model.layers.30.self_attn.v_proj.scales,<br>model.layers.30.self_attn.v_proj.weight,<br>model.layers.31.input_layernorm.weight,<br>model.layers.31.mlp.down_proj.biases,<br>model.layers.31.mlp.down_proj.scales,<br>model.layers.31.mlp.down_proj.weight,<br>model.layers.31.mlp.gate_proj.biases,<br>model.layers.31.mlp.gate_proj.scales,<br>model.layers.31.mlp.gate_proj.weight,<br>model.layers.31.mlp.up_proj.biases,<br>model.layers.31.mlp.up_proj.scales,<br>model.layers.31.mlp.up_proj.weight,<br>model.layers.31.post_attention_layernorm.weight,<br>model.layers.31.self_attn.k_norm.weight,<br>model.layers.31.self_attn.k_proj.biases,<br>model.layers.31.self_attn.k_proj.scales,<br>model.layers.31.self_attn.k_proj.weight,<br>model.layers.31.self_attn.o_proj.biases,<br>model.layers.31.self_attn.o_proj.scales,<br>model.layers.31.self_attn.o_proj.weight,<br>model.layers.31.self_attn.q_norm.weight,<br>model.layers.31.self_attn.q_proj.biases,<br>model.layers.31.self_attn.q_proj.scales,<br>model.layers.31.self_attn.q_proj.weight,<br>model.layers.31.self_attn.v_proj.biases,<br>model.layers.31.self_attn.v_proj.scales,<br>model.layers.31.self_attn.v_proj.weight,<br>model.layers.32.input_layernorm.weight,<br>model.layers.32.mlp.down_proj.biases,<br>model.layers.32.mlp.down_proj.scales,<br>model.layers.32.mlp.down_proj.weight,<br>model.layers.32.mlp.gate_proj.biases,<br>model.layers.32.mlp.gate_proj.scales,<br>model.layers.32.mlp.gate_proj.weight,<br>model.layers.32.mlp.up_proj.biases,<br>model.layers.32.mlp.up_proj.scales,<br>model.layers.32.mlp.up_proj.weight,<br>model.layers.32.post_attention_layernorm.weight,<br>model.layers.32.self_attn.k_norm.weight,<br>model.layers.32.self_attn.k_proj.biases,<br>model.layers.32.self_attn.k_proj.scales,<br>model.layers.32.self_attn.k_proj.weight,<br>model.layers.32.self_attn.o_proj.biases,<br>model.layers.32.self_attn.o_proj.scales,<br>model.layers.32.self_attn.o_proj.weight,<br>model.layers.32.self_attn.q_norm.weight,<br>model.layers.32.self_attn.q_proj.biases,<br>model.layers.32.self_attn.q_proj.scales,<br>model.layers.32.self_attn.q_proj.weight,<br>model.layers.32.self_attn.v_proj.biases,<br>model.layers.32.self_attn.v_proj.scales,<br>model.layers.32.self_attn.v_proj.weight,<br>model.layers.33.input_layernorm.weight,<br>model.layers.33.mlp.down_proj.biases,<br>model.layers.33.mlp.down_proj.scales,<br>model.layers.33.mlp.down_proj.weight,<br>model.layers.33.mlp.gate_proj.biases,<br>model.layers.33.mlp.gate_proj.scales,<br>model.layers.33.mlp.gate_proj.weight,<br>model.layers.33.mlp.up_proj.biases,<br>model.layers.33.mlp.up_proj.scales,<br>model.layers.33.mlp.up_proj.weight,<br>model.layers.33.post_attention_layernorm.weight,<br>model.layers.33.self_attn.k_norm.weight,<br>model.layers.33.self_attn.k_proj.biases,<br>model.layers.33.self_attn.k_proj.scales,<br>model.layers.33.self_attn.k_proj.weight,<br>model.layers.33.self_attn.o_proj.biases,<br>model.layers.33.self_attn.o_proj.scales,<br>model.layers.33.self_attn.o_proj.weight,<br>model.layers.33.self_attn.q_norm.weight,<br>model.layers.33.self_attn.q_proj.biases,<br>model.layers.33.self_attn.q_proj.scales,<br>model.layers.33.self_attn.q_proj.weight,<br>model.layers.33.self_attn.v_proj.biases,<br>model.layers.33.self_attn.v_proj.scales,<br>model.layers.33.self_attn.v_proj.weight,<br>model.layers.34.input_layernorm.weight,<br>model.layers.34.mlp.down_proj.biases,<br>model.layers.34.mlp.down_proj.scales,<br>model.layers.34.mlp.down_proj.weight,<br>model.layers.34.mlp.gate_proj.biases,<br>model.layers.34.mlp.gate_proj.scales,<br>model.layers.34.mlp.gate_proj.weight,<br>model.layers.34.mlp.up_proj.biases,<br>model.layers.34.mlp.up_proj.scales,<br>model.layers.34.mlp.up_proj.weight,<br>model.layers.34.post_attention_layernorm.weight,<br>model.layers.34.self_attn.k_norm.weight,<br>model.layers.34.self_attn.k_proj.biases,<br>model.layers.34.self_attn.k_proj.scales,<br>model.layers.34.self_attn.k_proj.weight,<br>model.layers.34.self_attn.o_proj.biases,<br>model.layers.34.self_attn.o_proj.scales,<br>model.layers.34.self_attn.o_proj.weight,<br>model.layers.34.self_attn.q_norm.weight,<br>model.layers.34.self_attn.q_proj.biases,<br>model.layers.34.self_attn.q_proj.scales,<br>model.layers.34.self_attn.q_proj.weight,<br>model.layers.34.self_attn.v_proj.biases,<br>model.layers.34.self_attn.v_proj.scales,<br>model.layers.34.self_attn.v_proj.weight,<br>model.layers.35.input_layernorm.weight,<br>model.layers.35.mlp.down_proj.biases,<br>model.layers.35.mlp.down_proj.scales,<br>model.layers.35.mlp.down_proj.weight,<br>model.layers.35.mlp.gate_proj.biases,<br>model.layers.35.mlp.gate_proj.scales,<br>model.layers.35.mlp.gate_proj.weight,<br>model.layers.35.mlp.up_proj.biases,<br>model.layers.35.mlp.up_proj.scales,<br>model.layers.35.mlp.up_proj.weight,<br>model.layers.35.post_attention_layernorm.weight,<br>model.layers.35.self_attn.k_norm.weight,<br>model.layers.35.self_attn.k_proj.biases,<br>model.layers.35.self_attn.k_proj.scales,<br>model.layers.35.self_attn.k_proj.weight,<br>model.layers.35.self_attn.o_proj.biases,<br>model.layers.35.self_attn.o_proj.scales,<br>model.layers.35.self_attn.o_proj.weight,<br>model.layers.35.self_attn.q_norm.weight,<br>model.layers.35.self_attn.q_proj.biases,<br>model.layers.35.self_attn.q_proj.scales,<br>model.layers.35.self_attn.q_proj.weight,<br>model.layers.35.self_attn.v_proj.biases,<br>model.layers.35.self_attn.v_proj.scales,<br>model.layers.35.self_attn.v_proj.weight,<br>model.layers.4.input_layernorm.weight,<br>model.layers.4.mlp.down_proj.biases,<br>model.layers.4.mlp.down_proj.scales,<br>model.layers.4.mlp.down_proj.weight,<br>model.layers.4.mlp.gate_proj.biases,<br>model.layers.4.mlp.gate_proj.scales,<br>model.layers.4.mlp.gate_proj.weight,<br>model.layers.4.mlp.up_proj.biases,<br>model.layers.4.mlp.up_proj.scales,<br>model.layers.4.mlp.up_proj.weight,<br>model.layers.4.post_attention_layernorm.weight,<br>model.layers.4.self_attn.k_norm.weight,<br>model.layers.4.self_attn.k_proj.biases,<br>model.layers.4.self_attn.k_proj.scales,<br>model.layers.4.self_attn.k_proj.weight,<br>model.layers.4.self_attn.o_proj.biases,<br>model.layers.4.self_attn.o_proj.scales,<br>model.layers.4.self_attn.o_proj.weight,<br>model.layers.4.self_attn.q_norm.weight,<br>model.layers.4.self_attn.q_proj.biases,<br>model.layers.4.self_attn.q_proj.scales,<br>model.layers.4.self_attn.q_proj.weight,<br>model.layers.4.self_attn.v_proj.biases,<br>model.layers.4.self_attn.v_proj.scales,<br>model.layers.4.self_attn.v_proj.weight,<br>model.layers.5.input_layernorm.weight,<br>model.layers.5.mlp.down_proj.biases,<br>model.layers.5.mlp.down_proj.scales,<br>model.layers.5.mlp.down_proj.weight,<br>model.layers.5.mlp.gate_proj.biases,<br>model.layers.5.mlp.gate_proj.scales,<br>model.layers.5.mlp.gate_proj.weight,<br>model.layers.5.mlp.up_proj.biases,<br>model.layers.5.mlp.up_proj.scales,<br>model.layers.5.mlp.up_proj.weight,<br>model.layers.5.post_attention_layernorm.weight,<br>model.layers.5.self_attn.k_norm.weight,<br>model.layers.5.self_attn.k_proj.biases,<br>model.layers.5.self_attn.k_proj.scales,<br>model.layers.5.self_attn.k_proj.weight,<br>model.layers.5.self_attn.o_proj.biases,<br>model.layers.5.self_attn.o_proj.scales,<br>model.layers.5.self_attn.o_proj.weight,<br>model.layers.5.self_attn.q_norm.weight,<br>model.layers.5.self_attn.q_proj.biases,<br>model.layers.5.self_attn.q_proj.scales,<br>model.layers.5.self_attn.q_proj.weight,<br>model.layers.5.self_attn.v_proj.biases,<br>model.layers.5.self_attn.v_proj.scales,<br>model.layers.5.self_attn.v_proj.weight,<br>model.layers.6.input_layernorm.weight,<br>model.layers.6.mlp.down_proj.biases,<br>model.layers.6.mlp.down_proj.scales,<br>model.layers.6.mlp.down_proj.weight,<br>model.layers.6.mlp.gate_proj.biases,<br>model.layers.6.mlp.gate_proj.scales,<br>model.layers.6.mlp.gate_proj.weight,<br>model.layers.6.mlp.up_proj.biases,<br>model.layers.6.mlp.up_proj.scales,<br>model.layers.6.mlp.up_proj.weight,<br>model.layers.6.post_attention_layernorm.weight,<br>model.layers.6.self_attn.k_norm.weight,<br>model.layers.6.self_attn.k_proj.biases,<br>model.layers.6.self_attn.k_proj.scales,<br>model.layers.6.self_attn.k_proj.weight,<br>model.layers.6.self_attn.o_proj.biases,<br>model.layers.6.self_attn.o_proj.scales,<br>model.layers.6.self_attn.o_proj.weight,<br>model.layers.6.self_attn.q_norm.weight,<br>model.layers.6.self_attn.q_proj.biases,<br>model.layers.6.self_attn.q_proj.scales,<br>model.layers.6.self_attn.q_proj.weight,<br>model.layers.6.self_attn.v_proj.biases,<br>model.layers.6.self_attn.v_proj.scales,<br>model.layers.6.self_attn.v_proj.weight,<br>model.layers.7.input_layernorm.weight,<br>model.layers.7.mlp.down_proj.biases,<br>model.layers.7.mlp.down_proj.scales,<br>model.layers.7.mlp.down_proj.weight,<br>model.layers.7.mlp.gate_proj.biases,<br>model.layers.7.mlp.gate_proj.scales,<br>model.layers.7.mlp.gate_proj.weight,<br>model.layers.7.mlp.up_proj.biases,<br>model.layers.7.mlp.up_proj.scales,<br>model.layers.7.mlp.up_proj.weight,<br>model.layers.7.post_attention_layernorm.weight,<br>model.layers.7.self_attn.k_norm.weight,<br>model.layers.7.self_attn.k_proj.biases,<br>model.layers.7.self_attn.k_proj.scales,<br>model.layers.7.self_attn.k_proj.weight,<br>model.layers.7.self_attn.o_proj.biases,<br>model.layers.7.self_attn.o_proj.scales,<br>model.layers.7.self_attn.o_proj.weight,<br>model.layers.7.self_attn.q_norm.weight,<br>model.layers.7.self_attn.q_proj.biases,<br>model.layers.7.self_attn.q_proj.scales,<br>model.layers.7.self_attn.q_proj.weight,<br>model.layers.7.self_attn.v_proj.biases,<br>model.layers.7.self_attn.v_proj.scales,<br>model.layers.7.self_attn.v_proj.weight,<br>model.layers.8.input_layernorm.weight,<br>model.layers.8.mlp.down_proj.biases,<br>model.layers.8.mlp.down_proj.scales,<br>model.layers.8.mlp.down_proj.weight,<br>model.layers.8.mlp.gate_proj.biases,<br>model.layers.8.mlp.gate_proj.scales,<br>model.layers.8.mlp.gate_proj.weight,<br>model.layers.8.mlp.up_proj.biases,<br>model.layers.8.mlp.up_proj.scales,<br>model.layers.8.mlp.up_proj.weight,<br>model.layers.8.post_attention_layernorm.weight,<br>model.layers.8.self_attn.k_norm.weight,<br>model.layers.8.self_attn.k_proj.biases,<br>model.layers.8.self_attn.k_proj.scales,<br>model.layers.8.self_attn.k_proj.weight,<br>model.layers.8.self_attn.o_proj.biases,<br>model.layers.8.self_attn.o_proj.scales,<br>model.layers.8.self_attn.o_proj.weight,<br>model.layers.8.self_attn.q_norm.weight,<br>model.layers.8.self_attn.q_proj.biases,<br>model.layers.8.self_attn.q_proj.scales,<br>model.layers.8.self_attn.q_proj.weight,<br>model.layers.8.self_attn.v_proj.biases,<br>model.layers.8.self_attn.v_proj.scales,<br>model.layers.8.self_attn.v_proj.weight,<br>model.layers.9.input_layernorm.weight,<br>model.layers.9.mlp.down_proj.biases,<br>model.layers.9.mlp.down_proj.scales,<br>model.layers.9.mlp.down_proj.weight,<br>model.layers.9.mlp.gate_proj.biases,<br>model.layers.9.mlp.gate_proj.scales,<br>model.layers.9.mlp.gate_proj.weight,<br>model.layers.9.mlp.up_proj.biases,<br>model.layers.9.mlp.up_proj.scales,<br>model.layers.9.mlp.up_proj.weight,<br>model.layers.9.post_attention_layernorm.weight,<br>model.layers.9.self_attn.k_norm.weight,<br>model.layers.9.self_attn.k_proj.biases,<br>model.layers.9.self_attn.k_proj.scales,<br>model.layers.9.self_attn.k_proj.weight,<br>model.layers.9.self_attn.o_proj.biases,<br>model.layers.9.self_attn.o_proj.scales,<br>model.layers.9.self_attn.o_proj.weight,<br>model.layers.9.self_attn.q_norm.weight,<br>model.layers.9.self_attn.q_proj.biases,<br>model.layers.9.self_attn.q_proj.scales,<br>model.layers.9.self_attn.q_proj.weight,<br>model.layers.9.self_attn.v_proj.biases,<br>model.layers.9.self_attn.v_proj.scales,<br>model.layers.9.self_attn.v_proj.weight,<br>model.norm.weight.
- *Arch supported by installed mlx-vlm:* yes (model_type mage_vl)
- *Model load time:* 0.19s
- *Generation time:* -
- *Total time:* 0.19s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0772
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00274
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* bde6c9c7146acff6af09e203245014f19306c5c5
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Mage-VL-OptiQ-4bit/snapshots/bde6c9c7146acff6af09e203245014f19306c5c5
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14168, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13080, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 823, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1306, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1171, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 904 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.layers.0.input_layernorm.weight,
model.layers.0.mlp.down_proj.biases,
model.layers.0.mlp.down_proj.scales,
model.layers.0.mlp.down_proj.weight,
model.layers.0.mlp.gate_proj.biases,
model.layers.0.mlp.gate_proj.scales,
model.layers.0.mlp.gate_proj.weight,
model.layers.0.mlp.up_proj.biases,
model.layers.0.mlp.up_proj.scales,
model.layers.0.mlp.up_proj.weight,
model.layers.0.post_attention_layernorm.weight,
model.layers.0.self_attn.k_norm.weight,
model.layers.0.self_attn.k_proj.biases,
model.layers.0.self_attn.k_proj.scales,
model.layers.0.self_attn.k_proj.weight,
model.layers.0.self_attn.o_proj.biases,
model.layers.0.self_attn.o_proj.scales,
model.layers.0.self_attn.o_proj.weight,
model.layers.0.self_attn.q_norm.weight,
model.layers.0.self_attn.q_proj.biases,
model.layers.0.self_attn.q_proj.scales,
model.layers.0.self_attn.q_proj.weight,
model.layers.0.self_attn.v_proj.biases,
model.layers.0.self_attn.v_proj.scales,
model.layers.0.self_attn.v_proj.weight,
model.layers.1.input_layernorm.weight,
model.layers.1.mlp.down_proj.biases,
model.layers.1.mlp.down_proj.scales,
model.layers.1.mlp.down_proj.weight,
model.layers.1.mlp.gate_proj.biases,
model.layers.1.mlp.gate_proj.scales,
model.layers.1.mlp.gate_proj.weight,
model.layers.1.mlp.up_proj.biases,
model.layers.1.mlp.up_proj.scales,
model.layers.1.mlp.up_proj.weight,
model.layers.1.post_attention_layernorm.weight,
model.layers.1.self_attn.k_norm.weight,
model.layers.1.self_attn.k_proj.biases,
model.layers.1.self_attn.k_proj.scales,
model.layers.1.self_attn.k_proj.weight,
model.layers.1.self_attn.o_proj.biases,
model.layers.1.self_attn.o_proj.scales,
model.layers.1.self_attn.o_proj.weight,
model.layers.1.self_attn.q_norm.weight,
model.layers.1.self_attn.q_proj.biases,
model.layers.1.self_attn.q_proj.scales,
model.layers.1.self_attn.q_proj.weight,
model.layers.1.self_attn.v_proj.biases,
model.layers.1.self_attn.v_proj.scales,
model.layers.1.self_attn.v_proj.weight,
model.layers.10.input_layernorm.weight,
model.layers.10.mlp.down_proj.biases,
model.layers.10.mlp.down_proj.scales,
model.layers.10.mlp.down_proj.weight,
model.layers.10.mlp.gate_proj.biases,
model.layers.10.mlp.gate_proj.scales,
model.layers.10.mlp.gate_proj.weight,
model.layers.10.mlp.up_proj.biases,
model.layers.10.mlp.up_proj.scales,
model.layers.10.mlp.up_proj.weight,
model.layers.10.post_attention_layernorm.weight,
model.layers.10.self_attn.k_norm.weight,
model.layers.10.self_attn.k_proj.biases,
model.layers.10.self_attn.k_proj.scales,
model.layers.10.self_attn.k_proj.weight,
model.layers.10.self_attn.o_proj.biases,
model.layers.10.self_attn.o_proj.scales,
model.layers.10.self_attn.o_proj.weight,
model.layers.10.self_attn.q_norm.weight,
model.layers.10.self_attn.q_proj.biases,
model.layers.10.self_attn.q_proj.scales,
model.layers.10.self_attn.q_proj.weight,
model.layers.10.self_attn.v_proj.biases,
model.layers.10.self_attn.v_proj.scales,
model.layers.10.self_attn.v_proj.weight,
model.layers.11.input_layernorm.weight,
model.layers.11.mlp.down_proj.biases,
model.layers.11.mlp.down_proj.scales,
model.layers.11.mlp.down_proj.weight,
model.layers.11.mlp.gate_proj.biases,
model.layers.11.mlp.gate_proj.scales,
model.layers.11.mlp.gate_proj.weight,
model.layers.11.mlp.up_proj.biases,
model.layers.11.mlp.up_proj.scales,
model.layers.11.mlp.up_proj.weight,
model.layers.11.post_attention_layernorm.weight,
model.layers.11.self_attn.k_norm.weight,
model.layers.11.self_attn.k_proj.biases,
model.layers.11.self_attn.k_proj.scales,
model.layers.11.self_attn.k_proj.weight,
model.layers.11.self_attn.o_proj.biases,
model.layers.11.self_attn.o_proj.scales,
model.layers.11.self_attn.o_proj.weight,
model.layers.11.self_attn.q_norm.weight,
model.layers.11.self_attn.q_proj.biases,
model.layers.11.self_attn.q_proj.scales,
model.layers.11.self_attn.q_proj.weight,
model.layers.11.self_attn.v_proj.biases,
model.layers.11.self_attn.v_proj.scales,
model.layers.11.self_attn.v_proj.weight,
model.layers.12.input_layernorm.weight,
model.layers.12.mlp.down_proj.biases,
model.layers.12.mlp.down_proj.scales,
model.layers.12.mlp.down_proj.weight,
model.layers.12.mlp.gate_proj.biases,
model.layers.12.mlp.gate_proj.scales,
model.layers.12.mlp.gate_proj.weight,
model.layers.12.mlp.up_proj.biases,
model.layers.12.mlp.up_proj.scales,
model.layers.12.mlp.up_proj.weight,
model.layers.12.post_attention_layernorm.weight,
model.layers.12.self_attn.k_norm.weight,
model.layers.12.self_attn.k_proj.biases,
model.layers.12.self_attn.k_proj.scales,
model.layers.12.self_attn.k_proj.weight,
model.layers.12.self_attn.o_proj.biases,
model.layers.12.self_attn.o_proj.scales,
model.layers.12.self_attn.o_proj.weight,
model.layers.12.self_attn.q_norm.weight,
model.layers.12.self_attn.q_proj.biases,
model.layers.12.self_attn.q_proj.scales,
model.layers.12.self_attn.q_proj.weight,
model.layers.12.self_attn.v_proj.biases,
model.layers.12.self_attn.v_proj.scales,
model.layers.12.self_attn.v_proj.weight,
model.layers.13.input_layernorm.weight,
model.layers.13.mlp.down_proj.biases,
model.layers.13.mlp.down_proj.scales,
model.layers.13.mlp.down_proj.weight,
model.layers.13.mlp.gate_proj.biases,
model.layers.13.mlp.gate_proj.scales,
model.layers.13.mlp.gate_proj.weight,
model.layers.13.mlp.up_proj.biases,
model.layers.13.mlp.up_proj.scales,
model.layers.13.mlp.up_proj.weight,
model.layers.13.post_attention_layernorm.weight,
model.layers.13.self_attn.k_norm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.o_proj.biases,
model.layers.13.self_attn.o_proj.scales,
model.layers.13.self_attn.o_proj.weight,
model.layers.13.self_attn.q_norm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.input_layernorm.weight,
model.layers.14.mlp.down_proj.biases,
model.layers.14.mlp.down_proj.scales,
model.layers.14.mlp.down_proj.weight,
model.layers.14.mlp.gate_proj.biases,
model.layers.14.mlp.gate_proj.scales,
model.layers.14.mlp.gate_proj.weight,
model.layers.14.mlp.up_proj.biases,
model.layers.14.mlp.up_proj.scales,
model.layers.14.mlp.up_proj.weight,
model.layers.14.post_attention_layernorm.weight,
model.layers.14.self_attn.k_norm.weight,
model.layers.14.self_attn.k_proj.biases,
model.layers.14.self_attn.k_proj.scales,
model.layers.14.self_attn.k_proj.weight,
model.layers.14.self_attn.o_proj.biases,
model.layers.14.self_attn.o_proj.scales,
model.layers.14.self_attn.o_proj.weight,
model.layers.14.self_attn.q_norm.weight,
model.layers.14.self_attn.q_proj.biases,
model.layers.14.self_attn.q_proj.scales,
model.layers.14.self_attn.q_proj.weight,
model.layers.14.self_attn.v_proj.biases,
model.layers.14.self_attn.v_proj.scales,
model.layers.14.self_attn.v_proj.weight,
model.layers.15.input_layernorm.weight,
model.layers.15.mlp.down_proj.biases,
model.layers.15.mlp.down_proj.scales,
model.layers.15.mlp.down_proj.weight,
model.layers.15.mlp.gate_proj.biases,
model.layers.15.mlp.gate_proj.scales,
model.layers.15.mlp.gate_proj.weight,
model.layers.15.mlp.up_proj.biases,
model.layers.15.mlp.up_proj.scales,
model.layers.15.mlp.up_proj.weight,
model.layers.15.post_attention_layernorm.weight,
model.layers.15.self_attn.k_norm.weight,
model.layers.15.self_attn.k_proj.biases,
model.layers.15.self_attn.k_proj.scales,
model.layers.15.self_attn.k_proj.weight,
model.layers.15.self_attn.o_proj.biases,
model.layers.15.self_attn.o_proj.scales,
model.layers.15.self_attn.o_proj.weight,
model.layers.15.self_attn.q_norm.weight,
model.layers.15.self_attn.q_proj.biases,
model.layers.15.self_attn.q_proj.scales,
model.layers.15.self_attn.q_proj.weight,
model.layers.15.self_attn.v_proj.biases,
model.layers.15.self_attn.v_proj.scales,
model.layers.15.self_attn.v_proj.weight,
model.layers.16.input_layernorm.weight,
model.layers.16.mlp.down_proj.biases,
model.layers.16.mlp.down_proj.scales,
model.layers.16.mlp.down_proj.weight,
model.layers.16.mlp.gate_proj.biases,
model.layers.16.mlp.gate_proj.scales,
model.layers.16.mlp.gate_proj.weight,
model.layers.16.mlp.up_proj.biases,
model.layers.16.mlp.up_proj.scales,
model.layers.16.mlp.up_proj.weight,
model.layers.16.post_attention_layernorm.weight,
model.layers.16.self_attn.k_norm.weight,
model.layers.16.self_attn.k_proj.biases,
model.layers.16.self_attn.k_proj.scales,
model.layers.16.self_attn.k_proj.weight,
model.layers.16.self_attn.o_proj.biases,
model.layers.16.self_attn.o_proj.scales,
model.layers.16.self_attn.o_proj.weight,
model.layers.16.self_attn.q_norm.weight,
model.layers.16.self_attn.q_proj.biases,
model.layers.16.self_attn.q_proj.scales,
model.layers.16.self_attn.q_proj.weight,
model.layers.16.self_attn.v_proj.biases,
model.layers.16.self_attn.v_proj.scales,
model.layers.16.self_attn.v_proj.weight,
model.layers.17.input_layernorm.weight,
model.layers.17.mlp.down_proj.biases,
model.layers.17.mlp.down_proj.scales,
model.layers.17.mlp.down_proj.weight,
model.layers.17.mlp.gate_proj.biases,
model.layers.17.mlp.gate_proj.scales,
model.layers.17.mlp.gate_proj.weight,
model.layers.17.mlp.up_proj.biases,
model.layers.17.mlp.up_proj.scales,
model.layers.17.mlp.up_proj.weight,
model.layers.17.post_attention_layernorm.weight,
model.layers.17.self_attn.k_norm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.o_proj.biases,
model.layers.17.self_attn.o_proj.scales,
model.layers.17.self_attn.o_proj.weight,
model.layers.17.self_attn.q_norm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.input_layernorm.weight,
model.layers.18.mlp.down_proj.biases,
model.layers.18.mlp.down_proj.scales,
model.layers.18.mlp.down_proj.weight,
model.layers.18.mlp.gate_proj.biases,
model.layers.18.mlp.gate_proj.scales,
model.layers.18.mlp.gate_proj.weight,
model.layers.18.mlp.up_proj.biases,
model.layers.18.mlp.up_proj.scales,
model.layers.18.mlp.up_proj.weight,
model.layers.18.post_attention_layernorm.weight,
model.layers.18.self_attn.k_norm.weight,
model.layers.18.self_attn.k_proj.biases,
model.layers.18.self_attn.k_proj.scales,
model.layers.18.self_attn.k_proj.weight,
model.layers.18.self_attn.o_proj.biases,
model.layers.18.self_attn.o_proj.scales,
model.layers.18.self_attn.o_proj.weight,
model.layers.18.self_attn.q_norm.weight,
model.layers.18.self_attn.q_proj.biases,
model.layers.18.self_attn.q_proj.scales,
model.layers.18.self_attn.q_proj.weight,
model.layers.18.self_attn.v_proj.biases,
model.layers.18.self_attn.v_proj.scales,
model.layers.18.self_attn.v_proj.weight,
model.layers.19.input_layernorm.weight,
model.layers.19.mlp.down_proj.biases,
model.layers.19.mlp.down_proj.scales,
model.layers.19.mlp.down_proj.weight,
model.layers.19.mlp.gate_proj.biases,
model.layers.19.mlp.gate_proj.scales,
model.layers.19.mlp.gate_proj.weight,
model.layers.19.mlp.up_proj.biases,
model.layers.19.mlp.up_proj.scales,
model.layers.19.mlp.up_proj.weight,
model.layers.19.post_attention_layernorm.weight,
model.layers.19.self_attn.k_norm.weight,
model.layers.19.self_attn.k_proj.biases,
model.layers.19.self_attn.k_proj.scales,
model.layers.19.self_attn.k_proj.weight,
model.layers.19.self_attn.o_proj.biases,
model.layers.19.self_attn.o_proj.scales,
model.layers.19.self_attn.o_proj.weight,
model.layers.19.self_attn.q_norm.weight,
model.layers.19.self_attn.q_proj.biases,
model.layers.19.self_attn.q_proj.scales,
model.layers.19.self_attn.q_proj.weight,
model.layers.19.self_attn.v_proj.biases,
model.layers.19.self_attn.v_proj.scales,
model.layers.19.self_attn.v_proj.weight,
model.layers.2.input_layernorm.weight,
model.layers.2.mlp.down_proj.biases,
model.layers.2.mlp.down_proj.scales,
model.layers.2.mlp.down_proj.weight,
model.layers.2.mlp.gate_proj.biases,
model.layers.2.mlp.gate_proj.scales,
model.layers.2.mlp.gate_proj.weight,
model.layers.2.mlp.up_proj.biases,
model.layers.2.mlp.up_proj.scales,
model.layers.2.mlp.up_proj.weight,
model.layers.2.post_attention_layernorm.weight,
model.layers.2.self_attn.k_norm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.o_proj.biases,
model.layers.2.self_attn.o_proj.scales,
model.layers.2.self_attn.o_proj.weight,
model.layers.2.self_attn.q_norm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.input_layernorm.weight,
model.layers.20.mlp.down_proj.biases,
model.layers.20.mlp.down_proj.scales,
model.layers.20.mlp.down_proj.weight,
model.layers.20.mlp.gate_proj.biases,
model.layers.20.mlp.gate_proj.scales,
model.layers.20.mlp.gate_proj.weight,
model.layers.20.mlp.up_proj.biases,
model.layers.20.mlp.up_proj.scales,
model.layers.20.mlp.up_proj.weight,
model.layers.20.post_attention_layernorm.weight,
model.layers.20.self_attn.k_norm.weight,
model.layers.20.self_attn.k_proj.biases,
model.layers.20.self_attn.k_proj.scales,
model.layers.20.self_attn.k_proj.weight,
model.layers.20.self_attn.o_proj.biases,
model.layers.20.self_attn.o_proj.scales,
model.layers.20.self_attn.o_proj.weight,
model.layers.20.self_attn.q_norm.weight,
model.layers.20.self_attn.q_proj.biases,
model.layers.20.self_attn.q_proj.scales,
model.layers.20.self_attn.q_proj.weight,
model.layers.20.self_attn.v_proj.biases,
model.layers.20.self_attn.v_proj.scales,
model.layers.20.self_attn.v_proj.weight,
model.layers.21.input_layernorm.weight,
model.layers.21.mlp.down_proj.biases,
model.layers.21.mlp.down_proj.scales,
model.layers.21.mlp.down_proj.weight,
model.layers.21.mlp.gate_proj.biases,
model.layers.21.mlp.gate_proj.scales,
model.layers.21.mlp.gate_proj.weight,
model.layers.21.mlp.up_proj.biases,
model.layers.21.mlp.up_proj.scales,
model.layers.21.mlp.up_proj.weight,
model.layers.21.post_attention_layernorm.weight,
model.layers.21.self_attn.k_norm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.o_proj.biases,
model.layers.21.self_attn.o_proj.scales,
model.layers.21.self_attn.o_proj.weight,
model.layers.21.self_attn.q_norm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.input_layernorm.weight,
model.layers.22.mlp.down_proj.biases,
model.layers.22.mlp.down_proj.scales,
model.layers.22.mlp.down_proj.weight,
model.layers.22.mlp.gate_proj.biases,
model.layers.22.mlp.gate_proj.scales,
model.layers.22.mlp.gate_proj.weight,
model.layers.22.mlp.up_proj.biases,
model.layers.22.mlp.up_proj.scales,
model.layers.22.mlp.up_proj.weight,
model.layers.22.post_attention_layernorm.weight,
model.layers.22.self_attn.k_norm.weight,
model.layers.22.self_attn.k_proj.biases,
model.layers.22.self_attn.k_proj.scales,
model.layers.22.self_attn.k_proj.weight,
model.layers.22.self_attn.o_proj.biases,
model.layers.22.self_attn.o_proj.scales,
model.layers.22.self_attn.o_proj.weight,
model.layers.22.self_attn.q_norm.weight,
model.layers.22.self_attn.q_proj.biases,
model.layers.22.self_attn.q_proj.scales,
model.layers.22.self_attn.q_proj.weight,
model.layers.22.self_attn.v_proj.biases,
model.layers.22.self_attn.v_proj.scales,
model.layers.22.self_attn.v_proj.weight,
model.layers.23.input_layernorm.weight,
model.layers.23.mlp.down_proj.biases,
model.layers.23.mlp.down_proj.scales,
model.layers.23.mlp.down_proj.weight,
model.layers.23.mlp.gate_proj.biases,
model.layers.23.mlp.gate_proj.scales,
model.layers.23.mlp.gate_proj.weight,
model.layers.23.mlp.up_proj.biases,
model.layers.23.mlp.up_proj.scales,
model.layers.23.mlp.up_proj.weight,
model.layers.23.post_attention_layernorm.weight,
model.layers.23.self_attn.k_norm.weight,
model.layers.23.self_attn.k_proj.biases,
model.layers.23.self_attn.k_proj.scales,
model.layers.23.self_attn.k_proj.weight,
model.layers.23.self_attn.o_proj.biases,
model.layers.23.self_attn.o_proj.scales,
model.layers.23.self_attn.o_proj.weight,
model.layers.23.self_attn.q_norm.weight,
model.layers.23.self_attn.q_proj.biases,
model.layers.23.self_attn.q_proj.scales,
model.layers.23.self_attn.q_proj.weight,
model.layers.23.self_attn.v_proj.biases,
model.layers.23.self_attn.v_proj.scales,
model.layers.23.self_attn.v_proj.weight,
model.layers.24.input_layernorm.weight,
model.layers.24.mlp.down_proj.biases,
model.layers.24.mlp.down_proj.scales,
model.layers.24.mlp.down_proj.weight,
model.layers.24.mlp.gate_proj.biases,
model.layers.24.mlp.gate_proj.scales,
model.layers.24.mlp.gate_proj.weight,
model.layers.24.mlp.up_proj.biases,
model.layers.24.mlp.up_proj.scales,
model.layers.24.mlp.up_proj.weight,
model.layers.24.post_attention_layernorm.weight,
model.layers.24.self_attn.k_norm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.o_proj.biases,
model.layers.24.self_attn.o_proj.scales,
model.layers.24.self_attn.o_proj.weight,
model.layers.24.self_attn.q_norm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.input_layernorm.weight,
model.layers.25.mlp.down_proj.biases,
model.layers.25.mlp.down_proj.scales,
model.layers.25.mlp.down_proj.weight,
model.layers.25.mlp.gate_proj.biases,
model.layers.25.mlp.gate_proj.scales,
model.layers.25.mlp.gate_proj.weight,
model.layers.25.mlp.up_proj.biases,
model.layers.25.mlp.up_proj.scales,
model.layers.25.mlp.up_proj.weight,
model.layers.25.post_attention_layernorm.weight,
model.layers.25.self_attn.k_norm.weight,
model.layers.25.self_attn.k_proj.biases,
model.layers.25.self_attn.k_proj.scales,
model.layers.25.self_attn.k_proj.weight,
model.layers.25.self_attn.o_proj.biases,
model.layers.25.self_attn.o_proj.scales,
model.layers.25.self_attn.o_proj.weight,
model.layers.25.self_attn.q_norm.weight,
model.layers.25.self_attn.q_proj.biases,
model.layers.25.self_attn.q_proj.scales,
model.layers.25.self_attn.q_proj.weight,
model.layers.25.self_attn.v_proj.biases,
model.layers.25.self_attn.v_proj.scales,
model.layers.25.self_attn.v_proj.weight,
model.layers.26.input_layernorm.weight,
model.layers.26.mlp.down_proj.biases,
model.layers.26.mlp.down_proj.scales,
model.layers.26.mlp.down_proj.weight,
model.layers.26.mlp.gate_proj.biases,
model.layers.26.mlp.gate_proj.scales,
model.layers.26.mlp.gate_proj.weight,
model.layers.26.mlp.up_proj.biases,
model.layers.26.mlp.up_proj.scales,
model.layers.26.mlp.up_proj.weight,
model.layers.26.post_attention_layernorm.weight,
model.layers.26.self_attn.k_norm.weight,
model.layers.26.self_attn.k_proj.biases,
model.layers.26.self_attn.k_proj.scales,
model.layers.26.self_attn.k_proj.weight,
model.layers.26.self_attn.o_proj.biases,
model.layers.26.self_attn.o_proj.scales,
model.layers.26.self_attn.o_proj.weight,
model.layers.26.self_attn.q_norm.weight,
model.layers.26.self_attn.q_proj.biases,
model.layers.26.self_attn.q_proj.scales,
model.layers.26.self_attn.q_proj.weight,
model.layers.26.self_attn.v_proj.biases,
model.layers.26.self_attn.v_proj.scales,
model.layers.26.self_attn.v_proj.weight,
model.layers.27.input_layernorm.weight,
model.layers.27.mlp.down_proj.biases,
model.layers.27.mlp.down_proj.scales,
model.layers.27.mlp.down_proj.weight,
model.layers.27.mlp.gate_proj.biases,
model.layers.27.mlp.gate_proj.scales,
model.layers.27.mlp.gate_proj.weight,
model.layers.27.mlp.up_proj.biases,
model.layers.27.mlp.up_proj.scales,
model.layers.27.mlp.up_proj.weight,
model.layers.27.post_attention_layernorm.weight,
model.layers.27.self_attn.k_norm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.o_proj.biases,
model.layers.27.self_attn.o_proj.scales,
model.layers.27.self_attn.o_proj.weight,
model.layers.27.self_attn.q_norm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.input_layernorm.weight,
model.layers.28.mlp.down_proj.biases,
model.layers.28.mlp.down_proj.scales,
model.layers.28.mlp.down_proj.weight,
model.layers.28.mlp.gate_proj.biases,
model.layers.28.mlp.gate_proj.scales,
model.layers.28.mlp.gate_proj.weight,
model.layers.28.mlp.up_proj.biases,
model.layers.28.mlp.up_proj.scales,
model.layers.28.mlp.up_proj.weight,
model.layers.28.post_attention_layernorm.weight,
model.layers.28.self_attn.k_norm.weight,
model.layers.28.self_attn.k_proj.biases,
model.layers.28.self_attn.k_proj.scales,
model.layers.28.self_attn.k_proj.weight,
model.layers.28.self_attn.o_proj.biases,
model.layers.28.self_attn.o_proj.scales,
model.layers.28.self_attn.o_proj.weight,
model.layers.28.self_attn.q_norm.weight,
model.layers.28.self_attn.q_proj.biases,
model.layers.28.self_attn.q_proj.scales,
model.layers.28.self_attn.q_proj.weight,
model.layers.28.self_attn.v_proj.biases,
model.layers.28.self_attn.v_proj.scales,
model.layers.28.self_attn.v_proj.weight,
model.layers.29.input_layernorm.weight,
model.layers.29.mlp.down_proj.biases,
model.layers.29.mlp.down_proj.scales,
model.layers.29.mlp.down_proj.weight,
model.layers.29.mlp.gate_proj.biases,
model.layers.29.mlp.gate_proj.scales,
model.layers.29.mlp.gate_proj.weight,
model.layers.29.mlp.up_proj.biases,
model.layers.29.mlp.up_proj.scales,
model.layers.29.mlp.up_proj.weight,
model.layers.29.post_attention_layernorm.weight,
model.layers.29.self_attn.k_norm.weight,
model.layers.29.self_attn.k_proj.biases,
model.layers.29.self_attn.k_proj.scales,
model.layers.29.self_attn.k_proj.weight,
model.layers.29.self_attn.o_proj.biases,
model.layers.29.self_attn.o_proj.scales,
model.layers.29.self_attn.o_proj.weight,
model.layers.29.self_attn.q_norm.weight,
model.layers.29.self_attn.q_proj.biases,
model.layers.29.self_attn.q_proj.scales,
model.layers.29.self_attn.q_proj.weight,
model.layers.29.self_attn.v_proj.biases,
model.layers.29.self_attn.v_proj.scales,
model.layers.29.self_attn.v_proj.weight,
model.layers.3.input_layernorm.weight,
model.layers.3.mlp.down_proj.biases,
model.layers.3.mlp.down_proj.scales,
model.layers.3.mlp.down_proj.weight,
model.layers.3.mlp.gate_proj.biases,
model.layers.3.mlp.gate_proj.scales,
model.layers.3.mlp.gate_proj.weight,
model.layers.3.mlp.up_proj.biases,
model.layers.3.mlp.up_proj.scales,
model.layers.3.mlp.up_proj.weight,
model.layers.3.post_attention_layernorm.weight,
model.layers.3.self_attn.k_norm.weight,
model.layers.3.self_attn.k_proj.biases,
model.layers.3.self_attn.k_proj.scales,
model.layers.3.self_attn.k_proj.weight,
model.layers.3.self_attn.o_proj.biases,
model.layers.3.self_attn.o_proj.scales,
model.layers.3.self_attn.o_proj.weight,
model.layers.3.self_attn.q_norm.weight,
model.layers.3.self_attn.q_proj.biases,
model.layers.3.self_attn.q_proj.scales,
model.layers.3.self_attn.q_proj.weight,
model.layers.3.self_attn.v_proj.biases,
model.layers.3.self_attn.v_proj.scales,
model.layers.3.self_attn.v_proj.weight,
model.layers.30.input_layernorm.weight,
model.layers.30.mlp.down_proj.biases,
model.layers.30.mlp.down_proj.scales,
model.layers.30.mlp.down_proj.weight,
model.layers.30.mlp.gate_proj.biases,
model.layers.30.mlp.gate_proj.scales,
model.layers.30.mlp.gate_proj.weight,
model.layers.30.mlp.up_proj.biases,
model.layers.30.mlp.up_proj.scales,
model.layers.30.mlp.up_proj.weight,
model.layers.30.post_attention_layernorm.weight,
model.layers.30.self_attn.k_norm.weight,
model.layers.30.self_attn.k_proj.biases,
model.layers.30.self_attn.k_proj.scales,
model.layers.30.self_attn.k_proj.weight,
model.layers.30.self_attn.o_proj.biases,
model.layers.30.self_attn.o_proj.scales,
model.layers.30.self_attn.o_proj.weight,
model.layers.30.self_attn.q_norm.weight,
model.layers.30.self_attn.q_proj.biases,
model.layers.30.self_attn.q_proj.scales,
model.layers.30.self_attn.q_proj.weight,
model.layers.30.self_attn.v_proj.biases,
model.layers.30.self_attn.v_proj.scales,
model.layers.30.self_attn.v_proj.weight,
model.layers.31.input_layernorm.weight,
model.layers.31.mlp.down_proj.biases,
model.layers.31.mlp.down_proj.scales,
model.layers.31.mlp.down_proj.weight,
model.layers.31.mlp.gate_proj.biases,
model.layers.31.mlp.gate_proj.scales,
model.layers.31.mlp.gate_proj.weight,
model.layers.31.mlp.up_proj.biases,
model.layers.31.mlp.up_proj.scales,
model.layers.31.mlp.up_proj.weight,
model.layers.31.post_attention_layernorm.weight,
model.layers.31.self_attn.k_norm.weight,
model.layers.31.self_attn.k_proj.biases,
model.layers.31.self_attn.k_proj.scales,
model.layers.31.self_attn.k_proj.weight,
model.layers.31.self_attn.o_proj.biases,
model.layers.31.self_attn.o_proj.scales,
model.layers.31.self_attn.o_proj.weight,
model.layers.31.self_attn.q_norm.weight,
model.layers.31.self_attn.q_proj.biases,
model.layers.31.self_attn.q_proj.scales,
model.layers.31.self_attn.q_proj.weight,
model.layers.31.self_attn.v_proj.biases,
model.layers.31.self_attn.v_proj.scales,
model.layers.31.self_attn.v_proj.weight,
model.layers.32.input_layernorm.weight,
model.layers.32.mlp.down_proj.biases,
model.layers.32.mlp.down_proj.scales,
model.layers.32.mlp.down_proj.weight,
model.layers.32.mlp.gate_proj.biases,
model.layers.32.mlp.gate_proj.scales,
model.layers.32.mlp.gate_proj.weight,
model.layers.32.mlp.up_proj.biases,
model.layers.32.mlp.up_proj.scales,
model.layers.32.mlp.up_proj.weight,
model.layers.32.post_attention_layernorm.weight,
model.layers.32.self_attn.k_norm.weight,
model.layers.32.self_attn.k_proj.biases,
model.layers.32.self_attn.k_proj.scales,
model.layers.32.self_attn.k_proj.weight,
model.layers.32.self_attn.o_proj.biases,
model.layers.32.self_attn.o_proj.scales,
model.layers.32.self_attn.o_proj.weight,
model.layers.32.self_attn.q_norm.weight,
model.layers.32.self_attn.q_proj.biases,
model.layers.32.self_attn.q_proj.scales,
model.layers.32.self_attn.q_proj.weight,
model.layers.32.self_attn.v_proj.biases,
model.layers.32.self_attn.v_proj.scales,
model.layers.32.self_attn.v_proj.weight,
model.layers.33.input_layernorm.weight,
model.layers.33.mlp.down_proj.biases,
model.layers.33.mlp.down_proj.scales,
model.layers.33.mlp.down_proj.weight,
model.layers.33.mlp.gate_proj.biases,
model.layers.33.mlp.gate_proj.scales,
model.layers.33.mlp.gate_proj.weight,
model.layers.33.mlp.up_proj.biases,
model.layers.33.mlp.up_proj.scales,
model.layers.33.mlp.up_proj.weight,
model.layers.33.post_attention_layernorm.weight,
model.layers.33.self_attn.k_norm.weight,
model.layers.33.self_attn.k_proj.biases,
model.layers.33.self_attn.k_proj.scales,
model.layers.33.self_attn.k_proj.weight,
model.layers.33.self_attn.o_proj.biases,
model.layers.33.self_attn.o_proj.scales,
model.layers.33.self_attn.o_proj.weight,
model.layers.33.self_attn.q_norm.weight,
model.layers.33.self_attn.q_proj.biases,
model.layers.33.self_attn.q_proj.scales,
model.layers.33.self_attn.q_proj.weight,
model.layers.33.self_attn.v_proj.biases,
model.layers.33.self_attn.v_proj.scales,
model.layers.33.self_attn.v_proj.weight,
model.layers.34.input_layernorm.weight,
model.layers.34.mlp.down_proj.biases,
model.layers.34.mlp.down_proj.scales,
model.layers.34.mlp.down_proj.weight,
model.layers.34.mlp.gate_proj.biases,
model.layers.34.mlp.gate_proj.scales,
model.layers.34.mlp.gate_proj.weight,
model.layers.34.mlp.up_proj.biases,
model.layers.34.mlp.up_proj.scales,
model.layers.34.mlp.up_proj.weight,
model.layers.34.post_attention_layernorm.weight,
model.layers.34.self_attn.k_norm.weight,
model.layers.34.self_attn.k_proj.biases,
model.layers.34.self_attn.k_proj.scales,
model.layers.34.self_attn.k_proj.weight,
model.layers.34.self_attn.o_proj.biases,
model.layers.34.self_attn.o_proj.scales,
model.layers.34.self_attn.o_proj.weight,
model.layers.34.self_attn.q_norm.weight,
model.layers.34.self_attn.q_proj.biases,
model.layers.34.self_attn.q_proj.scales,
model.layers.34.self_attn.q_proj.weight,
model.layers.34.self_attn.v_proj.biases,
model.layers.34.self_attn.v_proj.scales,
model.layers.34.self_attn.v_proj.weight,
model.layers.35.input_layernorm.weight,
model.layers.35.mlp.down_proj.biases,
model.layers.35.mlp.down_proj.scales,
model.layers.35.mlp.down_proj.weight,
model.layers.35.mlp.gate_proj.biases,
model.layers.35.mlp.gate_proj.scales,
model.layers.35.mlp.gate_proj.weight,
model.layers.35.mlp.up_proj.biases,
model.layers.35.mlp.up_proj.scales,
model.layers.35.mlp.up_proj.weight,
model.layers.35.post_attention_layernorm.weight,
model.layers.35.self_attn.k_norm.weight,
model.layers.35.self_attn.k_proj.biases,
model.layers.35.self_attn.k_proj.scales,
model.layers.35.self_attn.k_proj.weight,
model.layers.35.self_attn.o_proj.biases,
model.layers.35.self_attn.o_proj.scales,
model.layers.35.self_attn.o_proj.weight,
model.layers.35.self_attn.q_norm.weight,
model.layers.35.self_attn.q_proj.biases,
model.layers.35.self_attn.q_proj.scales,
model.layers.35.self_attn.q_proj.weight,
model.layers.35.self_attn.v_proj.biases,
model.layers.35.self_attn.v_proj.scales,
model.layers.35.self_attn.v_proj.weight,
model.layers.4.input_layernorm.weight,
model.layers.4.mlp.down_proj.biases,
model.layers.4.mlp.down_proj.scales,
model.layers.4.mlp.down_proj.weight,
model.layers.4.mlp.gate_proj.biases,
model.layers.4.mlp.gate_proj.scales,
model.layers.4.mlp.gate_proj.weight,
model.layers.4.mlp.up_proj.biases,
model.layers.4.mlp.up_proj.scales,
model.layers.4.mlp.up_proj.weight,
model.layers.4.post_attention_layernorm.weight,
model.layers.4.self_attn.k_norm.weight,
model.layers.4.self_attn.k_proj.biases,
model.layers.4.self_attn.k_proj.scales,
model.layers.4.self_attn.k_proj.weight,
model.layers.4.self_attn.o_proj.biases,
model.layers.4.self_attn.o_proj.scales,
model.layers.4.self_attn.o_proj.weight,
model.layers.4.self_attn.q_norm.weight,
model.layers.4.self_attn.q_proj.biases,
model.layers.4.self_attn.q_proj.scales,
model.layers.4.self_attn.q_proj.weight,
model.layers.4.self_attn.v_proj.biases,
model.layers.4.self_attn.v_proj.scales,
model.layers.4.self_attn.v_proj.weight,
model.layers.5.input_layernorm.weight,
model.layers.5.mlp.down_proj.biases,
model.layers.5.mlp.down_proj.scales,
model.layers.5.mlp.down_proj.weight,
model.layers.5.mlp.gate_proj.biases,
model.layers.5.mlp.gate_proj.scales,
model.layers.5.mlp.gate_proj.weight,
model.layers.5.mlp.up_proj.biases,
model.layers.5.mlp.up_proj.scales,
model.layers.5.mlp.up_proj.weight,
model.layers.5.post_attention_layernorm.weight,
model.layers.5.self_attn.k_norm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.o_proj.biases,
model.layers.5.self_attn.o_proj.scales,
model.layers.5.self_attn.o_proj.weight,
model.layers.5.self_attn.q_norm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.input_layernorm.weight,
model.layers.6.mlp.down_proj.biases,
model.layers.6.mlp.down_proj.scales,
model.layers.6.mlp.down_proj.weight,
model.layers.6.mlp.gate_proj.biases,
model.layers.6.mlp.gate_proj.scales,
model.layers.6.mlp.gate_proj.weight,
model.layers.6.mlp.up_proj.biases,
model.layers.6.mlp.up_proj.scales,
model.layers.6.mlp.up_proj.weight,
model.layers.6.post_attention_layernorm.weight,
model.layers.6.self_attn.k_norm.weight,
model.layers.6.self_attn.k_proj.biases,
model.layers.6.self_attn.k_proj.scales,
model.layers.6.self_attn.k_proj.weight,
model.layers.6.self_attn.o_proj.biases,
model.layers.6.self_attn.o_proj.scales,
model.layers.6.self_attn.o_proj.weight,
model.layers.6.self_attn.q_norm.weight,
model.layers.6.self_attn.q_proj.biases,
model.layers.6.self_attn.q_proj.scales,
model.layers.6.self_attn.q_proj.weight,
model.layers.6.self_attn.v_proj.biases,
model.layers.6.self_attn.v_proj.scales,
model.layers.6.self_attn.v_proj.weight,
model.layers.7.input_layernorm.weight,
model.layers.7.mlp.down_proj.biases,
model.layers.7.mlp.down_proj.scales,
model.layers.7.mlp.down_proj.weight,
model.layers.7.mlp.gate_proj.biases,
model.layers.7.mlp.gate_proj.scales,
model.layers.7.mlp.gate_proj.weight,
model.layers.7.mlp.up_proj.biases,
model.layers.7.mlp.up_proj.scales,
model.layers.7.mlp.up_proj.weight,
model.layers.7.post_attention_layernorm.weight,
model.layers.7.self_attn.k_norm.weight,
model.layers.7.self_attn.k_proj.biases,
model.layers.7.self_attn.k_proj.scales,
model.layers.7.self_attn.k_proj.weight,
model.layers.7.self_attn.o_proj.biases,
model.layers.7.self_attn.o_proj.scales,
model.layers.7.self_attn.o_proj.weight,
model.layers.7.self_attn.q_norm.weight,
model.layers.7.self_attn.q_proj.biases,
model.layers.7.self_attn.q_proj.scales,
model.layers.7.self_attn.q_proj.weight,
model.layers.7.self_attn.v_proj.biases,
model.layers.7.self_attn.v_proj.scales,
model.layers.7.self_attn.v_proj.weight,
model.layers.8.input_layernorm.weight,
model.layers.8.mlp.down_proj.biases,
model.layers.8.mlp.down_proj.scales,
model.layers.8.mlp.down_proj.weight,
model.layers.8.mlp.gate_proj.biases,
model.layers.8.mlp.gate_proj.scales,
model.layers.8.mlp.gate_proj.weight,
model.layers.8.mlp.up_proj.biases,
model.layers.8.mlp.up_proj.scales,
model.layers.8.mlp.up_proj.weight,
model.layers.8.post_attention_layernorm.weight,
model.layers.8.self_attn.k_norm.weight,
model.layers.8.self_attn.k_proj.biases,
model.layers.8.self_attn.k_proj.scales,
model.layers.8.self_attn.k_proj.weight,
model.layers.8.self_attn.o_proj.biases,
model.layers.8.self_attn.o_proj.scales,
model.layers.8.self_attn.o_proj.weight,
model.layers.8.self_attn.q_norm.weight,
model.layers.8.self_attn.q_proj.biases,
model.layers.8.self_attn.q_proj.scales,
model.layers.8.self_attn.q_proj.weight,
model.layers.8.self_attn.v_proj.biases,
model.layers.8.self_attn.v_proj.scales,
model.layers.8.self_attn.v_proj.weight,
model.layers.9.input_layernorm.weight,
model.layers.9.mlp.down_proj.biases,
model.layers.9.mlp.down_proj.scales,
model.layers.9.mlp.down_proj.weight,
model.layers.9.mlp.gate_proj.biases,
model.layers.9.mlp.gate_proj.scales,
model.layers.9.mlp.gate_proj.weight,
model.layers.9.mlp.up_proj.biases,
model.layers.9.mlp.up_proj.scales,
model.layers.9.mlp.up_proj.weight,
model.layers.9.post_attention_layernorm.weight,
model.layers.9.self_attn.k_norm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.o_proj.biases,
model.layers.9.self_attn.o_proj.scales,
model.layers.9.self_attn.o_proj.weight,
model.layers.9.self_attn.q_norm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight,
model.norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15198, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14183, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 904 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.layers.0.input_layernorm.weight,
model.layers.0.mlp.down_proj.biases,
model.layers.0.mlp.down_proj.scales,
model.layers.0.mlp.down_proj.weight,
model.layers.0.mlp.gate_proj.biases,
model.layers.0.mlp.gate_proj.scales,
model.layers.0.mlp.gate_proj.weight,
model.layers.0.mlp.up_proj.biases,
model.layers.0.mlp.up_proj.scales,
model.layers.0.mlp.up_proj.weight,
model.layers.0.post_attention_layernorm.weight,
model.layers.0.self_attn.k_norm.weight,
model.layers.0.self_attn.k_proj.biases,
model.layers.0.self_attn.k_proj.scales,
model.layers.0.self_attn.k_proj.weight,
model.layers.0.self_attn.o_proj.biases,
model.layers.0.self_attn.o_proj.scales,
model.layers.0.self_attn.o_proj.weight,
model.layers.0.self_attn.q_norm.weight,
model.layers.0.self_attn.q_proj.biases,
model.layers.0.self_attn.q_proj.scales,
model.layers.0.self_attn.q_proj.weight,
model.layers.0.self_attn.v_proj.biases,
model.layers.0.self_attn.v_proj.scales,
model.layers.0.self_attn.v_proj.weight,
model.layers.1.input_layernorm.weight,
model.layers.1.mlp.down_proj.biases,
model.layers.1.mlp.down_proj.scales,
model.layers.1.mlp.down_proj.weight,
model.layers.1.mlp.gate_proj.biases,
model.layers.1.mlp.gate_proj.scales,
model.layers.1.mlp.gate_proj.weight,
model.layers.1.mlp.up_proj.biases,
model.layers.1.mlp.up_proj.scales,
model.layers.1.mlp.up_proj.weight,
model.layers.1.post_attention_layernorm.weight,
model.layers.1.self_attn.k_norm.weight,
model.layers.1.self_attn.k_proj.biases,
model.layers.1.self_attn.k_proj.scales,
model.layers.1.self_attn.k_proj.weight,
model.layers.1.self_attn.o_proj.biases,
model.layers.1.self_attn.o_proj.scales,
model.layers.1.self_attn.o_proj.weight,
model.layers.1.self_attn.q_norm.weight,
model.layers.1.self_attn.q_proj.biases,
model.layers.1.self_attn.q_proj.scales,
model.layers.1.self_attn.q_proj.weight,
model.layers.1.self_attn.v_proj.biases,
model.layers.1.self_attn.v_proj.scales,
model.layers.1.self_attn.v_proj.weight,
model.layers.10.input_layernorm.weight,
model.layers.10.mlp.down_proj.biases,
model.layers.10.mlp.down_proj.scales,
model.layers.10.mlp.down_proj.weight,
model.layers.10.mlp.gate_proj.biases,
model.layers.10.mlp.gate_proj.scales,
model.layers.10.mlp.gate_proj.weight,
model.layers.10.mlp.up_proj.biases,
model.layers.10.mlp.up_proj.scales,
model.layers.10.mlp.up_proj.weight,
model.layers.10.post_attention_layernorm.weight,
model.layers.10.self_attn.k_norm.weight,
model.layers.10.self_attn.k_proj.biases,
model.layers.10.self_attn.k_proj.scales,
model.layers.10.self_attn.k_proj.weight,
model.layers.10.self_attn.o_proj.biases,
model.layers.10.self_attn.o_proj.scales,
model.layers.10.self_attn.o_proj.weight,
model.layers.10.self_attn.q_norm.weight,
model.layers.10.self_attn.q_proj.biases,
model.layers.10.self_attn.q_proj.scales,
model.layers.10.self_attn.q_proj.weight,
model.layers.10.self_attn.v_proj.biases,
model.layers.10.self_attn.v_proj.scales,
model.layers.10.self_attn.v_proj.weight,
model.layers.11.input_layernorm.weight,
model.layers.11.mlp.down_proj.biases,
model.layers.11.mlp.down_proj.scales,
model.layers.11.mlp.down_proj.weight,
model.layers.11.mlp.gate_proj.biases,
model.layers.11.mlp.gate_proj.scales,
model.layers.11.mlp.gate_proj.weight,
model.layers.11.mlp.up_proj.biases,
model.layers.11.mlp.up_proj.scales,
model.layers.11.mlp.up_proj.weight,
model.layers.11.post_attention_layernorm.weight,
model.layers.11.self_attn.k_norm.weight,
model.layers.11.self_attn.k_proj.biases,
model.layers.11.self_attn.k_proj.scales,
model.layers.11.self_attn.k_proj.weight,
model.layers.11.self_attn.o_proj.biases,
model.layers.11.self_attn.o_proj.scales,
model.layers.11.self_attn.o_proj.weight,
model.layers.11.self_attn.q_norm.weight,
model.layers.11.self_attn.q_proj.biases,
model.layers.11.self_attn.q_proj.scales,
model.layers.11.self_attn.q_proj.weight,
model.layers.11.self_attn.v_proj.biases,
model.layers.11.self_attn.v_proj.scales,
model.layers.11.self_attn.v_proj.weight,
model.layers.12.input_layernorm.weight,
model.layers.12.mlp.down_proj.biases,
model.layers.12.mlp.down_proj.scales,
model.layers.12.mlp.down_proj.weight,
model.layers.12.mlp.gate_proj.biases,
model.layers.12.mlp.gate_proj.scales,
model.layers.12.mlp.gate_proj.weight,
model.layers.12.mlp.up_proj.biases,
model.layers.12.mlp.up_proj.scales,
model.layers.12.mlp.up_proj.weight,
model.layers.12.post_attention_layernorm.weight,
model.layers.12.self_attn.k_norm.weight,
model.layers.12.self_attn.k_proj.biases,
model.layers.12.self_attn.k_proj.scales,
model.layers.12.self_attn.k_proj.weight,
model.layers.12.self_attn.o_proj.biases,
model.layers.12.self_attn.o_proj.scales,
model.layers.12.self_attn.o_proj.weight,
model.layers.12.self_attn.q_norm.weight,
model.layers.12.self_attn.q_proj.biases,
model.layers.12.self_attn.q_proj.scales,
model.layers.12.self_attn.q_proj.weight,
model.layers.12.self_attn.v_proj.biases,
model.layers.12.self_attn.v_proj.scales,
model.layers.12.self_attn.v_proj.weight,
model.layers.13.input_layernorm.weight,
model.layers.13.mlp.down_proj.biases,
model.layers.13.mlp.down_proj.scales,
model.layers.13.mlp.down_proj.weight,
model.layers.13.mlp.gate_proj.biases,
model.layers.13.mlp.gate_proj.scales,
model.layers.13.mlp.gate_proj.weight,
model.layers.13.mlp.up_proj.biases,
model.layers.13.mlp.up_proj.scales,
model.layers.13.mlp.up_proj.weight,
model.layers.13.post_attention_layernorm.weight,
model.layers.13.self_attn.k_norm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.o_proj.biases,
model.layers.13.self_attn.o_proj.scales,
model.layers.13.self_attn.o_proj.weight,
model.layers.13.self_attn.q_norm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.input_layernorm.weight,
model.layers.14.mlp.down_proj.biases,
model.layers.14.mlp.down_proj.scales,
model.layers.14.mlp.down_proj.weight,
model.layers.14.mlp.gate_proj.biases,
model.layers.14.mlp.gate_proj.scales,
model.layers.14.mlp.gate_proj.weight,
model.layers.14.mlp.up_proj.biases,
model.layers.14.mlp.up_proj.scales,
model.layers.14.mlp.up_proj.weight,
model.layers.14.post_attention_layernorm.weight,
model.layers.14.self_attn.k_norm.weight,
model.layers.14.self_attn.k_proj.biases,
model.layers.14.self_attn.k_proj.scales,
model.layers.14.self_attn.k_proj.weight,
model.layers.14.self_attn.o_proj.biases,
model.layers.14.self_attn.o_proj.scales,
model.layers.14.self_attn.o_proj.weight,
model.layers.14.self_attn.q_norm.weight,
model.layers.14.self_attn.q_proj.biases,
model.layers.14.self_attn.q_proj.scales,
model.layers.14.self_attn.q_proj.weight,
model.layers.14.self_attn.v_proj.biases,
model.layers.14.self_attn.v_proj.scales,
model.layers.14.self_attn.v_proj.weight,
model.layers.15.input_layernorm.weight,
model.layers.15.mlp.down_proj.biases,
model.layers.15.mlp.down_proj.scales,
model.layers.15.mlp.down_proj.weight,
model.layers.15.mlp.gate_proj.biases,
model.layers.15.mlp.gate_proj.scales,
model.layers.15.mlp.gate_proj.weight,
model.layers.15.mlp.up_proj.biases,
model.layers.15.mlp.up_proj.scales,
model.layers.15.mlp.up_proj.weight,
model.layers.15.post_attention_layernorm.weight,
model.layers.15.self_attn.k_norm.weight,
model.layers.15.self_attn.k_proj.biases,
model.layers.15.self_attn.k_proj.scales,
model.layers.15.self_attn.k_proj.weight,
model.layers.15.self_attn.o_proj.biases,
model.layers.15.self_attn.o_proj.scales,
model.layers.15.self_attn.o_proj.weight,
model.layers.15.self_attn.q_norm.weight,
model.layers.15.self_attn.q_proj.biases,
model.layers.15.self_attn.q_proj.scales,
model.layers.15.self_attn.q_proj.weight,
model.layers.15.self_attn.v_proj.biases,
model.layers.15.self_attn.v_proj.scales,
model.layers.15.self_attn.v_proj.weight,
model.layers.16.input_layernorm.weight,
model.layers.16.mlp.down_proj.biases,
model.layers.16.mlp.down_proj.scales,
model.layers.16.mlp.down_proj.weight,
model.layers.16.mlp.gate_proj.biases,
model.layers.16.mlp.gate_proj.scales,
model.layers.16.mlp.gate_proj.weight,
model.layers.16.mlp.up_proj.biases,
model.layers.16.mlp.up_proj.scales,
model.layers.16.mlp.up_proj.weight,
model.layers.16.post_attention_layernorm.weight,
model.layers.16.self_attn.k_norm.weight,
model.layers.16.self_attn.k_proj.biases,
model.layers.16.self_attn.k_proj.scales,
model.layers.16.self_attn.k_proj.weight,
model.layers.16.self_attn.o_proj.biases,
model.layers.16.self_attn.o_proj.scales,
model.layers.16.self_attn.o_proj.weight,
model.layers.16.self_attn.q_norm.weight,
model.layers.16.self_attn.q_proj.biases,
model.layers.16.self_attn.q_proj.scales,
model.layers.16.self_attn.q_proj.weight,
model.layers.16.self_attn.v_proj.biases,
model.layers.16.self_attn.v_proj.scales,
model.layers.16.self_attn.v_proj.weight,
model.layers.17.input_layernorm.weight,
model.layers.17.mlp.down_proj.biases,
model.layers.17.mlp.down_proj.scales,
model.layers.17.mlp.down_proj.weight,
model.layers.17.mlp.gate_proj.biases,
model.layers.17.mlp.gate_proj.scales,
model.layers.17.mlp.gate_proj.weight,
model.layers.17.mlp.up_proj.biases,
model.layers.17.mlp.up_proj.scales,
model.layers.17.mlp.up_proj.weight,
model.layers.17.post_attention_layernorm.weight,
model.layers.17.self_attn.k_norm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.o_proj.biases,
model.layers.17.self_attn.o_proj.scales,
model.layers.17.self_attn.o_proj.weight,
model.layers.17.self_attn.q_norm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.input_layernorm.weight,
model.layers.18.mlp.down_proj.biases,
model.layers.18.mlp.down_proj.scales,
model.layers.18.mlp.down_proj.weight,
model.layers.18.mlp.gate_proj.biases,
model.layers.18.mlp.gate_proj.scales,
model.layers.18.mlp.gate_proj.weight,
model.layers.18.mlp.up_proj.biases,
model.layers.18.mlp.up_proj.scales,
model.layers.18.mlp.up_proj.weight,
model.layers.18.post_attention_layernorm.weight,
model.layers.18.self_attn.k_norm.weight,
model.layers.18.self_attn.k_proj.biases,
model.layers.18.self_attn.k_proj.scales,
model.layers.18.self_attn.k_proj.weight,
model.layers.18.self_attn.o_proj.biases,
model.layers.18.self_attn.o_proj.scales,
model.layers.18.self_attn.o_proj.weight,
model.layers.18.self_attn.q_norm.weight,
model.layers.18.self_attn.q_proj.biases,
model.layers.18.self_attn.q_proj.scales,
model.layers.18.self_attn.q_proj.weight,
model.layers.18.self_attn.v_proj.biases,
model.layers.18.self_attn.v_proj.scales,
model.layers.18.self_attn.v_proj.weight,
model.layers.19.input_layernorm.weight,
model.layers.19.mlp.down_proj.biases,
model.layers.19.mlp.down_proj.scales,
model.layers.19.mlp.down_proj.weight,
model.layers.19.mlp.gate_proj.biases,
model.layers.19.mlp.gate_proj.scales,
model.layers.19.mlp.gate_proj.weight,
model.layers.19.mlp.up_proj.biases,
model.layers.19.mlp.up_proj.scales,
model.layers.19.mlp.up_proj.weight,
model.layers.19.post_attention_layernorm.weight,
model.layers.19.self_attn.k_norm.weight,
model.layers.19.self_attn.k_proj.biases,
model.layers.19.self_attn.k_proj.scales,
model.layers.19.self_attn.k_proj.weight,
model.layers.19.self_attn.o_proj.biases,
model.layers.19.self_attn.o_proj.scales,
model.layers.19.self_attn.o_proj.weight,
model.layers.19.self_attn.q_norm.weight,
model.layers.19.self_attn.q_proj.biases,
model.layers.19.self_attn.q_proj.scales,
model.layers.19.self_attn.q_proj.weight,
model.layers.19.self_attn.v_proj.biases,
model.layers.19.self_attn.v_proj.scales,
model.layers.19.self_attn.v_proj.weight,
model.layers.2.input_layernorm.weight,
model.layers.2.mlp.down_proj.biases,
model.layers.2.mlp.down_proj.scales,
model.layers.2.mlp.down_proj.weight,
model.layers.2.mlp.gate_proj.biases,
model.layers.2.mlp.gate_proj.scales,
model.layers.2.mlp.gate_proj.weight,
model.layers.2.mlp.up_proj.biases,
model.layers.2.mlp.up_proj.scales,
model.layers.2.mlp.up_proj.weight,
model.layers.2.post_attention_layernorm.weight,
model.layers.2.self_attn.k_norm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.o_proj.biases,
model.layers.2.self_attn.o_proj.scales,
model.layers.2.self_attn.o_proj.weight,
model.layers.2.self_attn.q_norm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.input_layernorm.weight,
model.layers.20.mlp.down_proj.biases,
model.layers.20.mlp.down_proj.scales,
model.layers.20.mlp.down_proj.weight,
model.layers.20.mlp.gate_proj.biases,
model.layers.20.mlp.gate_proj.scales,
model.layers.20.mlp.gate_proj.weight,
model.layers.20.mlp.up_proj.biases,
model.layers.20.mlp.up_proj.scales,
model.layers.20.mlp.up_proj.weight,
model.layers.20.post_attention_layernorm.weight,
model.layers.20.self_attn.k_norm.weight,
model.layers.20.self_attn.k_proj.biases,
model.layers.20.self_attn.k_proj.scales,
model.layers.20.self_attn.k_proj.weight,
model.layers.20.self_attn.o_proj.biases,
model.layers.20.self_attn.o_proj.scales,
model.layers.20.self_attn.o_proj.weight,
model.layers.20.self_attn.q_norm.weight,
model.layers.20.self_attn.q_proj.biases,
model.layers.20.self_attn.q_proj.scales,
model.layers.20.self_attn.q_proj.weight,
model.layers.20.self_attn.v_proj.biases,
model.layers.20.self_attn.v_proj.scales,
model.layers.20.self_attn.v_proj.weight,
model.layers.21.input_layernorm.weight,
model.layers.21.mlp.down_proj.biases,
model.layers.21.mlp.down_proj.scales,
model.layers.21.mlp.down_proj.weight,
model.layers.21.mlp.gate_proj.biases,
model.layers.21.mlp.gate_proj.scales,
model.layers.21.mlp.gate_proj.weight,
model.layers.21.mlp.up_proj.biases,
model.layers.21.mlp.up_proj.scales,
model.layers.21.mlp.up_proj.weight,
model.layers.21.post_attention_layernorm.weight,
model.layers.21.self_attn.k_norm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.o_proj.biases,
model.layers.21.self_attn.o_proj.scales,
model.layers.21.self_attn.o_proj.weight,
model.layers.21.self_attn.q_norm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.input_layernorm.weight,
model.layers.22.mlp.down_proj.biases,
model.layers.22.mlp.down_proj.scales,
model.layers.22.mlp.down_proj.weight,
model.layers.22.mlp.gate_proj.biases,
model.layers.22.mlp.gate_proj.scales,
model.layers.22.mlp.gate_proj.weight,
model.layers.22.mlp.up_proj.biases,
model.layers.22.mlp.up_proj.scales,
model.layers.22.mlp.up_proj.weight,
model.layers.22.post_attention_layernorm.weight,
model.layers.22.self_attn.k_norm.weight,
model.layers.22.self_attn.k_proj.biases,
model.layers.22.self_attn.k_proj.scales,
model.layers.22.self_attn.k_proj.weight,
model.layers.22.self_attn.o_proj.biases,
model.layers.22.self_attn.o_proj.scales,
model.layers.22.self_attn.o_proj.weight,
model.layers.22.self_attn.q_norm.weight,
model.layers.22.self_attn.q_proj.biases,
model.layers.22.self_attn.q_proj.scales,
model.layers.22.self_attn.q_proj.weight,
model.layers.22.self_attn.v_proj.biases,
model.layers.22.self_attn.v_proj.scales,
model.layers.22.self_attn.v_proj.weight,
model.layers.23.input_layernorm.weight,
model.layers.23.mlp.down_proj.biases,
model.layers.23.mlp.down_proj.scales,
model.layers.23.mlp.down_proj.weight,
model.layers.23.mlp.gate_proj.biases,
model.layers.23.mlp.gate_proj.scales,
model.layers.23.mlp.gate_proj.weight,
model.layers.23.mlp.up_proj.biases,
model.layers.23.mlp.up_proj.scales,
model.layers.23.mlp.up_proj.weight,
model.layers.23.post_attention_layernorm.weight,
model.layers.23.self_attn.k_norm.weight,
model.layers.23.self_attn.k_proj.biases,
model.layers.23.self_attn.k_proj.scales,
model.layers.23.self_attn.k_proj.weight,
model.layers.23.self_attn.o_proj.biases,
model.layers.23.self_attn.o_proj.scales,
model.layers.23.self_attn.o_proj.weight,
model.layers.23.self_attn.q_norm.weight,
model.layers.23.self_attn.q_proj.biases,
model.layers.23.self_attn.q_proj.scales,
model.layers.23.self_attn.q_proj.weight,
model.layers.23.self_attn.v_proj.biases,
model.layers.23.self_attn.v_proj.scales,
model.layers.23.self_attn.v_proj.weight,
model.layers.24.input_layernorm.weight,
model.layers.24.mlp.down_proj.biases,
model.layers.24.mlp.down_proj.scales,
model.layers.24.mlp.down_proj.weight,
model.layers.24.mlp.gate_proj.biases,
model.layers.24.mlp.gate_proj.scales,
model.layers.24.mlp.gate_proj.weight,
model.layers.24.mlp.up_proj.biases,
model.layers.24.mlp.up_proj.scales,
model.layers.24.mlp.up_proj.weight,
model.layers.24.post_attention_layernorm.weight,
model.layers.24.self_attn.k_norm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.o_proj.biases,
model.layers.24.self_attn.o_proj.scales,
model.layers.24.self_attn.o_proj.weight,
model.layers.24.self_attn.q_norm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.input_layernorm.weight,
model.layers.25.mlp.down_proj.biases,
model.layers.25.mlp.down_proj.scales,
model.layers.25.mlp.down_proj.weight,
model.layers.25.mlp.gate_proj.biases,
model.layers.25.mlp.gate_proj.scales,
model.layers.25.mlp.gate_proj.weight,
model.layers.25.mlp.up_proj.biases,
model.layers.25.mlp.up_proj.scales,
model.layers.25.mlp.up_proj.weight,
model.layers.25.post_attention_layernorm.weight,
model.layers.25.self_attn.k_norm.weight,
model.layers.25.self_attn.k_proj.biases,
model.layers.25.self_attn.k_proj.scales,
model.layers.25.self_attn.k_proj.weight,
model.layers.25.self_attn.o_proj.biases,
model.layers.25.self_attn.o_proj.scales,
model.layers.25.self_attn.o_proj.weight,
model.layers.25.self_attn.q_norm.weight,
model.layers.25.self_attn.q_proj.biases,
model.layers.25.self_attn.q_proj.scales,
model.layers.25.self_attn.q_proj.weight,
model.layers.25.self_attn.v_proj.biases,
model.layers.25.self_attn.v_proj.scales,
model.layers.25.self_attn.v_proj.weight,
model.layers.26.input_layernorm.weight,
model.layers.26.mlp.down_proj.biases,
model.layers.26.mlp.down_proj.scales,
model.layers.26.mlp.down_proj.weight,
model.layers.26.mlp.gate_proj.biases,
model.layers.26.mlp.gate_proj.scales,
model.layers.26.mlp.gate_proj.weight,
model.layers.26.mlp.up_proj.biases,
model.layers.26.mlp.up_proj.scales,
model.layers.26.mlp.up_proj.weight,
model.layers.26.post_attention_layernorm.weight,
model.layers.26.self_attn.k_norm.weight,
model.layers.26.self_attn.k_proj.biases,
model.layers.26.self_attn.k_proj.scales,
model.layers.26.self_attn.k_proj.weight,
model.layers.26.self_attn.o_proj.biases,
model.layers.26.self_attn.o_proj.scales,
model.layers.26.self_attn.o_proj.weight,
model.layers.26.self_attn.q_norm.weight,
model.layers.26.self_attn.q_proj.biases,
model.layers.26.self_attn.q_proj.scales,
model.layers.26.self_attn.q_proj.weight,
model.layers.26.self_attn.v_proj.biases,
model.layers.26.self_attn.v_proj.scales,
model.layers.26.self_attn.v_proj.weight,
model.layers.27.input_layernorm.weight,
model.layers.27.mlp.down_proj.biases,
model.layers.27.mlp.down_proj.scales,
model.layers.27.mlp.down_proj.weight,
model.layers.27.mlp.gate_proj.biases,
model.layers.27.mlp.gate_proj.scales,
model.layers.27.mlp.gate_proj.weight,
model.layers.27.mlp.up_proj.biases,
model.layers.27.mlp.up_proj.scales,
model.layers.27.mlp.up_proj.weight,
model.layers.27.post_attention_layernorm.weight,
model.layers.27.self_attn.k_norm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.o_proj.biases,
model.layers.27.self_attn.o_proj.scales,
model.layers.27.self_attn.o_proj.weight,
model.layers.27.self_attn.q_norm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.input_layernorm.weight,
model.layers.28.mlp.down_proj.biases,
model.layers.28.mlp.down_proj.scales,
model.layers.28.mlp.down_proj.weight,
model.layers.28.mlp.gate_proj.biases,
model.layers.28.mlp.gate_proj.scales,
model.layers.28.mlp.gate_proj.weight,
model.layers.28.mlp.up_proj.biases,
model.layers.28.mlp.up_proj.scales,
model.layers.28.mlp.up_proj.weight,
model.layers.28.post_attention_layernorm.weight,
model.layers.28.self_attn.k_norm.weight,
model.layers.28.self_attn.k_proj.biases,
model.layers.28.self_attn.k_proj.scales,
model.layers.28.self_attn.k_proj.weight,
model.layers.28.self_attn.o_proj.biases,
model.layers.28.self_attn.o_proj.scales,
model.layers.28.self_attn.o_proj.weight,
model.layers.28.self_attn.q_norm.weight,
model.layers.28.self_attn.q_proj.biases,
model.layers.28.self_attn.q_proj.scales,
model.layers.28.self_attn.q_proj.weight,
model.layers.28.self_attn.v_proj.biases,
model.layers.28.self_attn.v_proj.scales,
model.layers.28.self_attn.v_proj.weight,
model.layers.29.input_layernorm.weight,
model.layers.29.mlp.down_proj.biases,
model.layers.29.mlp.down_proj.scales,
model.layers.29.mlp.down_proj.weight,
model.layers.29.mlp.gate_proj.biases,
model.layers.29.mlp.gate_proj.scales,
model.layers.29.mlp.gate_proj.weight,
model.layers.29.mlp.up_proj.biases,
model.layers.29.mlp.up_proj.scales,
model.layers.29.mlp.up_proj.weight,
model.layers.29.post_attention_layernorm.weight,
model.layers.29.self_attn.k_norm.weight,
model.layers.29.self_attn.k_proj.biases,
model.layers.29.self_attn.k_proj.scales,
model.layers.29.self_attn.k_proj.weight,
model.layers.29.self_attn.o_proj.biases,
model.layers.29.self_attn.o_proj.scales,
model.layers.29.self_attn.o_proj.weight,
model.layers.29.self_attn.q_norm.weight,
model.layers.29.self_attn.q_proj.biases,
model.layers.29.self_attn.q_proj.scales,
model.layers.29.self_attn.q_proj.weight,
model.layers.29.self_attn.v_proj.biases,
model.layers.29.self_attn.v_proj.scales,
model.layers.29.self_attn.v_proj.weight,
model.layers.3.input_layernorm.weight,
model.layers.3.mlp.down_proj.biases,
model.layers.3.mlp.down_proj.scales,
model.layers.3.mlp.down_proj.weight,
model.layers.3.mlp.gate_proj.biases,
model.layers.3.mlp.gate_proj.scales,
model.layers.3.mlp.gate_proj.weight,
model.layers.3.mlp.up_proj.biases,
model.layers.3.mlp.up_proj.scales,
model.layers.3.mlp.up_proj.weight,
model.layers.3.post_attention_layernorm.weight,
model.layers.3.self_attn.k_norm.weight,
model.layers.3.self_attn.k_proj.biases,
model.layers.3.self_attn.k_proj.scales,
model.layers.3.self_attn.k_proj.weight,
model.layers.3.self_attn.o_proj.biases,
model.layers.3.self_attn.o_proj.scales,
model.layers.3.self_attn.o_proj.weight,
model.layers.3.self_attn.q_norm.weight,
model.layers.3.self_attn.q_proj.biases,
model.layers.3.self_attn.q_proj.scales,
model.layers.3.self_attn.q_proj.weight,
model.layers.3.self_attn.v_proj.biases,
model.layers.3.self_attn.v_proj.scales,
model.layers.3.self_attn.v_proj.weight,
model.layers.30.input_layernorm.weight,
model.layers.30.mlp.down_proj.biases,
model.layers.30.mlp.down_proj.scales,
model.layers.30.mlp.down_proj.weight,
model.layers.30.mlp.gate_proj.biases,
model.layers.30.mlp.gate_proj.scales,
model.layers.30.mlp.gate_proj.weight,
model.layers.30.mlp.up_proj.biases,
model.layers.30.mlp.up_proj.scales,
model.layers.30.mlp.up_proj.weight,
model.layers.30.post_attention_layernorm.weight,
model.layers.30.self_attn.k_norm.weight,
model.layers.30.self_attn.k_proj.biases,
model.layers.30.self_attn.k_proj.scales,
model.layers.30.self_attn.k_proj.weight,
model.layers.30.self_attn.o_proj.biases,
model.layers.30.self_attn.o_proj.scales,
model.layers.30.self_attn.o_proj.weight,
model.layers.30.self_attn.q_norm.weight,
model.layers.30.self_attn.q_proj.biases,
model.layers.30.self_attn.q_proj.scales,
model.layers.30.self_attn.q_proj.weight,
model.layers.30.self_attn.v_proj.biases,
model.layers.30.self_attn.v_proj.scales,
model.layers.30.self_attn.v_proj.weight,
model.layers.31.input_layernorm.weight,
model.layers.31.mlp.down_proj.biases,
model.layers.31.mlp.down_proj.scales,
model.layers.31.mlp.down_proj.weight,
model.layers.31.mlp.gate_proj.biases,
model.layers.31.mlp.gate_proj.scales,
model.layers.31.mlp.gate_proj.weight,
model.layers.31.mlp.up_proj.biases,
model.layers.31.mlp.up_proj.scales,
model.layers.31.mlp.up_proj.weight,
model.layers.31.post_attention_layernorm.weight,
model.layers.31.self_attn.k_norm.weight,
model.layers.31.self_attn.k_proj.biases,
model.layers.31.self_attn.k_proj.scales,
model.layers.31.self_attn.k_proj.weight,
model.layers.31.self_attn.o_proj.biases,
model.layers.31.self_attn.o_proj.scales,
model.layers.31.self_attn.o_proj.weight,
model.layers.31.self_attn.q_norm.weight,
model.layers.31.self_attn.q_proj.biases,
model.layers.31.self_attn.q_proj.scales,
model.layers.31.self_attn.q_proj.weight,
model.layers.31.self_attn.v_proj.biases,
model.layers.31.self_attn.v_proj.scales,
model.layers.31.self_attn.v_proj.weight,
model.layers.32.input_layernorm.weight,
model.layers.32.mlp.down_proj.biases,
model.layers.32.mlp.down_proj.scales,
model.layers.32.mlp.down_proj.weight,
model.layers.32.mlp.gate_proj.biases,
model.layers.32.mlp.gate_proj.scales,
model.layers.32.mlp.gate_proj.weight,
model.layers.32.mlp.up_proj.biases,
model.layers.32.mlp.up_proj.scales,
model.layers.32.mlp.up_proj.weight,
model.layers.32.post_attention_layernorm.weight,
model.layers.32.self_attn.k_norm.weight,
model.layers.32.self_attn.k_proj.biases,
model.layers.32.self_attn.k_proj.scales,
model.layers.32.self_attn.k_proj.weight,
model.layers.32.self_attn.o_proj.biases,
model.layers.32.self_attn.o_proj.scales,
model.layers.32.self_attn.o_proj.weight,
model.layers.32.self_attn.q_norm.weight,
model.layers.32.self_attn.q_proj.biases,
model.layers.32.self_attn.q_proj.scales,
model.layers.32.self_attn.q_proj.weight,
model.layers.32.self_attn.v_proj.biases,
model.layers.32.self_attn.v_proj.scales,
model.layers.32.self_attn.v_proj.weight,
model.layers.33.input_layernorm.weight,
model.layers.33.mlp.down_proj.biases,
model.layers.33.mlp.down_proj.scales,
model.layers.33.mlp.down_proj.weight,
model.layers.33.mlp.gate_proj.biases,
model.layers.33.mlp.gate_proj.scales,
model.layers.33.mlp.gate_proj.weight,
model.layers.33.mlp.up_proj.biases,
model.layers.33.mlp.up_proj.scales,
model.layers.33.mlp.up_proj.weight,
model.layers.33.post_attention_layernorm.weight,
model.layers.33.self_attn.k_norm.weight,
model.layers.33.self_attn.k_proj.biases,
model.layers.33.self_attn.k_proj.scales,
model.layers.33.self_attn.k_proj.weight,
model.layers.33.self_attn.o_proj.biases,
model.layers.33.self_attn.o_proj.scales,
model.layers.33.self_attn.o_proj.weight,
model.layers.33.self_attn.q_norm.weight,
model.layers.33.self_attn.q_proj.biases,
model.layers.33.self_attn.q_proj.scales,
model.layers.33.self_attn.q_proj.weight,
model.layers.33.self_attn.v_proj.biases,
model.layers.33.self_attn.v_proj.scales,
model.layers.33.self_attn.v_proj.weight,
model.layers.34.input_layernorm.weight,
model.layers.34.mlp.down_proj.biases,
model.layers.34.mlp.down_proj.scales,
model.layers.34.mlp.down_proj.weight,
model.layers.34.mlp.gate_proj.biases,
model.layers.34.mlp.gate_proj.scales,
model.layers.34.mlp.gate_proj.weight,
model.layers.34.mlp.up_proj.biases,
model.layers.34.mlp.up_proj.scales,
model.layers.34.mlp.up_proj.weight,
model.layers.34.post_attention_layernorm.weight,
model.layers.34.self_attn.k_norm.weight,
model.layers.34.self_attn.k_proj.biases,
model.layers.34.self_attn.k_proj.scales,
model.layers.34.self_attn.k_proj.weight,
model.layers.34.self_attn.o_proj.biases,
model.layers.34.self_attn.o_proj.scales,
model.layers.34.self_attn.o_proj.weight,
model.layers.34.self_attn.q_norm.weight,
model.layers.34.self_attn.q_proj.biases,
model.layers.34.self_attn.q_proj.scales,
model.layers.34.self_attn.q_proj.weight,
model.layers.34.self_attn.v_proj.biases,
model.layers.34.self_attn.v_proj.scales,
model.layers.34.self_attn.v_proj.weight,
model.layers.35.input_layernorm.weight,
model.layers.35.mlp.down_proj.biases,
model.layers.35.mlp.down_proj.scales,
model.layers.35.mlp.down_proj.weight,
model.layers.35.mlp.gate_proj.biases,
model.layers.35.mlp.gate_proj.scales,
model.layers.35.mlp.gate_proj.weight,
model.layers.35.mlp.up_proj.biases,
model.layers.35.mlp.up_proj.scales,
model.layers.35.mlp.up_proj.weight,
model.layers.35.post_attention_layernorm.weight,
model.layers.35.self_attn.k_norm.weight,
model.layers.35.self_attn.k_proj.biases,
model.layers.35.self_attn.k_proj.scales,
model.layers.35.self_attn.k_proj.weight,
model.layers.35.self_attn.o_proj.biases,
model.layers.35.self_attn.o_proj.scales,
model.layers.35.self_attn.o_proj.weight,
model.layers.35.self_attn.q_norm.weight,
model.layers.35.self_attn.q_proj.biases,
model.layers.35.self_attn.q_proj.scales,
model.layers.35.self_attn.q_proj.weight,
model.layers.35.self_attn.v_proj.biases,
model.layers.35.self_attn.v_proj.scales,
model.layers.35.self_attn.v_proj.weight,
model.layers.4.input_layernorm.weight,
model.layers.4.mlp.down_proj.biases,
model.layers.4.mlp.down_proj.scales,
model.layers.4.mlp.down_proj.weight,
model.layers.4.mlp.gate_proj.biases,
model.layers.4.mlp.gate_proj.scales,
model.layers.4.mlp.gate_proj.weight,
model.layers.4.mlp.up_proj.biases,
model.layers.4.mlp.up_proj.scales,
model.layers.4.mlp.up_proj.weight,
model.layers.4.post_attention_layernorm.weight,
model.layers.4.self_attn.k_norm.weight,
model.layers.4.self_attn.k_proj.biases,
model.layers.4.self_attn.k_proj.scales,
model.layers.4.self_attn.k_proj.weight,
model.layers.4.self_attn.o_proj.biases,
model.layers.4.self_attn.o_proj.scales,
model.layers.4.self_attn.o_proj.weight,
model.layers.4.self_attn.q_norm.weight,
model.layers.4.self_attn.q_proj.biases,
model.layers.4.self_attn.q_proj.scales,
model.layers.4.self_attn.q_proj.weight,
model.layers.4.self_attn.v_proj.biases,
model.layers.4.self_attn.v_proj.scales,
model.layers.4.self_attn.v_proj.weight,
model.layers.5.input_layernorm.weight,
model.layers.5.mlp.down_proj.biases,
model.layers.5.mlp.down_proj.scales,
model.layers.5.mlp.down_proj.weight,
model.layers.5.mlp.gate_proj.biases,
model.layers.5.mlp.gate_proj.scales,
model.layers.5.mlp.gate_proj.weight,
model.layers.5.mlp.up_proj.biases,
model.layers.5.mlp.up_proj.scales,
model.layers.5.mlp.up_proj.weight,
model.layers.5.post_attention_layernorm.weight,
model.layers.5.self_attn.k_norm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.o_proj.biases,
model.layers.5.self_attn.o_proj.scales,
model.layers.5.self_attn.o_proj.weight,
model.layers.5.self_attn.q_norm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.input_layernorm.weight,
model.layers.6.mlp.down_proj.biases,
model.layers.6.mlp.down_proj.scales,
model.layers.6.mlp.down_proj.weight,
model.layers.6.mlp.gate_proj.biases,
model.layers.6.mlp.gate_proj.scales,
model.layers.6.mlp.gate_proj.weight,
model.layers.6.mlp.up_proj.biases,
model.layers.6.mlp.up_proj.scales,
model.layers.6.mlp.up_proj.weight,
model.layers.6.post_attention_layernorm.weight,
model.layers.6.self_attn.k_norm.weight,
model.layers.6.self_attn.k_proj.biases,
model.layers.6.self_attn.k_proj.scales,
model.layers.6.self_attn.k_proj.weight,
model.layers.6.self_attn.o_proj.biases,
model.layers.6.self_attn.o_proj.scales,
model.layers.6.self_attn.o_proj.weight,
model.layers.6.self_attn.q_norm.weight,
model.layers.6.self_attn.q_proj.biases,
model.layers.6.self_attn.q_proj.scales,
model.layers.6.self_attn.q_proj.weight,
model.layers.6.self_attn.v_proj.biases,
model.layers.6.self_attn.v_proj.scales,
model.layers.6.self_attn.v_proj.weight,
model.layers.7.input_layernorm.weight,
model.layers.7.mlp.down_proj.biases,
model.layers.7.mlp.down_proj.scales,
model.layers.7.mlp.down_proj.weight,
model.layers.7.mlp.gate_proj.biases,
model.layers.7.mlp.gate_proj.scales,
model.layers.7.mlp.gate_proj.weight,
model.layers.7.mlp.up_proj.biases,
model.layers.7.mlp.up_proj.scales,
model.layers.7.mlp.up_proj.weight,
model.layers.7.post_attention_layernorm.weight,
model.layers.7.self_attn.k_norm.weight,
model.layers.7.self_attn.k_proj.biases,
model.layers.7.self_attn.k_proj.scales,
model.layers.7.self_attn.k_proj.weight,
model.layers.7.self_attn.o_proj.biases,
model.layers.7.self_attn.o_proj.scales,
model.layers.7.self_attn.o_proj.weight,
model.layers.7.self_attn.q_norm.weight,
model.layers.7.self_attn.q_proj.biases,
model.layers.7.self_attn.q_proj.scales,
model.layers.7.self_attn.q_proj.weight,
model.layers.7.self_attn.v_proj.biases,
model.layers.7.self_attn.v_proj.scales,
model.layers.7.self_attn.v_proj.weight,
model.layers.8.input_layernorm.weight,
model.layers.8.mlp.down_proj.biases,
model.layers.8.mlp.down_proj.scales,
model.layers.8.mlp.down_proj.weight,
model.layers.8.mlp.gate_proj.biases,
model.layers.8.mlp.gate_proj.scales,
model.layers.8.mlp.gate_proj.weight,
model.layers.8.mlp.up_proj.biases,
model.layers.8.mlp.up_proj.scales,
model.layers.8.mlp.up_proj.weight,
model.layers.8.post_attention_layernorm.weight,
model.layers.8.self_attn.k_norm.weight,
model.layers.8.self_attn.k_proj.biases,
model.layers.8.self_attn.k_proj.scales,
model.layers.8.self_attn.k_proj.weight,
model.layers.8.self_attn.o_proj.biases,
model.layers.8.self_attn.o_proj.scales,
model.layers.8.self_attn.o_proj.weight,
model.layers.8.self_attn.q_norm.weight,
model.layers.8.self_attn.q_proj.biases,
model.layers.8.self_attn.q_proj.scales,
model.layers.8.self_attn.q_proj.weight,
model.layers.8.self_attn.v_proj.biases,
model.layers.8.self_attn.v_proj.scales,
model.layers.8.self_attn.v_proj.weight,
model.layers.9.input_layernorm.weight,
model.layers.9.mlp.down_proj.biases,
model.layers.9.mlp.down_proj.scales,
model.layers.9.mlp.down_proj.weight,
model.layers.9.mlp.gate_proj.biases,
model.layers.9.mlp.gate_proj.scales,
model.layers.9.mlp.gate_proj.weight,
model.layers.9.mlp.up_proj.biases,
model.layers.9.mlp.up_proj.scales,
model.layers.9.mlp.up_proj.weight,
model.layers.9.post_attention_layernorm.weight,
model.layers.9.self_attn.k_norm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.o_proj.biases,
model.layers.9.self_attn.o_proj.scales,
model.layers.9.self_attn.o_proj.weight,
model.layers.9.self_attn.q_norm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight,
model.norm.weight.

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 3483.32it/s]
```

</details>

---
