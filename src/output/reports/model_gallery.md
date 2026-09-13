# Model Output Gallery

Generated on: 2026-09-13 21:59:50 BST

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

| Model                                                                                                                     | Mechanical checks      | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                      |
|---------------------------------------------------------------------------------------------------------------------------|------------------------|---------|---------------------|-----------------|---------|------------|---------|---------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)   | `no concerns detected` | 11.48s  | 29.5 tok/s          | 3.55            | 23      | 2,393      | 151     | none                                                                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)              | `no concerns detected` | 16.26s  | 78.8 tok/s          | 1.36            | 19      | 1,634      | 997     | none                                                                                              |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                            | `no concerns detected` | 9.62s   | 74.9 tok/s          | 5.69            | 8.7     | 6,450      | 140     | none                                                                                              |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                      | `no concerns detected` | 21.87s  | 41.1 tok/s          | 12.56           | 78      | 6,450      | 103     | none                                                                                              |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                   | `no concerns detected` | 9.01s   | 32.3 tok/s          | 1.87            | 18      | 2,612      | 138     | none                                                                                              |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                             | `no concerns detected` | 6.41s   | 55.4 tok/s          | 2.31            | 10      | 2,113      | 126     | none                                                                                              |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                               | `no concerns detected` | 6.39s   | 31.4 tok/s          | 1.60            | 17      | 2,113      | 77      | none                                                                                              |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                    | `no concerns detected` | 3.15s   | 206 tok/s           | 0.99            | 4.0     | 2,103      | 80      | none                                                                                              |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                              | `no concerns detected` | 3.40s   | 104 tok/s           | 0.73            | 7.0     | 391        | 101     | none                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)           | `no concerns detected` | 6.58s   | 64.7 tok/s          | 2.67            | 13      | 2,926      | 109     | none                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)           | `no concerns detected` | 8.50s   | 53.2 tok/s          | 2.83            | 13      | 2,926      | 187     | none                                                                                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)               | `no concerns detected` | 3.90s   | 181 tok/s           | 1.60            | 7.8     | 2,925      | 102     | none                                                                                              |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                 | `no concerns detected` | 5.00s   | 166 tok/s           | 2.74            | 3.9     | 4,083      | 84      | none                                                                                              |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                        | `no concerns detected` | 6.58s   | 75.7 tok/s          | 1.25            | 24      | 1,289      | 155     | none                                                                                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                          | `no concerns detected` | 4.20s   | 55.0 tok/s          | 0.79            | 9.3     | 1,133      | 107     | none                                                                                              |
| [`mlx-community/Qwen2-VL-7B-Instruct-4bit`](#model-mlx-community-qwen2-vl-7b-instruct-4bit)                               | `no concerns detected` | 45.84s  | 90.6 tok/s          | 42.87           | 9.3     | 16,558     | 117     | none                                                                                              |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                               | `no concerns detected` | 29.64s  | 85.9 tok/s          | 17.28           | 8.4     | 16,549     | 912     | none                                                                                              |
| [`mlx-community/Qwen3-VL-32B-Instruct-4bit`](#model-mlx-community-qwen3-vl-32b-instruct-4bit)                             | `no concerns detected` | 76.75s  | 18.6 tok/s          | 64.23           | 26      | 16,547     | 177     | none                                                                                              |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                               | `no concerns detected` | 42.99s  | 70.5 tok/s          | 39.76           | 11      | 16,547     | 93      | none                                                                                              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                          | `no concerns detected` | 38.97s  | 72.5 tok/s          | 33.87           | 25      | 16,563     | 97      | none                                                                                              |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                            | `no concerns detected` | 40.15s  | 89.0 tok/s          | 36.56           | 11      | 16,563     | 86      | none                                                                                              |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                  | `no concerns detected` | 62.11s  | 27.5 tok/s          | 54.48           | 21      | 16,563     | 124     | none                                                                                              |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                              | `no concerns detected` | 3.32s   | 125 tok/s           | 1.10            | 5.6     | 1,426      | 98      | none                                                                                              |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                            | `no concerns detected` | 40.68s  | 47.3 tok/s          | 23.62           | 92      | 3,491      | 121     | none                                                                                              |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                             | `no concerns detected` | 18.87s  | 58.1 tok/s          | 14.41           | 14      | 16,558     | 136     | none                                                                                              |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                   | `no concerns detected` | 9.38s   | 30.7 tok/s          | 1.48            | 17      | 590        | 152     | none                                                                                              |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                           | `no concerns detected` | 5.33s   | 60.7 tok/s          | 1.12            | 7.6     | 595        | 101     | none                                                                                              |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                   | `no concerns detected` | 5.00s   | 104 tok/s           | 1.00            | 16      | 595        | 96      | none                                                                                              |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                           | `no concerns detected` | 8.54s   | 25.8 tok/s          | 1.76            | 20      | 595        | 89      | none                                                                                              |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                           | `no concerns detected` | 3.67s   | 123 tok/s           | 0.76            | 5.9     | 591        | 73      | none                                                                                              |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                              | `no concerns detected` | 3.00s   | 172 tok/s           | 1.17            | 4.7     | 1,379      | 65      | none                                                                                              |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                 | `no concerns detected` | 7.61s   | 38.4 tok/s          | 2.54            | 16      | 3,116      | 102     | none                                                                                              |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                              | `concerns detected`    | 2.08s   | 475 tok/s           | 0.61            | 1.9     | 2,112      | 120     | duplicate keywords                                                                                |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                     | `concerns detected`    | 6.44s   | 68.9 tok/s          | 1.41            | 8.1     | 1,524      | 216     | duplicate keywords                                                                                |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                             | `concerns detected`    | 4.58s   | 102 tok/s           | 1.58            | 6.5     | 2,090      | 96      | control tokens visible                                                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                   | `concerns detected`    | 7.22s   | 43.6 tok/s          | 2.94            | 28      | 591        | 80      | duplicate keywords                                                                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                     | `major concerns`       | 22.86s  | 54.5 tok/s          | 1.48            | 20      | 1,324      | 1,000   | cut off at token limit                                                                            |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                           | `major concerns`       | 61.04s  | 21.4 tok/s          | 9.63            | 25      | 4,403      | 1,000   | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                     | `major concerns`       | 44.51s  | 74.4 tok/s          | 38.88           | 23      | 16,547     | 200     | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                             | `major concerns`       | 2.35s   | 315 tok/s           | 1.03            | 1.1     | 1,205      | 33      | labelled fields not detected                                                                      |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                         | `major concerns`       | 4.47s   | 70.5 tok/s          | 0.85            | 7.2     | 589        | 88      | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                   | `major concerns`       | 5.63s   | 99.5 tok/s          | 1.67            | 6.7     | 2,195      | 200     | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected     |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                              | `major concerns`       | 2.27s   | 202 tok/s           | 0.67            | 1.8     | 330        | 92      | labelled fields not detected                                                                      |
| [`mlx-community/paligemma2-10b-mix-448-4bit`](#model-mlx-community-paligemma2-10b-mix-448-4bit)                           | `major concerns`       | 4.73s   | insufficient sample | 2.27            | 9.7     | 1,344      | 7       | labelled fields not detected                                                                      |
| [`apple/FastVLM-7B-int4`](#model-apple-fastvlm-7b-int4)                                                                   | `not assessed`         | 0.14s   | -                   | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                            | `not assessed`         | 0.14s   | -                   | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)              | `not assessed`         | 2.49s   | -                   | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)              | `not assessed`         | 2.88s   | -                   | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                             | `not assessed`         | 0.25s   | -                   | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-mlx-community-mistral-small-32-24b-instruct-2506-4bit)  | `not assessed`         | 1.85s   | -                   | -               | -       | -          | -       | none                                                                                              |
| [`mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit`](#model-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) | `not assessed`         | 0.29s   | -                   | -               | -       | -          | -       | none                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `mlx-community/granite-4.0-3b-vision-4bit` at 3.00s

Lowest peak memory among completions without detected concerns: `mlx-community/North-Micro-Vision-Instruct-4bit` at 3.9 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                     | Mechanical checks | Observations                                                                                      |
|---------------------------------------------------------------------------------------------------------------------------|-------------------|---------------------------------------------------------------------------------------------------|
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                     | `major concerns`  | cut off at token limit                                                                            |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                           | `major concerns`  | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                     | `major concerns`  | stopped early: repeating; duplicate keywords                                                      |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                             | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                         | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                   | `major concerns`  | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected     |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                              | `major concerns`  | labelled fields not detected                                                                      |
| [`mlx-community/paligemma2-10b-mix-448-4bit`](#model-mlx-community-paligemma2-10b-mix-448-4bit)                           | `major concerns`  | labelled fields not detected                                                                      |
| [`apple/FastVLM-7B-int4`](#model-apple-fastvlm-7b-int4)                                                                   | `not assessed`    | none                                                                                              |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                            | `not assessed`    | none                                                                                              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)              | `not assessed`    | none                                                                                              |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)              | `not assessed`    | none                                                                                              |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                             | `not assessed`    | none                                                                                              |
| [`mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-mlx-community-mistral-small-32-24b-instruct-2506-4bit)  | `not assessed`    | none                                                                                              |
| [`mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit`](#model-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) | `not assessed`    | none                                                                                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

A compact preview of each model's final answer (or failure evidence for crashes), in chooser order. Where the requested catalogue fields were detected, the preview shows a little of each: the title, the start of the description, and the first keywords with their count, so the weakest field is not hidden behind a long description. Otherwise it is the first 280 characters. A closed reasoning trace is left out of the preview and reported as an omitted-character count; the complete output, trace included, is in the model's evidence block below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                     | Mechanical checks      | Output preview                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit)   | `no concerns detected` | Title: Swan on a calm river with moored boats \| Description: A solitary white swan swims gracefully on a calm river, with leisure boats and cruisers moored alongside r... \| Keywords (20): Swan, river, boats, mooring, residential buildings, greenery, water reflection, aquatic bird, marina, ...                                                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)              | `no concerns detected` | Title: Solitary White Swan Glides on Calm River with Boats and Lush Foliage \| Description: On 12 September 2026 at 17:41 UTC+01 (52.393850°N, 0.270830°E), a solitary wh... \| Keywords (17): White Swan, River, Boat, Marina, Greenery, Canal, Riverbank, Waterfowl, Lush Foliage, Leisure Boats, ...[2,490 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                            | `no concerns detected` | Title: Swan at River Marina \| Description: A white swan glides on calm river waters, with leisure boats moored alongside riverside residential buildings in the... \| Keywords (13): Swan, River, Marina, Leisure boats, Residential buildings, Calm waters, White swan, Riverbank, Boats moored, ...                                                                                |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                      | `no concerns detected` | Title: Solitary swan glides on calm river waters \| Description: A solitary white swan glides gracefully across calm river waters, framed by lush foliage, with leisu... \| Keywords (17): swan, river, water, foliage, boats, cruisers, residential buildings, moored, overcast, water reflection, ...                                                                               |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                   | `no concerns detected` | Title: Solitary Swan on Calm River Waters with Moored Boats and Riverside Buildings. \| Description: A solitary white swan glides gracefully across the calm river wate... \| Keywords (20): swan, river, boats, moored, residential buildings, foliage, calm waters, leisure, cruisers, riverside, ...                                                                               |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                             | `no concerns detected` | Title: Swan on Canal with Moored Boats \| Description: A white swan glides on a tranquil canal, surrounded by moored boats and riverside buildings, with lush greenery... \| Keywords (19): Swan, Canal, Boats, Mooring, Residential Buildings, Greenery, Trees, Water Reflection, Motorboat, Pier, ...                                                                               |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                               | `no concerns detected` | Title: Swan on Calm River \| Description: A white swan glides on tranquil river waters near a marina with moored boats and residential buildings, captured at dusk. \| Keywords (15): swan, river, marina, boats, residential buildings, foliage, trees, water reflection, architecture, motorboat, ...                                                                               |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                    | `no concerns detected` | Title: Swan gliding on river near moored boats and houses \| Description: A white swan swims gracefully on calm river waters framed by lush greenery, with leisure boats... \| Keywords (14): swan, river, boats, houses, greenery, water, reflection, leisure, architecture, waterfront, waterway, ...                                                                               |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                              | `no concerns detected` | Title: Swan Glides on Calm River Near Moored Boats \| Description: A white swan swims peacefully in a tranquil waterway, framed by greenery and moored leisure bo... \| Keywords (16): swan, river, waterway, mooring, motorboat, boats, greenery, trees, vegetation, waterfront, water reflection, ...                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)           | `no concerns detected` | Title: **Swan Gliding Past Moored Boats on a Serene River** \| Description: A solitary white swan swims calmly on a tranquil river, surrounded by reflections of... \| Keywords (16): swan, river, moored boats, marina, residential waterfront, greenery, calm water, reflections, leisure boats, ...                                                                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)           | `no concerns detected` | Title: *Elegant Swan Glides Past Moored Boats on Serene River* \| Description: A solitary white swan gracefully swims across calm river waters on **September 12, 2026,... \| Keywords (20): swan, river, leisure boats, moored boats, riverside residential buildings, greenery, golden sunlight, ...                                                                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)               | `no concerns detected` | Title: Swans by Riverside Marina \| Description: A white swan swims serenely in calm waters at dusk near a marina moored with leisure boats, framed by lush greenery... \| Keywords (17): swan, canal, marina, mooring, motorboat, pier, waterfront, waterway, greenery, residential architecture, ...                                                                                |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                 | `no concerns detected` | Title: Swan on the River \| Description: A solitary white swan glides gracefully across calm river waters, framed by lush greenery. In the background, leisure boa... \| Keywords (13): Swan, River, Greenery, Marina, Boat, Waterfront, Water, Aquatic Bird, Architecture, Leisure Boats, Foliage, ...                                                                               |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                        | `no concerns detected` | Title: White Swan Glides Past Moored Boats On Calm River \| Description: A solitary white swan glades across a calm river in the foreground, its reflection ripplin... \| Keywords (19): white swan, river, moored boats, motorboats, cruisers, marina, pontoon, water reflection, swan, waterfowl, ...                                                                               |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                          | `no concerns detected` | Title: Serene Swan on the River \| Description: A white swan is seen swimming peacefully on a calm river, with boats and houses in the background, captured on S... \| Keywords (18): swan, river, calm, boats, houses, foliage, waterfront, mooring, leisure, tranquility, architecture, waterway, ...                                                                               |
| [`mlx-community/Qwen2-VL-7B-Instruct-4bit`](#model-mlx-community-qwen2-vl-7b-instruct-4bit)                               | `no concerns detected` | Title: Graceful Swan on the River \| Description: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruis... \| Keywords (26): swan, river, calm water, lush foliage, leisure boats, cruisers, riverside buildings, residential area, ...                                                                               |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                               | `no concerns detected` | Title: Solitary Swan on Riverfront Marina \| Description: A solitary white swan glides across calm river waters in a canal with moored leisure boats and reside... \| Keywords (15): Canal, Riverbank, Marina, Mooring, Motorboat, Pier, Boat, Architecture, Waterway, Swimming, Trees, Vegetation, ...[2,859 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/Qwen3-VL-32B-Instruct-4bit`](#model-mlx-community-qwen3-vl-32b-instruct-4bit)                             | `no concerns detected` | Title: Swan Gliding on River Near Moored Boats \| Description: A solitary white swan glides across calm river waters, framed by overhanging green foliage, with se... \| Keywords (18): swan, river, moored boats, leisure craft, residential buildings, greenery, water reflection, foliage, pier, ...                                                                               |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                               | `no concerns detected` | Title: Swan on River with Moored Boats \| Description: A solitary white swan glides calmly on a river, framed by green foliage, with moored leisure boats and resid... \| Keywords (17): swan, river, boats, mooring, canal, greenery, trees, water reflection, waterfront, residential, motorboat, ...                                                                               |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                          | `no concerns detected` | Title: Swan Swimming Past Moored Boats \| Description: A solitary white swan glides across calm river waters near a riverside mooring of leisure cruisers and resid... \| Keywords (18): swan, river, moored boats, riverbank, vegetation, water reflection, cruiser, residential buildings, trees, ...                                                                               |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                            | `no concerns detected` | Title: Swan Glides Past Moored Boats on Calm Riverbank \| Description: A solitary white swan swims peacefully on a reflective river, with leisure boats moored along t... \| Keywords (15): swan, river, boats, mooring, residential buildings, water reflection, foliage, trees, motorboat, canal, ...                                                                               |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                  | `no concerns detected` | Title: Swan swimming near moored boats on the River Thames \| Description: A solitary white swan swims on the calm water of the River Thames at Lambeth, with its re... \| Keywords (15): Swan, River Thames, Boat, Waterfowl, Water, Reflection, Mooring, Residential building, Foliage, Greenery, ...                                                                               |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                              | `no concerns detected` | Title: Swan on River \| Description: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers mo... \| Keywords (20): Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, ...                                                                               |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                            | `no concerns detected` | Title: Swan glides past moored boats on a calm river \| Description: A solitary white swan swims across calm river waters, framed by lush foreground foliage and... \| Keywords (18): Swan, waterfowl, river, marina, mooring, motorboat, pier, waterfront, waterway, canal, bird, swimming, trees, ...                                                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                             | `no concerns detected` | Title: Swan Gliding on Calm River with Moored Boats \| Description: A serene scene captures a solitary white swan gracefully swimming across calm river waters, frame... \| Keywords (23): Swan, River, Calm, Moored Boats, Residential Buildings, Greenery, Waterfront, Waterway, Motorboat, Pier, ...                                                                               |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                   | `no concerns detected` | Title: Swan on River Cam, Cambridge – September 2026 \| Description: A white swan swims on the River Cam at 52.393850°N, 0.270830°E, captured on 12 September... \| Keywords (18): Swan, River Cam, Cambridge, Waterfowl, Boat, Mooring, Marina, Riverbank, Waterway, Residential building, Trees, ...                                                                                |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                           | `no concerns detected` | Title: White Swan Gliding on a River Near Moored Boats \| Description: A solitary white swan glides across a calm river surface reflecting leisure boats and residential... \| Keywords (19): Swan, Waterfowl, River, Canal, Boat, Motorboat, Marina, Mooring, Architecture, Residential, Waterway, ...                                                                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                   | `no concerns detected` | Title: White swan swimming on calm river with moored boats \| Description: A solitary white swan glides across calm river waters near a wooden pier, with several... \| Keywords (15): white swan, river, motorboat, waterfowl, waterfront, greenery, reflection, residential building, boat, pier, ...                                                                               |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                           | `no concerns detected` | Title: White Swan Swimming Past Moored River Boats \| Description: A single white swan swims across calm river waters in front of leisure boats moored beside rivers... \| Keywords (18): swan, river, boats, marina, moored, waterfront, waterway, residential, architecture, greenery, waterfowl, ...                                                                               |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                           | `no concerns detected` | Title: Swan glides on calm water beneath residential buildings \| Description: A solitary white swan gracefully floats on the calm river water, framed by dense greenery... \| Keywords (16): swan, waterfowl, river, canal, marina, boats, trees, greenery, quiet, architecture, moored, peaceful, ...                                                                               |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                              | `no concerns detected` | Title: "Swan on the Canal" \| Description: A white swan glides gracefully across a calm river, framed by lush foliage and moored boats, with a serene residential backdrop. \| Keywords (10): swan, canal, water, foliage, boats, mooring, residential, reflection, serene, waterway                                                                                                  |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                 | `no concerns detected` | Title: Swan in River with Moored Boats \| Description: A solitary white swan gracefully swims across calm river waters, framed by lush foliage, with leisure boats... \| Keywords (18): Swan, River, Boats, Marina, Water, Foliage, Residential, Mooring, Waterfront, Waterway, Trees, Vegetation, ...                                                                                |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                              | `concerns detected`    | Title: Swans glide across calm river waters in a serene canal setting. \| Description: A solitary white swan glides gracefully across calm river waters framed by lu... \| Keywords (24): Swans, canal, water, serene, leisure, boats, water, architecture, greenery, water, reflection, waterfowl, ...                                                                               |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                     | `concerns detected`    | Title: Serene River Scene with White Swan and Moored Boats \| Description: A tranquil riverside view features a solitary white swan gliding across calm water, surro... \| Keywords (42): swan, river, boats, mooring, canal, greenery, trees, vegetation, water reflection, waterfowl, waterfront, ...                                                                               |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                             | `concerns detected`    | Title: (not detected) \| Description: A serene scene captures a graceful white swan effortlessly gliding across the calm waters of a river, framed by lush greenery... \| Keywords (12): Swan, River, Greenery, Boats, Cruisers, Buildings, Lighting, Afternoon, Tranquil, Waterfowl, Architecture, ...                                                                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                   | `concerns detected`    | Title: White Swan Gliding on River Near Moored Boats \| Description: A solitary white swan glides across calm river waters, with leisure boats and cruisers moored al... \| Keywords (16): swan, waterfowl, river, boat, marina, mooring, motorboat, cruiser, pier, waterfront, waterway, greenery, ...                                                                               |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                     | `major concerns`       | Title: Swan on canal with moored boats at marina \| Description: A white swan glides across a calm canal, surrounded by moored motorboats and residential buildings with lush greenery, at 2026-09-12 17:41:02 UTC+01:00 near 52.393850°N, 0.270830°E. \| Keywords (3): Swans, Canal, Motorboat[3,708 characters of reasoning omitted; complete output in the evidence block]         |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                           | `major concerns`       | Title: (not detected) \| Description: (not detected) \| Keywords (1): We need title 5-10 words. Concrete. Maybe "Mute Swan on River with Moored Boats and Apartments". Count words. Mute(1) Swan(2) on(3) River(4) with(5) Moored(6) Boats(7) and(8) Apartments(9). Good 9 words.                                                                                                     |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                     | `major concerns`       | Title: Solitary swan on a calm river with moored boats \| Description: A white swan glides across the calm waters of a river, framed by green foliage in the foregrou... \| Keywords (57): swan, river, boats, mooring, motorboat, residential, building, balcony, trees, foliage, greenery, water, ...                                                                               |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                             | `major concerns`       | A solitary white swan gliding across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.                                                                                                                                                                                                    |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                         | `major concerns`       | A serene scene unfolds on a calm river, where a graceful swan glides across the water. Several boats, including motorboats and sailboats, are moored along the riverbank, nestled beneath a row of residential buildings. Lush greenery frames the scene, with trees and bushes linin...                                                                                              |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                   | `major concerns`       | <\|channel\|> analysis<\|message\|> The image shows a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a...                                                                                          |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                              | `major concerns`       | "Amidst the Riverbank, a solitary white swan glides gracefully across calm waters, framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background. The serene setting is complemented by the presence of greenery, archit...                                                                                              |
| [`mlx-community/paligemma2-10b-mix-448-4bit`](#model-mlx-community-paligemma2-10b-mix-448-4bit)                           | `major concerns`       | - title, description, keywords                                                                                                                                                                                                                                                                                                                                                        |
| [`apple/FastVLM-7B-int4`](#model-apple-fastvlm-7b-int4)                                                                   | `not assessed`         | Model loading failed: 'vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight'                                                                                                                                                                                                                                                                                            |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                            | `not assessed`         | Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'                                                                                                                                                                                                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-4bit`](#model-mlx-community-llama-32-11b-vision-instruct-4bit)              | `not assessed`         | Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.                                                                                                                                                                                                                       |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)              | `not assessed`         | Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-8bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.                                                                                                                                                                                                                       |
| [`mlx-community/Mage-VL-OptiQ-4bit`](#model-mlx-community-mage-vl-optiq-4bit)                                             | `not assessed`         | Model loading failed: Received 904 parameters not in model:<br>model.embed_tokens.biases,<br>model.embed_tokens.scales,<br>model.embed_tokens.weight,<br>model.layers.0.input_layernorm.weight,<br>model.layers.0.mlp.down_proj.biases,<br>model.layers.0.mlp.down_proj.scales,<br>model.layers.0.mlp.down...                                                                         |
| [`mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-mlx-community-mistral-small-32-24b-instruct-2506-4bit)  | `not assessed`         | Prompt prefill failed for mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit: can only concatenate str (not "list") to str                                                                                                                                                                                                                                                        |
| [`mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit`](#model-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) | `not assessed`         | Model loading failed: Received 729 parameters not in model:<br>backbone.embeddings.biases,<br>backbone.embeddings.scales,<br>backbone.embeddings.weight,<br>backbone.layers.0.mixer.A_log,<br>backbone.layers.0.mixer.D,<br>backbone.layers.0.mixer.conv1d.bias,<br>backbone.layers.0.mixer.conv1d.weight,...                                                                         |
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
- *Model load time:* 2.29s
- *Generation time:* 8.66s
- *Total time:* 11.48s
- *Prompt preparation time:* 0.521
- *First-token latency:* 2.97
- *Cleanup time:* 0.116
- *Prompt tokens:* 2,393
- *Generation tokens:* 151
- *Total tokens:* 2,544
- *Prompt throughput (raw):* 805 tok/s
- *Generation throughput (raw):* 29.5 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 18
- *Post-cleanup active memory (GB):* 0.000394
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
- *Model load time:* 1.61s
- *Generation time:* 14.01s
- *Total time:* 16.26s
- *Prompt preparation time:* 0.635
- *First-token latency:* 0.747
- *Cleanup time:* 0.117
- *Prompt tokens:* 1,634
- *Generation tokens:* 997
- *Total tokens:* 2,631
- *Prompt throughput (raw):* 2,188 tok/s
- *Generation throughput (raw):* 78.8 tok/s
- *Peak memory (GB):* 19
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.18
- *Model-load active memory (GB):* 17.4
- *Post-cleanup active memory (GB):* 0.000607
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
- *Generation time:* 7.57s
- *Total time:* 9.62s
- *Prompt preparation time:* 0.564
- *First-token latency:* 5.18
- *Cleanup time:* 0.0976
- *Prompt tokens:* 6,450
- *Generation tokens:* 140
- *Total tokens:* 6,590
- *Prompt throughput (raw):* 1,244 tok/s
- *Generation throughput (raw):* 74.9 tok/s
- *Peak memory (GB):* 8.7
- *Active memory (GB):* 7.1
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 7.07
- *Post-cleanup active memory (GB):* 0.000919
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
- *Model load time:* 6.22s
- *Generation time:* 15.07s
- *Total time:* 21.87s
- *Prompt preparation time:* 0.573
- *First-token latency:* 11.6
- *Cleanup time:* 0.276
- *Prompt tokens:* 6,450
- *Generation tokens:* 103
- *Total tokens:* 6,553
- *Prompt throughput (raw):* 558 tok/s
- *Generation throughput (raw):* 41.1 tok/s
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
- *Model load time:* 2.12s
- *Generation time:* 6.14s
- *Total time:* 9.01s
- *Prompt preparation time:* 0.746
- *First-token latency:* 1.15
- *Cleanup time:* 0.113
- *Prompt tokens:* 2,612
- *Generation tokens:* 138
- *Total tokens:* 2,750
- *Prompt throughput (raw):* 2,272 tok/s
- *Generation throughput (raw):* 32.3 tok/s
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
- *Model load time:* 1.14s
- *Generation time:* 4.59s
- *Total time:* 6.41s
- *Prompt preparation time:* 0.668
- *First-token latency:* 1.66
- *Cleanup time:* 0.118
- *Prompt tokens:* 2,113
- *Generation tokens:* 126
- *Total tokens:* 2,239
- *Prompt throughput (raw):* 1,270 tok/s
- *Generation throughput (raw):* 55.4 tok/s
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
- *Model load time:* 1.64s
- *Generation time:* 4.05s
- *Total time:* 6.39s
- *Prompt preparation time:* 0.694
- *First-token latency:* 0.945
- *Cleanup time:* 0.121
- *Prompt tokens:* 2,113
- *Generation tokens:* 77
- *Total tokens:* 2,190
- *Prompt throughput (raw):* 2,237 tok/s
- *Generation throughput (raw):* 31.4 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 15.9
- *Post-cleanup active memory (GB):* 0.00211
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
- *Model load time:* 1.20s
- *Generation time:* 1.38s
- *Total time:* 3.15s
- *Prompt preparation time:* 0.566
- *First-token latency:* 0.445
- *Cleanup time:* 0.0894
- *Prompt tokens:* 2,103
- *Generation tokens:* 80
- *Total tokens:* 2,183
- *Prompt throughput (raw):* 4,727 tok/s
- *Generation throughput (raw):* 206 tok/s
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
- *Model load time:* 1.16s
- *Generation time:* 1.71s
- *Total time:* 3.40s
- *Prompt preparation time:* 0.526
- *First-token latency:* 0.244
- *Cleanup time:* 0.101
- *Prompt tokens:* 391
- *Generation tokens:* 101
- *Total tokens:* 492
- *Prompt throughput (raw):* 1,599 tok/s
- *Generation throughput (raw):* 104 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 6.1
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 6.15
- *Post-cleanup active memory (GB):* 0.00334
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
- *Model load time:* 1.65s
- *Generation time:* 4.36s
- *Total time:* 6.58s
- *Prompt preparation time:* 0.565
- *First-token latency:* 2.15
- *Cleanup time:* 0.112
- *Prompt tokens:* 2,926
- *Generation tokens:* 109
- *Total tokens:* 3,035
- *Prompt throughput (raw):* 1,362 tok/s
- *Generation throughput (raw):* 64.7 tok/s
- *Peak memory (GB):* 13
- *Active memory (GB):* 8.0
- *Cache memory (GB):* 0.56
- *Model-load active memory (GB):* 8
- *Post-cleanup active memory (GB):* 0.00361
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
- *Model load time:* 1.60s
- *Generation time:* 6.35s
- *Total time:* 8.50s
- *Prompt preparation time:* 0.547
- *First-token latency:* 2.3
- *Cleanup time:* 0.144
- *Prompt tokens:* 2,926
- *Generation tokens:* 187
- *Total tokens:* 3,113
- *Prompt throughput (raw):* 1,269 tok/s
- *Generation throughput (raw):* 53.2 tok/s
- *Peak memory (GB):* 13
- *Active memory (GB):* 8.4
- *Cache memory (GB):* 0.60
- *Model-load active memory (GB):* 8.43
- *Post-cleanup active memory (GB):* 0.00387
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
- *Model load time:* 1.18s
- *Generation time:* 2.17s
- *Total time:* 3.90s
- *Prompt preparation time:* 0.551
- *First-token latency:* 1.08
- *Cleanup time:* 0.0924
- *Prompt tokens:* 2,925
- *Generation tokens:* 102
- *Total tokens:* 3,027
- *Prompt throughput (raw):* 2,721 tok/s
- *Generation throughput (raw):* 181 tok/s
- *Peak memory (GB):* 7.8
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.38
- *Model-load active memory (GB):* 2.75
- *Post-cleanup active memory (GB):* 0.00413
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
- *Model load time:* 0.98s
- *Generation time:* 3.25s
- *Total time:* 5.00s
- *Prompt preparation time:* 0.77
- *First-token latency:* 2.01
- *Cleanup time:* 0.0991
- *Prompt tokens:* 4,083
- *Generation tokens:* 84
- *Total tokens:* 4,167
- *Prompt throughput (raw):* 2,036 tok/s
- *Generation throughput (raw):* 166 tok/s
- *Peak memory (GB):* 3.9
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.65
- *Model-load active memory (GB):* 2.18
- *Post-cleanup active memory (GB):* 0.00538
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
- *Generation time:* 3.30s
- *Total time:* 6.58s
- *Prompt preparation time:* 0.661
- *First-token latency:* 0.65
- *Cleanup time:* 0.135
- *Prompt tokens:* 1,289
- *Generation tokens:* 155
- *Total tokens:* 1,444
- *Prompt throughput (raw):* 1,983 tok/s
- *Generation throughput (raw):* 75.7 tok/s
- *Peak memory (GB):* 24
- *Active memory (GB):* 23
- *Cache memory (GB):* 0.14
- *Model-load active memory (GB):* 23.1
- *Post-cleanup active memory (GB):* 0.00588
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
- *Model load time:* 0.94s
- *Generation time:* 2.74s
- *Total time:* 4.20s
- *Prompt preparation time:* 0.508
- *First-token latency:* 0.302
- *Cleanup time:* 0.0863
- *Prompt tokens:* 1,133
- *Generation tokens:* 107
- *Total tokens:* 1,240
- *Prompt throughput (raw):* 3,746 tok/s
- *Generation throughput (raw):* 55.0 tok/s
- *Peak memory (GB):* 9.3
- *Active memory (GB):* 8.3
- *Cache memory (GB):* 0.52
- *Model-load active memory (GB):* 8.3
- *Post-cleanup active memory (GB):* 0.00595
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

<a id="model-mlx-community-qwen2-vl-7b-instruct-4bit"></a>

### mlx-community/Qwen2-VL-7B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-7B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.80s
- *Generation time:* 44.17s
- *Total time:* 45.84s
- *Prompt preparation time:* 0.875
- *First-token latency:* 42
- *Cleanup time:* 0.105
- *Prompt tokens:* 16,558
- *Generation tokens:* 117
- *Total tokens:* 16,675
- *Prompt throughput (raw):* 394 tok/s
- *Generation throughput (raw):* 90.6 tok/s
- *Peak memory (GB):* 9.3
- *Active memory (GB):* 4.7
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 4.68
- *Post-cleanup active memory (GB):* 0.00626
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
- *Model load time:* 0.78s
- *Generation time:* 27.89s
- *Total time:* 29.64s
- *Prompt preparation time:* 0.962
- *First-token latency:* 16.4
- *Cleanup time:* 0.153
- *Prompt tokens:* 16,549
- *Generation tokens:* 912
- *Total tokens:* 17,461
- *Prompt throughput (raw):* 1,009 tok/s
- *Generation throughput (raw):* 85.9 tok/s
- *Peak memory (GB):* 8.4
- *Active memory (GB):* 4.3
- *Cache memory (GB):* 2.7
- *Model-load active memory (GB):* 4.26
- *Post-cleanup active memory (GB):* 0.00657
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
- *Generation time:* 73.76s
- *Total time:* 76.75s
- *Prompt preparation time:* 0.997
- *First-token latency:* 63.3
- *Cleanup time:* 0.154
- *Prompt tokens:* 16,547
- *Generation tokens:* 177
- *Total tokens:* 16,724
- *Prompt throughput (raw):* 261 tok/s
- *Generation throughput (raw):* 18.6 tok/s
- *Peak memory (GB):* 26
- *Active memory (GB):* 20
- *Cache memory (GB):* 5.5
- *Model-load active memory (GB):* 19.6
- *Post-cleanup active memory (GB):* 0.00719
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
- *Model load time:* 0.92s
- *Generation time:* 41.08s
- *Total time:* 42.99s
- *Prompt preparation time:* 0.982
- *First-token latency:* 38.9
- *Cleanup time:* 0.108
- *Prompt tokens:* 16,547
- *Generation tokens:* 93
- *Total tokens:* 16,640
- *Prompt throughput (raw):* 426 tok/s
- *Generation throughput (raw):* 70.5 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 3.4
- *Model-load active memory (GB):* 5.77
- *Post-cleanup active memory (GB):* 0.00751
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
- *Model load time:* 2.73s
- *Generation time:* 35.21s
- *Total time:* 38.97s
- *Prompt preparation time:* 1.01
- *First-token latency:* 33
- *Cleanup time:* 0.141
- *Prompt tokens:* 16,563
- *Generation tokens:* 97
- *Total tokens:* 16,660
- *Prompt throughput (raw):* 502 tok/s
- *Generation throughput (raw):* 72.5 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.89
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.00801
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
- *Generation time:* 37.53s
- *Total time:* 40.15s
- *Prompt preparation time:* 0.971
- *First-token latency:* 35.7
- *Cleanup time:* 0.131
- *Prompt tokens:* 16,563
- *Generation tokens:* 86
- *Total tokens:* 16,649
- *Prompt throughput (raw):* 464 tok/s
- *Generation throughput (raw):* 89.0 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 6.0
- *Cache memory (GB):* 1.2
- *Model-load active memory (GB):* 5.96
- *Post-cleanup active memory (GB):* 0.00854
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
- *Model load time:* 2.10s
- *Generation time:* 58.99s
- *Total time:* 62.11s
- *Prompt preparation time:* 1.01
- *First-token latency:* 53.6
- *Cleanup time:* 0.177
- *Prompt tokens:* 16,563
- *Generation tokens:* 124
- *Total tokens:* 16,687
- *Prompt throughput (raw):* 309 tok/s
- *Generation throughput (raw):* 27.5 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 16
- *Cache memory (GB):* 1.9
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.00905
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
- *Model load time:* 0.66s
- *Generation time:* 1.89s
- *Total time:* 3.32s
- *Prompt preparation time:* 0.768
- *First-token latency:* 0.377
- *Cleanup time:* 0.0892
- *Prompt tokens:* 1,426
- *Generation tokens:* 98
- *Total tokens:* 1,524
- *Prompt throughput (raw):* 3,782 tok/s
- *Generation throughput (raw):* 125 tok/s
- *Peak memory (GB):* 5.6
- *Active memory (GB):* 4.5
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 4.5
- *Post-cleanup active memory (GB):* 0.00927
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
- *Model load time:* 13.66s
- *Generation time:* 26.18s
- *Total time:* 40.68s
- *Prompt preparation time:* 0.843
- *First-token latency:* 21.9
- *Cleanup time:* 0.366
- *Prompt tokens:* 3,491
- *Generation tokens:* 121
- *Total tokens:* 3,612
- *Prompt throughput (raw):* 159 tok/s
- *Generation throughput (raw):* 47.3 tok/s
- *Peak memory (GB):* 92
- *Active memory (GB):* 85
- *Cache memory (GB):* 0.41
- *Model-load active memory (GB):* 85
- *Post-cleanup active memory (GB):* 0.00954
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
- *Model load time:* 1.19s
- *Generation time:* 16.76s
- *Total time:* 18.87s
- *Prompt preparation time:* 0.925
- *First-token latency:* 13.6
- *Cleanup time:* 0.103
- *Prompt tokens:* 16,558
- *Generation tokens:* 136
- *Total tokens:* 16,694
- *Prompt throughput (raw):* 1,219 tok/s
- *Generation throughput (raw):* 58.1 tok/s
- *Peak memory (GB):* 14
- *Active memory (GB):* 9.5
- *Cache memory (GB):* 1.4
- *Model-load active memory (GB):* 9.46
- *Post-cleanup active memory (GB):* 0.00985
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
- *Model load time:* 2.36s
- *Generation time:* 6.43s
- *Total time:* 9.38s
- *Prompt preparation time:* 0.57
- *First-token latency:* 0.93
- *Cleanup time:* 0.137
- *Prompt tokens:* 590
- *Generation tokens:* 152
- *Total tokens:* 742
- *Prompt throughput (raw):* 635 tok/s
- *Generation throughput (raw):* 30.7 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.44
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.0109
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
- *Model load time:* 1.88s
- *Generation time:* 2.79s
- *Total time:* 5.33s
- *Prompt preparation time:* 0.648
- *First-token latency:* 0.52
- *Cleanup time:* 0.134
- *Prompt tokens:* 595
- *Generation tokens:* 101
- *Total tokens:* 696
- *Prompt throughput (raw):* 1,144 tok/s
- *Generation throughput (raw):* 60.7 tok/s
- *Peak memory (GB):* 7.6
- *Active memory (GB):* 6.8
- *Cache memory (GB):* 0.32
- *Model-load active memory (GB):* 6.75
- *Post-cleanup active memory (GB):* 0.012
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
- *Model load time:* 2.47s
- *Generation time:* 1.92s
- *Total time:* 5.00s
- *Prompt preparation time:* 0.6
- *First-token latency:* 0.412
- *Cleanup time:* 0.152
- *Prompt tokens:* 595
- *Generation tokens:* 96
- *Total tokens:* 691
- *Prompt throughput (raw):* 1,443 tok/s
- *Generation throughput (raw):* 104 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 15.4
- *Post-cleanup active memory (GB):* 0.0125
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
- *Model load time:* 2.70s
- *Generation time:* 5.22s
- *Total time:* 8.54s
- *Prompt preparation time:* 0.614
- *First-token latency:* 1.17
- *Cleanup time:* 0.176
- *Prompt tokens:* 595
- *Generation tokens:* 89
- *Total tokens:* 684
- *Prompt throughput (raw):* 508 tok/s
- *Generation throughput (raw):* 25.8 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.79
- *Model-load active memory (GB):* 18.4
- *Post-cleanup active memory (GB):* 0.013
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
- *Model load time:* 1.70s
- *Generation time:* 1.36s
- *Total time:* 3.67s
- *Prompt preparation time:* 0.604
- *First-token latency:* 0.187
- *Cleanup time:* 0.122
- *Prompt tokens:* 591
- *Generation tokens:* 73
- *Total tokens:* 664
- *Prompt throughput (raw):* 3,161 tok/s
- *Generation throughput (raw):* 123 tok/s
- *Peak memory (GB):* 5.9
- *Active memory (GB):* 5.2
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 5.16
- *Post-cleanup active memory (GB):* 0.0136
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
- *Generation time:* 1.55s
- *Total time:* 3.00s
- *Prompt preparation time:* 0.829
- *First-token latency:* 0.397
- *Cleanup time:* 0.103
- *Prompt tokens:* 1,379
- *Generation tokens:* 65
- *Total tokens:* 1,444
- *Prompt throughput (raw):* 3,475 tok/s
- *Generation throughput (raw):* 172 tok/s
- *Peak memory (GB):* 4.7
- *Active memory (GB):* 3.0
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 3.03
- *Post-cleanup active memory (GB):* 0.0138
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
- *Model load time:* 1.86s
- *Generation time:* 5.20s
- *Total time:* 7.61s
- *Prompt preparation time:* 0.553
- *First-token latency:* 2.01
- *Cleanup time:* 0.134
- *Prompt tokens:* 3,116
- *Generation tokens:* 102
- *Total tokens:* 3,218
- *Prompt throughput (raw):* 1,547 tok/s
- *Generation throughput (raw):* 38.4 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 13
- *Cache memory (GB):* 0.59
- *Model-load active memory (GB):* 13.5
- *Post-cleanup active memory (GB):* 0.0161
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
- *Model load time:* 0.37s
- *Generation time:* 0.87s
- *Total time:* 2.08s
- *Prompt preparation time:* 0.496
- *First-token latency:* 0.121
- *Cleanup time:* 0.0739
- *Prompt tokens:* 2,112
- *Generation tokens:* 120
- *Total tokens:* 2,232
- *Prompt throughput (raw):* 17,442 tok/s
- *Generation throughput (raw):* 475 tok/s
- *Peak memory (GB):* 1.9
- *Active memory (GB):* 0.90
- *Cache memory (GB):* 0.06
- *Model-load active memory (GB):* 0.897
- *Post-cleanup active memory (GB):* 0.000132
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
- *Model load time:* 1.20s
- *Generation time:* 4.55s
- *Total time:* 6.44s
- *Prompt preparation time:* 0.687
- *First-token latency:* 0.766
- *Cleanup time:* 0.106
- *Prompt tokens:* 1,524
- *Generation tokens:* 216
- *Total tokens:* 1,740
- *Prompt throughput (raw):* 1,989 tok/s
- *Generation throughput (raw):* 68.9 tok/s
- *Peak memory (GB):* 8.1
- *Active memory (GB):* 6.4
- *Cache memory (GB):* 0.31
- *Model-load active memory (GB):* 6.45
- *Post-cleanup active memory (GB):* 0.00444
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
- *Model load time:* 1.18s
- *Generation time:* 2.53s
- *Total time:* 4.58s
- *Prompt preparation time:* 0.865
- *First-token latency:* 0.734
- *Cleanup time:* 0.111
- *Prompt tokens:* 2,090
- *Generation tokens:* 96
- *Total tokens:* 2,186
- *Prompt throughput (raw):* 2,847 tok/s
- *Generation throughput (raw):* 102 tok/s
- *Peak memory (GB):* 6.5
- *Active memory (GB):* 5.5
- *Cache memory (GB):* 0.35
- *Model-load active memory (GB):* 5.46
- *Post-cleanup active memory (GB):* 0.0104
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
- *Model load time:* 3.62s
- *Generation time:* 2.95s
- *Total time:* 7.22s
- *Prompt preparation time:* 0.639
- *First-token latency:* 0.331
- *Cleanup time:* 0.145
- *Prompt tokens:* 591
- *Generation tokens:* 80
- *Total tokens:* 671
- *Prompt throughput (raw):* 1,785 tok/s
- *Generation throughput (raw):* 43.6 tok/s
- *Peak memory (GB):* 28
- *Active memory (GB):* 27
- *Cache memory (GB):* 0.01
- *Model-load active memory (GB):* 27.2
- *Post-cleanup active memory (GB):* 0.0104
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
- *Model load time:* 2.43s
- *Generation time:* 19.83s
- *Total time:* 22.86s
- *Prompt preparation time:* 0.601
- *First-token latency:* 0.805
- *Cleanup time:* 0.132
- *Prompt tokens:* 1,324
- *Generation tokens:* 1,000
- *Total tokens:* 2,324
- *Prompt throughput (raw):* 1,645 tok/s
- *Generation throughput (raw):* 54.5 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 1.5
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
- *Model load time:* 3.89s
- *Generation time:* 56.44s
- *Total time:* 61.04s
- *Prompt preparation time:* 0.692
- *First-token latency:* 8.98
- *Cleanup time:* 0.235
- *Prompt tokens:* 4,403
- *Generation tokens:* 1,000
- *Total tokens:* 5,403
- *Prompt throughput (raw):* 490 tok/s
- *Generation throughput (raw):* 21.4 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.25
- *Model-load active memory (GB):* 23.9
- *Post-cleanup active memory (GB):* 0.00485
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
- *Model load time:* 1.92s
- *Generation time:* 41.57s
- *Total time:* 44.51s
- *Prompt preparation time:* 1.02
- *First-token latency:* 38
- *Cleanup time:* 0.146
- *Prompt tokens:* 16,547
- *Generation tokens:* 200
- *Total tokens:* 16,747
- *Prompt throughput (raw):* 436 tok/s
- *Generation throughput (raw):* 74.4 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 18
- *Cache memory (GB):* 4.0
- *Model-load active memory (GB):* 18.3
- *Post-cleanup active memory (GB):* 0.00688
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
- *Model load time:* 0.32s
- *Generation time:* 1.14s
- *Total time:* 2.35s
- *Prompt preparation time:* 0.879
- *First-token latency:* 0.205
- *Cleanup time:* 0.0846
- *Prompt tokens:* 1,205
- *Generation tokens:* 33
- *Total tokens:* 1,238
- *Prompt throughput (raw):* 5,887 tok/s
- *Generation throughput (raw):* 315 tok/s
- *Peak memory (GB):* 1.1
- *Active memory (GB):* 0.16
- *Cache memory (GB):* 0.10
- *Model-load active memory (GB):* 0.159
- *Post-cleanup active memory (GB):* 0.00916
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
- *Model load time:* 1.79s
- *Generation time:* 2.10s
- *Total time:* 4.47s
- *Prompt preparation time:* 0.568
- *First-token latency:* 0.31
- *Cleanup time:* 0.126
- *Prompt tokens:* 589
- *Generation tokens:* 88
- *Total tokens:* 677
- *Prompt throughput (raw):* 1,900 tok/s
- *Generation throughput (raw):* 70.5 tok/s
- *Peak memory (GB):* 7.2
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 0.04
- *Model-load active memory (GB):* 5.83
- *Post-cleanup active memory (GB):* 0.0115
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
- *Model load time:* 1.26s
- *Generation time:* 3.68s
- *Total time:* 5.63s
- *Prompt preparation time:* 0.685
- *First-token latency:* 1.01
- *Cleanup time:* 0.179
- *Prompt tokens:* 2,195
- *Generation tokens:* 200
- *Total tokens:* 2,395
- *Prompt throughput (raw):* 2,166 tok/s
- *Generation throughput (raw):* 99.5 tok/s
- *Peak memory (GB):* 6.7
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.70
- *Model-load active memory (GB):* 5.69
- *Post-cleanup active memory (GB):* 0.0142
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
- *Model load time:* 0.51s
- *Generation time:* 1.13s
- *Total time:* 2.27s
- *Prompt preparation time:* 0.63
- *First-token latency:* 0.09
- *Cleanup time:* 0.106
- *Prompt tokens:* 330
- *Generation tokens:* 92
- *Total tokens:* 422
- *Prompt throughput (raw):* 3,668 tok/s
- *Generation throughput (raw):* 202 tok/s
- *Peak memory (GB):* 1.8
- *Active memory (GB):* 0.61
- *Cache memory (GB):* 0.28
- *Model-load active memory (GB):* 0.613
- *Post-cleanup active memory (GB):* 0.0148
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
- *Model load time:* 1.63s
- *Generation time:* 2.48s
- *Total time:* 4.73s
- *Prompt preparation time:* 0.62
- *First-token latency:* 1.69
- *Cleanup time:* 0.135
- *Prompt tokens:* 1,344
- *Generation tokens:* 7
- *Total tokens:* 1,351
- *Prompt throughput (raw):* 794 tok/s
- *Generation throughput (raw):* 35.8 tok/s
- *Peak memory (GB):* 9.7
- *Active memory (GB):* 6.0
- *Cache memory (GB):* 1.8
- *Model-load active memory (GB):* 6.05
- *Post-cleanup active memory (GB):* 0.0158
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

<a id="model-apple-fastvlm-7b-int4"></a>

### apple/FastVLM-7B-int4

<details>
<summary>Complete evidence: apple/FastVLM-7B-int4</summary>

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
- *Error message:* Model loading failed:
  'vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight'
- *Root exception type:* KeyError
- *Root exception module:* builtins
- *Root exception message:* 'vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight'
- *Arch supported by installed mlx-vlm:* yes (model_type llava_qwen2 via
  fastvlm)
- *Model load time:* 0.14s
- *Generation time:* -
- *Total time:* 0.14s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0693
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.000132
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* 1aeadbaaba011276f3dcda9582e5e64e2a90873a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--apple--FastVLM-7B-int4/snapshots/1aeadbaaba011276f3dcda9582e5e64e2a90873a
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14197, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13109, in _load_model
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1072, in load_model
    weights = sanitize_weights(
        model_class.VisionModel, weights, model_config.vision_config
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1241, in sanitize_weights
    weights = model_obj.sanitize(weights)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/fastvlm/vision.py", line 650, in sanitize
    W, C = weights[
           ~~~~~~~^
        "vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight"
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ].shape[-2:]
    ^
KeyError: 'vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14212, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: 'vision_tower.vision_model.patch_embed.blocks.1.reparam_conv.weight'

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 4166.87it/s]
```

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
- *Model load time:* 0.14s
- *Generation time:* -
- *Total time:* 0.14s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0776
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00211
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14197, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13109, in _load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14212, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 14 files:   0%|          | 0/14 [00:00<?, ?it/s]
Fetching 14 files: 100%|##########| 14/14 [00:00<00:00, 4551.25it/s]
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
- *Model load time:* 1.35s
- *Generation time:* 0.54s
- *Total time:* 2.49s
- *Prompt preparation time:* 0.593
- *First-token latency:* -
- *Cleanup time:* 0.128
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00303
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13443, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14120, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<5 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14015, in _generate_with_repetition_guard
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14224, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14140, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13452, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 3714.33it/s]
[21:49:05] Generation error for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13443, in
           _run_generation_guarded
               return generate_once()
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14120, in
           _generate_once
               return _generate_with_repetition_guard(
                   model=prepared.model,
               ...<5 lines>...
                   **prepared.generate_kwargs,
               )
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14015, in
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

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

<details>
<summary>Complete evidence: mlx-community/Llama-3.2-11B-Vision-Instruct-8bit</summary>

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
  mlx-community/Llama-3.2-11B-Vision-Instruct-8bit: [broadcast_shapes] Shapes
  (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* [broadcast_shapes] Shapes (1,1,301,6404) and
  (1,32,300,6404) cannot be broadcast.
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 1.77s
- *Generation time:* 0.54s
- *Total time:* 2.88s
- *Prompt preparation time:* 0.57
- *First-token latency:* -
- *Cleanup time:* 0.176
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00303
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
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

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13443, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14120, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<5 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14015, in _generate_with_repetition_guard
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14224, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14140, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13452, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-8bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 3483.35it/s]
[21:49:08] Generation error for mlx-community/Llama-3.2-11B-Vision-Instruct-8bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13443, in
           _run_generation_guarded
               return generate_once()
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14120, in
           _generate_once
               return _generate_with_repetition_guard(
                   model=prepared.model,
               ...<5 lines>...
                   **prepared.generate_kwargs,
               )
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14015, in
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
- *Model load time:* 0.25s
- *Generation time:* -
- *Total time:* 0.25s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0726
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00303
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14197, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13109, in _load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14212, in _run_model_generation
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
Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 2518.09it/s]
```

</details>

---

<a id="model-mlx-community-mistral-small-32-24b-instruct-2506-4bit"></a>

### mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit

<details>
<summary>Complete evidence: mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit</summary>

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* prefill
- *Error stage:* Error
- *Error code:* TRANSFORMERS_PREFILL_ERROR
- *Error type:* ValueError
- *Error package:* transformers
- *Error message:* Prompt prefill failed for
  mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit: can only concatenate
  str (not "list") to str
- *Root exception type:* TypeError
- *Root exception module:* builtins
- *Root exception message:* can only concatenate str (not "list") to str
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.84s
- *Generation time:* -
- *Total time:* 1.85s
- *Prompt preparation time:* 0.00411
- *First-token latency:* -
- *Cleanup time:* 0.184
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00413
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* exception
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* not captured
- *Processor:* not captured
- *Tokenizer:* not captured
- *Requested model revision:* not requested
- *Resolved model revision:* 2a1d5eabfc504747bdc24178394821a1efc0edde
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Mistral-Small-3.2-24B-Instruct-2506-4bit/snapshots/2a1d5eabfc504747bdc24178394821a1efc0edde
- *Generation settings:* not captured
- *EOS token:* not captured

#### Complete traceback

```python
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13521, in _prepare_generation_prompt
    apply_chat_template(
    ~~~~~~~~~~~~~~~~~~~^
        processor=processor,
        ^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **chat_template_kwargs,
        ^^^^^^^^^^^^^^^^^^^^^^^
    ),
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 1026, in apply_chat_template
    return get_chat_template(processor, messages, add_generation_prompt, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 824, in get_chat_template
    return template_processor.apply_chat_template(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        messages,
        ^^^^^^^^^
    ...<2 lines>...
        **template_kwargs,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/tokenization_utils_base.py", line 3108, in apply_chat_template
    rendered_chat, generation_indices = render_jinja_template(
                                        ~~~~~~~~~~~~~~~~~~~~~^
        conversations=conversations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        **template_kwargs,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/utils/chat_template_utils.py", line 581, in render_jinja_template
    rendered_chat = compiled_template.render(
        messages=chat,
    ...<3 lines>...
        **kwargs,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py", line 1295, in render
    self.environment.handle_exception()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py", line 942, in handle_exception
    raise rewrite_traceback_stack(source=source)
  File "<template>", line 1, in top-level template code
TypeError: can only concatenate str (not "list") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14216, in _run_model_generation
    prepared = _prepare_generation(
        params,
    ...<4 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13767, in _prepare_generation
    prepared_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13540, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(msg), "prefill") from prefill_err
ValueError: Prompt prefill failed for mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit: can only concatenate str (not "list") to str

```

#### Captured upstream output

```text
=== STDERR ===
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 3538.60it/s]
[21:49:34] Prompt prefill failed for mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13521, in
           _prepare_generation_prompt
               apply_chat_template(
               ~~~~~~~~~~~~~~~~~~~^
                   processor=processor,
                   ^^^^^^^^^^^^^^^^^^^^
               ...<3 lines>...
                   **chat_template_kwargs,
                   ^^^^^^^^^^^^^^^^^^^^^^^
               ),
               ^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 1026, in
           apply_chat_template
               return get_chat_template(processor, messages, add_generation_prompt, **kwargs)
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 824, in
           get_chat_template
               return template_processor.apply_chat_template(
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
                   messages,
                   ^^^^^^^^^
               ...<2 lines>...
                   **template_kwargs,
                   ^^^^^^^^^^^^^^^^^^
               )
               ^
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/tokenizatio
           n_utils_base.py", line 3108, in apply_chat_template
               rendered_chat, generation_indices = render_jinja_template(
                                                   ~~~~~~~~~~~~~~~~~~~~~^
                   conversations=conversations,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               ...<6 lines>...
                   **template_kwargs,
                   ^^^^^^^^^^^^^^^^^^
               )
               ^
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/utils/chat_
           template_utils.py", line 581, in render_jinja_template
               rendered_chat = compiled_template.render(
                   messages=chat,
               ...<3 lines>...
                   **kwargs,
               )
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py",
           line 1295, in render
               self.environment.handle_exception()
               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py",
           line 942, in handle_exception
               raise rewrite_traceback_stack(source=source)
             File "<template>", line 1, in top-level template code
           TypeError: can only concatenate str (not "list") to str
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
- *Model load time:* 0.28s
- *Generation time:* -
- *Total time:* 0.29s
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0956
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory (GB):* -
- *Active memory (GB):* -
- *Cache memory (GB):* -
- *Model-load active memory (GB):* -
- *Post-cleanup active memory (GB):* 0.00485
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14197, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13109, in _load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14212, in _run_model_generation
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
Fetching 25 files: 100%|##########| 25/25 [00:00<00:00, 3080.69it/s]
```

</details>

---
