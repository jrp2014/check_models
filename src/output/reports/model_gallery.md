# Model Output Gallery

Generated on: 2026-08-30 01:47:24 BST

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

| Model                                                                                                                   | Usability             | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------|---------------------|-----------------|---------|------------|---------|-----------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | 2.51s   | 208 tok/s           | 0.56            | 4.0     | 2,619      | 106     | none                                                                                                                        |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | 6.06s   | 67.7 tok/s          | 1.92            | 12      | 2,577      | 157     | none                                                                                                                        |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 2.97s   | 192 tok/s           | 0.79            | 6.4     | 2,576      | 138     | none                                                                                                                        |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | 71.92s  | 66.0 tok/s          | 58.85           | 74      | 16,639     | 112     | none                                                                                                                        |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | 27.13s  | 90.3 tok/s          | 15.55           | 8.4     | 16,625     | 903     | none                                                                                                                        |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 57.20s  | 111 tok/s           | 52.33           | 24      | 16,639     | 150     | none                                                                                                                        |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 57.53s  | 91.3 tok/s          | 54.03           | 10      | 16,639     | 115     | none                                                                                                                        |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 81.61s  | 17.5 tok/s          | 71.50           | 33      | 16,639     | 106     | none                                                                                                                        |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.09s   | 129 tok/s           | 0.40            | 16      | 579        | 101     | none                                                                                                                        |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 8.29s   | 26.1 tok/s          | 1.11            | 20      | 579        | 107     | none                                                                                                                        |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable with caveats` | 16.28s  | 92.3 tok/s          | 13.37           | 8.4     | 16,623     | 135     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable with caveats` | 10.38s  | 30.2 tok/s          | 2.67            | 22      | 2,044      | 152     | title/keyword constraints failed                                                                                            |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable with caveats` | 28.65s  | 63.9 tok/s          | 4.44            | 60      | 1,596      | 711     | title/keyword constraints failed                                                                                            |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable with caveats` | 29.52s  | 43.5 tok/s          | 17.01           | 78      | 6,325      | 126     | control tokens visible; title/keyword constraints failed                                                                    |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable with caveats` | 6.00s   | 34.3 tok/s          | 0.99            | 17      | 2,605      | 95      | title/keyword constraints failed                                                                                            |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable with caveats` | 2.33s   | 187 tok/s           | 0.41            | 4.0     | 2,627      | 149     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable with caveats` | 6.92s   | 65.2 tok/s          | 1.68            | 12      | 2,577      | 215     | title/keyword constraints failed                                                                                            |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable with caveats` | 5.34s   | 231 tok/s           | 2.99            | 3.9     | 4,039      | 150     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable with caveats` | 4.24s   | 55.7 tok/s          | 0.23            | 9.4     | 1,113      | 143     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable with caveats` | 58.84s  | 86.8 tok/s          | 54.70           | 23      | 16,623     | 122     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable with caveats` | 77.95s  | 30.5 tok/s          | 70.68           | 21      | 16,639     | 131     | title/keyword constraints failed                                                                                            |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable with caveats` | 2.62s   | 123 tok/s           | 0.39            | 5.5     | 1,411      | 112     | title/keyword constraints failed; draft hints copied unchanged                                                              |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable with caveats` | 26.98s  | 45.9 tok/s          | 16.23           | 70      | 3,472      | 118     | title/keyword constraints failed                                                                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable with caveats` | 22.10s  | 57.8 tok/s          | 17.49           | 14      | 16,634     | 149     | title/keyword constraints failed                                                                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable with caveats` | 5.85s   | 66.2 tok/s          | 0.34            | 29      | 575        | 88      | control tokens visible                                                                                                      |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable with caveats` | 5.40s   | 80.3 tok/s          | 0.33            | 28      | 575        | 89      | control tokens visible                                                                                                      |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable with caveats` | 7.86s   | 31.4 tok/s          | 0.98            | 17      | 570        | 127     | title/keyword constraints failed                                                                                            |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable with caveats` | 2.49s   | 171 tok/s           | 0.45            | 4.8     | 1,510      | 118     | title/keyword constraints failed                                                                                            |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable with caveats` | 6.72s   | 39.5 tok/s          | 1.45            | 15      | 2,648      | 125     | title/keyword constraints failed                                                                                            |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `unusable`            | 1.65s   | 470 tok/s           | 0.13            | 1.7     | 2,627      | 200     | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | 4.21s   | 138 tok/s           | 0.93            | 3.9     | 2,047      | 275     | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 27.35s  | 42.5 tok/s          | 1.72            | 14      | 2,739      | 1,000   | missing required fields; echoes instructions; cut off at token limit                                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 1.96s   | 339 tok/s           | 0.09            | 2.2     | 314        | 41      | missing required fields                                                                                                     |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 29.01s  | 46.5 tok/s          | 5.35            | 13      | 6,321      | 1,000   | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | 9.91s   | 80.5 tok/s          | 5.61            | 8.4     | 6,325      | 200     | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `unusable`            | 11.44s  | 32.0 tok/s          | 1.09            | 18      | 2,603      | 250     | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | 219.33s | 4.67 tok/s          | 1.35            | 40      | 1,307      | 1,000   | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 1.89s   | 271 tok/s           | 0.24            | 3.8     | 916        | 50      | missing required fields; extra text before Title                                                                            |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `unusable`            | 16.30s  | 72.1 tok/s          | 0.73            | 9.0     | 1,504      | 1,000   | repeated text; missing required fields; cut off at token limit                                                              |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 3.17s   | insufficient sample | 0.38            | 17      | 569        | 1       | empty response; missing required fields                                                                                     |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not evaluated`       | 1.03s   | -                   | -               | -       | -          | -       | none                                                                                                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 208 tok/s

Average clean-completion throughput: 100.0 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 4.0 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                  | Usability       | Observations                                                                                                                |
|--------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                           | `unusable`      | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                    | `unusable`      | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx) | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                             | `unusable`      | missing required fields                                                                                                     |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)             | `unusable`      | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                       | `unusable`      | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                | `unusable`      | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)  | `unusable`      | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                           | `unusable`      | missing required fields; extra text before Title                                                                            |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                  | `unusable`      | repeated text; missing required fields; cut off at token limit                                                              |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                      | `unusable`      | empty response; missing required fields                                                                                     |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                  | `not evaluated` | none                                                                                                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | Title: White motor cruiser boat cruising on calm waterway<br>Description: A white motor cruiser boat named 'Wavey Katey II' with a blue canopy and British flag cruises past a riverside cottage and lush foliage.<br>Keywords: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Can...                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable`              | Title:<br>**"Wavey Katey II – British Motor Cruiser on Serene Waterway"**<br><br>Description:<br>A white Yamaha-powered motor cruiser, *Wavey Katey II*, cruises at dusk along a tranquil waterway near a rustic riverside cottage, flying a British maritime flag. The boat features a blue cano...          |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>A British Cruiser Boat on a Riverbank Serenade<br><br>Description:<br>A white motor cruiser named *Wavey Katey II* glides calmly along a river, adorned with a British maritime flag, under a rustic wooden cottage and lush greenery. The midday sunlight casts soft shadows on the bo...          |
| [`mlx-community/Ornith-1.0-35B-bf16`](#model-mlx-community-ornith-10-35b-bf16)                                          | `usable`              | Title: Wavey Katey II motor cruiser on a river<br>Description: The white 4Plus2 motor cruiser 'Wavey Katey II' navigates a calm waterway under a blue canvas canopy, flying a Red Ensign past a rustic wooden house with a balcony.<br>Keywords: Boat, Cabin cruiser, Canopy, Cottage, Crui...                |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable`              | Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The main subject is the boat named "Wavey Katey II" on a river. So maybe "Wavey Katey II Motor Cruiser on Riverbank" but let's check word count. "Wavey Katey II Motor Cruiser Cruising on River" –...                       |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Wavey Katey II Motor Cruiser on River<br><br>Description: Captured on 28 August 2026, this white cabin cruiser named 'Wavey Katey II' navigates a calm waterway, flying a Red Ensign flag past a rustic wooden riverside house and lush green trees. The vessel features a blue cano...                |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title:<br>Wavey Katey II Cruises Past Riverside Cottage on Calm River<br><br>Description:<br>A white cabin cruiser named 'Wavey Katey II' glides along a tranquil river, flying a British maritime flag, with a rustic wooden riverside cottage and lush green trees in the background under soft...          |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | Title:<br>White Cabin Cruiser Cruising Past Riverside Cottage<br><br>Description:<br>The white cabin cruiser 'Wavey Katey II' moves along a calm waterway, flying a red ensign past a rustic wooden riverside house surrounded by lush greenery.<br><br>Keywords:<br>Cabin cruiser, Motorboat, River, Cott... |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: White motor cruiser Wavey Katey II cruising on river<br>Description: A white motor cruiser named Wavey Katey II cruises along a calm river past a rustic wooden house and lush green foliage. The boat features a blue canopy and flies a small British maritime flag under brigh...                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: White Motor Cruiser Wavey Katey II on a River<br>Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green foliage. The boat features a blue canopy and flies a British maritime flag under bri...                   |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title: Wavey Katey II motorboat on river<br><br>Description: A white motorboat named 'Wavey Katey II' with a blue canopy and a British flag sails along a calm river, passing a rustic wooden cottage surrounded by lush greenery and trees. The boat is equipped with fenders and an outbo...                |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>White motor cruiser 'Wavey Katey II' on calm waterway<br><br>Description:<br>A white motor cruiser named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees on a sunny afternoon in August 2026....          |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable_with_caveats` | Alright, let's get to work. I've got an image of a boat, and I need to generate some metadata for it. First things first, I need to break down the request. It's a straightforward task: title, description, and keywords. The image is of a white motor cruiser, and I have some hin...                      |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Wavey Katey II Motor Cruiser on River<br>Description: A white motor cruiser named "Wavey Katey II" cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees, with a blue canopy and visible fende...                 |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable_with_caveats` | Title: Boating on Calm Waterway<br><br>Description: A white motor cruiser named 'Wavey Katey II' cruises past a rustic wooden house, flying a British maritime flag, with lush greenery in the background.<br><br>Keywords: Boat, Wavey Katey II, Cruiser, British flag, Wooden house, Lush green...          |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable_with_caveats` | Title: Wavey Katey II Motor Cruiser Cruising<br><br>Description: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees. The boat is equipped with a blue canopy, a mot...                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Wavey Katey II Cruising Along British Waterway (2026-08-28)*<br><br>**Description:**<br>A white Yamaha motor cruiser, *Wavey Katey II* (with blue canopy and fenders attached to its hull railings) cruises past a rustic wooden riverside cottage, flying a British maritime flag...          |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable_with_caveats` | Title: Wavey Katey II - A White Motor Cruiser Boat<br><br>Description: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees. The boat features a blue canopy, an outb...                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable_with_caveats` | Title: "Wavey Katey II Boat Cruising on a Serene River"<br><br>Description: The white motorboat 'Wavey Katey II' is seen cruising on a calm river, with a British flag flying from its mast, past a rustic riverside house and lush greenery.<br><br>Keywords: Wavey Katey II, motorboat, British...          |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable_with_caveats` | Title: White motor cruiser cruising on a river<br>Description: A white motor cruiser named 'Wavey Katey II' cruises on a calm river, its outboard motor creating a wake. The boat, featuring a blue canopy and a British flag, passes a wooden riverside cottage surrounded by lush gree...                   |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable_with_caveats` | Title:<br>Wavey Katey II Cruising Past a Rustic Riverside Cottage<br><br>Description:<br>The white cabin cruiser Wavey Katey II moves along a calm river, flying a red ensign past a wooden riverside cottage and lush green trees. The boat features a blue canvas canopy and white fenders, wit...          |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `usable_with_caveats` | Title: Cruising Motorboat<br>Description: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.<br>Keywords: Boat, Boat driver, Boat fender, Boating, Cabin cruiser,...                 |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable_with_caveats` | Title:<br>British motor cruiser 'Wavey Katey II' cruising on a calm river<br><br>Description:<br>A white motor cruiser named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.<br><br>Keywords:<br>Boat, Boat dri... |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `usable_with_caveats` | Title:<br>"Wavey Katey II Cruising Along the River"<br><br>Description:<br>A white motor cruiser named "Wavey Katey II" glides through calm waters, its British flag fluttering in the breeze, as it passes a rustic wooden riverside cottage surrounded by lush greenery. The scene captures lei...          |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: White motor cruiser Wavey Katey II cruising on a river<br>Description: A white cabin motor cruiser named Wavey Katey II cruises along a calm waterway, past a British maritime flag and a rustic wooden cottage.<br>Keywords: Boat, cabin cruiser, motor...           |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable_with_caveats` | <\|channel>thought<br><channel\|>Title: White motor cruiser Wavey Katey II cruising on a river<br>Description: A white cabin motor cruiser named 'Wavey Katey II' travels along a calm waterway past a rustic wooden riverside cottage and lush green foliage.<br>Keywords: Boat, cabin cruiser,...           |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable_with_caveats` | Title: Wavey Katey II on River Cruise, August 2026<br><br>Description: Captured on 28th August 2026, the white motor cruiser 'Wavey Katey II' cruises along a calm river, passing a wooden house and trees in bright daylight. A British maritime flag is flown from the stern of the boat....                |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable_with_caveats` | Title: "Wavey Katey II Cruising Along a Serene Waterway"<br>Description: "A white motor cruiser named 'Wavey Katey II' glides through calm waters, passing by a rustic riverside cottage and lush greenery, under a British maritime flag."<br>Keywords: Boat, Motor cruiser, British flag,...                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable_with_caveats` | Title:<br>Motor Cruiser "Wavey Katey II" on Calm River<br><br>Description:<br>The white motor cruiser "Wavey Katey II" cruises along a serene river, displaying a British maritime flag, with a rustic riverside cottage and lush greenery in the background.<br><br>Keywords:<br>Boat, Boat driver, Boat...  |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `unusable`            | Title:<br>Boat Cruising on a Calm River<br><br>Description:<br>A white motor cruiser boat named 'Wavey Katey II' cruises along a calm river, with a British maritime flag flying and lush green trees in the background.<br><br>Keywords:<br>Boat, Cruising, River, Riverbank, Foliage, Nautical, Outboard... |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | Title:<br>A White Motor Cruiser named 'Wavey Katey II' Cruises Along a Calm Waterway<br><br>Description:<br>A white motor cruiser named 'Wavey Katey II' is seen cruising along a calm waterway, flying a British maritime flag. The boat is passing by a rustic wooden riverside house and lush...           |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time: 2026-08-28 16:17:42 UTC+01:00. The descriptive hints: description hint: "A white motor cruiser boat named 'Wavey Kate...                   |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A serene white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.                                                                                                                            |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The boat is named "Wavey Katey II" and it's a motor cruiser. So maybe "Wavey Katey II Motor Cruiser on Waterway".<br><br>Next, the description: need to combine context (capture dat...          |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `unusable`            | Title: Wavey Katey II Motor Cruiser<br>Description: The white motor cruiser 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.<br>Keywords: Wavey Katey II, motor cruiser, British maritime flag, c...                |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `unusable`            | Title: "Wavey Katey II Cruises Along Calm Waterway".<br><br>Description: "The 'Wavey Katey II' motor cruiser boat, adorned with a British maritime flag, glides smoothly along a serene waterway. The boat, painted white with a blue canopy, is accompanied by a rustic wooden riverside h...                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `unusable`            | ◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and context provided. Let's start by analyzing the image.<br><br>The image shows a white motor cruiser named "Wavey Katey II" on a calm waterway. There's a rustic wooden...                |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | White motorboat cruising near rustic cottage on calm water.<br><br>Keywords: boat, cruiser, waterway, cottage, flag, fenders, canopy, leisure, outboard, trees, railing, foliage, nautical, river, cabin, passenger                                                                                           |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `unusable`            | Wavey Katey II Cruises Calm River Past Rustic Cottage<br><br>A white motor cruiser named Wavey Katey II glides smoothly along a tranquil waterway, its blue canopy and red flag visible against the backdrop of a rustic wooden riverside house and lush green trees. The boat's name is cl...                |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | empty output                                                                                                                                                                                                                                                                                                  |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not_evaluated`       | Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)                                                                                                                                                                 |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.0rc0`
- `mlx`: `0.32.3.dev20260829+052e77db9`
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

- *Description:* A white motor cruiser boat named 'Wavey Katey II' cruises
  along a calm waterway, flying a British maritime flag, past a rustic wooden
  riverside house and lush green trees.
- *Keywords:* Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy,
  Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard
  motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Vessel,
  Watercraft, Waterfront, Waterway, Wooden house, boathouse, british, calm
  water, canal, day trip, england, fenders, holiday, hull, landscape,
  lifestyle, motor cruiser, nature, navigation, outdoor, powerboat,
  recreation, relaxation, riverboat, riverfront, riverside, scenic, summer,
  tourism, tranquil, travel, united kingdom, vacation, water, water transport
- *Date:* 2026-08-28 16:17:42 UTC+01:00
- *Time:* 16:17:42

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
> &#45; Capture date/time: 2026-08-28 16:17:42 UTC+01:00
>
> &#8203;Descriptive hints:
> &#45; Description hint: A white motor cruiser boat named 'Wavey Katey II'
> cruises along a calm waterway, flying a British maritime flag, past a rustic
> wooden riverside house and lush green trees.
> &#45; Keyword hints: Boat, Boat driver, Boat fender, Boating, Cabin cruiser,
> Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical,
> Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees
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

<a id="model-mlx-community-lfm25-vl-3b-optiq-4bit"></a>

### mlx-community/LFM2.5-VL-3B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-3B-OptiQ-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.98s
- *Generation time:* 1.52s
- *Total time:* 2.51s
- *Input validation time:* 4.54e-05
- *Prompt preparation time:* 0.00951
- *First-token latency:* 0.559
- *Cleanup time:* 0.0807
- *Prompt tokens:* 2,619
- *Generation tokens:* 106
- *Total tokens:* 2,725
- *Prompt throughput (raw):* 4,686 tok/s
- *Generation throughput (raw):* 208 tok/s
- *Peak memory:* 4.0
- *Active memory:* 2.8
- *Cache memory:* 0.10
- *Model-load active memory:* 2.82
- *Post-cleanup active memory:* 0.00436
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1325
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
Title: White motor cruiser boat cruising on calm waterway
Description: A white motor cruiser boat named 'Wavey Katey II' with a blue canopy and British flag cruises past a riverside cottage and lush foliage.
Keywords: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Passenger, Railing, River, Riverbank, Shrubs, Trees
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
- *Generation time:* 4.67s
- *Total time:* 6.06s
- *Input validation time:* 4.83e-05
- *Prompt preparation time:* 0.00811
- *First-token latency:* 1.92
- *Cleanup time:* 0.0934
- *Prompt tokens:* 2,577
- *Generation tokens:* 157
- *Total tokens:* 2,734
- *Prompt throughput (raw):* 1,343 tok/s
- *Generation throughput (raw):* 67.7 tok/s
- *Peak memory:* 12
- *Active memory:* 8.0
- *Cache memory:* 0.51
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00515
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3712
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
**"Wavey Katey II – British Motor Cruiser on Serene Waterway"**

Description:
A white Yamaha-powered motor cruiser, *Wavey Katey II*, cruises at dusk along a tranquil waterway near a rustic riverside cottage, flying a British maritime flag. The boat features a blue canopy, two passengers, and protective fenders, with lush greenery and calm lighting enhancing the nautical scene.

Keywords:
Boat, Yamaha motor cruiser, Wavey Katey II, British maritime flag, motorboat, cabin cruiser, leisure boating, riverside cottage, calm waterway, blue canopy, passenger boat, outboard motor, fenders, dusk lighting, foliage, riverbank, nautical, rustic setting
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
- *Model load time:* 1.01s
- *Generation time:* 1.94s
- *Total time:* 2.97s
- *Input validation time:* 4.83e-05
- *Prompt preparation time:* 0.00789
- *First-token latency:* 0.786
- *Cleanup time:* 0.0826
- *Prompt tokens:* 2,576
- *Generation tokens:* 138
- *Total tokens:* 2,714
- *Prompt throughput (raw):* 3,277 tok/s
- *Generation throughput (raw):* 192 tok/s
- *Peak memory:* 6.4
- *Active memory:* 2.8
- *Cache memory:* 0.35
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00567
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3711
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
A British Cruiser Boat on a Riverbank Serenade

Description:
A white motor cruiser named *Wavey Katey II* glides calmly along a river, adorned with a British maritime flag, under a rustic wooden cottage and lush greenery. The midday sunlight casts soft shadows on the boat’s canopy and fenders, with a lone driver navigating leisurely.

Keywords:
British maritime flag, cabin cruiser, calm river cruising, fenders, motor cruiser *Wavey Katey II*, nautical leisure, outboard motor, passenger boat, red-and-white flag, riverbank cottage, shrubs, trees, UK boating scene
</pre>

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
- *Model load time:* 10.40s
- *Generation time:* 61.49s
- *Total time:* 71.92s
- *Input validation time:* 6.36e-05
- *Prompt preparation time:* 0.0191
- *First-token latency:* 58.8
- *Cleanup time:* 0.261
- *Prompt tokens:* 16,639
- *Generation tokens:* 112
- *Total tokens:* 16,751
- *Prompt throughput (raw):* 283 tok/s
- *Generation throughput (raw):* 66.0 tok/s
- *Peak memory:* 74
- *Active memory:* 70
- *Cache memory:* 0.89
- *Model-load active memory:* 70.2
- *Post-cleanup active memory:* 0.00701
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1365
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
Title: Wavey Katey II motor cruiser on a river
Description: The white 4Plus2 motor cruiser 'Wavey Katey II' navigates a calm waterway under a blue canvas canopy, flying a Red Ensign past a rustic wooden house with a balcony.
Keywords: Boat, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Railing, River, Riverbank, Shrubs, Trees, Fender, Wavey Katey II
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
- *Generation time:* 26.40s
- *Total time:* 27.13s
- *Input validation time:* 6.27e-05
- *Prompt preparation time:* 0.0117
- *First-token latency:* 15.5
- *Cleanup time:* 0.0975
- *Prompt tokens:* 16,625
- *Generation tokens:* 903
- *Total tokens:* 17,528
- *Prompt throughput (raw):* 1,069 tok/s
- *Generation throughput (raw):* 90.3 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.7
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00739
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1354
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
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The main subject is the boat named "Wavey Katey II" on a river. So maybe "Wavey Katey II Motor Cruiser on Riverbank" but let's check word count. "Wavey Katey II Motor Cruiser Cruising on River" – that's 8 words, good.

Next, the description. It has to combine context (capture date/time is given but we don't claim it's visible, so just mention the time as part of the context). The boat is white, named Wavey Katey II, flying a British flag (the red flag with Union Jack), past a rustic wooden house, lush green trees, calm water. Action is cruising. Lighting: probably daylight, since it's a clear day. Distinctive details: blue canopy, Yamaha outboard motor (visible on the back), fenders, the house with a deck. Let's structure it: "White motor cruiser Wavey Katey II, with blue canopy and British flag, cruises along a calm river past rustic wooden cottage with lush greenery, under daylight lighting."

Then keywords. Need 10-18 unique, comma-separated. From context: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage. Let's check the image: the boat has "4PLUS2" on it, so maybe "4PLUS2" is a keyword? Wait, the context says "Boat" is a keyword, so include that. Also, the house is a cottage, so "Cottage" is there. Let's list: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha (since the outboard is Yamaha), Wavey Katey II (but maybe not, since it's a name, but the context says "Boat" is a keyword, so include the name as a keyword? Wait, the context says "Keyword hints" include "Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees". So we need to pick from these. Let's check the image: the boat has "4PLUS2" on it, so "4PLUS2" is a detail. The boat is white, has a blue canopy, the house is a cottage, there's a Yamaha outboard motor (visible on the back), the flag is British (Union Jack), the water is calm, trees and shrubs. So keywords: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha, Wavey Katey II (but maybe not, since it's a name, but the context says "Keyword hints" include "Boat" so maybe include the name as a keyword? Wait, the problem says "10-18 unique, comma-separated keywords covering relevant context and visible details." So from the context and image: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha, Wavey Katey II. Wait, but "Wavey Katey II" is the name of the boat, so it's a visible detail. Let's check the image: the boat is named "Wavey Katey
&lt;/think&gt;

Title: Wavey Katey II Motor Cruiser on Riverbank
Description: White motor cruiser Wavey Katey II with blue canopy and British flag cruises along a calm river past rustic wooden cottage, lush greenery, and daylight lighting.
Keywords: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha, Wavey Katey II
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The main subject is the boat named "Wavey Katey II" on a river. So maybe "Wavey Katey II Motor Cruiser on Riverbank" but let's check word count. "Wavey Katey II Motor Cruiser Cruising on River" – that's 8 words, good.

Next, the description. It has to combine context (capture date/time is given but we don't claim it's visible, so just mention the time as part of the context). The boat is white, named Wavey Katey II, flying a British flag (the red flag with Union Jack), past a rustic wooden house, lush green trees, calm water. Action is cruising. Lighting: probably daylight, since it's a clear day. Distinctive details: blue canopy, Yamaha outboard motor (visible on the back), fenders, the house with a deck. Let's structure it: "White motor cruiser Wavey Katey II, with blue canopy and British flag, cruises along a calm river past rustic wooden cottage with lush greenery, under daylight lighting."

Then keywords. Need 10-18 unique, comma-separated. From context: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage. Let's check the image: the boat has "4PLUS2" on it, so maybe "4PLUS2" is a keyword? Wait, the context says "Boat" is a keyword, so include that. Also, the house is a cottage, so "Cottage" is there. Let's list: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha (since the outboard is Yamaha), Wavey Katey II (but maybe not, since it's a name, but the context says "Boat" is a keyword, so include the name as a keyword? Wait, the context says "Keyword hints" include "Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees". So we need to pick from these. Let's check the image: the boat has "4PLUS2" on it, so "4PLUS2" is a detail. The boat is white, has a blue canopy, the house is a cottage, there's a Yamaha outboard motor (visible on the back), the flag is British (Union Jack), the water is calm, trees and shrubs. So keywords: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha, Wavey Katey II (but maybe not, since it's a name, but the context says "Keyword hints" include "Boat" so maybe include the name as a keyword? Wait, the problem says "10-18 unique, comma-separated keywords covering relevant context and visible details." So from the context and image: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha, Wavey Katey II. Wait, but "Wavey Katey II" is the name of the boat, so it's a visible detail. Let's check the image: the boat is named "Wavey Katey
</think>

Title: Wavey Katey II Motor Cruiser on Riverbank  
Description: White motor cruiser Wavey Katey II with blue canopy and British flag cruises along a calm river past rustic wooden cottage, lush greenery, and daylight lighting.  
Keywords: Boat, Motorboat, Cabin cruiser, Canopy, Flag, River, Riverbank, Foliage, Leisure, Boating, Outboard motor, Passenger, Railing, Trees, Cottage, Yamaha, Wavey Katey II
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
- *Model load time:* 2.65s
- *Generation time:* 54.52s
- *Total time:* 57.20s
- *Input validation time:* 6.31e-05
- *Prompt preparation time:* 0.0159
- *First-token latency:* 52.3
- *Cleanup time:* 0.124
- *Prompt tokens:* 16,639
- *Generation tokens:* 150
- *Total tokens:* 16,789
- *Prompt throughput (raw):* 318 tok/s
- *Generation throughput (raw):* 111 tok/s
- *Peak memory:* 24
- *Active memory:* 20
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00821
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1365
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
Title: Wavey Katey II Motor Cruiser on River

Description: Captured on 28 August 2026, this white cabin cruiser named 'Wavey Katey II' navigates a calm waterway, flying a Red Ensign flag past a rustic wooden riverside house and lush green trees. The vessel features a blue canopy, Yamaha outboard motor, and white fenders, with passengers visible inside the cabin.

Keywords: Wavey Katey II, motor cruiser, cabin cruiser, British maritime flag, Red Ensign, Yamaha outboard motor, blue canopy, river, riverside house, wooden cabin, green trees, foliage, leisure boating, nautical, passenger, fender, railing, shrubs
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
- *Generation time:* 56.15s
- *Total time:* 57.53s
- *Input validation time:* 7.17e-05
- *Prompt preparation time:* 0.00159
- *First-token latency:* 54
- *Cleanup time:* 0.0918
- *Prompt tokens:* 16,639
- *Generation tokens:* 115
- *Total tokens:* 16,754
- *Prompt throughput (raw):* 308 tok/s
- *Generation throughput (raw):* 91.3 tok/s
- *Peak memory:* 10
- *Active memory:* 6.0
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00873
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1365
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
Wavey Katey II Cruises Past Riverside Cottage on Calm River

Description:
A white cabin cruiser named 'Wavey Katey II' glides along a tranquil river, flying a British maritime flag, with a rustic wooden riverside cottage and lush green trees in the background under soft daylight.

Keywords:
Boat, Cabin cruiser, River, Riverbank, Cottage, Trees, Foliage, Flag, Fender, Outboard motor, Passenger, Railing, Canopy, Leisure, Boating, Nautical, Motorboat
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:  
Wavey Katey II Cruises Past Riverside Cottage on Calm River

Description:  
A white cabin cruiser named 'Wavey Katey II' glides along a tranquil river, flying a British maritime flag, with a rustic wooden riverside cottage and lush green trees in the background under soft daylight.

Keywords:  
Boat, Cabin cruiser, River, Riverbank, Cottage, Trees, Foliage, Flag, Fender, Outboard motor, Passenger, Railing, Canopy, Leisure, Boating, Nautical, Motorboat
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
- *Model load time:* 3.19s
- *Generation time:* 78.40s
- *Total time:* 81.61s
- *Input validation time:* 6.3e-05
- *Prompt preparation time:* 0.0153
- *First-token latency:* 71.5
- *Cleanup time:* 0.15
- *Prompt tokens:* 16,639
- *Generation tokens:* 106
- *Total tokens:* 16,745
- *Prompt throughput (raw):* 233 tok/s
- *Generation throughput (raw):* 17.5 tok/s
- *Peak memory:* 33
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00924
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1365
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
White Cabin Cruiser Cruising Past Riverside Cottage

Description:
The white cabin cruiser 'Wavey Katey II' moves along a calm waterway, flying a red ensign past a rustic wooden riverside house surrounded by lush greenery.

Keywords:
Cabin cruiser, Motorboat, River, Cottage, British flag, Fenders, Outboard motor, Canopy, Riverbank, Trees, Leisure, Nautical, White boat, Waterway, House, Railing, Shrubs, Boating
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
- *Generation time:* 1.68s
- *Total time:* 4.09s
- *Input validation time:* 8.39e-05
- *Prompt preparation time:* 0.0226
- *First-token latency:* 0.402
- *Cleanup time:* 0.121
- *Prompt tokens:* 579
- *Generation tokens:* 101
- *Total tokens:* 680
- *Prompt throughput (raw):* 1,440 tok/s
- *Generation throughput (raw):* 129 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.20
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.012
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1328
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
Title: White motor cruiser Wavey Katey II cruising on river
Description: A white motor cruiser named Wavey Katey II cruises along a calm river past a rustic wooden house and lush green foliage. The boat features a blue canopy and flies a small British maritime flag under bright daylight.
Keywords: Boat, Cabin cruiser, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, River, Riverbank, Shrubs, Trees, Waterway, Watercraft
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
- *Model load time:* 2.57s
- *Generation time:* 5.71s
- *Total time:* 8.29s
- *Input validation time:* 7.49e-05
- *Prompt preparation time:* 0.00191
- *First-token latency:* 1.11
- *Cleanup time:* 0.139
- *Prompt tokens:* 579
- *Generation tokens:* 107
- *Total tokens:* 686
- *Prompt throughput (raw):* 520 tok/s
- *Generation throughput (raw):* 26.1 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.76
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0126
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1328
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
Title: White Motor Cruiser Wavey Katey II on a River
Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green foliage. The boat features a blue canopy and flies a British maritime flag under bright daylight.
Keywords: Boat, Wavey Katey II, cabin cruiser, motorboat, river, riverbank, cottage, British maritime flag, blue canopy, outboard motor, boat fender, nautical, leisure, foliage, trees, cruising
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
- *Observations:* Keyword list has 22 terms (requested 10-18); Duplicate
  keywords: river, trees, passenger
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.67s
- *Generation time:* 15.59s
- *Total time:* 16.28s
- *Input validation time:* 4.21e-05
- *Prompt preparation time:* 0.00975
- *First-token latency:* 13.4
- *Cleanup time:* 0.083
- *Prompt tokens:* 16,623
- *Generation tokens:* 135
- *Total tokens:* 16,758
- *Prompt throughput (raw):* 1,243 tok/s
- *Generation throughput (raw):* 92.3 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1346
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
Title: Wavey Katey II motorboat on river

Description: A white motorboat named 'Wavey Katey II' with a blue canopy and a British flag sails along a calm river, passing a rustic wooden cottage surrounded by lush greenery and trees. The boat is equipped with fenders and an outboard motor, and a passenger is visible in the cabin.

Keywords: motorboat, Wavey Katey II, British flag, river, riverbank, cottage, foliage, trees, leisure, boating, cabin cruiser, canopy, passenger, outboard motor, fender, nautical, river, passenger, railing, shrubs, trees, cruising
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
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.20s
- *Generation time:* 8.17s
- *Total time:* 10.38s
- *Input validation time:* 5.81e-05
- *Prompt preparation time:* 0.00743
- *First-token latency:* 2.67
- *Cleanup time:* 0.107
- *Prompt tokens:* 2,044
- *Generation tokens:* 152
- *Total tokens:* 2,196
- *Prompt throughput (raw):* 765 tok/s
- *Generation throughput (raw):* 30.2 tok/s
- *Peak memory:* 22
- *Active memory:* 18
- *Cache memory:* 0.42
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.00128
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1274
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
White motor cruiser 'Wavey Katey II' on calm waterway

Description:
A white motor cruiser named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees on a sunny afternoon in August 2026.

Keywords:
Boat, Motor cruiser, 'Wavey Katey II', British maritime flag, Calm waterway, Rustic wooden house, Lush green trees, Sunny afternoon, August 2026, Cabin cruiser, Canopy, Outboard motor, Boat fender, Passenger, Railing, Riverbank, Shrubs, Leisure, Nautical
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
- *Observations:* Keyword list has 21 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 11.04s
- *Generation time:* 17.59s
- *Total time:* 28.65s
- *Input validation time:* 6.41e-05
- *Prompt preparation time:* 0.0128
- *First-token latency:* 4.44
- *Cleanup time:* 0.246
- *Prompt tokens:* 1,596
- *Generation tokens:* 711
- *Total tokens:* 2,307
- *Prompt throughput (raw):* 359 tok/s
- *Generation throughput (raw):* 63.9 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.17
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1464
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
Alright, let's get to work. I've got an image of a boat, and I need to generate some metadata for it. First things first, I need to break down the request. It's a straightforward task: title, description, and keywords. The image is of a white motor cruiser, and I have some hints to work with.

Okay, the image itself. It's a white motor cruiser, a "Wavey Katey II," judging by the name on the stern. It's got a blue canopy, and I can see a British flag flying. The setting is a calm river or lake, with a wooden house and some greenery in the background. The lighting seems to be a bit overcast, which is important to note.

Now, the hints. The title hint is a bit of a mess, but I can extract the key elements: boat, flag, house, trees. The description hint is much better, but it has some errors. I need to be careful here. The keywords are a good starting point, but I need to refine them.

Let's start with the title. I need something concise and descriptive. "Wavey Katey II Cruising on the River" is a good start.

Next, the description. I need to combine the context, the image, and the hints. I'll start with the boat's name and the setting. Then I'll add the details about the flag, the house, and the weather. I'll make sure to mention the calm water and the overcast sky.

Finally, the keywords. I need to be comprehensive. I'll include the boat's name, the type of boat, the setting, the key features, and the overall mood. I'll make sure to include the British flag, the wooden house, and the overcast sky.

Okay, let's put it all together.

**Title:** Wavey Katey II Cruising on the River

**Description:** A white motor cruiser named 'Wavey Katey II' sails on a calm river, flying a British flag past a wooden house and lush green trees under an overcast sky.

**Keywords:** Wavey Katey II, boat, motor cruiser, river, wooden house, British flag, overcast sky, white, blue canopy, cabin cruiser, leisure, nautical, riverbank, trees, foliage, houseboat, outboard motor, passenger, cabin, calm water, UK.

That should be it. I've followed the instructions, used the image evidence, and avoided making claims about things I can't see. I'm confident this is a good representation of the image.
&lt;/think&gt;

Title: Wavey Katey II Cruising on the River

Description: A white motor cruiser named 'Wavey Katey II' sails on a calm river, flying a British flag past a wooden house and lush green trees under an overcast sky.

Keywords: Wavey Katey II, boat, motor cruiser, river, wooden house, British flag, overcast sky, white, blue canopy, cabin cruiser, leisure, nautical, riverbank, trees, foliage, houseboat, outboard motor, passenger, cabin, calm water, UK.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got an image of a boat, and I need to generate some metadata for it. First things first, I need to break down the request. It's a straightforward task: title, description, and keywords. The image is of a white motor cruiser, and I have some hints to work with.

Okay, the image itself. It's a white motor cruiser, a "Wavey Katey II," judging by the name on the stern. It's got a blue canopy, and I can see a British flag flying. The setting is a calm river or lake, with a wooden house and some greenery in the background. The lighting seems to be a bit overcast, which is important to note.

Now, the hints. The title hint is a bit of a mess, but I can extract the key elements: boat, flag, house, trees. The description hint is much better, but it has some errors. I need to be careful here. The keywords are a good starting point, but I need to refine them.

Let's start with the title. I need something concise and descriptive. "Wavey Katey II Cruising on the River" is a good start.

Next, the description. I need to combine the context, the image, and the hints. I'll start with the boat's name and the setting. Then I'll add the details about the flag, the house, and the weather. I'll make sure to mention the calm water and the overcast sky.

Finally, the keywords. I need to be comprehensive. I'll include the boat's name, the type of boat, the setting, the key features, and the overall mood. I'll make sure to include the British flag, the wooden house, and the overcast sky.

Okay, let's put it all together.

**Title:** Wavey Katey II Cruising on the River

**Description:** A white motor cruiser named 'Wavey Katey II' sails on a calm river, flying a British flag past a wooden house and lush green trees under an overcast sky.

**Keywords:** Wavey Katey II, boat, motor cruiser, river, wooden house, British flag, overcast sky, white, blue canopy, cabin cruiser, leisure, nautical, riverbank, trees, foliage, houseboat, outboard motor, passenger, cabin, calm water, UK.

That should be it. I've followed the instructions, used the image evidence, and avoided making claims about things I can't see. I'm confident this is a good representation of the image.
</think>

Title: Wavey Katey II Cruising on the River

Description: A white motor cruiser named 'Wavey Katey II' sails on a calm river, flying a British flag past a wooden house and lush green trees under an overcast sky.

Keywords: Wavey Katey II, boat, motor cruiser, river, wooden house, British flag, overcast sky, white, blue canopy, cabin cruiser, leisure, nautical, riverbank, trees, foliage, houseboat, outboard motor, passenger, cabin, calm water, UK.
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
- *Observations:* Unrecognised model control tokens remain visible; Keyword
  list has 21 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 9.14s
- *Generation time:* 20.37s
- *Total time:* 29.52s
- *Input validation time:* 7.16e-05
- *Prompt preparation time:* 0.00918
- *First-token latency:* 17
- *Cleanup time:* 0.255
- *Prompt tokens:* 6,325
- *Generation tokens:* 126
- *Total tokens:* 6,451
- *Prompt throughput (raw):* 372 tok/s
- *Generation throughput (raw):* 43.5 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00274
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1355
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
&lt;|begin_of_box|&gt;Title: Wavey Katey II Motor Cruiser on River
Description: A white motor cruiser named "Wavey Katey II" cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees, with a blue canopy and visible fenders.
Keywords: Boat, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, White boat, British flag, Blue canopy, Rustic house, Lush greenery&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Wavey Katey II Motor Cruiser on River
Description: A white motor cruiser named "Wavey Katey II" cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees, with a blue canopy and visible fenders.
Keywords: Boat, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, White boat, British flag, Blue canopy, Rustic house, Lush greenery<|end_of_box|>
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
- *Model load time:* 1.66s
- *Generation time:* 4.32s
- *Total time:* 6.00s
- *Input validation time:* 6.08e-05
- *Prompt preparation time:* 0.0029
- *First-token latency:* 0.99
- *Cleanup time:* 0.113
- *Prompt tokens:* 2,605
- *Generation tokens:* 95
- *Total tokens:* 2,700
- *Prompt throughput (raw):* 2,631 tok/s
- *Generation throughput (raw):* 34.3 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.21
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00331
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1279
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
Title: Boating on Calm Waterway

Description: A white motor cruiser named 'Wavey Katey II' cruises past a rustic wooden house, flying a British maritime flag, with lush greenery in the background.

Keywords: Boat, Wavey Katey II, Cruiser, British flag, Wooden house, Lush greenery, River, Fender, Cabin, Canopy, Railing, Leisure, Motorboat, Foliage, Outboard motor
</pre>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.65s
- *Generation time:* 1.66s
- *Total time:* 2.33s
- *Input validation time:* 5.89e-05
- *Prompt preparation time:* 0.00449
- *First-token latency:* 0.413
- *Cleanup time:* 0.0791
- *Prompt tokens:* 2,627
- *Generation tokens:* 149
- *Total tokens:* 2,776
- *Prompt throughput (raw):* 6,361 tok/s
- *Generation throughput (raw):* 187 tok/s
- *Peak memory:* 4.0
- *Active memory:* 3.2
- *Cache memory:* 0.08
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.0041
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1325
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
Title: Wavey Katey II Motor Cruiser Cruising

Description: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees. The boat is equipped with a blue canopy, a motor fender, and a cabin cruiser design, offering a leisurely boating experience.

Keywords: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, River, Riverbank, Shrubs, Trees
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
- *Observations:* Keyword list has 21 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.46s
- *Generation time:* 5.45s
- *Total time:* 6.92s
- *Input validation time:* 6.23e-05
- *Prompt preparation time:* 0.00127
- *First-token latency:* 1.68
- *Cleanup time:* 0.0944
- *Prompt tokens:* 2,577
- *Generation tokens:* 215
- *Total tokens:* 2,792
- *Prompt throughput (raw):* 1,535 tok/s
- *Generation throughput (raw):* 65.2 tok/s
- *Peak memory:* 12
- *Active memory:* 8.4
- *Cache memory:* 0.51
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00541
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3712
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
*Wavey Katey II Cruising Along British Waterway (2026-08-28)*

**Description:**
A white Yamaha motor cruiser, *Wavey Katey II* (with blue canopy and fenders attached to its hull railings) cruises past a rustic wooden riverside cottage, flying a British maritime flag. Captured on **2026-08-28 at 16:17:42 UTC+01:00** under calm, reflective sunlight, the boat navigates a serene waterway lined with lush green foliage and shrubs.

**Keywords:**
Boat, *Wavey Katey II*, Yamaha motor cruiser, British maritime flag, cruising, waterway, riverside cottage, blue canopy, boat fenders, hull railings, calm water, serene sunlight, lush foliage, shrubs, leisure boating, nautical scene, wooden cottage, passenger boat, outboard motor, riverbank, leisure cruising
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type cohere_compass)
- *Model load time:* 1.02s
- *Generation time:* 4.31s
- *Total time:* 5.34s
- *Input validation time:* 5.2e-05
- *Prompt preparation time:* 0.00466
- *First-token latency:* 2.99
- *Cleanup time:* 0.0864
- *Prompt tokens:* 4,039
- *Generation tokens:* 150
- *Total tokens:* 4,189
- *Prompt throughput (raw):* 1,350 tok/s
- *Generation throughput (raw):* 231 tok/s
- *Peak memory:* 3.9
- *Active memory:* 2.2
- *Cache memory:* 0.64
- *Model-load active memory:* 2.18
- *Post-cleanup active memory:* 0.00651
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1405
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
Title: Wavey Katey II - A White Motor Cruiser Boat

Description: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees. The boat features a blue canopy, an outboard motor, and a cabin cruiser setup with a visible driver. The surrounding environment includes a serene riverbank with trees and shrubs, and a quaint wooden house in the background.

Keywords: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees
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
- *Observations:* Keyword list has 19 terms (requested 10-18); Duplicate
  keywords: nautical, outboard motor
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Model load time:* 0.96s
- *Generation time:* 3.28s
- *Total time:* 4.24s
- *Input validation time:* 6.85e-05
- *Prompt preparation time:* 0.00184
- *First-token latency:* 0.228
- *Cleanup time:* 0.0876
- *Prompt tokens:* 1,113
- *Generation tokens:* 143
- *Total tokens:* 1,256
- *Prompt throughput (raw):* 4,875 tok/s
- *Generation throughput (raw):* 55.7 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.52
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00708
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1295
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
Title: "Wavey Katey II Boat Cruising on a Serene River"

Description: The white motorboat 'Wavey Katey II' is seen cruising on a calm river, with a British flag flying from its mast, past a rustic riverside house and lush greenery.

Keywords: Wavey Katey II, motorboat, British flag, calm river, rustic house, lush greenery, nautical, outboard motor, passenger, riverbank, foliage, leisure, boating, canopy, cabin cruiser, nautical, outboard motor, flag, river.
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl_moe)
- *Model load time:* 1.88s
- *Generation time:* 56.95s
- *Total time:* 58.84s
- *Input validation time:* 6.23e-05
- *Prompt preparation time:* 0.0014
- *First-token latency:* 54.7
- *Cleanup time:* 0.128
- *Prompt tokens:* 16,623
- *Generation tokens:* 122
- *Total tokens:* 16,745
- *Prompt throughput (raw):* 304 tok/s
- *Generation throughput (raw):* 86.8 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 2.3
- *Model-load active memory:* 18.3
- *Post-cleanup active memory:* 0.0077
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1346
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
Title: White motor cruiser cruising on a river
Description: A white motor cruiser named 'Wavey Katey II' cruises on a calm river, its outboard motor creating a wake. The boat, featuring a blue canopy and a British flag, passes a wooden riverside cottage surrounded by lush green trees and foliage under bright daylight.
Keywords: motor cruiser, boat, river, cottage, outboard motor, canopy, British flag, trees, foliage, shrubs, water, boating, leisure, nautical, cabin cruiser, passenger, railing, fender, lifebuoy, motorboat
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
- *Model load time:* 2.10s
- *Generation time:* 75.83s
- *Total time:* 77.95s
- *Input validation time:* 7.15e-05
- *Prompt preparation time:* 0.0176
- *First-token latency:* 70.7
- *Cleanup time:* 0.122
- *Prompt tokens:* 16,639
- *Generation tokens:* 131
- *Total tokens:* 16,770
- *Prompt throughput (raw):* 235 tok/s
- *Generation throughput (raw):* 30.5 tok/s
- *Peak memory:* 21
- *Active memory:* 16
- *Cache memory:* 1.9
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.00975
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1365
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
Wavey Katey II Cruising Past a Rustic Riverside Cottage

Description:
The white cabin cruiser Wavey Katey II moves along a calm river, flying a red ensign past a wooden riverside cottage and lush green trees. The boat features a blue canvas canopy and white fenders, with passengers visible inside the cabin.

Keywords:
Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees
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
- *Model load time:* 0.69s
- *Generation time:* 1.92s
- *Total time:* 2.62s
- *Input validation time:* 6.66e-05
- *Prompt preparation time:* 0.00292
- *First-token latency:* 0.388
- *Cleanup time:* 0.0896
- *Prompt tokens:* 1,411
- *Generation tokens:* 112
- *Total tokens:* 1,523
- *Prompt throughput (raw):* 3,639 tok/s
- *Generation throughput (raw):* 123 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.35
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.00986
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1306
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
 Title: Cruising Motorboat
Description: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.
Keywords: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees
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
- *Observations:* Title has 11 words (requested 5-10); Keyword list has 20
  terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 7.42s
- *Generation time:* 19.53s
- *Total time:* 26.98s
- *Input validation time:* 7.3e-05
- *Prompt preparation time:* 0.0167
- *First-token latency:* 16.2
- *Cleanup time:* 0.244
- *Prompt tokens:* 3,472
- *Generation tokens:* 118
- *Total tokens:* 3,590
- *Prompt throughput (raw):* 214 tok/s
- *Generation throughput (raw):* 45.9 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.0101
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1352
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
British motor cruiser 'Wavey Katey II' cruising on a calm river

Description:
A white motor cruiser named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.

Keywords:
Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees
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
- *Observations:* Keyword list has 25 terms (requested 10-18); Duplicate
  keywords: cabin cruiser
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.23s
- *Generation time:* 20.87s
- *Total time:* 22.10s
- *Input validation time:* 7.84e-05
- *Prompt preparation time:* 0.00436
- *First-token latency:* 17.5
- *Cleanup time:* 0.102
- *Prompt tokens:* 16,634
- *Generation tokens:* 149
- *Total tokens:* 16,783
- *Prompt throughput (raw):* 951 tok/s
- *Generation throughput (raw):* 57.8 tok/s
- *Peak memory:* 14
- *Active memory:* 9.5
- *Cache memory:* 1.4
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0104
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1404
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
"Wavey Katey II Cruising Along the River"

Description:
A white motor cruiser named "Wavey Katey II" glides through calm waters, its British flag fluttering in the breeze, as it passes a rustic wooden riverside cottage surrounded by lush greenery. The scene captures leisurely boating activity under soft, diffused lighting.

Keywords: Boat, Wavey Katey II, British flag, motor cruiser, riverside cottage, wooden structure, lush foliage, calm water, leisure boating, outboard motor, cabin cruiser, canopy, fenders, shrubs, trees, riverbank, nautical, cabin, passenger, driver, motorboat, cabin cruiser, boating, river, countryside.
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
- *Model load time:* 3.47s
- *Generation time:* 2.35s
- *Total time:* 5.85s
- *Input validation time:* 7.65e-05
- *Prompt preparation time:* 0.0236
- *First-token latency:* 0.336
- *Cleanup time:* 0.15
- *Prompt tokens:* 575
- *Generation tokens:* 88
- *Total tokens:* 663
- *Prompt throughput (raw):* 1,709 tok/s
- *Generation throughput (raw):* 66.2 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0104
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1300
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
&lt;channel|&gt;Title: White motor cruiser Wavey Katey II cruising on a river
Description: A white cabin motor cruiser named Wavey Katey II cruises along a calm waterway, past a British maritime flag and a rustic wooden cottage.
Keywords: Boat, cabin cruiser, motorboat, river, outboard motor, canopy, flag, fender, railing, riverbank, nautical, leisure, foliage, trees, shrubs, cottage
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: White motor cruiser Wavey Katey II cruising on a river
Description: A white cabin motor cruiser named Wavey Katey II cruises along a calm waterway, past a British maritime flag and a rustic wooden cottage.
Keywords: Boat, cabin cruiser, motorboat, river, outboard motor, canopy, flag, fender, railing, riverbank, nautical, leisure, foliage, trees, shrubs, cottage
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
- *Model load time:* 3.30s
- *Generation time:* 2.09s
- *Total time:* 5.40s
- *Input validation time:* 6.31e-05
- *Prompt preparation time:* 0.00181
- *First-token latency:* 0.326
- *Cleanup time:* 0.141
- *Prompt tokens:* 575
- *Generation tokens:* 89
- *Total tokens:* 664
- *Prompt throughput (raw):* 1,766 tok/s
- *Generation throughput (raw):* 80.3 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0104
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1300
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
&lt;channel|&gt;Title: White motor cruiser Wavey Katey II cruising on a river
Description: A white cabin motor cruiser named 'Wavey Katey II' travels along a calm waterway past a rustic wooden riverside cottage and lush green foliage.
Keywords: Boat, cabin cruiser, motorboat, river, riverbank, cruising, outboard motor, canopy, fender, flag, railing, foliage, trees, shrubs, leisure, nautical
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel>thought
<channel|>Title: White motor cruiser Wavey Katey II cruising on a river
Description: A white cabin motor cruiser named 'Wavey Katey II' travels along a calm waterway past a rustic wooden riverside cottage and lush green foliage.
Keywords: Boat, cabin cruiser, motorboat, river, riverbank, cruising, outboard motor, canopy, fender, flag, railing, foliage, trees, shrubs, leisure, nautical
```

</details>

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
- *Model load time:* 2.35s
- *Generation time:* 5.50s
- *Total time:* 7.86s
- *Input validation time:* 7.34e-05
- *Prompt preparation time:* 0.00455
- *First-token latency:* 0.983
- *Cleanup time:* 0.13
- *Prompt tokens:* 570
- *Generation tokens:* 127
- *Total tokens:* 697
- *Prompt throughput (raw):* 580 tok/s
- *Generation throughput (raw):* 31.4 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.43
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.011
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1329
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
Title: Wavey Katey II on River Cruise, August 2026

Description: Captured on 28th August 2026, the white motor cruiser 'Wavey Katey II' cruises along a calm river, passing a wooden house and trees in bright daylight. A British maritime flag is flown from the stern of the boat.

Keywords: Boat, Motorboat, Cabin cruiser, Cruising, River, Riverbank, Trees, Foliage, Cottage, House, Flag, Wavey Katey II, Outboard motor, Railing, Fender, Leisure, Nautical, Canopy, Plus2
</pre>

</details>

---

<a id="model-mlx-community-granite-40-3b-vision-4bit"></a>

### mlx-community/granite-4.0-3b-vision-4bit

<details>
<summary>Complete evidence: mlx-community/granite-4.0-3b-vision-4bit</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Duplicate keywords: outboard motor
- *Arch supported by installed mlx-vlm:* yes (model_type granite4_vision)
- *Model load time:* 0.68s
- *Generation time:* 1.79s
- *Total time:* 2.49s
- *Input validation time:* 5.72e-05
- *Prompt preparation time:* 0.0103
- *First-token latency:* 0.446
- *Cleanup time:* 0.0913
- *Prompt tokens:* 1,510
- *Generation tokens:* 118
- *Total tokens:* 1,628
- *Prompt throughput (raw):* 3,383 tok/s
- *Generation throughput (raw):* 171 tok/s
- *Peak memory:* 4.8
- *Active memory:* 3.0
- *Cache memory:* 0.25
- *Model-load active memory:* 3.03
- *Post-cleanup active memory:* 0.0128
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1498
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
Title: "Wavey Katey II Cruising Along a Serene Waterway"
Description: "A white motor cruiser named 'Wavey Katey II' glides through calm waters, passing by a rustic riverside cottage and lush greenery, under a British maritime flag."
Keywords: Boat, Motor cruiser, British flag, Cottage, Waterway, Trees, Canopy, Outboard motor, Nautical, Leisure, YAMAHA, Outboard motor, British-English, 2026-08-28 16:17:42 UTC+01:00
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.67s
- *Generation time:* 5.05s
- *Total time:* 6.72s
- *Input validation time:* 5.87e-05
- *Prompt preparation time:* 0.00389
- *First-token latency:* 1.45
- *Cleanup time:* 0.105
- *Prompt tokens:* 2,648
- *Generation tokens:* 125
- *Total tokens:* 2,773
- *Prompt throughput (raw):* 1,829 tok/s
- *Generation throughput (raw):* 39.5 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.50
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0131
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1274
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
Motor Cruiser "Wavey Katey II" on Calm River

Description:
The white motor cruiser "Wavey Katey II" cruises along a serene river, displaying a British maritime flag, with a rustic riverside cottage and lush greenery in the background.

Keywords:
Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees
</pre>

</details>

---

<a id="model-liquidai-lfm25-vl-450m-mlx-bf16"></a>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

<details>
<summary>Complete evidence: LiquidAI/LFM2.5-VL-450M-MLX-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Keyword list has 42 terms (requested
  10-18); Duplicate keywords: boat fender, boat driver
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.43s
- *Generation time:* 0.95s
- *Total time:* 1.65s
- *Input validation time:* 0.256
- *Prompt preparation time:* 0.00659
- *First-token latency:* 0.126
- *Cleanup time:* 0.0625
- *Prompt tokens:* 2,627
- *Generation tokens:* 200
- *Total tokens:* 2,827
- *Prompt throughput (raw):* 20,896 tok/s
- *Generation throughput (raw):* 470 tok/s
- *Peak memory:* 1.7
- *Active memory:* 0.90
- *Cache memory:* 0.23
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1325
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
Boat Cruising on a Calm River

Description:
A white motor cruiser boat named 'Wavey Katey II' cruises along a calm river, with a British maritime flag flying and lush green trees in the background.

Keywords:
Boat, Cruising, River, Riverbank, Foliage, Nautical, Outboard motor, Passenger, Cabin cruiser, Canopy, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver,
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
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Title has 13 words (requested 5-10);
  Keyword list has 54 terms (requested 10-18); Duplicate keywords: outboard
  motor, passenger, railing, river, riverbank, shrubs, trees, leisure,
  cruising, nautical
- *Arch supported by installed mlx-vlm:* yes (model_type jvlm via jina_vlm)
- *Model load time:* 0.57s
- *Generation time:* 3.63s
- *Total time:* 4.21s
- *Input validation time:* 4.36e-05
- *Prompt preparation time:* 0.00205
- *First-token latency:* 0.926
- *Cleanup time:* 0.0728
- *Prompt tokens:* 2,047
- *Generation tokens:* 275
- *Total tokens:* 2,322
- *Prompt throughput (raw):* 2,210 tok/s
- *Generation throughput (raw):* 138 tok/s
- *Peak memory:* 3.9
- *Active memory:* 2.1
- *Cache memory:* 1.2
- *Model-load active memory:* 2.11
- *Post-cleanup active memory:* 0.000755
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1286
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
A White Motor Cruiser named 'Wavey Katey II' Cruises Along a Calm Waterway

Description:
A white motor cruiser named 'Wavey Katey II' is seen cruising along a calm waterway, flying a British maritime flag. The boat is passing by a rustic wooden riverside house and lush green trees. The boat driver is visible inside the cabin, enjoying the peaceful surroundings. The boat is equipped with a canopy, fenders, and an outboard motor. The scene is set during the daytime under clear skies.

Keywords:
Boat, Wavey Katey II, Motor Cruiser, British Maritime Flag, Calm Waterway, Rustic Wooden Riverbank, Lush Green Trees, Boat Driver, Cabin Cruiser, Canopy, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River,
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
- *Model load time:* 1.64s
- *Generation time:* 25.69s
- *Total time:* 27.35s
- *Input validation time:* 4.31e-05
- *Prompt preparation time:* 0.008
- *First-token latency:* 1.72
- *Cleanup time:* 0.101
- *Prompt tokens:* 2,739
- *Generation tokens:* 1,000
- *Total tokens:* 3,739
- *Prompt throughput (raw):* 1,595 tok/s
- *Generation throughput (raw):* 42.5 tok/s
- *Peak memory:* 14
- *Active memory:* 13
- *Cache memory:* 0.77
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.00102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1648
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
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time: 2026-08-28 16:17:42 UTC+01:00. The descriptive hints: description hint: "A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees." Keyword hints: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees.

We need to produce:

- a concrete 5-10-word title (so 5-10 words, presumably a short title).
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

We must treat capture date/time as authoritative fact, but not claim they are visible. So we can incorporate date/time in description if needed, but not claim it's visible. The description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. So we need to mention the boat name "Wavey Katey II", the British maritime flag (the Union Jack?), the white motor cruiser, the calm waterway, the rustic wooden riverside house, lush green trees, the canopy (blue canopy), fenders, etc. Also mention the time of day: 16:17:42 UTC+01:00, which is 4:17 PM BST (British Summer Time). Lighting: likely bright daylight, maybe warm afternoon light. The boat appears to be moving, leaving a wake. The boat has a blue canopy, a white hull, a small cabin, a flag on the stern (red? Actually the hint says British maritime flag, which is the Red Ensign with the Union Jack in the canton? But the image shows a red flag? The hint says British maritime flag, but the image shows a red flag? Actually the image shows a red flag on the stern, which could be the British Red Ensign (a red field with the Union Jack in the canton). The hint says British maritime flag, so we can mention that.

We need to produce 10-18 unique keywords. The hints provide many, but we can also add others like "cabin cruiser", "fender", "riverbank", "shrubs", "trees", "cottage", "railings", "outboard motor", "passenger", "leisure", "nautical", "motorboat", "boat driver", "flag", "canopy", "calm water", "wake". We need 10-18 unique keywords, separated by commas.

We need to ensure we have 5-10 words in the title. Title: "Wavey Katey II cruising past riverside cottage". That's 7 words. Or "White cruiser Wavey Katey II on calm river". That's 9 words. Or "Wavey Katey II glides past wooden cottage". That's 7 words. Title must be concrete 5-10 words. Let's choose "White cruiser Wavey Katey II on calm river". That's 9 words.

Now description: 1-2 sentences. Must combine relevant context with main visible subject, setting, action, lighting, distinctive details. So something like: "On 28 August 2026 at 16:17 BST, the white cabin cruiser Wavey Katey II glides along a tranquil river, its blue canopy and British maritime flag contrasting against the lush green foliage and a rustic wooden riverside cottage in the background." That's one sentence. Could add second sentence about lighting: "Bright afternoon light highlights the boat’s polished hull and the gentle wake trailing behind." That's two sentences. That's fine.

Now keywords: 10-18 unique, comma-separated. Let's list: Boat, Cabin cruiser, Wavey Katey II, British flag, Blue canopy, River, Calm water, Riverside cottage, Wooden house, Lush foliage, Trees, Shrubs, Fender, Outboard motor, Passenger, Nautical, Leisure, Wake. That's 18 keywords. Ensure they are unique and separated by commas. That's fine.

We need to ensure we
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
- *Model load time:* 0.63s
- *Generation time:* 1.32s
- *Total time:* 1.96s
- *Input validation time:* 6.38e-05
- *Prompt preparation time:* 0.00214
- *First-token latency:* 0.0948
- *Cleanup time:* 0.081
- *Prompt tokens:* 314
- *Generation tokens:* 41
- *Total tokens:* 355
- *Prompt throughput (raw):* 3,312 tok/s
- *Generation throughput (raw):* 339 tok/s
- *Peak memory:* 2.2
- *Active memory:* 1.2
- *Cache memory:* 0.03
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1369
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
A serene white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.
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
- *Observations:* Extra text appears before the Title field; Response appears
  cut off at the token limit; Internal reasoning block appears incomplete;
  Title has 11 words (requested 5-10); Keyword list has 29 terms (requested
  10-18); Duplicate keywords: trees
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.69s
- *Generation time:* 27.31s
- *Total time:* 29.01s
- *Input validation time:* 4.52e-05
- *Prompt preparation time:* 0.00271
- *First-token latency:* 5.35
- *Cleanup time:* 0.102
- *Prompt tokens:* 6,321
- *Generation tokens:* 1,000
- *Total tokens:* 7,321
- *Prompt throughput (raw):* 1,182 tok/s
- *Generation throughput (raw):* 46.5 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.43
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1331
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
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The boat is named "Wavey Katey II" and it's a motor cruiser. So maybe "Wavey Katey II Motor Cruiser on Waterway".

Next, the description: need to combine context (capture date, maybe) with main subject, setting, action, lighting, distinctive details. The boat is cruising, past a wooden riverside house, flying a British flag, calm waterway, etc. Let's make it a 1-2 sentence factual description.

Then keywords: need 10-18 unique ones from the hints, covering context and visible details. Let's list them out: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs. Wait, let's check the hints. The hints include Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees. So pick unique ones, maybe 12-15.

Now, let's structure each part:

Title: "Wavey Katey II Motor Cruiser on Waterway" (about 8 words, concrete)

Description: Combine context (capture date) with subject, setting, action. So "The motor cruiser 'Wavey Katey II' cruises along a calm waterway on 2026-08-28, passing a rustic wooden riverside house with lush greenery, flying a British maritime flag." Wait, need to make sure it's 1-2 sentences. Let's make it: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, with a British maritime flag visible."

Keywords: Let's list them comma-separated. Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs. That's 17, which is within 10-18.

Wait, let's check the image details. The boat has "4PLUS2" on it, "Wavey Katey II" on the side, Yamaha outboard motor, blue canopy, fenders, British flag. The setting is a river with a wooden cottage, trees, etc.

So finalizing:

Title: "Wavey Katey II Motor Cruiser"

Description: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, flying a British maritime flag."

Keywords: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical

Wait, let's count keywords: 19, maybe trim one. Let's do Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical. That's 19, maybe remove "Nautical" to make 18.

Wait, the hints include "Nautical" as a keyword, so maybe include it. Let's check the image: the boat is on a river, has maritime elements, so Nautical is appropriate.

Now, let's make sure the description is factual, combining context (capture date) with subject, setting, action, lighting (calm waterway, maybe daylight), distinctive details (British flag, wooden house, lush trees).

Final check:

Title: "Wavey Katey II Motor Cruiser" (5 words, concrete)

Description: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, flying a British maritime flag." (1 sentence, combines context, subject, setting, action, distinctive details)

Keywords: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical (18
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The boat is named "Wavey Katey II" and it's a motor cruiser. So maybe "Wavey Katey II Motor Cruiser on Waterway". 

Next, the description: need to combine context (capture date, maybe) with main subject, setting, action, lighting, distinctive details. The boat is cruising, past a wooden riverside house, flying a British flag, calm waterway, etc. Let's make it a 1-2 sentence factual description. 

Then keywords: need 10-18 unique ones from the hints, covering context and visible details. Let's list them out: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs. Wait, let's check the hints. The hints include Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees. So pick unique ones, maybe 12-15. 

Now, let's structure each part:

Title: "Wavey Katey II Motor Cruiser on Waterway" (about 8 words, concrete)

Description: Combine context (capture date) with subject, setting, action. So "The motor cruiser 'Wavey Katey II' cruises along a calm waterway on 2026-08-28, passing a rustic wooden riverside house with lush greenery, flying a British maritime flag." Wait, need to make sure it's 1-2 sentences. Let's make it: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, with a British maritime flag visible." 

Keywords: Let's list them comma-separated. Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs. That's 17, which is within 10-18. 

Wait, let's check the image details. The boat has "4PLUS2" on it, "Wavey Katey II" on the side, Yamaha outboard motor, blue canopy, fenders, British flag. The setting is a river with a wooden cottage, trees, etc. 

So finalizing:

Title: "Wavey Katey II Motor Cruiser"

Description: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, flying a British maritime flag."

Keywords: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical

Wait, let's count keywords: 19, maybe trim one. Let's do Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical. That's 19, maybe remove "Nautical" to make 18. 

Wait, the hints include "Nautical" as a keyword, so maybe include it. Let's check the image: the boat is on a river, has maritime elements, so Nautical is appropriate. 

Now, let's make sure the description is factual, combining context (capture date) with subject, setting, action, lighting (calm waterway, maybe daylight), distinctive details (British flag, wooden house, lush trees). 

Final check:

Title: "Wavey Katey II Motor Cruiser" (5 words, concrete)

Description: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, flying a British maritime flag." (1 sentence, combines context, subject, setting, action, distinctive details)

Keywords: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical (18
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
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Keyword list has 50 terms (requested
  10-18); Duplicate keywords: boat, boat driver, boat fender
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.35s
- *Generation time:* 8.55s
- *Total time:* 9.91s
- *Input validation time:* 6.33e-05
- *Prompt preparation time:* 0.00795
- *First-token latency:* 5.61
- *Cleanup time:* 0.0911
- *Prompt tokens:* 6,325
- *Generation tokens:* 200
- *Total tokens:* 6,525
- *Prompt throughput (raw):* 1,127 tok/s
- *Generation throughput (raw):* 80.5 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.73
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1355
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
Title: Wavey Katey II Motor Cruiser
Description: The white motor cruiser 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.
Keywords: Wavey Katey II, motor cruiser, British maritime flag, calm waterway, rustic wooden riverside house, lush green trees, boat, boating, leisure, motorboat, nautical, outboard motor, river, riverbank, shrubs, trees, canopy, cabin cruiser, fender, passenger, railing, flag, foliage, cottage, cruising, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat
</pre>

</details>

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### mlx-community/Idefics3-8B-Llama3-bf16

<details>
<summary>Complete evidence: mlx-community/Idefics3-8B-Llama3-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Keyword list has 61 terms (requested
  10-18); Duplicate keywords: boat, waterway, flag, motor, cruiser, maritime,
  british
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 1.94s
- *Generation time:* 9.49s
- *Total time:* 11.44s
- *Input validation time:* 6.32e-05
- *Prompt preparation time:* 0.0022
- *First-token latency:* 1.09
- *Cleanup time:* 0.108
- *Prompt tokens:* 2,603
- *Generation tokens:* 250
- *Total tokens:* 2,853
- *Prompt throughput (raw):* 2,383 tok/s
- *Generation throughput (raw):* 32.0 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.83
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1311
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
 Title: "Wavey Katey II Cruises Along Calm Waterway".

Description: "The 'Wavey Katey II' motor cruiser boat, adorned with a British maritime flag, glides smoothly along a serene waterway. The boat, painted white with a blue canopy, is accompanied by a rustic wooden riverside house and lush green trees, creating a picturesque scene. The boat's outboard motor propels it forward, and the calm water reflects the surrounding foliage, adding to the tranquil atmosphere."

Keywords: motor cruiser, boat, Wavey Katey II, British maritime flag, calm waterway, rustic wooden riverside house, lush green trees, serene, outboard motor, picturesque, tranquil, waterway, boat, flag, canopy, motor, boat, motor, cruiser, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British,
</pre>

</details>

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

<details>
<summary>Complete evidence: mlx-community/Kimi-VL-A3B-Thinking-2506-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Extra text appears before
  the Title field; Response appears cut off at the token limit; Internal
  reasoning block appears incomplete; Keyword list has 73 terms (requested
  10-18); Duplicate keywords: motorboat, cruising, river, calm water, rustic
  house, lush foliage, british flag, fenders, wait, boating, cabin cruiser,
  leisure, nautical
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Model load time:* 3.27s
- *Generation time:* 216.04s
- *Total time:* 219.33s
- *Input validation time:* 5.53e-05
- *Prompt preparation time:* 0.00353
- *First-token latency:* 1.35
- *Cleanup time:* 0.161
- *Prompt tokens:* 1,307
- *Generation tokens:* 1,000
- *Total tokens:* 2,307
- *Prompt throughput (raw):* 969 tok/s
- *Generation throughput (raw):* 4.67 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.9
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00397
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1461
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
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and context provided. Let's start by analyzing the image.

The image shows a white motor cruiser named "Wavey Katey II" on a calm waterway. There's a rustic wooden riverside house in the background, lush green trees, and the boat has a British maritime flag. The capture date/time is 2026-08-28 16:17:42 UTC+01:00.

For the title, I need a 5-10 word British-English title. Maybe "Wavey Katey II Cruising on a Calm River". That's concise and descriptive.

Next, the description: combine context and main subject. The boat is a motor cruiser, name, on a calm river, with a rustic house and trees. So: "A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying."

Keywords: Need 10-18 unique keywords. Let's list them: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Motor Cruiser, Fenders, Cabin Cruiser. That's 10 keywords.

Wait, check if all hints are covered. The hints include Boat, Boat driver (but no driver visible), Boat fender (yes, the fenders are there), Boating, Cabin cruiser, Canopy (the blue canopy), Cottage (the house), Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor (maybe the engine), Passenger (no visible passenger), Railing (the house's railing), River, Riverbank, Shrubs, Trees. But the image might not show all. Let's pick the most visible: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical. That's 10. Maybe add "Leisure" as well. Let's adjust.

Wait, the hints mention "Boating" and "Leisure" as keywords. So include those. Let's check again. The image has the boat, motorboat, cruising, river, calm water, rustic house, lush foliage, British flag, fenders, cabin cruiser, nautical, boating, leisure. That's 12. Maybe "Outboard motor" if visible. The boat has an outboard? The image shows the engine area, so maybe. Let's include that. So: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Outboard Motor. That's 12. Maybe that's enough.

Now, structure:

Title: Wavey Katey II Cruising on a Calm River

Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying.

Keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Outboard Motor

Wait, the hints also mention "Passenger" but no passenger is visible. So maybe omit. Let's stick to what's visible. Let's check the image again. The boat has a canopy (blue), so include Canopy. So add Canopy. Now keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Canopy, Outboard Motor. That's 13. Maybe that's okay.

Wait, the hints also have "Shrubs" and "Trees" but the image has trees in the background. Maybe "Trees" is covered under Lush Foliage. So maybe not necessary. Let's finalize.

Title: Wavey Katey II Cruising on a Calm River

Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying.

Keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Canopy, Outboard Motor

Wait, the hints mention "Boating" and "Leisure" as keywords. Let's check if all are covered. Yes. The boat is a cabin cruiser, so Cabin Cruiser is good. The outboard
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and context provided. Let's start by analyzing the image.

The image shows a white motor cruiser named "Wavey Katey II" on a calm waterway. There's a rustic wooden riverside house in the background, lush green trees, and the boat has a British maritime flag. The capture date/time is 2026-08-28 16:17:42 UTC+01:00. 

For the title, I need a 5-10 word British-English title. Maybe "Wavey Katey II Cruising on a Calm River". That's concise and descriptive.

Next, the description: combine context and main subject. The boat is a motor cruiser, name, on a calm river, with a rustic house and trees. So: "A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying."

Keywords: Need 10-18 unique keywords. Let's list them: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Motor Cruiser, Fenders, Cabin Cruiser. That's 10 keywords.

Wait, check if all hints are covered. The hints include Boat, Boat driver (but no driver visible), Boat fender (yes, the fenders are there), Boating, Cabin cruiser, Canopy (the blue canopy), Cottage (the house), Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor (maybe the engine), Passenger (no visible passenger), Railing (the house's railing), River, Riverbank, Shrubs, Trees. But the image might not show all. Let's pick the most visible: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical. That's 10. Maybe add "Leisure" as well. Let's adjust.

Wait, the hints mention "Boating" and "Leisure" as keywords. So include those. Let's check again. The image has the boat, motorboat, cruising, river, calm water, rustic house, lush foliage, British flag, fenders, cabin cruiser, nautical, boating, leisure. That's 12. Maybe "Outboard motor" if visible. The boat has an outboard? The image shows the engine area, so maybe. Let's include that. So: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Outboard Motor. That's 12. Maybe that's enough.

Now, structure:

Title: Wavey Katey II Cruising on a Calm River

Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying.

Keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Outboard Motor

Wait, the hints also mention "Passenger" but no passenger is visible. So maybe omit. Let's stick to what's visible. Let's check the image again. The boat has a canopy (blue), so include Canopy. So add Canopy. Now keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Canopy, Outboard Motor. That's 13. Maybe that's okay.

Wait, the hints also have "Shrubs" and "Trees" but the image has trees in the background. Maybe "Trees" is covered under Lush Foliage. So maybe not necessary. Let's finalize.

Title: Wavey Katey II Cruising on a Calm River

Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying.

Keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Canopy, Outboard Motor

Wait, the hints mention "Boating" and "Leisure" as keywords. Let's check if all are covered. Yes. The boat is a cabin cruiser, so Cabin Cruiser is good. The outboard
```

</details>

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
- *Model load time:* 0.88s
- *Generation time:* 0.99s
- *Total time:* 1.89s
- *Input validation time:* 5.94e-05
- *Prompt preparation time:* 0.0094
- *First-token latency:* 0.243
- *Cleanup time:* 0.0925
- *Prompt tokens:* 916
- *Generation tokens:* 50
- *Total tokens:* 966
- *Prompt throughput (raw):* 3,776 tok/s
- *Generation throughput (raw):* 271 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00488
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1342
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
White motorboat cruising near rustic cottage on calm water.

Keywords: boat, cruiser, waterway, cottage, flag, fenders, canopy, leisure, outboard, trees, railing, foliage, nautical, river, cabin, passenger
</pre>

</details>

---

<a id="model-mlx-community-molmo2-8b-4bit"></a>

### mlx-community/Molmo2-8B-4bit

<details>
<summary>Complete evidence: mlx-community/Molmo2-8B-4bit</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Missing or empty fields:
  Title, Description, Keywords; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Model load time:* 1.15s
- *Generation time:* 15.14s
- *Total time:* 16.30s
- *Input validation time:* 5.44e-05
- *Prompt preparation time:* 0.00596
- *First-token latency:* 0.734
- *Cleanup time:* 0.0897
- *Prompt tokens:* 1,504
- *Generation tokens:* 1,000
- *Total tokens:* 2,504
- *Prompt throughput (raw):* 2,049 tok/s
- *Generation throughput (raw):* 72.1 tok/s
- *Peak memory:* 9.0
- *Active memory:* 6.5
- *Cache memory:* 0.40
- *Model-load active memory:* 6.45
- *Post-cleanup active memory:* 0.00598
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1312
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
Wavey Katey II Cruises Calm River Past Rustic Cottage

A white motor cruiser named Wavey Katey II glides smoothly along a tranquil waterway, its blue canopy and red flag visible against the backdrop of a rustic wooden riverside house and lush green trees. The boat's name is clearly displayed on its side, and the scene captures a peaceful moment of leisure boating.

Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Waterway, Rustic house, Blue canopy, Red flag, White boat, Calm water, Rustic wooden house, Lush green trees, Boat name, Boat side, Boat side name, Boat side display, Boat side text, Boat side writing, Boat side lettering, Boat side branding, Boat side logo, Boat side emblem, Boat side insignia, Boat side marking, Boat side identification, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text,
</pre>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### mlx-community/gemma-3n-E4B-it-bf16

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-bf16</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* No response text was returned; Missing or empty fields:
  Title, Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 2.31s
- *Generation time:* 0.84s
- *Total time:* 3.17s
- *Input validation time:* 0.000163
- *Prompt preparation time:* 0.00427
- *First-token latency:* 0.376
- *Cleanup time:* 0.126
- *Prompt tokens:* 569
- *Generation tokens:* 1
- *Total tokens:* 570
- *Prompt throughput (raw):* 1,513 tok/s
- *Generation throughput (raw):* 62,174 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0115
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1331
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

#### Complete generated output

empty output

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
- *Model load time:* 1.03s
- *Generation time:* -
- *Total time:* 1.03s
- *Input validation time:* 5.2e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.145
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.0131
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13593, in _run_model_generation
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 818, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1217, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1362, in load_processor
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14606, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13608, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)

```

#### Captured upstream output

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 19 files:   0%|          | 0/19 [00:00<?, ?it/s]
Fetching 19 files: 100%|##########| 19/19 [00:00<00:00, 2785.16it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[01:47:22] DEBUG    HF Cache Info for tencent/Youtu-VL-4B-Instruct: size=10229.6 MB, files=27
```

</details>

---
