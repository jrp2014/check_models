# Model Output Gallery

Generated on: 2026-08-30 22:59:16 BST

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
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable`              | 18.61s  | 62.8 tok/s          | 1.35            | 60      | 1,669      | 706     | none                                                                                                                        |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | 5.94s   | 34.3 tok/s          | 0.82            | 17      | 2,147      | 94      | none                                                                                                                        |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | 2.31s   | 184 tok/s           | 0.34            | 4.0     | 2,150      | 142     | none                                                                                                                        |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | 2.39s   | 211 tok/s           | 0.46            | 4.0     | 2,141      | 90      | none                                                                                                                        |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | 3.19s   | 189 tok/s           | 0.96            | 7.8     | 2,958      | 123     | none                                                                                                                        |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                      | `usable`              | 5.27s   | 106 tok/s           | 0.79            | 25      | 1,323      | 129     | none                                                                                                                        |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | 4.43s   | 56.5 tok/s          | 0.26            | 9.4     | 1,174      | 150     | none                                                                                                                        |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable`              | 51.82s  | 86.5 tok/s          | 47.65           | 23      | 16,581     | 113     | none                                                                                                                        |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | 54.11s  | 111 tok/s           | 49.37           | 24      | 16,597     | 134     | none                                                                                                                        |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | 54.88s  | 91.7 tok/s          | 51.34           | 10.0    | 16,597     | 103     | none                                                                                                                        |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | 79.30s  | 17.6 tok/s          | 68.97           | 33      | 16,597     | 110     | none                                                                                                                        |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | 75.26s  | 30.7 tok/s          | 68.43           | 21      | 16,597     | 115     | none                                                                                                                        |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | 22.50s  | 46.5 tok/s          | 13.90           | 70      | 3,525      | 102     | none                                                                                                                        |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable`              | 6.16s   | 46.6 tok/s          | 0.33            | 28      | 627        | 85      | none                                                                                                                        |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | 7.89s   | 31.5 tok/s          | 1.02            | 17      | 626        | 126     | none                                                                                                                        |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | 4.04s   | 128 tok/s           | 0.41            | 16      | 631        | 87      | none                                                                                                                        |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | 7.65s   | 26.3 tok/s          | 1.12            | 20      | 631        | 89      | none                                                                                                                        |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | 6.16s   | 39.6 tok/s          | 1.70            | 16      | 3,149      | 91      | none                                                                                                                        |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable with caveats` | 1.73s   | 480 tok/s           | 0.12            | 1.9     | 2,150      | 118     | title/keyword constraints failed                                                                                            |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable with caveats` | 16.08s  | 92.2 tok/s          | 13.24           | 8.4     | 16,581     | 121     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable with caveats` | 9.29s   | 30.4 tok/s          | 3.17            | 23      | 2,426      | 102     | title/keyword constraints failed                                                                                            |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable with caveats` | 8.97s   | 80.1 tok/s          | 5.78            | 8.4     | 6,484      | 99      | title/keyword constraints failed                                                                                            |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable with caveats` | 20.04s  | 43.8 tok/s          | 11.54           | 78      | 6,484      | 100     | control tokens visible; title/keyword constraints failed                                                                    |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable with caveats` | 8.59s   | 32.2 tok/s          | 1.10            | 18      | 2,646      | 156     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable with caveats` | 166.01s | 4.67 tok/s          | 1.26            | 40      | 1,356      | 750     | role tokens visible                                                                                                         |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable with caveats` | 6.22s   | 67.2 tok/s          | 1.97            | 13      | 2,959      | 146     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable with caveats` | 6.82s   | 64.7 tok/s          | 2.02            | 13      | 2,959      | 179     | title/keyword constraints failed                                                                                            |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `usable with caveats` | 4.39s   | 72.7 tok/s          | 0.75            | 9.1     | 1,558      | 135     | title/keyword constraints failed                                                                                            |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable with caveats` | 5.20s   | 233 tok/s           | 3.05            | 3.9     | 4,118      | 77      | title/keyword constraints failed                                                                                            |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable with caveats` | 24.81s  | 92.6 tok/s          | 13.29           | 8.4     | 16,583     | 902     | title/keyword constraints failed                                                                                            |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable with caveats` | 5.78s   | 61.3 tok/s          | 0.34            | 29      | 627        | 83      | title/keyword constraints failed                                                                                            |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable with caveats` | 2.21s   | 171 tok/s           | 0.40            | 4.7     | 1,411      | 66      | title/keyword constraints failed                                                                                            |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | 4.64s   | 138 tok/s           | 0.96            | 3.7     | 2,101      | 300     | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | 28.08s  | 42.1 tok/s          | 1.98            | 15      | 3,240      | 1,000   | missing required fields; echoes instructions; cut off at token limit                                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | 2.42s   | 352 tok/s           | 0.10            | 2.2     | 368        | 169     | missing required fields; extra text before Title                                                                            |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | 29.30s  | 46.6 tok/s          | 5.51            | 13      | 6,480      | 1,000   | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | 1.98s   | 273 tok/s           | 0.25            | 3.8     | 966        | 46      | missing required fields; extra text before Title                                                                            |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `unusable`            | 3.44s   | 124 tok/s           | 0.39            | 5.5     | 1,461      | 200     | repeated text; stopped early: repeating; missing required fields                                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | 36.59s  | 55.6 tok/s          | 16.57           | 13      | 16,592     | 1,000   | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | 3.26s   | insufficient sample | 0.40            | 17      | 625        | 1       | empty response; missing required fields                                                                                     |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `not evaluated`       | 0.50s   | -                   | -               | -       | -          | -       | none                                                                                                                        |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not evaluated`       | 1.17s   | -                   | -               | -       | -          | -       | none                                                                                                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Fastest clean completion: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 211 tok/s

Average clean-completion throughput: 83.3 tok/s (indicative only: tokenizers and architectures differ across models)

Lowest peak memory among clean completions: `mlx-community/LFM2.5-VL-3B-OptiQ-4bit` at 4.0 GB

## Avoid for This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                  | Usability       | Observations                                                                                                                |
|--------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------|
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                    | `unusable`      | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx) | `unusable`      | missing required fields; echoes instructions; cut off at token limit                                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                             | `unusable`      | missing required fields; extra text before Title                                                                            |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)             | `unusable`      | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                           | `unusable`      | missing required fields; extra text before Title                                                                            |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)           | `unusable`      | repeated text; stopped early: repeating; missing required fields                                                            |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                          | `unusable`      | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                      | `unusable`      | empty response; missing required fields                                                                                     |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)        | `not evaluated` | none                                                                                                                        |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                  | `not evaluated` | none                                                                                                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

The first 280 characters of each model's actual output (or failure evidence for crashes), in chooser order. Complete exact evidence for every model follows below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                   | Usability             | Output preview                                                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | `usable`              | Alright, let's get to work. I've got the image and the hints, and I need to generate some metadata. First, the title. It needs to be concise, descriptive, and accurate. "Scaffolding on Princes Street, Edinburgh" is a good start, but I can do better. "Extensive Scaffolding on E...                         |
| [`mlx-community/InternVL3-8B-bf16`](#model-mlx-community-internvl3-8b-bf16)                                             | `usable`              | Title: Edinburgh City Centre Princes Street<br>Description: Extensive scaffolding covers a building under renovation on Princes Street, with pedestrians walking past a Boots pharmacy under an overcast sky.<br>Keywords: Edinburgh, City Centre, Princes Street, Scaffolding, Renovation,...                   |
| [`mlx-community/LFM2.5-VL-1.6B-bf16`](#model-mlx-community-lfm25-vl-16b-bf16)                                           | `usable`              | Title: "Edinburgh's Urban Renewal: Scaffolding and Construction on Princes Street"<br><br>Description: A bustling city centre in Edinburgh, Scotland, is undergoing significant redevelopment. Extensive scaffolding covers a building along Princes Street, indicating major renovation an...                   |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                  | `usable`              | Title: Edinburgh Princes Street Construction Site<br>Description: Extensive scaffolding covers a building undergoing renovation along Princes Street, with pedestrians walking past temporary fences and a Boots pharmacy under an overcast sky.<br>Keywords: Edinburgh, Scotland, UK, Euro...                   |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)             | `usable`              | Title:<br>**Princes Street Construction: Edinburgh’s Facade in Renewal**<br><br>Description:<br>A major renovation under scaffolding transforms a historic building in Edinburgh’s bustling city centre along Princes Street, with cranes and protective barriers visible under an overcast sky....              |
| [`mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit`](#model-mlx-community-ornith-15-35b-a3b-optiq-4bit)                      | `usable`              | Title:<br>Scaffolding Covers Building Renovation on Princes Street, Edinburgh<br><br>Description:<br>Extensive scaffolding envelops the facade of a building undergoing major redevelopment along Princes Street in Edinburgh's city centre, beside a Boots pharmacy beneath an overcast sky. Ped...             |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                        | `usable`              | Title: Edinburgh's Princes Street Undergoing Major Renovation<br><br>Description: Pedestrians stroll past a building undergoing extensive renovation on Princes Street in Edinburgh, with scaffolding and construction fences visible.<br><br>Keywords: Edinburgh, Princes Street, renovation, sc...             |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                   | `usable`              | Title: Extensive scaffolding on Princes Street, Edinburgh<br>Description: A wide shot captures a building under major renovation on Princes Street in Edinburgh, Scotland, its facade entirely covered by scaffolding. Pedestrians walk along the pavement past a Boots pharmacy and tem...                      |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                        | `usable`              | Title: Princes Street Building Renovation with Scaffolding<br><br>Description: An extensive construction site on Princes Street in Edinburgh features a multi-story building fully encased in scaffolding and protective sheeting, situated next to a Boots pharmacy under an overcast sky....                   |
| [`mlx-community/Qwen3.5-9B-MLX-4bit`](#model-mlx-community-qwen35-9b-mlx-4bit)                                          | `usable`              | Title:<br>Scaffolding-covered building under renovation on Princes Street, Edinburgh<br><br>Description:<br>Extensive scaffolding envelops a historic building undergoing redevelopment along Princes Street in Edinburgh, with pedestrians walking past temporary fencing and a Boots pharmacy u...             |
| [`mlx-community/Qwen3.6-27B-mxfp8`](#model-mlx-community-qwen36-27b-mxfp8)                                              | `usable`              | Title:<br>Edinburgh Princes Street Construction and Boots Pharmacy<br><br>Description:<br>Extensive scaffolding covers a building undergoing major renovation on Princes Street in Edinburgh, situated next to a Boots pharmacy. Pedestrians walk past temporary construction fences and a yellow...             |
| [`mlx-community/Qwen3.8-27B-4bit`](#model-mlx-community-qwen38-27b-4bit)                                                | `usable`              | Title:<br>Scaffolding on Princes Street, Edinburgh, Scotland<br><br>Description:<br>A large building on Princes Street in Edinburgh is covered in extensive scaffolding and green safety netting, indicating major renovation works. Pedestrians walk past a Boots pharmacy and temporary constru...             |
| [`mlx-community/Step-3.7-Flash-oQ2e`](#model-mlx-community-step-37-flash-oq2e)                                          | `usable`              | Title:<br>Edinburgh City Centre Scaffolding and Construction Works<br><br>Description:<br>Extensive scaffolding covers a building undergoing renovation along Princes Street in Edinburgh, Scotland, with pedestrians walking past temporary construction fences and a Boots pharmacy under an ov...             |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | `usable`              | Title: Building Renovation on Princes Street, Edinburgh City Centre<br>Description: Extensive scaffolding and a crane cover a building undergoing renovation along Princes Street in Edinburgh while pedestrians walk past a Boots pharmacy under an overcast sky.<br>Keywords: Edinburgh,...                    |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                 | `usable`              | Title: Princes Street Building Renovation, Edinburgh<br><br>Description: Captured on 18th August 2026, this view shows extensive scaffolding covering a building undergoing renovation on Princes Street, Edinburgh, with pedestrians passing a Boots pharmacy under an overcast sky. A yel...                   |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                 | `usable`              | Title: Scaffolding on building on Princes Street, Edinburgh<br>Description: Extensive scaffolding covers a building undergoing renovation on Princes Street in Edinburgh under an overcast sky. Pedestrians walk past construction fences and a Boots pharmacy on the street level.<br>Keyw...                   |
| [`mlx-community/gemma-4-31b-it-4bit`](#model-mlx-community-gemma-4-31b-it-4bit)                                         | `usable`              | Title: Construction and Scaffolding on Princes Street, Edinburgh<br>Description: Extensive scaffolding covers a building undergoing renovation along Princes Street in Edinburgh, Scotland, under an overcast sky. Pedestrians walk past construction fences and a Boots pharmacy storef...                      |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                               | `usable`              | Title:<br>Princes Street Redevelopment, Edinburgh, Scotland<br><br>Description:<br>Scaffolding covers a building undergoing renovation along Princes Street in Edinburgh, Scotland, with pedestrians walking past under an overcast sky.<br><br>Keywords:<br>Princes Street, Edinburgh, Scotland, Scaffold...    |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                            | `usable_with_caveats` | Title:<br>City Centre, Edinburgh, Scotland, UK, GBR, Europe<br><br>Description:<br>A bustling city street scene in Edinburgh, Scotland, featuring a construction site on Princes Street with scaffolding and a Boots pharmacy. Pedestrians walk along the sidewalk, and a crane is visible in the...             |
| [`Qwen/Qwen3-VL-2B-Instruct`](#model-qwen-qwen3-vl-2b-instruct)                                                         | `usable_with_caveats` | Title:<br>Edinburgh Princes Street Construction<br><br>Description:<br>Extensive scaffolding covers a building undergoing renovation on Princes Street in Edinburgh, Scotland, under an overcast sky, with pedestrians walking past a Boots pharmacy and construction fences. A yellow crane is v...             |
| [`mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`](#model-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | `usable_with_caveats` | Title:<br>Edinburgh Princes Street Construction<br><br>Description:<br>Extensive scaffolding covers a building undergoing major renovation along Princes Street in Edinburgh, Scotland, under an overcast sky, with pedestrians walking past temporary construction fences and a Boots pharmacy....              |
| [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)                                        | `usable_with_caveats` | Title: Princes Street Renovation, Edinburgh<br>Description: The image shows a construction site on Princes Street in Edinburgh, Scotland, with extensive scaffolding covering a building under renovation. Pedestrians walk past temporary construction fences, and a Boots pharmacy is...                       |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                    | `usable_with_caveats` | <\|begin_of_box\|>Title: Princes Street Construction, Edinburgh<br>Description: A construction site with extensive scaffolding and a crane on Princes Street in Edinburgh, Scotland, under an overcast sky, with pedestrians passing a Boots pharmacy and temporary fences.<br>Keywords: Prin...                 |
| [`mlx-community/Idefics3-8B-Llama3-bf16`](#model-mlx-community-idefics3-8b-llama3-bf16)                                 | `usable_with_caveats` | Title: Extensive Scaffolding on Princes Street Building Renovation<br><br>Description: A building on Princes Street in Edinburgh, Scotland, is undergoing significant renovation and redevelopment, as evidenced by the extensive scaffolding covering its facade. Pedestrians are seen wal...                   |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`](#model-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | `usable_with_caveats` | ◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the given context and image. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords.<br><br>Starting with the title: The image shows scaffolding on a build...                   |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)         | `usable_with_caveats` | Title:<br>**Princes Street, Edinburgh – Major Building Renovation Underway**<br><br>Description:<br>A large multi-storey building on Princes Street, Edinburgh, is undergoing extensive renovation with full scaffolding coverage, while construction fences and a yellow crane are visible. Pede...             |
| [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)         | `usable_with_caveats` | **Title:**<br>*Princes Street, Edinburgh: Scaffolding &amp; Renovation*<br><br>**Description:**<br>On **18 August 2026** in Edinburgh’s city centre (GPS: **55.951722°N, 3.201417°W**), extensive scaffolding envelops a multi-storey building under major renovation along **Princes Street**. Pedes...         |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                   | `usable_with_caveats` | Title: Princes Street Renovation: Scaffolding and Boots Pharmacy<br><br>Description: The image captures a bustling scene in Edinburgh's city centre, featuring extensive scaffolding on a building undergoing major renovation. A Boots pharmacy is visible beneath the scaffolding, while...                    |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)               | `usable_with_caveats` | Title: Edinburgh City Centre Renovation<br><br>Description: Extensive scaffolding covers the facade of a building undergoing major renovation and redevelopment along Princes Street in Edinburgh, Scotland. Pedestrians walk past temporary construction fences and a Boots pharmacy benea...                   |
| [`mlx-community/Qwen3-VL-2B-Thinking-bf16`](#model-mlx-community-qwen3-vl-2b-thinking-bf16)                             | `usable_with_caveats` | Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The context says it's Princes Street in Edinburgh, Scotland. The main subject is scaffolding on a building, renovation. So maybe "Edinburgh Princes Street Renovation Scaffolding" but let's check w...                         |
| [`mlx-community/diffusiongemma-26B-A4B-it-8bit`](#model-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | `usable_with_caveats` | Title: Scaffolding on Princes Street in Edinburgh City Centre<br>Description: Extensive scaffolding covers the facade of a building undergoing renovation along Princes Street in Edinburgh, while pedestrians walk past construction fences and a Boots pharmacy.<br>Keywords: Edinburgh,...                    |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                            | `usable_with_caveats` | Title: Edinburgh City Centre Renovation<br>Description: Scaffolding and construction activity on a building along Princes Street, Edinburgh, with pedestrians navigating temporary barriers.<br>Keywords: Edinburgh, Princes Street, Construction, Scaffolding, Overcast, Pedestrians, Boot...                   |
| [`jinaai/jina-vlm-mlx`](#model-jinaai-jina-vlm-mlx)                                                                     | `unusable`            | Title:<br>City Centre, Edinburgh, Scotland, UK, GBR, Europe<br>Description:<br>A bustling city centre scene in Edinburgh, Scotland, UK, under an overcast sky. Pedestrians walk by temporary construction fences, while a building undergoes major renovation and redevelopment along Princes...                 |
| [`mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`](#model-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | `unusable`            | Here are my reasoning steps:<br>The user wants British-English catalogue metadata from the image and supplied context. They want:<br><br>- a concrete 5-10-word title (so 5-10 words)<br>- a 1-2-sentence factual description combining relevant context with the main visible subject, setting,...              |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                              | `unusable`            | A concrete 5-10-word title: "City Centre Renovation in Edinburgh, Scotland, UK, GBR, Europe"<br>Description: The image captures a bustling city street in Edinburgh, Scotland, featuring extensive scaffolding covering the facade of a building undergoing major renovation and redevel...                      |
| [`mlx-community/GLM-4.1V-9B-Thinking-8bit`](#model-mlx-community-glm-41v-9b-thinking-8bit)                              | `unusable`            | &lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words. Looking at the image, the main elements are scaffolding, Princes Street, Edinburgh, construction, etc. So maybe "Edinburgh Princes Street Construction Site" or something similar. Wait,...                   |
| [`mlx-community/MiniCPM-V-4.6-8bit`](#model-mlx-community-minicpm-v-46-8bit)                                            | `unusable`            | Urban construction with scaffolding on city buildings.<br><br>Keywords: urban construction, scaffolding, city buildings, Princes Street, Edinburgh, construction fence, crane, overcast sky, pedestrians, modern architecture, civil engineering, Europe                                                         |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                            | `unusable`            | Edinburgh, Scotland, UK, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland,...                          |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                           | `unusable`            | Title: Edinburgh Princes Street Construction<br><br>Description: A Boots pharmacy on Princes Street in Edinburgh, Scotland, remains open as pedestrians walk past a major renovation site with extensive scaffolding and a construction crane under an overcast sky.<br><br>Keywords: Edinburgh,...              |
| [`mlx-community/gemma-3n-E4B-it-bf16`](#model-mlx-community-gemma-3n-e4b-it-bf16)                                       | `unusable`            | empty output                                                                                                                                                                                                                                                                                                     |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                         | `not_evaluated`       | Model loading failed: Received 1460 parameters not in model:<br>embed_tokens.biases,<br>embed_tokens.scales,<br>embed_tokens.weight,<br>layers.0.input_layernorm.weight,<br>layers.0.mlp.down_proj.biases,<br>layers.0.mlp.down_proj.scales,<br>layers.0.mlp.down_proj.weight,<br>layers.0.mlp.gate_proj.bias... |
| [`tencent/Youtu-VL-4B-Instruct`](#model-tencent-youtu-vl-4b-instruct)                                                   | `not_evaluated`       | Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)                                                                                                                                                                    |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.0rc0`
- `mlx`: `0.32.3.dev20260830+37c26e575`
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

- *Title:* , City Centre, Edinburgh, Scotland, UK, GBR, Europe
- *Description:* Extensive scaffolding covers the facade of a building
  undergoing major renovation and redevelopment along Princes Street in
  Edinburgh, Scotland, while pedestrians walk past temporary construction
  fences and a Boots pharmacy beneath an overcast sky.
- *Keywords:* 10 Best (structured), Adobe Stock, Any Vision, City Centre,
  Civil engineering, Construction fence, Construction site, Crane, Edinburgh,
  Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky,
  Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland, Traffic Cone,
  Tree branch, UK, United Kingdom, Urban architecture, architecture, building
  exterior, building renovation, city architecture, city life, city street,
  cityscape, cloudy sky, construction, construction work, daytime,
  engineering, european architecture, facade, great britain, historic
  building, infrastructure, maintenance, mobile crane, outdoor, redevelopment,
  renovation, restoration, safety barrier, sidewalk, stone building, street,
  street light, streetscape, tram tracks, tramway, urban development, urban
  landscape, urban regeneration, urban street, walking
- *Date:* 2026-08-18 17:46:05 UTC+01:00
- *Time:* 17:46:05
- *GPS:* 55.951722°N, 3.201417°W

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
> &#45; Capture date/time: 2026-08-18 17:46:05 UTC+01:00
> &#45; GPS: 55.951722°N, 3.201417°W
>
> &#8203;Descriptive hints:
> &#45; Title hint: City Centre, Edinburgh, Scotland, UK, GBR, Europe
> &#45; Description hint: Extensive scaffolding covers the facade of a building
> undergoing major renovation and redevelopment along Princes Street in
> Edinburgh, Scotland, while pedestrians walk past temporary construction
> fences and a Boots pharmacy beneath an overcast sky.
> &#45; Keyword hints: 10 Best (structured), Adobe Stock, Any Vision, City Centre,
> Civil engineering, Construction fence, Construction site, Crane, Edinburgh,
> Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky,
> Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland
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

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 5.13s
- *Generation time:* 13.47s
- *Total time:* 18.61s
- *Input validation time:* 6.29e-05
- *Prompt preparation time:* 0.00961
- *First-token latency:* 1.35
- *Cleanup time:* 0.212
- *Prompt tokens:* 1,669
- *Generation tokens:* 706
- *Total tokens:* 2,375
- *Prompt throughput (raw):* 1,233 tok/s
- *Generation throughput (raw):* 62.8 tok/s
- *Peak memory:* 60
- *Active memory:* 59
- *Cache memory:* 0.17
- *Model-load active memory:* 58.8
- *Post-cleanup active memory:* 0.00149
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1707
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
Alright, let's get to work. I've got the image and the hints, and I need to generate some metadata. First, the title. It needs to be concise, descriptive, and accurate. "Scaffolding on Princes Street, Edinburgh" is a good start, but I can do better. "Extensive Scaffolding on Edinburgh's Princes Street" is more evocative. I'll go with that.

Now, the description. I need to weave in the context and the hints. The image shows a major construction site on Princes Street in Edinburgh. The key elements are the scaffolding, the overcast sky, the pedestrians, and the Boots pharmacy. I'll start with the location and the main subject, then add the details. "On a grey day in August 2026, extensive scaffolding covers the facade of a building undergoing renovation on Princes Street in Edinburgh. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath the overcast sky." That's a solid, factual description.

Finally, the keywords. I need to cover the location, the subject, the details, and the context. I'll start with the obvious: Edinburgh, Princes Street, Scotland, UK, Europe. Then the key elements: scaffolding, construction, building, street, pedestrians, Boots, pharmacy, overcast sky, and the date. I'll add some more general terms like city centre, modern architecture, and roadworks. I'll make sure they are all comma-separated and unique.

Let's see... "Edinburgh, Princes Street, Scotland, UK, Europe, Scaffolding, Construction, Building, Street, Pedestrians, Boots, Pharmacy, Overcast Sky, City Centre, Modern Architecture, Roadworks, 2026-08-18". Perfect. That should be a good, comprehensive set of keywords.

Okay, final check. Title: "Extensive Scaffolding on Edinburgh's Princes Street." Description: "On a grey day in August 2026, extensive scaffolding covers the facade of a building undergoing renovation on Princes Street in Edinburgh. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath the overcast sky." Keywords: "Edinburgh, Princes Street, Scotland, UK, Europe, Scaffolding, Construction, Building, Street, Pedestrians, Boots, Pharmacy, Overcast Sky, City Centre, Modern Architecture, Roadworks, 2026-08-18".

All done. Ready to output.
&lt;/think&gt;

Title: Extensive Scaffolding on Edinburgh's Princes Street

Description: On a grey day in August 2026, extensive scaffolding covers the facade of a building undergoing renovation on Princes Street in Edinburgh. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath the overcast sky.

Keywords: Edinburgh, Princes Street, Scotland, UK, Europe, Scaffolding, Construction, Building, Street, Pedestrians, Boots, Pharmacy, Overcast Sky, City Centre, Modern Architecture, Roadworks, 2026-08-18
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get to work. I've got the image and the hints, and I need to generate some metadata. First, the title. It needs to be concise, descriptive, and accurate. "Scaffolding on Princes Street, Edinburgh" is a good start, but I can do better. "Extensive Scaffolding on Edinburgh's Princes Street" is more evocative. I'll go with that.

Now, the description. I need to weave in the context and the hints. The image shows a major construction site on Princes Street in Edinburgh. The key elements are the scaffolding, the overcast sky, the pedestrians, and the Boots pharmacy. I'll start with the location and the main subject, then add the details. "On a grey day in August 2026, extensive scaffolding covers the facade of a building undergoing renovation on Princes Street in Edinburgh. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath the overcast sky." That's a solid, factual description.

Finally, the keywords. I need to cover the location, the subject, the details, and the context. I'll start with the obvious: Edinburgh, Princes Street, Scotland, UK, Europe. Then the key elements: scaffolding, construction, building, street, pedestrians, Boots, pharmacy, overcast sky, and the date. I'll add some more general terms like city centre, modern architecture, and roadworks. I'll make sure they are all comma-separated and unique.

Let's see... "Edinburgh, Princes Street, Scotland, UK, Europe, Scaffolding, Construction, Building, Street, Pedestrians, Boots, Pharmacy, Overcast Sky, City Centre, Modern Architecture, Roadworks, 2026-08-18". Perfect. That should be a good, comprehensive set of keywords.

Okay, final check. Title: "Extensive Scaffolding on Edinburgh's Princes Street." Description: "On a grey day in August 2026, extensive scaffolding covers the facade of a building undergoing renovation on Princes Street in Edinburgh. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath the overcast sky." Keywords: "Edinburgh, Princes Street, Scotland, UK, Europe, Scaffolding, Construction, Building, Street, Pedestrians, Boots, Pharmacy, Overcast Sky, City Centre, Modern Architecture, Roadworks, 2026-08-18".

All done. Ready to output.
</think>

Title: Extensive Scaffolding on Edinburgh's Princes Street

Description: On a grey day in August 2026, extensive scaffolding covers the facade of a building undergoing renovation on Princes Street in Edinburgh. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath the overcast sky.

Keywords: Edinburgh, Princes Street, Scotland, UK, Europe, Scaffolding, Construction, Building, Street, Pedestrians, Boots, Pharmacy, Overcast Sky, City Centre, Modern Architecture, Roadworks, 2026-08-18
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
- *Model load time:* 1.76s
- *Generation time:* 4.17s
- *Total time:* 5.94s
- *Input validation time:* 0.000113
- *Prompt preparation time:* 0.00262
- *First-token latency:* 0.815
- *Cleanup time:* 0.112
- *Prompt tokens:* 2,147
- *Generation tokens:* 94
- *Total tokens:* 2,241
- *Prompt throughput (raw):* 2,633 tok/s
- *Generation throughput (raw):* 34.3 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.17
- *Model-load active memory:* 15.9
- *Post-cleanup active memory:* 0.00331
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1522
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
Title: Edinburgh City Centre Princes Street
Description: Extensive scaffolding covers a building under renovation on Princes Street, with pedestrians walking past a Boots pharmacy under an overcast sky.
Keywords: Edinburgh, City Centre, Princes Street, Scaffolding, Renovation, Construction Site, Boots Pharmacy, Pedestrians, Overcast Sky, Modern Architecture, Roadworks, Scaffolding Fences, Civil Engineering, Pedestrian Walkway, Urban Development
</pre>

</details>

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### mlx-community/LFM2.5-VL-1.6B-bf16

<details>
<summary>Complete evidence: mlx-community/LFM2.5-VL-1.6B-bf16</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Model load time:* 0.68s
- *Generation time:* 1.62s
- *Total time:* 2.31s
- *Input validation time:* 7.11e-05
- *Prompt preparation time:* 0.00481
- *First-token latency:* 0.335
- *Cleanup time:* 0.0741
- *Prompt tokens:* 2,150
- *Generation tokens:* 142
- *Total tokens:* 2,292
- *Prompt throughput (raw):* 6,410 tok/s
- *Generation throughput (raw):* 184 tok/s
- *Peak memory:* 4.0
- *Active memory:* 3.2
- *Cache memory:* 0.06
- *Model-load active memory:* 3.2
- *Post-cleanup active memory:* 0.0041
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1568
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
Title: "Edinburgh's Urban Renewal: Scaffolding and Construction on Princes Street"

Description: A bustling city centre in Edinburgh, Scotland, is undergoing significant redevelopment. Extensive scaffolding covers a building along Princes Street, indicating major renovation and redevelopment work. Pedestrians walk past temporary construction fences, while a Boots pharmacy remains open beneath the construction site. The overcast sky adds a moody atmosphere to the scene.

Keywords: Edinburgh, Scotland, Princes Street, Scaffolding, Construction, Urban Renewal, City Centre, Pedestrians, Overcast Sky, Construction Fence, Boots Pharmacy, Civil Engineering
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
- *Model load time:* 0.97s
- *Generation time:* 1.41s
- *Total time:* 2.39s
- *Input validation time:* 5.26e-05
- *Prompt preparation time:* 0.00899
- *First-token latency:* 0.463
- *Cleanup time:* 0.0781
- *Prompt tokens:* 2,141
- *Generation tokens:* 90
- *Total tokens:* 2,231
- *Prompt throughput (raw):* 4,619 tok/s
- *Generation throughput (raw):* 211 tok/s
- *Peak memory:* 4.0
- *Active memory:* 2.8
- *Cache memory:* 0.08
- *Model-load active memory:* 2.82
- *Post-cleanup active memory:* 0.00436
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1568
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
Title: Edinburgh Princes Street Construction Site
Description: Extensive scaffolding covers a building undergoing renovation along Princes Street, with pedestrians walking past temporary fences and a Boots pharmacy under an overcast sky.
Keywords: Edinburgh, Scotland, UK, Europe, City Centre, Civil engineering, Construction fence, Construction site, Crane, Overcast, Pedestrians, Princes Street, Roadworks, Scaffolding, Modern Architecture
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
- *Model load time:* 1.03s
- *Generation time:* 2.15s
- *Total time:* 3.19s
- *Input validation time:* 4.66e-05
- *Prompt preparation time:* 0.00753
- *First-token latency:* 0.96
- *Cleanup time:* 0.0808
- *Prompt tokens:* 2,958
- *Generation tokens:* 123
- *Total tokens:* 3,081
- *Prompt throughput (raw):* 3,082 tok/s
- *Generation throughput (raw):* 189 tok/s
- *Peak memory:* 7.8
- *Active memory:* 2.8
- *Cache memory:* 0.41
- *Model-load active memory:* 2.75
- *Post-cleanup active memory:* 0.00567
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3954
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
**Princes Street Construction: Edinburgh’s Facade in Renewal**

Description:
A major renovation under scaffolding transforms a historic building in Edinburgh’s bustling city centre along Princes Street, with cranes and protective barriers visible under an overcast sky. Pedestrians navigate the temporary roadworks, including a Boots pharmacy and construction fences.

Keywords:
Princes Street construction site, Edinburgh urban renewal, scaffolding facade, historic building restoration, cranes at work, Boots pharmacy under construction, temporary pedestrian paths, overcast cityscape, construction safety barriers, roadworks in city centre.
</pre>

</details>

---

<a id="model-mlx-community-ornith-15-35b-a3b-optiq-4bit"></a>

### mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.62s
- *Generation time:* 2.63s
- *Total time:* 5.27s
- *Input validation time:* 5.5e-05
- *Prompt preparation time:* 0.0185
- *First-token latency:* 0.793
- *Cleanup time:* 0.127
- *Prompt tokens:* 1,323
- *Generation tokens:* 129
- *Total tokens:* 1,452
- *Prompt throughput (raw):* 1,669 tok/s
- *Generation throughput (raw):* 106 tok/s
- *Peak memory:* 25
- *Active memory:* 23
- *Cache memory:* 0.14
- *Model-load active memory:* 23.1
- *Post-cleanup active memory:* 0.00701
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1608
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
Title:
Scaffolding Covers Building Renovation on Princes Street, Edinburgh

Description:
Extensive scaffolding envelops the facade of a building undergoing major redevelopment along Princes Street in Edinburgh's city centre, beside a Boots pharmacy beneath an overcast sky. Pedestrians walk past temporary construction fencing and roadworks while a yellow crane operates in the distance.

Keywords:
Edinburgh, Princes Street, Scotland, City Centre, Scaffolding, Construction site, Construction fence, Roadworks, Crane, Pedestrians, Boots, Pharmacy, Overcast sky, Modern architecture, Renovation, Urban street, United Kingdom
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
- *Model load time:* 1.00s
- *Generation time:* 3.42s
- *Total time:* 4.43s
- *Input validation time:* 6.08e-05
- *Prompt preparation time:* 0.00166
- *First-token latency:* 0.255
- *Cleanup time:* 0.0905
- *Prompt tokens:* 1,174
- *Generation tokens:* 150
- *Total tokens:* 1,324
- *Prompt throughput (raw):* 4,604 tok/s
- *Generation throughput (raw):* 56.5 tok/s
- *Peak memory:* 9.4
- *Active memory:* 8.3
- *Cache memory:* 0.62
- *Model-load active memory:* 8.3
- *Post-cleanup active memory:* 0.00708
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1538
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
Title: Edinburgh's Princes Street Undergoing Major Renovation

Description: Pedestrians stroll past a building undergoing extensive renovation on Princes Street in Edinburgh, with scaffolding and construction fences visible.

Keywords: Edinburgh, Princes Street, renovation, scaffolding, construction, pedestrians, overcast sky, Boots pharmacy, civil engineering, modern architecture, roadworks, European city centre, GBR, 2026, 17:46:05 UTC+01:00, 55.951722°N, 3.201417°W
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
- *Model load time:* 1.93s
- *Generation time:* 49.87s
- *Total time:* 51.82s
- *Input validation time:* 6.16e-05
- *Prompt preparation time:* 0.00154
- *First-token latency:* 47.6
- *Cleanup time:* 0.144
- *Prompt tokens:* 16,581
- *Generation tokens:* 113
- *Total tokens:* 16,694
- *Prompt throughput (raw):* 348 tok/s
- *Generation throughput (raw):* 86.5 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 2.3
- *Model-load active memory:* 18.3
- *Post-cleanup active memory:* 0.0077
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1589
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
Title: Extensive scaffolding on Princes Street, Edinburgh
Description: A wide shot captures a building under major renovation on Princes Street in Edinburgh, Scotland, its facade entirely covered by scaffolding. Pedestrians walk along the pavement past a Boots pharmacy and temporary construction fencing under an overcast sky.
Keywords: Edinburgh, Princes Street, Scotland, UK, Europe, construction site, scaffolding, construction fence, roadworks, crane, modern architecture, civil engineering, overcast sky, pedestrians, building renovation, city centre, objects, Boots pharmacy
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
- *Model load time:* 2.57s
- *Generation time:* 51.52s
- *Total time:* 54.11s
- *Input validation time:* 7.34e-05
- *Prompt preparation time:* 0.0174
- *First-token latency:* 49.4
- *Cleanup time:* 0.128
- *Prompt tokens:* 16,597
- *Generation tokens:* 134
- *Total tokens:* 16,731
- *Prompt throughput (raw):* 336 tok/s
- *Generation throughput (raw):* 111 tok/s
- *Peak memory:* 24
- *Active memory:* 20
- *Cache memory:* 0.89
- *Model-load active memory:* 20.4
- *Post-cleanup active memory:* 0.00821
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1608
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
Title: Princes Street Building Renovation with Scaffolding

Description: An extensive construction site on Princes Street in Edinburgh features a multi-story building fully encased in scaffolding and protective sheeting, situated next to a Boots pharmacy under an overcast sky. Pedestrians walk along the sidewalk past temporary fencing and roadworks, while a yellow crane is visible in the background.

Keywords: Edinburgh, Princes Street, Scaffolding, Construction site, Roadworks, Boots pharmacy, Pedestrians, Overcast sky, Crane, Civil engineering, City Centre, Scotland, UK, Modern architecture, Fence, Objects, Europe, 2026
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
- *Model load time:* 1.47s
- *Generation time:* 53.40s
- *Total time:* 54.88s
- *Input validation time:* 6.91e-05
- *Prompt preparation time:* 0.00166
- *First-token latency:* 51.3
- *Cleanup time:* 0.0935
- *Prompt tokens:* 16,597
- *Generation tokens:* 103
- *Total tokens:* 16,700
- *Prompt throughput (raw):* 323 tok/s
- *Generation throughput (raw):* 91.7 tok/s
- *Peak memory:* 10.0
- *Active memory:* 6.0
- *Cache memory:* 1.2
- *Model-load active memory:* 5.96
- *Post-cleanup active memory:* 0.00873
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1608
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
Scaffolding-covered building under renovation on Princes Street, Edinburgh

Description:
Extensive scaffolding envelops a historic building undergoing redevelopment along Princes Street in Edinburgh, with pedestrians walking past temporary fencing and a Boots pharmacy under an overcast sky.

Keywords:
Princes Street, Edinburgh, Scotland, UK, scaffolding, construction site, roadworks, Boots pharmacy, pedestrians, overcast sky, modern architecture, civil engineering, temporary fence, crane, historic building, urban renewal
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:  
Scaffolding-covered building under renovation on Princes Street, Edinburgh

Description:  
Extensive scaffolding envelops a historic building undergoing redevelopment along Princes Street in Edinburgh, with pedestrians walking past temporary fencing and a Boots pharmacy under an overcast sky.

Keywords:  
Princes Street, Edinburgh, Scotland, UK, scaffolding, construction site, roadworks, Boots pharmacy, pedestrians, overcast sky, modern architecture, civil engineering, temporary fence, crane, historic building, urban renewal
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
- *Model load time:* 3.13s
- *Generation time:* 76.15s
- *Total time:* 79.30s
- *Input validation time:* 7.39e-05
- *Prompt preparation time:* 0.0174
- *First-token latency:* 69
- *Cleanup time:* 0.146
- *Prompt tokens:* 16,597
- *Generation tokens:* 110
- *Total tokens:* 16,707
- *Prompt throughput (raw):* 241 tok/s
- *Generation throughput (raw):* 17.6 tok/s
- *Peak memory:* 33
- *Active memory:* 29
- *Cache memory:* 1.9
- *Model-load active memory:* 28.7
- *Post-cleanup active memory:* 0.00924
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1608
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
Edinburgh Princes Street Construction and Boots Pharmacy

Description:
Extensive scaffolding covers a building undergoing major renovation on Princes Street in Edinburgh, situated next to a Boots pharmacy. Pedestrians walk past temporary construction fences and a yellow crane under an overcast sky.

Keywords:
Edinburgh, Princes Street, Scotland, Construction, Scaffolding, Boots, Pharmacy, Renovation, Crane, Pedestrians, City Centre, Overcast, Roadworks, Urban, Architecture, Development, Street Scene, UK
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
- *Model load time:* 2.13s
- *Generation time:* 73.10s
- *Total time:* 75.26s
- *Input validation time:* 7.26e-05
- *Prompt preparation time:* 0.0186
- *First-token latency:* 68.4
- *Cleanup time:* 0.126
- *Prompt tokens:* 16,597
- *Generation tokens:* 115
- *Total tokens:* 16,712
- *Prompt throughput (raw):* 243 tok/s
- *Generation throughput (raw):* 30.7 tok/s
- *Peak memory:* 21
- *Active memory:* 16
- *Cache memory:* 1.9
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.00975
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1608
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
Scaffolding on Princes Street, Edinburgh, Scotland

Description:
A large building on Princes Street in Edinburgh is covered in extensive scaffolding and green safety netting, indicating major renovation works. Pedestrians walk past a Boots pharmacy and temporary construction fencing along the street under an overcast sky.

Keywords:
Scaffolding, Construction, Edinburgh, Princes Street, Renovation, Boots, Pharmacy, Pedestrians, Overcast, City Centre, Scotland, Roadworks, Urban, Architecture, Fencing, Crane, Street Scene
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
- *Model load time:* 5.60s
- *Generation time:* 16.88s
- *Total time:* 22.50s
- *Input validation time:* 6.21e-05
- *Prompt preparation time:* 0.0158
- *First-token latency:* 13.9
- *Cleanup time:* 0.24
- *Prompt tokens:* 3,525
- *Generation tokens:* 102
- *Total tokens:* 3,627
- *Prompt throughput (raw):* 254 tok/s
- *Generation throughput (raw):* 46.5 tok/s
- *Peak memory:* 70
- *Active memory:* 63
- *Cache memory:* 0.41
- *Model-load active memory:* 63
- *Post-cleanup active memory:* 0.0101
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1595
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
Edinburgh City Centre Scaffolding and Construction Works

Description:
Extensive scaffolding covers a building undergoing renovation along Princes Street in Edinburgh, Scotland, with pedestrians walking past temporary construction fences and a Boots pharmacy under an overcast sky.

Keywords:
Edinburgh, Princes Street, Scaffolding, Construction site, Boots pharmacy, Overcast sky, Pedestrians, Roadworks, City Centre, Scotland, UK, Europe, Civil engineering, Construction fence, Crane, Modern architecture, Temporary signage
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
- *Model load time:* 3.25s
- *Generation time:* 2.89s
- *Total time:* 6.16s
- *Input validation time:* 5.53e-05
- *Prompt preparation time:* 0.00112
- *First-token latency:* 0.331
- *Cleanup time:* 0.162
- *Prompt tokens:* 627
- *Generation tokens:* 85
- *Total tokens:* 712
- *Prompt throughput (raw):* 1,894 tok/s
- *Generation throughput (raw):* 46.6 tok/s
- *Peak memory:* 28
- *Active memory:* 27
- *Cache memory:* 0.01
- *Model-load active memory:* 27.2
- *Post-cleanup active memory:* 0.0104
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1543
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
Title: Building Renovation on Princes Street, Edinburgh City Centre
Description: Extensive scaffolding and a crane cover a building undergoing renovation along Princes Street in Edinburgh while pedestrians walk past a Boots pharmacy under an overcast sky.
Keywords: Edinburgh, Scotland, Princes Street, City Centre, construction, scaffolding, crane, renovation, architecture, pedestrians, pharmacy, urban, overcast, civil engineering, roadworks, Europe
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
- *Model load time:* 2.29s
- *Generation time:* 5.58s
- *Total time:* 7.89s
- *Input validation time:* 6.3e-05
- *Prompt preparation time:* 0.00459
- *First-token latency:* 1.02
- *Cleanup time:* 0.13
- *Prompt tokens:* 626
- *Generation tokens:* 126
- *Total tokens:* 752
- *Prompt throughput (raw):* 611 tok/s
- *Generation throughput (raw):* 31.5 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.46
- *Model-load active memory:* 16.1
- *Post-cleanup active memory:* 0.011
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1572
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
Title: Princes Street Building Renovation, Edinburgh

Description: Captured on 18th August 2026, this view shows extensive scaffolding covering a building undergoing renovation on Princes Street, Edinburgh, with pedestrians passing a Boots pharmacy under an overcast sky. A yellow crane is visible amidst the construction site, and temporary fencing secures the area.

Keywords: Edinburgh, Princes Street, Scotland, Renovation, Construction, Scaffolding, Crane, Boots, Pedestrians, Roadworks, Civil engineering, Overcast Sky, Modern Architecture, Building, Facade, Construction fence, City Centre, 2026
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
- *Model load time:* 2.35s
- *Generation time:* 1.66s
- *Total time:* 4.04s
- *Input validation time:* 6.16e-05
- *Prompt preparation time:* 0.023
- *First-token latency:* 0.407
- *Cleanup time:* 0.126
- *Prompt tokens:* 631
- *Generation tokens:* 87
- *Total tokens:* 718
- *Prompt throughput (raw):* 1,549 tok/s
- *Generation throughput (raw):* 128 tok/s
- *Peak memory:* 16
- *Active memory:* 15
- *Cache memory:* 0.21
- *Model-load active memory:* 15.4
- *Post-cleanup active memory:* 0.012
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1571
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
Title: Scaffolding on building on Princes Street, Edinburgh
Description: Extensive scaffolding covers a building undergoing renovation on Princes Street in Edinburgh under an overcast sky. Pedestrians walk past construction fences and a Boots pharmacy on the street level.
Keywords: Edinburgh, Scotland, Princes Street, scaffolding, construction, renovation, urban, city centre, pedestrians, Boots, pharmacy, architecture, overcast sky, street scene, Europe, UK
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
- *Generation time:* 5.07s
- *Total time:* 7.65s
- *Input validation time:* 5.81e-05
- *Prompt preparation time:* 0.00175
- *First-token latency:* 1.12
- *Cleanup time:* 0.142
- *Prompt tokens:* 631
- *Generation tokens:* 89
- *Total tokens:* 720
- *Prompt throughput (raw):* 563 tok/s
- *Generation throughput (raw):* 26.3 tok/s
- *Peak memory:* 20
- *Active memory:* 18
- *Cache memory:* 0.81
- *Model-load active memory:* 18.4
- *Post-cleanup active memory:* 0.0126
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1571
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
Title: Construction and Scaffolding on Princes Street, Edinburgh
Description: Extensive scaffolding covers a building undergoing renovation along Princes Street in Edinburgh, Scotland, under an overcast sky. Pedestrians walk past construction fences and a Boots pharmacy storefront.
Keywords: Edinburgh, Scotland, Princes Street, construction site, scaffolding, crane, Boots pharmacy, pedestrians, roadworks, city centre, overcast sky, civil engineering, urban redevelopment, Europe, UK
</pre>

</details>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### mlx-community/pixtral-12b-8bit

<details>
<summary>Complete evidence: mlx-community/pixtral-12b-8bit</summary>

- *Execution:* completed
- *Usability:* usable
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type pixtral)
- *Model load time:* 1.64s
- *Generation time:* 4.51s
- *Total time:* 6.16s
- *Input validation time:* 5.93e-05
- *Prompt preparation time:* 0.00371
- *First-token latency:* 1.7
- *Cleanup time:* 0.105
- *Prompt tokens:* 3,149
- *Generation tokens:* 91
- *Total tokens:* 3,240
- *Prompt throughput (raw):* 1,850 tok/s
- *Generation throughput (raw):* 39.6 tok/s
- *Peak memory:* 16
- *Active memory:* 13
- *Cache memory:* 0.59
- *Model-load active memory:* 13.5
- *Post-cleanup active memory:* 0.0131
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1517
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
Princes Street Redevelopment, Edinburgh, Scotland

Description:
Scaffolding covers a building undergoing renovation along Princes Street in Edinburgh, Scotland, with pedestrians walking past under an overcast sky.

Keywords:
Princes Street, Edinburgh, Scotland, Scaffolding, Construction, Renovation, Pedestrians, Overcast, Boots Pharmacy, Civil Engineering, Roadworks, Crane, Modern Architecture, Fence, City Centre, Europe
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
- *Observations:* Duplicate keywords: edinburgh, scotland
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Model load time:* 0.56s
- *Generation time:* 0.84s
- *Total time:* 1.73s
- *Input validation time:* 0.319
- *Prompt preparation time:* 0.00673
- *First-token latency:* 0.121
- *Cleanup time:* 0.0626
- *Prompt tokens:* 2,150
- *Generation tokens:* 118
- *Total tokens:* 2,268
- *Prompt throughput (raw):* 17,807 tok/s
- *Generation throughput (raw):* 480 tok/s
- *Peak memory:* 1.9
- *Active memory:* 0.90
- *Cache memory:* 0.08
- *Model-load active memory:* 0.897
- *Post-cleanup active memory:* 0.000132
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1568
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
City Centre, Edinburgh, Scotland, UK, GBR, Europe

Description:
A bustling city street scene in Edinburgh, Scotland, featuring a construction site on Princes Street with scaffolding and a Boots pharmacy. Pedestrians walk along the sidewalk, and a crane is visible in the background.

Keywords:
City Centre, Edinburgh, Scotland, UK, GBR, Europe, Construction site, Boots pharmacy, Edinburgh, Scotland, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Modern Architecture
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
- *Observations:* Title has 4 words (requested 5-10); Keyword list has 19
  terms (requested 10-18); Duplicate keywords: scaffolding, roadworks
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.69s
- *Generation time:* 15.37s
- *Total time:* 16.08s
- *Input validation time:* 4.62e-05
- *Prompt preparation time:* 0.00999
- *First-token latency:* 13.2
- *Cleanup time:* 0.0855
- *Prompt tokens:* 16,581
- *Generation tokens:* 121
- *Total tokens:* 16,702
- *Prompt throughput (raw):* 1,252 tok/s
- *Generation throughput (raw):* 92.2 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.6
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.000443
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1589
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
Title:
Edinburgh Princes Street Construction

Description:
Extensive scaffolding covers a building undergoing renovation on Princes Street in Edinburgh, Scotland, under an overcast sky, with pedestrians walking past a Boots pharmacy and construction fences. A yellow crane is visible near the site, and the scene is marked by modern and historic architecture.

Keywords: Edinburgh, Princes Street, construction, scaffolding, crane, overcast, pedestrians, Boots pharmacy, roadworks, modern architecture, city centre, Scotland, European city, civil engineering, construction site, fence, roadworks, scaffolding, overcast sky
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
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 2.19s
- *Generation time:* 7.08s
- *Total time:* 9.29s
- *Input validation time:* 5.46e-05
- *Prompt preparation time:* 0.00815
- *First-token latency:* 3.17
- *Cleanup time:* 0.101
- *Prompt tokens:* 2,426
- *Generation tokens:* 102
- *Total tokens:* 2,528
- *Prompt throughput (raw):* 765 tok/s
- *Generation throughput (raw):* 30.4 tok/s
- *Peak memory:* 23
- *Active memory:* 18
- *Cache memory:* 0.47
- *Model-load active memory:* 18
- *Post-cleanup active memory:* 0.00128
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1517
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
Edinburgh Princes Street Construction

Description:
Extensive scaffolding covers a building undergoing major renovation along Princes Street in Edinburgh, Scotland, under an overcast sky, with pedestrians walking past temporary construction fences and a Boots pharmacy.

Keywords:
Edinburgh, Princes Street, construction, scaffolding, renovation, overcast sky, pedestrians, Boots pharmacy, temporary fences, urban development, city centre, Scotland, UK, Europe, modern architecture, civil engineering, roadworks, crane
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
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.41s
- *Generation time:* 7.55s
- *Total time:* 8.97s
- *Input validation time:* 6.51e-05
- *Prompt preparation time:* 0.00916
- *First-token latency:* 5.78
- *Cleanup time:* 0.0879
- *Prompt tokens:* 6,484
- *Generation tokens:* 99
- *Total tokens:* 6,583
- *Prompt throughput (raw):* 1,123 tok/s
- *Generation throughput (raw):* 80.1 tok/s
- *Peak memory:* 8.4
- *Active memory:* 6.8
- *Cache memory:* 0.46
- *Model-load active memory:* 6.78
- *Post-cleanup active memory:* 0.00243
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1598
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
Title: Princes Street Renovation, Edinburgh
Description: The image shows a construction site on Princes Street in Edinburgh, Scotland, with extensive scaffolding covering a building under renovation. Pedestrians walk past temporary construction fences, and a Boots pharmacy is visible beneath an overcast sky.
Keywords: Princes Street renovation, Edinburgh construction site, scaffolding, temporary construction fences, Boots pharmacy, overcast sky, pedestrians, civil engineering, roadworks, Scotland, Europe, city centre.
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
- *Observations:* Unrecognised model control tokens remain visible; Title has
  4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Model load time:* 5.65s
- *Generation time:* 14.37s
- *Total time:* 20.04s
- *Input validation time:* 6.09e-05
- *Prompt preparation time:* 0.0123
- *First-token latency:* 11.5
- *Cleanup time:* 0.251
- *Prompt tokens:* 6,484
- *Generation tokens:* 100
- *Total tokens:* 6,584
- *Prompt throughput (raw):* 562 tok/s
- *Generation throughput (raw):* 43.8 tok/s
- *Peak memory:* 78
- *Active memory:* 62
- *Cache memory:* 1.4
- *Model-load active memory:* 61.9
- *Post-cleanup active memory:* 0.00274
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1598
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
&lt;|begin_of_box|&gt;Title: Princes Street Construction, Edinburgh
Description: A construction site with extensive scaffolding and a crane on Princes Street in Edinburgh, Scotland, under an overcast sky, with pedestrians passing a Boots pharmacy and temporary fences.
Keywords: Princes Street, Edinburgh, Scotland, Construction site, Scaffolding, Crane, Boots pharmacy, Pedestrians, Construction fence, Overcast sky, City Centre, Roadworks, Civil engineering, Modern Architecture, Fence, Europe, GBR&lt;|end_of_box|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|begin_of_box|>Title: Princes Street Construction, Edinburgh
Description: A construction site with extensive scaffolding and a crane on Princes Street in Edinburgh, Scotland, under an overcast sky, with pedestrians passing a Boots pharmacy and temporary fences.
Keywords: Princes Street, Edinburgh, Scotland, Construction site, Scaffolding, Crane, Boots pharmacy, Pedestrians, Construction fence, Overcast sky, City Centre, Roadworks, Civil engineering, Modern Architecture, Fence, Europe, GBR<|end_of_box|>
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
- *Maintainer status:* none
- *Observations:* Keyword list has 20 terms (requested 10-18); Duplicate
  keywords: scotland, renovation, city centre
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Model load time:* 1.91s
- *Generation time:* 6.67s
- *Total time:* 8.59s
- *Input validation time:* 5.22e-05
- *Prompt preparation time:* 0.00221
- *First-token latency:* 1.1
- *Cleanup time:* 0.107
- *Prompt tokens:* 2,646
- *Generation tokens:* 156
- *Total tokens:* 2,802
- *Prompt throughput (raw):* 2,398 tok/s
- *Generation throughput (raw):* 32.2 tok/s
- *Peak memory:* 18
- *Active memory:* 17
- *Cache memory:* 0.43
- *Model-load active memory:* 16.9
- *Post-cleanup active memory:* 0.003
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1554
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
 Title: Extensive Scaffolding on Princes Street Building Renovation

Description: A building on Princes Street in Edinburgh, Scotland, is undergoing significant renovation and redevelopment, as evidenced by the extensive scaffolding covering its facade. Pedestrians are seen walking past temporary construction fences and a Boots pharmacy beneath an overcast sky. The building's modern architecture is partially obscured by the scaffolding, which is a common sight in city centers where construction and development are ongoing.

Keywords: Edinburgh, Scotland, Princes Street, Scaffolding, Construction, Renovation, Modern Architecture, Boots Pharmacy, City Centre, Overcast Sky, Pedestrians, Construction Fences, Building Facade, Renovation, Development, City Centre, Scotland, Europe, GBR, UK.
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
- *Model load time:* 3.24s
- *Generation time:* 162.75s
- *Total time:* 166.01s
- *Input validation time:* 5.2e-05
- *Prompt preparation time:* 0.00414
- *First-token latency:* 1.26
- *Cleanup time:* 0.19
- *Prompt tokens:* 1,356
- *Generation tokens:* 750
- *Total tokens:* 2,106
- *Prompt throughput (raw):* 1,077 tok/s
- *Generation throughput (raw):* 4.67 tok/s
- *Peak memory:* 40
- *Active memory:* 33
- *Cache memory:* 5.8
- *Model-load active memory:* 32.8
- *Post-cleanup active memory:* 0.00397
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1704
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
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the given context and image. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords.

Starting with the title: The image shows scaffolding on a building in Edinburgh, Scotland, with pedestrians and a Boots pharmacy. The title should be concise. Maybe "Edinburgh Construction Scaffolding and Pedestrian Activity on Princes Street" – that's 8 words.

For the description: It needs to combine context, main subject, setting, action, lighting, and details. The scene is a building under renovation with scaffolding, pedestrians, Boots pharmacy, overcast sky. So something like "Major building renovation with extensive scaffolding on Princes Street in Edinburgh, Scotland, featuring pedestrians, a Boots pharmacy, and an overcast sky."

Keywords: Let's list visible elements. Scaffolding, Edinburgh, Scotland, UK, construction, pedestrians, Boots pharmacy, Princes Street, overcast sky, crane, construction site, roadworks. That's 10 keywords. Need to check if they cover all key points. Maybe add "Civil engineering" and "Modern Architecture" as per the hints. Wait, the hints include those. So include those. Let me count: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture. That's 11. Maybe add "Fence" from the hints. So 12. Check if all hints are covered. The hints mention "Construction fence" and "Fence" so include those. Now 14. Maybe "Fence" is part of the construction site. Hmm, but the image shows temporary fencing. So include "Fence" as a keyword. Now 14. Let me check again. The hints also have "10 Best (structured)" but that's probably a category. The keywords should be unique and relevant. Final list: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture, Fence. That's 12. Maybe add "Construction fence" as a separate keyword? But the image shows a fence, so maybe "Fence" is enough. Alternatively, "Construction fence" is in the hints. Let me check the hints again. The hints include "Construction fence" and "Fence". So include both? But the user said 10-18 unique keywords. Let's see: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture, Fence, Construction fence. That's 13. Maybe that's okay. Alternatively, combine "Construction fence" into "Fence" if it's redundant. But the hints mention both, so better to include both. Wait, the hints say "Construction fence" and "Fence" are separate. So include both. So 13 keywords. Let me check the count. Yes, that's 13. Okay.◁/think▷Title: Edinburgh Construction Scaffolding and Pedestrian Activity on Princes Street
Description: Major building renovation with extensive scaffolding on Princes Street in Edinburgh, Scotland, featuring pedestrians, a Boots pharmacy, and an overcast sky.
Keywords: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture, Fence, Construction fence&lt;|im_assistant|&gt;
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the given context and image. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords. 

Starting with the title: The image shows scaffolding on a building in Edinburgh, Scotland, with pedestrians and a Boots pharmacy. The title should be concise. Maybe "Edinburgh Construction Scaffolding and Pedestrian Activity on Princes Street" – that's 8 words.

For the description: It needs to combine context, main subject, setting, action, lighting, and details. The scene is a building under renovation with scaffolding, pedestrians, Boots pharmacy, overcast sky. So something like "Major building renovation with extensive scaffolding on Princes Street in Edinburgh, Scotland, featuring pedestrians, a Boots pharmacy, and an overcast sky."

Keywords: Let's list visible elements. Scaffolding, Edinburgh, Scotland, UK, construction, pedestrians, Boots pharmacy, Princes Street, overcast sky, crane, construction site, roadworks. That's 10 keywords. Need to check if they cover all key points. Maybe add "Civil engineering" and "Modern Architecture" as per the hints. Wait, the hints include those. So include those. Let me count: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture. That's 11. Maybe add "Fence" from the hints. So 12. Check if all hints are covered. The hints mention "Construction fence" and "Fence" so include those. Now 14. Maybe "Fence" is part of the construction site. Hmm, but the image shows temporary fencing. So include "Fence" as a keyword. Now 14. Let me check again. The hints also have "10 Best (structured)" but that's probably a category. The keywords should be unique and relevant. Final list: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture, Fence. That's 12. Maybe add "Construction fence" as a separate keyword? But the image shows a fence, so maybe "Fence" is enough. Alternatively, "Construction fence" is in the hints. Let me check the hints again. The hints include "Construction fence" and "Fence". So include both? But the user said 10-18 unique keywords. Let's see: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture, Fence, Construction fence. That's 13. Maybe that's okay. Alternatively, combine "Construction fence" into "Fence" if it's redundant. But the hints mention both, so better to include both. Wait, the hints say "Construction fence" and "Fence" are separate. So include both. So 13 keywords. Let me check the count. Yes, that's 13. Okay.◁/think▷Title: Edinburgh Construction Scaffolding and Pedestrian Activity on Princes Street  
Description: Major building renovation with extensive scaffolding on Princes Street in Edinburgh, Scotland, featuring pedestrians, a Boots pharmacy, and an overcast sky.  
Keywords: Scaffolding, Edinburgh, Scotland, UK, Civil engineering, Construction site, Crane, Pedestrians, Princes Street, Roadworks, Boots pharmacy, Overcast sky, Modern Architecture, Fence, Construction fence<|im_assistant|>
```

</details>

</details>

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

<details>
<summary>Complete evidence: mlx-community/Ministral-3-14B-Instruct-2512-mxfp4</summary>

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* none
- *Observations:* Keyword list has 19 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.49s
- *Generation time:* 4.71s
- *Total time:* 6.22s
- *Input validation time:* 5.29e-05
- *Prompt preparation time:* 0.00743
- *First-token latency:* 1.97
- *Cleanup time:* 0.0918
- *Prompt tokens:* 2,959
- *Generation tokens:* 146
- *Total tokens:* 3,105
- *Prompt throughput (raw):* 1,500 tok/s
- *Generation throughput (raw):* 67.2 tok/s
- *Peak memory:* 13
- *Active memory:* 8.0
- *Cache memory:* 0.60
- *Model-load active memory:* 8.01
- *Post-cleanup active memory:* 0.00515
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3955
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
**Princes Street, Edinburgh – Major Building Renovation Underway**

Description:
A large multi-storey building on Princes Street, Edinburgh, is undergoing extensive renovation with full scaffolding coverage, while construction fences and a yellow crane are visible. Pedestrians walk past the site, including a Boots pharmacy, under an overcast sky on 18 August 2026.

Keywords:
Princes Street, Edinburgh, Scotland, UK, GBR, Europe, construction site, scaffolding, renovation, crane, temporary fences, Boots pharmacy, city centre, overcast sky, pedestrian activity, civil engineering, modern architecture, roadworks, 2026-08-18
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
- *Observations:* Keyword list has 20 terms (requested 10-18)
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 1.50s
- *Generation time:* 5.31s
- *Total time:* 6.82s
- *Input validation time:* 5.31e-05
- *Prompt preparation time:* 0.000893
- *First-token latency:* 2.02
- *Cleanup time:* 0.0943
- *Prompt tokens:* 2,959
- *Generation tokens:* 179
- *Total tokens:* 3,138
- *Prompt throughput (raw):* 1,464 tok/s
- *Generation throughput (raw):* 64.7 tok/s
- *Peak memory:* 13
- *Active memory:* 8.4
- *Cache memory:* 0.60
- *Model-load active memory:* 8.43
- *Post-cleanup active memory:* 0.00541
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3955
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
*Princes Street, Edinburgh: Scaffolding &amp; Renovation*

**Description:**
On **18 August 2026** in Edinburgh’s city centre (GPS: **55.951722°N, 3.201417°W**), extensive scaffolding envelops a multi-storey building under major renovation along **Princes Street**. Pedestrians pass temporary construction barriers, including a **Boots Pharmacy**, under an overcast sky, while a crane and debris netting highlight ongoing civil engineering work.

**Keywords:**
Princes Street, Edinburgh, Scotland, GBR, Europe, scaffolding, construction site, renovation, civil engineering, crane, construction fence, temporary barriers, Boots Pharmacy, city centre, overcast sky, pedestrians, roadworks, modern architecture, debris netting, urban development
</pre>

<details>
<summary>Exact raw output</summary>

```text
**Title:**
*Princes Street, Edinburgh: Scaffolding & Renovation*

**Description:**
On **18 August 2026** in Edinburgh’s city centre (GPS: **55.951722°N, 3.201417°W**), extensive scaffolding envelops a multi-storey building under major renovation along **Princes Street**. Pedestrians pass temporary construction barriers, including a **Boots Pharmacy**, under an overcast sky, while a crane and debris netting highlight ongoing civil engineering work.

**Keywords:**
Princes Street, Edinburgh, Scotland, GBR, Europe, scaffolding, construction site, renovation, civil engineering, crane, construction fence, temporary barriers, Boots Pharmacy, city centre, overcast sky, pedestrians, roadworks, modern architecture, debris netting, urban development
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
- *Model load time:* 1.10s
- *Generation time:* 3.27s
- *Total time:* 4.39s
- *Input validation time:* 4.36e-05
- *Prompt preparation time:* 0.00619
- *First-token latency:* 0.747
- *Cleanup time:* 0.0865
- *Prompt tokens:* 1,558
- *Generation tokens:* 135
- *Total tokens:* 1,693
- *Prompt throughput (raw):* 2,086 tok/s
- *Generation throughput (raw):* 72.7 tok/s
- *Peak memory:* 9.1
- *Active memory:* 6.5
- *Cache memory:* 0.31
- *Model-load active memory:* 6.45
- *Post-cleanup active memory:* 0.00598
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1555
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
Title: Princes Street Renovation: Scaffolding and Boots Pharmacy

Description: The image captures a bustling scene in Edinburgh's city centre, featuring extensive scaffolding on a building undergoing major renovation. A Boots pharmacy is visible beneath the scaffolding, while pedestrians navigate the area under an overcast sky. The scene is set along Princes Street, with temporary construction fences and roadworks evident.

Keywords: Princes Street, Edinburgh, Scotland, GBR, Europe, scaffolding, renovation, Boots pharmacy, overcast sky, pedestrians, city centre, construction site, roadworks, modern architecture, civil engineering, urban development, temporary fencing, street scene, architectural transformation
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
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type cohere_compass)
- *Model load time:* 1.05s
- *Generation time:* 4.14s
- *Total time:* 5.20s
- *Input validation time:* 5.28e-05
- *Prompt preparation time:* 0.00432
- *First-token latency:* 3.05
- *Cleanup time:* 0.0888
- *Prompt tokens:* 4,118
- *Generation tokens:* 77
- *Total tokens:* 4,195
- *Prompt throughput (raw):* 1,351 tok/s
- *Generation throughput (raw):* 233 tok/s
- *Peak memory:* 3.9
- *Active memory:* 2.2
- *Cache memory:* 0.64
- *Model-load active memory:* 2.18
- *Post-cleanup active memory:* 0.00651
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1648
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
Title: Edinburgh City Centre Renovation

Description: Extensive scaffolding covers the facade of a building undergoing major renovation and redevelopment along Princes Street in Edinburgh, Scotland. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath an overcast sky.

Keywords: Edinburgh, City Centre, Construction, Scaffolding, Princes Street, Boots, Overcast, Pedestrians, Roadworks, Europe
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
- *Observations:* Duplicate keywords: scaffolding, scotland
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Model load time:* 0.85s
- *Generation time:* 23.94s
- *Total time:* 24.81s
- *Input validation time:* 6.28e-05
- *Prompt preparation time:* 0.012
- *First-token latency:* 13.3
- *Cleanup time:* 0.0963
- *Prompt tokens:* 16,583
- *Generation tokens:* 902
- *Total tokens:* 17,485
- *Prompt throughput (raw):* 1,248 tok/s
- *Generation throughput (raw):* 92.6 tok/s
- *Peak memory:* 8.4
- *Active memory:* 4.3
- *Cache memory:* 2.7
- *Model-load active memory:* 4.26
- *Post-cleanup active memory:* 0.00739
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1597
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
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The context says it's Princes Street in Edinburgh, Scotland. The main subject is scaffolding on a building, renovation. So maybe "Edinburgh Princes Street Renovation Scaffolding" but let's check word count. "Edinburgh Princes Street Renovation with Scaffolding" – that's 10 words. Wait, "Princes Street Renovation Scaffolding" is 7, but maybe "Edinburgh Princes Street Renovation Scaffolding" is 10. Let's confirm.

Next, the description. Needs 1-2 sentences. Context: capture date/time, GPS, but we don't claim those. The main visible subject: scaffolding on a building (the one with the Boots pharmacy), pedestrians, construction fences, Boots pharmacy, overcast sky. So: "Overcast sky illuminates Edinburgh's Princes Street during renovation, where scaffolding covers a building with a Boots pharmacy and pedestrians walk past temporary construction fences."

Wait, the description should combine context with main visible subject, setting, action, lighting, distinctive details. Let's see: "Overcast sky, Edinburgh's Princes Street, scaffolding covers a building under renovation, with Boots pharmacy and pedestrians walking past construction fences."

Now keywords: need 10-18 unique, comma-separated. From context: City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland. But need to pick relevant ones. Let's check the image: scaffolding, crane (yellow crane), Boots pharmacy, pedestrians, overcast sky, Princes Street, Edinburgh, Scotland, construction fence, roadworks (since it's a construction site), scaffolding, modern architecture (the building with the facade), overcast sky. Let's list:

1. Edinburgh
2. Princes Street
3. Scaffolding
4. Construction site
5. Crane
6. Overcast Sky
7. Pedestrians
8. Boots Pharmacy
9. Roadworks
10. Modern Architecture
11. Construction fence
12. Civil engineering
13. Europe
14. Scotland
15. City Centre
16. Roadworks (wait, roadworks is a bit redundant, but maybe)

Wait, the context says "Roadworks" is a keyword hint. Let's check the hints: "Roadworks" is in the keyword hints. So include that. Also, "Scaffolding" is there. Let's make sure to have 10-18. Let's list:

Edinburgh, Princes Street, Scaffolding, Construction site, Crane, Overcast Sky, Pedestrians, Boots Pharmacy, Roadworks, Modern Architecture, Construction fence, Civil engineering, Europe, Scotland, City Centre, Roadworks (wait, no, roadworks is already there). Wait, the hints include "Roadworks" as a keyword hint. Let's check the list:

Keyword hints: 10 Best (structured), Adobe Stock, Any Vision, City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland

So we need to pick from these, but the description should include visible details. Let's see: the image has scaffolding, crane (yellow), Boots pharmacy, pedestrians, overcast sky, Princes Street, Edinburgh, Scotland, construction fence, roadworks (since it's a construction site), scaffolding, modern architecture (the building with the facade), overcast sky.

So the keywords should be:

Ed
&lt;/think&gt;

Title: Edinburgh Princes Street Renovation Scaffolding

Description: Overcast sky illuminates Edinburgh's Princes Street during renovation, where scaffolding covers a building with a Boots pharmacy and pedestrians walk past temporary construction fences.

Keywords: Edinburgh, Princes Street, Scaffolding, Construction site, Crane, Overcast Sky, Pedestrians, Boots Pharmacy, Roadworks, Modern Architecture, Construction fence, Civil engineering, Europe, Scotland, City Centre, Scaffolding, Scotland
</pre>

<details>
<summary>Exact raw output</summary>

```text
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The context says it's Princes Street in Edinburgh, Scotland. The main subject is scaffolding on a building, renovation. So maybe "Edinburgh Princes Street Renovation Scaffolding" but let's check word count. "Edinburgh Princes Street Renovation with Scaffolding" – that's 10 words. Wait, "Princes Street Renovation Scaffolding" is 7, but maybe "Edinburgh Princes Street Renovation Scaffolding" is 10. Let's confirm.

Next, the description. Needs 1-2 sentences. Context: capture date/time, GPS, but we don't claim those. The main visible subject: scaffolding on a building (the one with the Boots pharmacy), pedestrians, construction fences, Boots pharmacy, overcast sky. So: "Overcast sky illuminates Edinburgh's Princes Street during renovation, where scaffolding covers a building with a Boots pharmacy and pedestrians walk past temporary construction fences."

Wait, the description should combine context with main visible subject, setting, action, lighting, distinctive details. Let's see: "Overcast sky, Edinburgh's Princes Street, scaffolding covers a building under renovation, with Boots pharmacy and pedestrians walking past construction fences."

Now keywords: need 10-18 unique, comma-separated. From context: City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland. But need to pick relevant ones. Let's check the image: scaffolding, crane (yellow crane), Boots pharmacy, pedestrians, overcast sky, Princes Street, Edinburgh, Scotland, construction fence, roadworks (since it's a construction site), scaffolding, modern architecture (the building with the facade), overcast sky. Let's list:

1. Edinburgh
2. Princes Street
3. Scaffolding
4. Construction site
5. Crane
6. Overcast Sky
7. Pedestrians
8. Boots Pharmacy
9. Roadworks
10. Modern Architecture
11. Construction fence
12. Civil engineering
13. Europe
14. Scotland
15. City Centre
16. Roadworks (wait, roadworks is a bit redundant, but maybe)

Wait, the context says "Roadworks" is a keyword hint. Let's check the hints: "Roadworks" is in the keyword hints. So include that. Also, "Scaffolding" is there. Let's make sure to have 10-18. Let's list:

Edinburgh, Princes Street, Scaffolding, Construction site, Crane, Overcast Sky, Pedestrians, Boots Pharmacy, Roadworks, Modern Architecture, Construction fence, Civil engineering, Europe, Scotland, City Centre, Roadworks (wait, no, roadworks is already there). Wait, the hints include "Roadworks" as a keyword hint. Let's check the list:

Keyword hints: 10 Best (structured), Adobe Stock, Any Vision, City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland

So we need to pick from these, but the description should include visible details. Let's see: the image has scaffolding, crane (yellow), Boots pharmacy, pedestrians, overcast sky, Princes Street, Edinburgh, Scotland, construction fence, roadworks (since it's a construction site), scaffolding, modern architecture (the building with the facade), overcast sky.

So the keywords should be:

Ed
</think>

Title: Edinburgh Princes Street Renovation Scaffolding

Description: Overcast sky illuminates Edinburgh's Princes Street during renovation, where scaffolding covers a building with a Boots pharmacy and pedestrians walk past temporary construction fences.

Keywords: Edinburgh, Princes Street, Scaffolding, Construction site, Crane, Overcast Sky, Pedestrians, Boots Pharmacy, Roadworks, Modern Architecture, Construction fence, Civil engineering, Europe, Scotland, City Centre, Scaffolding, Scotland
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
- *Maintainer status:* none
- *Observations:* Duplicate keywords: urban
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Model load time:* 3.33s
- *Generation time:* 2.42s
- *Total time:* 5.78s
- *Input validation time:* 5.97e-05
- *Prompt preparation time:* 0.0239
- *First-token latency:* 0.341
- *Cleanup time:* 0.131
- *Prompt tokens:* 627
- *Generation tokens:* 83
- *Total tokens:* 710
- *Prompt throughput (raw):* 1,839 tok/s
- *Generation throughput (raw):* 61.3 tok/s
- *Peak memory:* 29
- *Active memory:* 28
- *Cache memory:* 0.01
- *Model-load active memory:* 28
- *Post-cleanup active memory:* 0.0104
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1543
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
Title: Scaffolding on Princes Street in Edinburgh City Centre
Description: Extensive scaffolding covers the facade of a building undergoing renovation along Princes Street in Edinburgh, while pedestrians walk past construction fences and a Boots pharmacy.
Keywords: Edinburgh, Scotland, Princes Street, city centre, construction, scaffolding, renovation, urban, architecture, urban, overcast, pedestrians, crane, pharmacy, Boots, Europe
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
- *Observations:* Title has 4 words (requested 5-10)
- *Arch supported by installed mlx-vlm:* yes (model_type granite4_vision)
- *Model load time:* 0.63s
- *Generation time:* 1.56s
- *Total time:* 2.21s
- *Input validation time:* 6.82e-05
- *Prompt preparation time:* 0.0104
- *First-token latency:* 0.401
- *Cleanup time:* 0.0888
- *Prompt tokens:* 1,411
- *Generation tokens:* 66
- *Total tokens:* 1,477
- *Prompt throughput (raw):* 3,515 tok/s
- *Generation throughput (raw):* 171 tok/s
- *Peak memory:* 4.7
- *Active memory:* 3.0
- *Cache memory:* 0.22
- *Model-load active memory:* 3.03
- *Post-cleanup active memory:* 0.0128
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1741
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
Title: Edinburgh City Centre Renovation
Description: Scaffolding and construction activity on a building along Princes Street, Edinburgh, with pedestrians navigating temporary barriers.
Keywords: Edinburgh, Princes Street, Construction, Scaffolding, Overcast, Pedestrians, Boots Pharmacy, Roadworks, Civil Engineering, Modern Architecture
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
  after sustained repeated output; Keyword list has 49 terms (requested
  10-18); Duplicate keywords: european city centre renewal project, european
  city centre redevelopment project, european city centre transformation
  project
- *Arch supported by installed mlx-vlm:* yes (model_type jvlm via jina_vlm)
- *Model load time:* 0.67s
- *Generation time:* 3.97s
- *Total time:* 4.64s
- *Input validation time:* 5.77e-05
- *Prompt preparation time:* 0.00204
- *First-token latency:* 0.96
- *Cleanup time:* 0.0752
- *Prompt tokens:* 2,101
- *Generation tokens:* 300
- *Total tokens:* 2,401
- *Prompt throughput (raw):* 2,188 tok/s
- *Generation throughput (raw):* 138 tok/s
- *Peak memory:* 3.7
- *Active memory:* 2.1
- *Cache memory:* 0.95
- *Model-load active memory:* 2.11
- *Post-cleanup active memory:* 0.000755
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1529
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
City Centre, Edinburgh, Scotland, UK, GBR, Europe
Description:
A bustling city centre scene in Edinburgh, Scotland, UK, under an overcast sky. Pedestrians walk by temporary construction fences, while a building undergoes major renovation and redevelopment along Princes Street. A Boots pharmacy is visible beneath the scaffolding, and a crane stands ready for the next phase of the project.
Keywords:
Princes Street, Edinburgh, Scotland, UK, Europe, City Centre, Construction site, Crane, Boots pharmacy, Overcast sky, Pedestrians, Roadworks, Scaffolding, Modern Architecture, Objects, Temporary construction fence, Road construction, Pedestrian walkway, Urban landscape, Construction equipment, City development, European cityscape, Overcast weather, City centre atmosphere, Edinburgh urban renewal, Scotland's architectural transformation, European city centre, European urban landscape, European city centre scene, European city centre development, European city centre renovation, European city centre redevelopment, European city centre transformation, European city centre renewal project, European city centre redevelopment project, European city centre transformation project, European city centre renewal project, European city centre redevelopment project, European city centre transformation project, European city centre renewal project, European city centre redevelopment project, European city centre transformation project, European city centre renewal project, European city centre redevelopment project, European city centre transformation project, European city centre renewal project, European city centre redevelopment project, European city centre transformation project, European city centre renewal project,
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
- *Model load time:* 1.83s
- *Generation time:* 26.23s
- *Total time:* 28.08s
- *Input validation time:* 5.14e-05
- *Prompt preparation time:* 0.00795
- *First-token latency:* 1.98
- *Cleanup time:* 0.0968
- *Prompt tokens:* 3,240
- *Generation tokens:* 1,000
- *Total tokens:* 4,240
- *Prompt throughput (raw):* 1,635 tok/s
- *Generation throughput (raw):* 42.1 tok/s
- *Peak memory:* 15
- *Active memory:* 13
- *Cache memory:* 0.87
- *Model-load active memory:* 12.6
- *Post-cleanup active memory:* 0.00102
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1891
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
The user wants British-English catalogue metadata from the image and supplied context. They want:

- a concrete 5-10-word title (so 5-10 words)
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

We need to produce exactly three sections: Title:, Description:, Keywords:.

We must treat capture date/time and GPS as authoritative facts, but not claim they are visible. So we can incorporate date/time and location in description if needed. The description should be 1-2 sentences, factual, combining relevant context with main visible subject, setting, action, lighting, distinctive details.

We need to ensure British English spelling (e.g., "behaviour", "organisation", "programme", etc.) but not necessarily needed.

We need to incorporate the visible details: scaffolding covering a building undergoing major renovation, construction fences, a Boots pharmacy, overcast sky, a crane, pedestrians, street sign "Princes Street", etc.

We need to incorporate the context: location is City Centre, Edinburgh, Scotland, UK, Europe. The date/time is 2026-08-18 17:46:05 UTC+01:00. GPS coordinates: 55.951722°N, 3.201417°W.

We need to produce a title of 5-10 words. Something like "Princes Street Construction Site, Edinburgh". That's 5 words? Let's count: "Princes" (1) "Street" (2) "Construction" (3) "Site," (4) "Edinburgh" (5). That's 5 words. Could also be "Edinburgh Princes Street Renovation 2026". That's 5 words? "Edinburgh" (1) "Princes" (2) "Street" (3) "Renovation" (4) "2026" (5). That's 5 words. Or "Edinburgh Princes Street Building Under Scaffolding". That's 6 words? "Edinburgh" (1) "Princes" (2) "Street" (3) "Building" (4) "Under" (5) "Scaffolding" (6). That's 6 words. That works.

We need to ensure it's 5-10 words. Let's pick "Edinburgh Princes Street Building Under Scaffolding". That's 6 words.

Now description: 1-2 sentences. Should mention date/time? Possibly incorporate date/time: "Captured on 18 August 2026 at 17:46 UTC+01:00, the image shows extensive scaffolding covering a building undergoing major renovation on Princes Street in Edinburgh's city centre, with pedestrians walking past construction fences and a Boots pharmacy beneath an overcast sky." That's one sentence. Could be two sentences: "Captured on 18 August 2026 at 17:46 UTC+01:00, the image shows extensive scaffolding covering a building undergoing major renovation on Princes Street in Edinburgh's city centre. Pedestrians walk past temporary construction fences and a Boots pharmacy beneath an overcast sky." That's two sentences. That's fine.

Now keywords: 10-18 unique, comma-separated. Should include: Edinburgh, Scotland, United Kingdom, Europe, Princes Street, construction site, scaffolding, renovation, building, crane, construction fence, temporary fencing, Boots pharmacy, pedestrians, overcast sky, city centre, modern architecture, civil engineering, roadworks, 2026, August, 17:46, UTC+01:00, GPS coordinates, etc. But we need 10-18 unique keywords. Let's pick 14 maybe.

Potential keywords: Edinburgh, Scotland, United Kingdom, Europe, Princes Street, construction site, scaffolding, renovation, building, crane, construction fence, temporary fencing, Boots pharmacy, pedestrians, overcast sky, city centre, modern architecture, civil engineering, roadworks, 2026.

But we need 10-18 unique, comma-separated. Let's count: Edinburgh (1), Scotland (2), United Kingdom (3), Europe (4), Princes Street (5), construction site (6), scaffolding (7), renovation (8), building (9), crane (10), construction fence (11), temporary fencing (12), Boots pharmacy (
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
- *Model load time:* 0.73s
- *Generation time:* 1.67s
- *Total time:* 2.42s
- *Input validation time:* 6.11e-05
- *Prompt preparation time:* 0.00246
- *First-token latency:* 0.0972
- *Cleanup time:* 0.0816
- *Prompt tokens:* 368
- *Generation tokens:* 169
- *Total tokens:* 537
- *Prompt throughput (raw):* 3,788 tok/s
- *Generation throughput (raw):* 352 tok/s
- *Peak memory:* 2.2
- *Active memory:* 1.2
- *Cache memory:* 0.04
- *Model-load active memory:* 1.25
- *Post-cleanup active memory:* 0.0018
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1612
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
A concrete 5-10-word title: "City Centre Renovation in Edinburgh, Scotland, UK, GBR, Europe"
Description: The image captures a bustling city street in Edinburgh, Scotland, featuring extensive scaffolding covering the facade of a building undergoing major renovation and redevelopment along Princes Street. The scene is set under an overcast sky, with pedestrians walking past temporary construction fences and a Boots pharmacy beneath an overcast sky. The image also includes a crane and a fence, highlighting the ongoing construction work. The photo was taken on 2026-08-18 at 17:46:05 UTC+01:00, with a GPS coordinates of 55.951722°N, 3.201417°W.
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
  reasoning block appears incomplete; Keyword list has 53 terms (requested
  10-18); Duplicate keywords: princes street, construction site, scaffolding,
  pedestrians, overcast, boots pharmacy, construction fence, crane, city
  centre, scotland, europe, roadworks, modern architecture, fence, civil
  engineering
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Model load time:* 1.79s
- *Generation time:* 27.50s
- *Total time:* 29.30s
- *Input validation time:* 5.41e-05
- *Prompt preparation time:* 0.00255
- *First-token latency:* 5.51
- *Cleanup time:* 0.102
- *Prompt tokens:* 6,480
- *Generation tokens:* 1,000
- *Total tokens:* 7,480
- *Prompt throughput (raw):* 1,177 tok/s
- *Generation throughput (raw):* 46.6 tok/s
- *Peak memory:* 13
- *Active memory:* 12
- *Cache memory:* 0.44
- *Model-load active memory:* 11.8
- *Post-cleanup active memory:* 0.00211
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1574
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
&lt;think&gt;Got it, let's tackle this task step by step. First, the title needs to be 5-10 words. Looking at the image, the main elements are scaffolding, Princes Street, Edinburgh, construction, etc. So maybe "Edinburgh Princes Street Construction Site" or something similar. Wait, need to keep it concise. Let's go with "Edinburgh Princes Street Renovation Site".

Then the description: need to combine context (date/time, location) with main visible subject (scaffolding, building renovation), setting (city centre, Princes Street), action (pedestrians walking), lighting (overcast sky), distinctive details (Boots pharmacy, crane). So something like "On 2026-08-18, scaffolding covers a building under renovation along Edinburgh's Princes Street, with pedestrians passing construction fences near a Boots pharmacy beneath an overcast sky."

Keywords need to be 10-18 unique, comma-separated. Let's list the hints and add visible details. From hints: City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland. Also, Boots pharmacy is visible, so add "Boots Pharmacy" maybe. Let's check the hints and image. The image has scaffolding, construction site, crane, pedestrians, Princes Street, Edinburgh, overcast sky, construction fence, Boots pharmacy. So keywords would include those. Let's make sure they are unique.

Now, let's structure each part:

Title: Need 5-10 words. Let's do "Edinburgh Princes Street Construction Renovation".

Description: Combine context (date/time, location) with main subject (scaffolding, building renovation), setting (city centre, Princes Street), action (pedestrians walking), lighting (overcast sky), details (Boots pharmacy, crane). So: "On 2026-08-18, scaffolding surrounds a building undergoing renovation along Edinburgh's Princes Street, with pedestrians walking past construction barriers near a Boots pharmacy under an overcast sky."

Keywords: Let's list them: Edinburgh, Princes Street, Construction site, Scaffolding, Pedestrians, Overcast, Boots Pharmacy, Construction fence, Crane, City Centre, Scotland, Europe, Roadworks, Modern Architecture, Fence, Civil engineering, Overcast Sky. Wait, need to count. Let's check the hints and image. The hints include City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland. Add Boots Pharmacy as a visible detail. So keywords: Edinburgh, Princes Street, Construction site, Scaffolding, Pedestrians, Overcast, Boots Pharmacy, Construction fence, Crane, City Centre, Scotland, Europe, Roadworks, Modern Architecture, Fence, Civil engineering, Overcast Sky. That's 17, which is within 10-18. Let's make sure they are unique.

Now, check each part:

Title: "Edinburgh Princes Street Construction Renovation" (7 words, within 5-10)

Description: "On 2026-08-18, scaffolding covers a building under renovation along Edinburgh's Princes Street, with pedestrians walking past construction barriers near a Boots pharmacy beneath an overcast sky." (2 sentences, factual, combines context and visible details)

Keywords: "Edinburgh, Princes Street, Construction site, Scaffolding, Pedestrians, Overcast, Boots Pharmacy, Construction fence, Crane, City Centre, Scotland, Europe, Roadworks, Modern Architecture, Fence, Civil engineering, Overcast Sky" (17 keywords, unique, covers context and visible details)

Wait, need to make sure the description is 1-2 sentences. Let's adjust. "On 2026-08-18, scaffolding covers a building undergoing major renovation along Edinburgh's Princes Street, with pedestrians walking past temporary construction fences near a Boots pharmacy under an overcast sky." That's one sentence, maybe. Or two. Let's do two sentences for clarity. "On 2026-08-18, extensive scaffolding covers the facade of a building undergoing major renovation along Edinburgh's Princes Street. Pedestrians walk past temporary construction fences near a Boots pharmacy under an overcast sky." That's two sentences, factual.

Keywords need to be comma-separated, unique. Let's list again, ensuring no duplicates. From hints and image:

- Edinburgh (location)
- Princes Street (street)
- Construction site (main subject)
- Scaffolding (visible detail)
- Pedestrians (action)
- Overcast (lighting)
-
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words. Looking at the image, the main elements are scaffolding, Princes Street, Edinburgh, construction, etc. So maybe "Edinburgh Princes Street Construction Site" or something similar. Wait, need to keep it concise. Let's go with "Edinburgh Princes Street Renovation Site".

Then the description: need to combine context (date/time, location) with main visible subject (scaffolding, building renovation), setting (city centre, Princes Street), action (pedestrians walking), lighting (overcast sky), distinctive details (Boots pharmacy, crane). So something like "On 2026-08-18, scaffolding covers a building under renovation along Edinburgh's Princes Street, with pedestrians passing construction fences near a Boots pharmacy beneath an overcast sky."

Keywords need to be 10-18 unique, comma-separated. Let's list the hints and add visible details. From hints: City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland. Also, Boots pharmacy is visible, so add "Boots Pharmacy" maybe. Let's check the hints and image. The image has scaffolding, construction site, crane, pedestrians, Princes Street, Edinburgh, overcast sky, construction fence, Boots pharmacy. So keywords would include those. Let's make sure they are unique.

Now, let's structure each part:

Title: Need 5-10 words. Let's do "Edinburgh Princes Street Construction Renovation".

Description: Combine context (date/time, location) with main subject (scaffolding, building renovation), setting (city centre, Princes Street), action (pedestrians walking), lighting (overcast sky), details (Boots pharmacy, crane). So: "On 2026-08-18, scaffolding surrounds a building undergoing renovation along Edinburgh's Princes Street, with pedestrians walking past construction barriers near a Boots pharmacy under an overcast sky."

Keywords: Let's list them: Edinburgh, Princes Street, Construction site, Scaffolding, Pedestrians, Overcast, Boots Pharmacy, Construction fence, Crane, City Centre, Scotland, Europe, Roadworks, Modern Architecture, Fence, Civil engineering, Overcast Sky. Wait, need to count. Let's check the hints and image. The hints include City Centre, Civil engineering, Construction fence, Construction site, Crane, Edinburgh, Europe, Fence, Modern Architecture, Objects, Overcast, Overcast Sky, Pedestrians, Princes Street, Roadworks, Scaffolding, Scotland. Add Boots Pharmacy as a visible detail. So keywords: Edinburgh, Princes Street, Construction site, Scaffolding, Pedestrians, Overcast, Boots Pharmacy, Construction fence, Crane, City Centre, Scotland, Europe, Roadworks, Modern Architecture, Fence, Civil engineering, Overcast Sky. That's 17, which is within 10-18. Let's make sure they are unique.

Now, check each part:

Title: "Edinburgh Princes Street Construction Renovation" (7 words, within 5-10)

Description: "On 2026-08-18, scaffolding covers a building under renovation along Edinburgh's Princes Street, with pedestrians walking past construction barriers near a Boots pharmacy beneath an overcast sky." (2 sentences, factual, combines context and visible details)

Keywords: "Edinburgh, Princes Street, Construction site, Scaffolding, Pedestrians, Overcast, Boots Pharmacy, Construction fence, Crane, City Centre, Scotland, Europe, Roadworks, Modern Architecture, Fence, Civil engineering, Overcast Sky" (17 keywords, unique, covers context and visible details)

Wait, need to make sure the description is 1-2 sentences. Let's adjust. "On 2026-08-18, scaffolding covers a building undergoing major renovation along Edinburgh's Princes Street, with pedestrians walking past temporary construction fences near a Boots pharmacy under an overcast sky." That's one sentence, maybe. Or two. Let's do two sentences for clarity. "On 2026-08-18, extensive scaffolding covers the facade of a building undergoing major renovation along Edinburgh's Princes Street. Pedestrians walk past temporary construction fences near a Boots pharmacy under an overcast sky." That's two sentences, factual.

Keywords need to be comma-separated, unique. Let's list again, ensuring no duplicates. From hints and image:

- Edinburgh (location)
- Princes Street (street)
- Construction site (main subject)
- Scaffolding (visible detail)
- Pedestrians (action)
- Overcast (lighting)
-
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
- *Model load time:* 0.89s
- *Generation time:* 1.08s
- *Total time:* 1.98s
- *Input validation time:* 4.76e-05
- *Prompt preparation time:* 0.00929
- *First-token latency:* 0.25
- *Cleanup time:* 0.0913
- *Prompt tokens:* 966
- *Generation tokens:* 46
- *Total tokens:* 1,012
- *Prompt throughput (raw):* 3,861 tok/s
- *Generation throughput (raw):* 273 tok/s
- *Peak memory:* 3.8
- *Active memory:* 2.3
- *Cache memory:* 0.05
- *Model-load active memory:* 2.32
- *Post-cleanup active memory:* 0.00488
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1585
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
Urban construction with scaffolding on city buildings.

Keywords: urban construction, scaffolding, city buildings, Princes Street, Edinburgh, construction fence, crane, overcast sky, pedestrians, modern architecture, civil engineering, Europe
</pre>

</details>

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

<details>
<summary>Complete evidence: mlx-community/SmolVLM2-2.2B-Instruct-mlx</summary>

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Missing or empty fields: Title,
  Description, Keywords
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Model load time:* 0.72s
- *Generation time:* 2.71s
- *Total time:* 3.44s
- *Input validation time:* 5.96e-05
- *Prompt preparation time:* 0.00296
- *First-token latency:* 0.387
- *Cleanup time:* 0.0866
- *Prompt tokens:* 1,461
- *Generation tokens:* 200
- *Total tokens:* 1,661
- *Prompt throughput (raw):* 3,771 tok/s
- *Generation throughput (raw):* 124 tok/s
- *Peak memory:* 5.5
- *Active memory:* 4.5
- *Cache memory:* 0.72
- *Model-load active memory:* 4.5
- *Post-cleanup active memory:* 0.00986
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1549
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
 Edinburgh, Scotland, UK, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland, Edinburgh, Scotland,
</pre>

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
  the token limit; Title has 4 words (requested 5-10); Keyword list has 308
  terms (requested 10-18); Duplicate keywords: scaffolding, construction site,
  construction equipment, construction progress, building maintenance,
  construction safety, building renovation, building restoration, commercial
  building, construction activity, pedestrian walkway, construction site
  signage, building exterior, construction materials, building facade
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.15s
- *Generation time:* 35.42s
- *Total time:* 36.59s
- *Input validation time:* 9.05e-05
- *Prompt preparation time:* 0.00426
- *First-token latency:* 16.6
- *Cleanup time:* 0.0979
- *Prompt tokens:* 16,592
- *Generation tokens:* 1,000
- *Total tokens:* 17,592
- *Prompt throughput (raw):* 1,001 tok/s
- *Generation throughput (raw):* 55.6 tok/s
- *Peak memory:* 13
- *Active memory:* 9.5
- *Cache memory:* 1.3
- *Model-load active memory:* 9.46
- *Post-cleanup active memory:* 0.0104
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1647
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
Title: Edinburgh Princes Street Construction

Description: A Boots pharmacy on Princes Street in Edinburgh, Scotland, remains open as pedestrians walk past a major renovation site with extensive scaffolding and a construction crane under an overcast sky.

Keywords: Edinburgh, Princes Street, Boots, pharmacy, construction, scaffolding, renovation, pedestrians, overcast, Scotland, Europe, roadworks, modern architecture, civil engineering, fence, crane, building, facelift, urban development, retail, retail store, safety barriers, city centre, scaffolding, construction site, roadwork, pedestrian activity, commercial area, urban renewal, scaffolding, construction equipment, building facades, retail space, public access, safety signage, construction progress, urban landscape, city infrastructure, building maintenance, commercial district, retail environment, construction safety, building renovation, public space, urban planning, construction industry, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation, construction equipment, construction site, building maintenance, construction safety, building restoration, commercial building, construction activity, pedestrian walkway, construction site signage, building exterior, construction materials, building facade, construction progress, building renovation
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
- *Model load time:* 2.29s
- *Generation time:* 0.95s
- *Total time:* 3.26s
- *Input validation time:* 4.85e-05
- *Prompt preparation time:* 0.00426
- *First-token latency:* 0.396
- *Cleanup time:* 0.131
- *Prompt tokens:* 625
- *Generation tokens:* 1
- *Total tokens:* 626
- *Prompt throughput (raw):* 1,578 tok/s
- *Generation throughput (raw):* 56,208 tok/s
- *Peak memory:* 17
- *Active memory:* 16
- *Cache memory:* 0.05
- *Model-load active memory:* 15.7
- *Post-cleanup active memory:* 0.0115
- *Post-cleanup cache memory:* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 1574
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

<a id="model-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Muse-Glimmer-30B-OptiQ-4bit</summary>

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Failure phase:* model_load
- *Error stage:* Model Error
- *Error code:* MLX_VLM_MODEL_LOAD_MODEL
- *Error type:* ValueError
- *Error package:* mlx-vlm
- *Error message:* Model loading failed: Received 1460 parameters not in
  model:
  <br>embed_tokens.biases,<br>embed_tokens.scales,<br>embed_tokens.weight,<br>layers.0.input_layernorm.weight,<br>layers.0.mlp.down_proj.biases,<br>layers.0.mlp.down_proj.scales,<br>layers.0.mlp.down_proj.weight,<br>layers.0.mlp.gate_proj.biases,<br>layers.0.mlp.gate_proj.scales,<br>layers.0.mlp.gate_proj.weight,<br>layers.0.mlp.up_proj.biases,<br>layers.0.mlp.up_proj.scales,<br>layers.0.mlp.up_proj.weight,<br>layers.0.post_attention_layernorm.weight,<br>layers.0.post_feedforward_layernorm.weight,<br>layers.0.pre_feedforward_layernorm.weight,<br>layers.0.self_attn.gate_proj.biases,<br>layers.0.self_attn.gate_proj.scales,<br>layers.0.self_attn.gate_proj.weight,<br>layers.0.self_attn.k_proj.biases,<br>layers.0.self_attn.k_proj.scales,<br>layers.0.self_attn.k_proj.weight,<br>layers.0.self_attn.o_proj.biases,<br>layers.0.self_attn.o_proj.scales,<br>layers.0.self_attn.o_proj.weight,<br>layers.0.self_attn.q_proj.biases,<br>layers.0.self_attn.q_proj.scales,<br>layers.0.self_attn.q_proj.weight,<br>layers.0.self_attn.v_proj.biases,<br>layers.0.self_attn.v_proj.scales,<br>layers.0.self_attn.v_proj.weight,<br>layers.1.input_layernorm.weight,<br>layers.1.mlp.down_proj.biases,<br>layers.1.mlp.down_proj.scales,<br>layers.1.mlp.down_proj.weight,<br>layers.1.mlp.gate_proj.biases,<br>layers.1.mlp.gate_proj.scales,<br>layers.1.mlp.gate_proj.weight,<br>layers.1.mlp.up_proj.biases,<br>layers.1.mlp.up_proj.scales,<br>layers.1.mlp.up_proj.weight,<br>layers.1.post_attention_layernorm.weight,<br>layers.1.post_feedforward_layernorm.weight,<br>layers.1.pre_feedforward_layernorm.weight,<br>layers.1.self_attn.gate_proj.biases,<br>layers.1.self_attn.gate_proj.scales,<br>layers.1.self_attn.gate_proj.weight,<br>layers.1.self_attn.k_proj.biases,<br>layers.1.self_attn.k_proj.scales,<br>layers.1.self_attn.k_proj.weight,<br>layers.1.self_attn.o_proj.biases,<br>layers.1.self_attn.o_proj.scales,<br>layers.1.self_attn.o_proj.weight,<br>layers.1.self_attn.q_proj.biases,<br>layers.1.self_attn.q_proj.scales,<br>layers.1.self_attn.q_proj.weight,<br>layers.1.self_attn.v_proj.biases,<br>layers.1.self_attn.v_proj.scales,<br>layers.1.self_attn.v_proj.weight,<br>layers.10.input_layernorm.weight,<br>layers.10.mlp.down_proj.biases,<br>layers.10.mlp.down_proj.scales,<br>layers.10.mlp.down_proj.weight,<br>layers.10.mlp.gate_proj.biases,<br>layers.10.mlp.gate_proj.scales,<br>layers.10.mlp.gate_proj.weight,<br>layers.10.mlp.up_proj.biases,<br>layers.10.mlp.up_proj.scales,<br>layers.10.mlp.up_proj.weight,<br>layers.10.post_attention_layernorm.weight,<br>layers.10.post_feedforward_layernorm.weight,<br>layers.10.pre_feedforward_layernorm.weight,<br>layers.10.self_attn.gate_proj.biases,<br>layers.10.self_attn.gate_proj.scales,<br>layers.10.self_attn.gate_proj.weight,<br>layers.10.self_attn.k_proj.biases,<br>layers.10.self_attn.k_proj.scales,<br>layers.10.self_attn.k_proj.weight,<br>layers.10.self_attn.o_proj.biases,<br>layers.10.self_attn.o_proj.scales,<br>layers.10.self_attn.o_proj.weight,<br>layers.10.self_attn.q_proj.biases,<br>layers.10.self_attn.q_proj.scales,<br>layers.10.self_attn.q_proj.weight,<br>layers.10.self_attn.v_proj.biases,<br>layers.10.self_attn.v_proj.scales,<br>layers.10.self_attn.v_proj.weight,<br>layers.11.input_layernorm.weight,<br>layers.11.mlp.down_proj.biases,<br>layers.11.mlp.down_proj.scales,<br>layers.11.mlp.down_proj.weight,<br>layers.11.mlp.gate_proj.biases,<br>layers.11.mlp.gate_proj.scales,<br>layers.11.mlp.gate_proj.weight,<br>layers.11.mlp.up_proj.biases,<br>layers.11.mlp.up_proj.scales,<br>layers.11.mlp.up_proj.weight,<br>layers.11.post_attention_layernorm.weight,<br>layers.11.post_feedforward_layernorm.weight,<br>layers.11.pre_feedforward_layernorm.weight,<br>layers.11.self_attn.gate_proj.biases,<br>layers.11.self_attn.gate_proj.scales,<br>layers.11.self_attn.gate_proj.weight,<br>layers.11.self_attn.k_proj.biases,<br>layers.11.self_attn.k_proj.scales,<br>layers.11.self_attn.k_proj.weight,<br>layers.11.self_attn.o_proj.biases,<br>layers.11.self_attn.o_proj.scales,<br>layers.11.self_attn.o_proj.weight,<br>layers.11.self_attn.q_proj.biases,<br>layers.11.self_attn.q_proj.scales,<br>layers.11.self_attn.q_proj.weight,<br>layers.11.self_attn.v_proj.biases,<br>layers.11.self_attn.v_proj.scales,<br>layers.11.self_attn.v_proj.weight,<br>layers.12.input_layernorm.weight,<br>layers.12.mlp.down_proj.biases,<br>layers.12.mlp.down_proj.scales,<br>layers.12.mlp.down_proj.weight,<br>layers.12.mlp.gate_proj.biases,<br>layers.12.mlp.gate_proj.scales,<br>layers.12.mlp.gate_proj.weight,<br>layers.12.mlp.up_proj.biases,<br>layers.12.mlp.up_proj.scales,<br>layers.12.mlp.up_proj.weight,<br>layers.12.post_attention_layernorm.weight,<br>layers.12.post_feedforward_layernorm.weight,<br>layers.12.pre_feedforward_layernorm.weight,<br>layers.12.self_attn.gate_proj.biases,<br>layers.12.self_attn.gate_proj.scales,<br>layers.12.self_attn.gate_proj.weight,<br>layers.12.self_attn.k_proj.biases,<br>layers.12.self_attn.k_proj.scales,<br>layers.12.self_attn.k_proj.weight,<br>layers.12.self_attn.o_proj.biases,<br>layers.12.self_attn.o_proj.scales,<br>layers.12.self_attn.o_proj.weight,<br>layers.12.self_attn.q_proj.biases,<br>layers.12.self_attn.q_proj.scales,<br>layers.12.self_attn.q_proj.weight,<br>layers.12.self_attn.v_proj.biases,<br>layers.12.self_attn.v_proj.scales,<br>layers.12.self_attn.v_proj.weight,<br>layers.13.input_layernorm.weight,<br>layers.13.mlp.down_proj.biases,<br>layers.13.mlp.down_proj.scales,<br>layers.13.mlp.down_proj.weight,<br>layers.13.mlp.gate_proj.biases,<br>layers.13.mlp.gate_proj.scales,<br>layers.13.mlp.gate_proj.weight,<br>layers.13.mlp.up_proj.biases,<br>layers.13.mlp.up_proj.scales,<br>layers.13.mlp.up_proj.weight,<br>layers.13.post_attention_layernorm.weight,<br>layers.13.post_feedforward_layernorm.weight,<br>layers.13.pre_feedforward_layernorm.weight,<br>layers.13.self_attn.gate_proj.biases,<br>layers.13.self_attn.gate_proj.scales,<br>layers.13.self_attn.gate_proj.weight,<br>layers.13.self_attn.k_proj.biases,<br>layers.13.self_attn.k_proj.scales,<br>layers.13.self_attn.k_proj.weight,<br>layers.13.self_attn.o_proj.biases,<br>layers.13.self_attn.o_proj.scales,<br>layers.13.self_attn.o_proj.weight,<br>layers.13.self_attn.q_proj.biases,<br>layers.13.self_attn.q_proj.scales,<br>layers.13.self_attn.q_proj.weight,<br>layers.13.self_attn.v_proj.biases,<br>layers.13.self_attn.v_proj.scales,<br>layers.13.self_attn.v_proj.weight,<br>layers.14.input_layernorm.weight,<br>layers.14.mlp.down_proj.biases,<br>layers.14.mlp.down_proj.scales,<br>layers.14.mlp.down_proj.weight,<br>layers.14.mlp.gate_proj.biases,<br>layers.14.mlp.gate_proj.scales,<br>layers.14.mlp.gate_proj.weight,<br>layers.14.mlp.up_proj.biases,<br>layers.14.mlp.up_proj.scales,<br>layers.14.mlp.up_proj.weight,<br>layers.14.post_attention_layernorm.weight,<br>layers.14.post_feedforward_layernorm.weight,<br>layers.14.pre_feedforward_layernorm.weight,<br>layers.14.self_attn.gate_proj.biases,<br>layers.14.self_attn.gate_proj.scales,<br>layers.14.self_attn.gate_proj.weight,<br>layers.14.self_attn.k_proj.biases,<br>layers.14.self_attn.k_proj.scales,<br>layers.14.self_attn.k_proj.weight,<br>layers.14.self_attn.o_proj.biases,<br>layers.14.self_attn.o_proj.scales,<br>layers.14.self_attn.o_proj.weight,<br>layers.14.self_attn.q_proj.biases,<br>layers.14.self_attn.q_proj.scales,<br>layers.14.self_attn.q_proj.weight,<br>layers.14.self_attn.v_proj.biases,<br>layers.14.self_attn.v_proj.scales,<br>layers.14.self_attn.v_proj.weight,<br>layers.15.input_layernorm.weight,<br>layers.15.mlp.down_proj.biases,<br>layers.15.mlp.down_proj.scales,<br>layers.15.mlp.down_proj.weight,<br>layers.15.mlp.gate_proj.biases,<br>layers.15.mlp.gate_proj.scales,<br>layers.15.mlp.gate_proj.weight,<br>layers.15.mlp.up_proj.biases,<br>layers.15.mlp.up_proj.scales,<br>layers.15.mlp.up_proj.weight,<br>layers.15.post_attention_layernorm.weight,<br>layers.15.post_feedforward_layernorm.weight,<br>layers.15.pre_feedforward_layernorm.weight,<br>layers.15.self_attn.gate_proj.biases,<br>layers.15.self_attn.gate_proj.scales,<br>layers.15.self_attn.gate_proj.weight,<br>layers.15.self_attn.k_proj.biases,<br>layers.15.self_attn.k_proj.scales,<br>layers.15.self_attn.k_proj.weight,<br>layers.15.self_attn.o_proj.biases,<br>layers.15.self_attn.o_proj.scales,<br>layers.15.self_attn.o_proj.weight,<br>layers.15.self_attn.q_proj.biases,<br>layers.15.self_attn.q_proj.scales,<br>layers.15.self_attn.q_proj.weight,<br>layers.15.self_attn.v_proj.biases,<br>layers.15.self_attn.v_proj.scales,<br>layers.15.self_attn.v_proj.weight,<br>layers.16.input_layernorm.weight,<br>layers.16.mlp.down_proj.biases,<br>layers.16.mlp.down_proj.scales,<br>layers.16.mlp.down_proj.weight,<br>layers.16.mlp.gate_proj.biases,<br>layers.16.mlp.gate_proj.scales,<br>layers.16.mlp.gate_proj.weight,<br>layers.16.mlp.up_proj.biases,<br>layers.16.mlp.up_proj.scales,<br>layers.16.mlp.up_proj.weight,<br>layers.16.post_attention_layernorm.weight,<br>layers.16.post_feedforward_layernorm.weight,<br>layers.16.pre_feedforward_layernorm.weight,<br>layers.16.self_attn.gate_proj.biases,<br>layers.16.self_attn.gate_proj.scales,<br>layers.16.self_attn.gate_proj.weight,<br>layers.16.self_attn.k_proj.biases,<br>layers.16.self_attn.k_proj.scales,<br>layers.16.self_attn.k_proj.weight,<br>layers.16.self_attn.o_proj.biases,<br>layers.16.self_attn.o_proj.scales,<br>layers.16.self_attn.o_proj.weight,<br>layers.16.self_attn.q_proj.biases,<br>layers.16.self_attn.q_proj.scales,<br>layers.16.self_attn.q_proj.weight,<br>layers.16.self_attn.v_proj.biases,<br>layers.16.self_attn.v_proj.scales,<br>layers.16.self_attn.v_proj.weight,<br>layers.17.input_layernorm.weight,<br>layers.17.mlp.down_proj.biases,<br>layers.17.mlp.down_proj.scales,<br>layers.17.mlp.down_proj.weight,<br>layers.17.mlp.gate_proj.biases,<br>layers.17.mlp.gate_proj.scales,<br>layers.17.mlp.gate_proj.weight,<br>layers.17.mlp.up_proj.biases,<br>layers.17.mlp.up_proj.scales,<br>layers.17.mlp.up_proj.weight,<br>layers.17.post_attention_layernorm.weight,<br>layers.17.post_feedforward_layernorm.weight,<br>layers.17.pre_feedforward_layernorm.weight,<br>layers.17.self_attn.gate_proj.biases,<br>layers.17.self_attn.gate_proj.scales,<br>layers.17.self_attn.gate_proj.weight,<br>layers.17.self_attn.k_proj.biases,<br>layers.17.self_attn.k_proj.scales,<br>layers.17.self_attn.k_proj.weight,<br>layers.17.self_attn.o_proj.biases,<br>layers.17.self_attn.o_proj.scales,<br>layers.17.self_attn.o_proj.weight,<br>layers.17.self_attn.q_proj.biases,<br>layers.17.self_attn.q_proj.scales,<br>layers.17.self_attn.q_proj.weight,<br>layers.17.self_attn.v_proj.biases,<br>layers.17.self_attn.v_proj.scales,<br>layers.17.self_attn.v_proj.weight,<br>layers.18.input_layernorm.weight,<br>layers.18.mlp.down_proj.biases,<br>layers.18.mlp.down_proj.scales,<br>layers.18.mlp.down_proj.weight,<br>layers.18.mlp.gate_proj.biases,<br>layers.18.mlp.gate_proj.scales,<br>layers.18.mlp.gate_proj.weight,<br>layers.18.mlp.up_proj.biases,<br>layers.18.mlp.up_proj.scales,<br>layers.18.mlp.up_proj.weight,<br>layers.18.post_attention_layernorm.weight,<br>layers.18.post_feedforward_layernorm.weight,<br>layers.18.pre_feedforward_layernorm.weight,<br>layers.18.self_attn.gate_proj.biases,<br>layers.18.self_attn.gate_proj.scales,<br>layers.18.self_attn.gate_proj.weight,<br>layers.18.self_attn.k_proj.biases,<br>layers.18.self_attn.k_proj.scales,<br>layers.18.self_attn.k_proj.weight,<br>layers.18.self_attn.o_proj.biases,<br>layers.18.self_attn.o_proj.scales,<br>layers.18.self_attn.o_proj.weight,<br>layers.18.self_attn.q_proj.biases,<br>layers.18.self_attn.q_proj.scales,<br>layers.18.self_attn.q_proj.weight,<br>layers.18.self_attn.v_proj.biases,<br>layers.18.self_attn.v_proj.scales,<br>layers.18.self_attn.v_proj.weight,<br>layers.19.input_layernorm.weight,<br>layers.19.mlp.down_proj.biases,<br>layers.19.mlp.down_proj.scales,<br>layers.19.mlp.down_proj.weight,<br>layers.19.mlp.gate_proj.biases,<br>layers.19.mlp.gate_proj.scales,<br>layers.19.mlp.gate_proj.weight,<br>layers.19.mlp.up_proj.biases,<br>layers.19.mlp.up_proj.scales,<br>layers.19.mlp.up_proj.weight,<br>layers.19.post_attention_layernorm.weight,<br>layers.19.post_feedforward_layernorm.weight,<br>layers.19.pre_feedforward_layernorm.weight,<br>layers.19.self_attn.gate_proj.biases,<br>layers.19.self_attn.gate_proj.scales,<br>layers.19.self_attn.gate_proj.weight,<br>layers.19.self_attn.k_proj.biases,<br>layers.19.self_attn.k_proj.scales,<br>layers.19.self_attn.k_proj.weight,<br>layers.19.self_attn.o_proj.biases,<br>layers.19.self_attn.o_proj.scales,<br>layers.19.self_attn.o_proj.weight,<br>layers.19.self_attn.q_proj.biases,<br>layers.19.self_attn.q_proj.scales,<br>layers.19.self_attn.q_proj.weight,<br>layers.19.self_attn.v_proj.biases,<br>layers.19.self_attn.v_proj.scales,<br>layers.19.self_attn.v_proj.weight,<br>layers.2.input_layernorm.weight,<br>layers.2.mlp.down_proj.biases,<br>layers.2.mlp.down_proj.scales,<br>layers.2.mlp.down_proj.weight,<br>layers.2.mlp.gate_proj.biases,<br>layers.2.mlp.gate_proj.scales,<br>layers.2.mlp.gate_proj.weight,<br>layers.2.mlp.up_proj.biases,<br>layers.2.mlp.up_proj.scales,<br>layers.2.mlp.up_proj.weight,<br>layers.2.post_attention_layernorm.weight,<br>layers.2.post_feedforward_layernorm.weight,<br>layers.2.pre_feedforward_layernorm.weight,<br>layers.2.self_attn.gate_proj.biases,<br>layers.2.self_attn.gate_proj.scales,<br>layers.2.self_attn.gate_proj.weight,<br>layers.2.self_attn.k_proj.biases,<br>layers.2.self_attn.k_proj.scales,<br>layers.2.self_attn.k_proj.weight,<br>layers.2.self_attn.o_proj.biases,<br>layers.2.self_attn.o_proj.scales,<br>layers.2.self_attn.o_proj.weight,<br>layers.2.self_attn.q_proj.biases,<br>layers.2.self_attn.q_proj.scales,<br>layers.2.self_attn.q_proj.weight,<br>layers.2.self_attn.v_proj.biases,<br>layers.2.self_attn.v_proj.scales,<br>layers.2.self_attn.v_proj.weight,<br>layers.20.input_layernorm.weight,<br>layers.20.mlp.down_proj.biases,<br>layers.20.mlp.down_proj.scales,<br>layers.20.mlp.down_proj.weight,<br>layers.20.mlp.gate_proj.biases,<br>layers.20.mlp.gate_proj.scales,<br>layers.20.mlp.gate_proj.weight,<br>layers.20.mlp.up_proj.biases,<br>layers.20.mlp.up_proj.scales,<br>layers.20.mlp.up_proj.weight,<br>layers.20.post_attention_layernorm.weight,<br>layers.20.post_feedforward_layernorm.weight,<br>layers.20.pre_feedforward_layernorm.weight,<br>layers.20.self_attn.gate_proj.biases,<br>layers.20.self_attn.gate_proj.scales,<br>layers.20.self_attn.gate_proj.weight,<br>layers.20.self_attn.k_proj.biases,<br>layers.20.self_attn.k_proj.scales,<br>layers.20.self_attn.k_proj.weight,<br>layers.20.self_attn.o_proj.biases,<br>layers.20.self_attn.o_proj.scales,<br>layers.20.self_attn.o_proj.weight,<br>layers.20.self_attn.q_proj.biases,<br>layers.20.self_attn.q_proj.scales,<br>layers.20.self_attn.q_proj.weight,<br>layers.20.self_attn.v_proj.biases,<br>layers.20.self_attn.v_proj.scales,<br>layers.20.self_attn.v_proj.weight,<br>layers.21.input_layernorm.weight,<br>layers.21.mlp.down_proj.biases,<br>layers.21.mlp.down_proj.scales,<br>layers.21.mlp.down_proj.weight,<br>layers.21.mlp.gate_proj.biases,<br>layers.21.mlp.gate_proj.scales,<br>layers.21.mlp.gate_proj.weight,<br>layers.21.mlp.up_proj.biases,<br>layers.21.mlp.up_proj.scales,<br>layers.21.mlp.up_proj.weight,<br>layers.21.post_attention_layernorm.weight,<br>layers.21.post_feedforward_layernorm.weight,<br>layers.21.pre_feedforward_layernorm.weight,<br>layers.21.self_attn.gate_proj.biases,<br>layers.21.self_attn.gate_proj.scales,<br>layers.21.self_attn.gate_proj.weight,<br>layers.21.self_attn.k_proj.biases,<br>layers.21.self_attn.k_proj.scales,<br>layers.21.self_attn.k_proj.weight,<br>layers.21.self_attn.o_proj.biases,<br>layers.21.self_attn.o_proj.scales,<br>layers.21.self_attn.o_proj.weight,<br>layers.21.self_attn.q_proj.biases,<br>layers.21.self_attn.q_proj.scales,<br>layers.21.self_attn.q_proj.weight,<br>layers.21.self_attn.v_proj.biases,<br>layers.21.self_attn.v_proj.scales,<br>layers.21.self_attn.v_proj.weight,<br>layers.22.input_layernorm.weight,<br>layers.22.mlp.down_proj.biases,<br>layers.22.mlp.down_proj.scales,<br>layers.22.mlp.down_proj.weight,<br>layers.22.mlp.gate_proj.biases,<br>layers.22.mlp.gate_proj.scales,<br>layers.22.mlp.gate_proj.weight,<br>layers.22.mlp.up_proj.biases,<br>layers.22.mlp.up_proj.scales,<br>layers.22.mlp.up_proj.weight,<br>layers.22.post_attention_layernorm.weight,<br>layers.22.post_feedforward_layernorm.weight,<br>layers.22.pre_feedforward_layernorm.weight,<br>layers.22.self_attn.gate_proj.biases,<br>layers.22.self_attn.gate_proj.scales,<br>layers.22.self_attn.gate_proj.weight,<br>layers.22.self_attn.k_proj.biases,<br>layers.22.self_attn.k_proj.scales,<br>layers.22.self_attn.k_proj.weight,<br>layers.22.self_attn.o_proj.biases,<br>layers.22.self_attn.o_proj.scales,<br>layers.22.self_attn.o_proj.weight,<br>layers.22.self_attn.q_proj.biases,<br>layers.22.self_attn.q_proj.scales,<br>layers.22.self_attn.q_proj.weight,<br>layers.22.self_attn.v_proj.biases,<br>layers.22.self_attn.v_proj.scales,<br>layers.22.self_attn.v_proj.weight,<br>layers.23.input_layernorm.weight,<br>layers.23.mlp.down_proj.biases,<br>layers.23.mlp.down_proj.scales,<br>layers.23.mlp.down_proj.weight,<br>layers.23.mlp.gate_proj.biases,<br>layers.23.mlp.gate_proj.scales,<br>layers.23.mlp.gate_proj.weight,<br>layers.23.mlp.up_proj.biases,<br>layers.23.mlp.up_proj.scales,<br>layers.23.mlp.up_proj.weight,<br>layers.23.post_attention_layernorm.weight,<br>layers.23.post_feedforward_layernorm.weight,<br>layers.23.pre_feedforward_layernorm.weight,<br>layers.23.self_attn.gate_proj.biases,<br>layers.23.self_attn.gate_proj.scales,<br>layers.23.self_attn.gate_proj.weight,<br>layers.23.self_attn.k_proj.biases,<br>layers.23.self_attn.k_proj.scales,<br>layers.23.self_attn.k_proj.weight,<br>layers.23.self_attn.o_proj.biases,<br>layers.23.self_attn.o_proj.scales,<br>layers.23.self_attn.o_proj.weight,<br>layers.23.self_attn.q_proj.biases,<br>layers.23.self_attn.q_proj.scales,<br>layers.23.self_attn.q_proj.weight,<br>layers.23.self_attn.v_proj.biases,<br>layers.23.self_attn.v_proj.scales,<br>layers.23.self_attn.v_proj.weight,<br>layers.24.input_layernorm.weight,<br>layers.24.mlp.down_proj.biases,<br>layers.24.mlp.down_proj.scales,<br>layers.24.mlp.down_proj.weight,<br>layers.24.mlp.gate_proj.biases,<br>layers.24.mlp.gate_proj.scales,<br>layers.24.mlp.gate_proj.weight,<br>layers.24.mlp.up_proj.biases,<br>layers.24.mlp.up_proj.scales,<br>layers.24.mlp.up_proj.weight,<br>layers.24.post_attention_layernorm.weight,<br>layers.24.post_feedforward_layernorm.weight,<br>layers.24.pre_feedforward_layernorm.weight,<br>layers.24.self_attn.gate_proj.biases,<br>layers.24.self_attn.gate_proj.scales,<br>layers.24.self_attn.gate_proj.weight,<br>layers.24.self_attn.k_proj.biases,<br>layers.24.self_attn.k_proj.scales,<br>layers.24.self_attn.k_proj.weight,<br>layers.24.self_attn.o_proj.biases,<br>layers.24.self_attn.o_proj.scales,<br>layers.24.self_attn.o_proj.weight,<br>layers.24.self_attn.q_proj.biases,<br>layers.24.self_attn.q_proj.scales,<br>layers.24.self_attn.q_proj.weight,<br>layers.24.self_attn.v_proj.biases,<br>layers.24.self_attn.v_proj.scales,<br>layers.24.self_attn.v_proj.weight,<br>layers.25.input_layernorm.weight,<br>layers.25.mlp.down_proj.biases,<br>layers.25.mlp.down_proj.scales,<br>layers.25.mlp.down_proj.weight,<br>layers.25.mlp.gate_proj.biases,<br>layers.25.mlp.gate_proj.scales,<br>layers.25.mlp.gate_proj.weight,<br>layers.25.mlp.up_proj.biases,<br>layers.25.mlp.up_proj.scales,<br>layers.25.mlp.up_proj.weight,<br>layers.25.post_attention_layernorm.weight,<br>layers.25.post_feedforward_layernorm.weight,<br>layers.25.pre_feedforward_layernorm.weight,<br>layers.25.self_attn.gate_proj.biases,<br>layers.25.self_attn.gate_proj.scales,<br>layers.25.self_attn.gate_proj.weight,<br>layers.25.self_attn.k_proj.biases,<br>layers.25.self_attn.k_proj.scales,<br>layers.25.self_attn.k_proj.weight,<br>layers.25.self_attn.o_proj.biases,<br>layers.25.self_attn.o_proj.scales,<br>layers.25.self_attn.o_proj.weight,<br>layers.25.self_attn.q_proj.biases,<br>layers.25.self_attn.q_proj.scales,<br>layers.25.self_attn.q_proj.weight,<br>layers.25.self_attn.v_proj.biases,<br>layers.25.self_attn.v_proj.scales,<br>layers.25.self_attn.v_proj.weight,<br>layers.26.input_layernorm.weight,<br>layers.26.mlp.down_proj.biases,<br>layers.26.mlp.down_proj.scales,<br>layers.26.mlp.down_proj.weight,<br>layers.26.mlp.gate_proj.biases,<br>layers.26.mlp.gate_proj.scales,<br>layers.26.mlp.gate_proj.weight,<br>layers.26.mlp.up_proj.biases,<br>layers.26.mlp.up_proj.scales,<br>layers.26.mlp.up_proj.weight,<br>layers.26.post_attention_layernorm.weight,<br>layers.26.post_feedforward_layernorm.weight,<br>layers.26.pre_feedforward_layernorm.weight,<br>layers.26.self_attn.gate_proj.biases,<br>layers.26.self_attn.gate_proj.scales,<br>layers.26.self_attn.gate_proj.weight,<br>layers.26.self_attn.k_proj.biases,<br>layers.26.self_attn.k_proj.scales,<br>layers.26.self_attn.k_proj.weight,<br>layers.26.self_attn.o_proj.biases,<br>layers.26.self_attn.o_proj.scales,<br>layers.26.self_attn.o_proj.weight,<br>layers.26.self_attn.q_proj.biases,<br>layers.26.self_attn.q_proj.scales,<br>layers.26.self_attn.q_proj.weight,<br>layers.26.self_attn.v_proj.biases,<br>layers.26.self_attn.v_proj.scales,<br>layers.26.self_attn.v_proj.weight,<br>layers.27.input_layernorm.weight,<br>layers.27.mlp.down_proj.biases,<br>layers.27.mlp.down_proj.scales,<br>layers.27.mlp.down_proj.weight,<br>layers.27.mlp.gate_proj.biases,<br>layers.27.mlp.gate_proj.scales,<br>layers.27.mlp.gate_proj.weight,<br>layers.27.mlp.up_proj.biases,<br>layers.27.mlp.up_proj.scales,<br>layers.27.mlp.up_proj.weight,<br>layers.27.post_attention_layernorm.weight,<br>layers.27.post_feedforward_layernorm.weight,<br>layers.27.pre_feedforward_layernorm.weight,<br>layers.27.self_attn.gate_proj.biases,<br>layers.27.self_attn.gate_proj.scales,<br>layers.27.self_attn.gate_proj.weight,<br>layers.27.self_attn.k_proj.biases,<br>layers.27.self_attn.k_proj.scales,<br>layers.27.self_attn.k_proj.weight,<br>layers.27.self_attn.o_proj.biases,<br>layers.27.self_attn.o_proj.scales,<br>layers.27.self_attn.o_proj.weight,<br>layers.27.self_attn.q_proj.biases,<br>layers.27.self_attn.q_proj.scales,<br>layers.27.self_attn.q_proj.weight,<br>layers.27.self_attn.v_proj.biases,<br>layers.27.self_attn.v_proj.scales,<br>layers.27.self_attn.v_proj.weight,<br>layers.28.input_layernorm.weight,<br>layers.28.mlp.down_proj.biases,<br>layers.28.mlp.down_proj.scales,<br>layers.28.mlp.down_proj.weight,<br>layers.28.mlp.gate_proj.biases,<br>layers.28.mlp.gate_proj.scales,<br>layers.28.mlp.gate_proj.weight,<br>layers.28.mlp.up_proj.biases,<br>layers.28.mlp.up_proj.scales,<br>layers.28.mlp.up_proj.weight,<br>layers.28.post_attention_layernorm.weight,<br>layers.28.post_feedforward_layernorm.weight,<br>layers.28.pre_feedforward_layernorm.weight,<br>layers.28.self_attn.gate_proj.biases,<br>layers.28.self_attn.gate_proj.scales,<br>layers.28.self_attn.gate_proj.weight,<br>layers.28.self_attn.k_proj.biases,<br>layers.28.self_attn.k_proj.scales,<br>layers.28.self_attn.k_proj.weight,<br>layers.28.self_attn.o_proj.biases,<br>layers.28.self_attn.o_proj.scales,<br>layers.28.self_attn.o_proj.weight,<br>layers.28.self_attn.q_proj.biases,<br>layers.28.self_attn.q_proj.scales,<br>layers.28.self_attn.q_proj.weight,<br>layers.28.self_attn.v_proj.biases,<br>layers.28.self_attn.v_proj.scales,<br>layers.28.self_attn.v_proj.weight,<br>layers.29.input_layernorm.weight,<br>layers.29.mlp.down_proj.biases,<br>layers.29.mlp.down_proj.scales,<br>layers.29.mlp.down_proj.weight,<br>layers.29.mlp.gate_proj.biases,<br>layers.29.mlp.gate_proj.scales,<br>layers.29.mlp.gate_proj.weight,<br>layers.29.mlp.up_proj.biases,<br>layers.29.mlp.up_proj.scales,<br>layers.29.mlp.up_proj.weight,<br>layers.29.post_attention_layernorm.weight,<br>layers.29.post_feedforward_layernorm.weight,<br>layers.29.pre_feedforward_layernorm.weight,<br>layers.29.self_attn.gate_proj.biases,<br>layers.29.self_attn.gate_proj.scales,<br>layers.29.self_attn.gate_proj.weight,<br>layers.29.self_attn.k_proj.biases,<br>layers.29.self_attn.k_proj.scales,<br>layers.29.self_attn.k_proj.weight,<br>layers.29.self_attn.o_proj.biases,<br>layers.29.self_attn.o_proj.scales,<br>layers.29.self_attn.o_proj.weight,<br>layers.29.self_attn.q_proj.biases,<br>layers.29.self_attn.q_proj.scales,<br>layers.29.self_attn.q_proj.weight,<br>layers.29.self_attn.v_proj.biases,<br>layers.29.self_attn.v_proj.scales,<br>layers.29.self_attn.v_proj.weight,<br>layers.3.input_layernorm.weight,<br>layers.3.mlp.down_proj.biases,<br>layers.3.mlp.down_proj.scales,<br>layers.3.mlp.down_proj.weight,<br>layers.3.mlp.gate_proj.biases,<br>layers.3.mlp.gate_proj.scales,<br>layers.3.mlp.gate_proj.weight,<br>layers.3.mlp.up_proj.biases,<br>layers.3.mlp.up_proj.scales,<br>layers.3.mlp.up_proj.weight,<br>layers.3.post_attention_layernorm.weight,<br>layers.3.post_feedforward_layernorm.weight,<br>layers.3.pre_feedforward_layernorm.weight,<br>layers.3.self_attn.gate_proj.biases,<br>layers.3.self_attn.gate_proj.scales,<br>layers.3.self_attn.gate_proj.weight,<br>layers.3.self_attn.k_proj.biases,<br>layers.3.self_attn.k_proj.scales,<br>layers.3.self_attn.k_proj.weight,<br>layers.3.self_attn.o_proj.biases,<br>layers.3.self_attn.o_proj.scales,<br>layers.3.self_attn.o_proj.weight,<br>layers.3.self_attn.q_proj.biases,<br>layers.3.self_attn.q_proj.scales,<br>layers.3.self_attn.q_proj.weight,<br>layers.3.self_attn.v_proj.biases,<br>layers.3.self_attn.v_proj.scales,<br>layers.3.self_attn.v_proj.weight,<br>layers.30.input_layernorm.weight,<br>layers.30.mlp.down_proj.biases,<br>layers.30.mlp.down_proj.scales,<br>layers.30.mlp.down_proj.weight,<br>layers.30.mlp.gate_proj.biases,<br>layers.30.mlp.gate_proj.scales,<br>layers.30.mlp.gate_proj.weight,<br>layers.30.mlp.up_proj.biases,<br>layers.30.mlp.up_proj.scales,<br>layers.30.mlp.up_proj.weight,<br>layers.30.post_attention_layernorm.weight,<br>layers.30.post_feedforward_layernorm.weight,<br>layers.30.pre_feedforward_layernorm.weight,<br>layers.30.self_attn.gate_proj.biases,<br>layers.30.self_attn.gate_proj.scales,<br>layers.30.self_attn.gate_proj.weight,<br>layers.30.self_attn.k_proj.biases,<br>layers.30.self_attn.k_proj.scales,<br>layers.30.self_attn.k_proj.weight,<br>layers.30.self_attn.o_proj.biases,<br>layers.30.self_attn.o_proj.scales,<br>layers.30.self_attn.o_proj.weight,<br>layers.30.self_attn.q_proj.biases,<br>layers.30.self_attn.q_proj.scales,<br>layers.30.self_attn.q_proj.weight,<br>layers.30.self_attn.v_proj.biases,<br>layers.30.self_attn.v_proj.scales,<br>layers.30.self_attn.v_proj.weight,<br>layers.31.input_layernorm.weight,<br>layers.31.mlp.down_proj.biases,<br>layers.31.mlp.down_proj.scales,<br>layers.31.mlp.down_proj.weight,<br>layers.31.mlp.gate_proj.biases,<br>layers.31.mlp.gate_proj.scales,<br>layers.31.mlp.gate_proj.weight,<br>layers.31.mlp.up_proj.biases,<br>layers.31.mlp.up_proj.scales,<br>layers.31.mlp.up_proj.weight,<br>layers.31.post_attention_layernorm.weight,<br>layers.31.post_feedforward_layernorm.weight,<br>layers.31.pre_feedforward_layernorm.weight,<br>layers.31.self_attn.gate_proj.biases,<br>layers.31.self_attn.gate_proj.scales,<br>layers.31.self_attn.gate_proj.weight,<br>layers.31.self_attn.k_proj.biases,<br>layers.31.self_attn.k_proj.scales,<br>layers.31.self_attn.k_proj.weight,<br>layers.31.self_attn.o_proj.biases,<br>layers.31.self_attn.o_proj.scales,<br>layers.31.self_attn.o_proj.weight,<br>layers.31.self_attn.q_proj.biases,<br>layers.31.self_attn.q_proj.scales,<br>layers.31.self_attn.q_proj.weight,<br>layers.31.self_attn.v_proj.biases,<br>layers.31.self_attn.v_proj.scales,<br>layers.31.self_attn.v_proj.weight,<br>layers.32.input_layernorm.weight,<br>layers.32.mlp.down_proj.biases,<br>layers.32.mlp.down_proj.scales,<br>layers.32.mlp.down_proj.weight,<br>layers.32.mlp.gate_proj.biases,<br>layers.32.mlp.gate_proj.scales,<br>layers.32.mlp.gate_proj.weight,<br>layers.32.mlp.up_proj.biases,<br>layers.32.mlp.up_proj.scales,<br>layers.32.mlp.up_proj.weight,<br>layers.32.post_attention_layernorm.weight,<br>layers.32.post_feedforward_layernorm.weight,<br>layers.32.pre_feedforward_layernorm.weight,<br>layers.32.self_attn.gate_proj.biases,<br>layers.32.self_attn.gate_proj.scales,<br>layers.32.self_attn.gate_proj.weight,<br>layers.32.self_attn.k_proj.biases,<br>layers.32.self_attn.k_proj.scales,<br>layers.32.self_attn.k_proj.weight,<br>layers.32.self_attn.o_proj.biases,<br>layers.32.self_attn.o_proj.scales,<br>layers.32.self_attn.o_proj.weight,<br>layers.32.self_attn.q_proj.biases,<br>layers.32.self_attn.q_proj.scales,<br>layers.32.self_attn.q_proj.weight,<br>layers.32.self_attn.v_proj.biases,<br>layers.32.self_attn.v_proj.scales,<br>layers.32.self_attn.v_proj.weight,<br>layers.33.input_layernorm.weight,<br>layers.33.mlp.down_proj.biases,<br>layers.33.mlp.down_proj.scales,<br>layers.33.mlp.down_proj.weight,<br>layers.33.mlp.gate_proj.biases,<br>layers.33.mlp.gate_proj.scales,<br>layers.33.mlp.gate_proj.weight,<br>layers.33.mlp.up_proj.biases,<br>layers.33.mlp.up_proj.scales,<br>layers.33.mlp.up_proj.weight,<br>layers.33.post_attention_layernorm.weight,<br>layers.33.post_feedforward_layernorm.weight,<br>layers.33.pre_feedforward_layernorm.weight,<br>layers.33.self_attn.gate_proj.biases,<br>layers.33.self_attn.gate_proj.scales,<br>layers.33.self_attn.gate_proj.weight,<br>layers.33.self_attn.k_proj.biases,<br>layers.33.self_attn.k_proj.scales,<br>layers.33.self_attn.k_proj.weight,<br>layers.33.self_attn.o_proj.biases,<br>layers.33.self_attn.o_proj.scales,<br>layers.33.self_attn.o_proj.weight,<br>layers.33.self_attn.q_proj.biases,<br>layers.33.self_attn.q_proj.scales,<br>layers.33.self_attn.q_proj.weight,<br>layers.33.self_attn.v_proj.biases,<br>layers.33.self_attn.v_proj.scales,<br>layers.33.self_attn.v_proj.weight,<br>layers.34.input_layernorm.weight,<br>layers.34.mlp.down_proj.biases,<br>layers.34.mlp.down_proj.scales,<br>layers.34.mlp.down_proj.weight,<br>layers.34.mlp.gate_proj.biases,<br>layers.34.mlp.gate_proj.scales,<br>layers.34.mlp.gate_proj.weight,<br>layers.34.mlp.up_proj.biases,<br>layers.34.mlp.up_proj.scales,<br>layers.34.mlp.up_proj.weight,<br>layers.34.post_attention_layernorm.weight,<br>layers.34.post_feedforward_layernorm.weight,<br>layers.34.pre_feedforward_layernorm.weight,<br>layers.34.self_attn.gate_proj.biases,<br>layers.34.self_attn.gate_proj.scales,<br>layers.34.self_attn.gate_proj.weight,<br>layers.34.self_attn.k_proj.biases,<br>layers.34.self_attn.k_proj.scales,<br>layers.34.self_attn.k_proj.weight,<br>layers.34.self_attn.o_proj.biases,<br>layers.34.self_attn.o_proj.scales,<br>layers.34.self_attn.o_proj.weight,<br>layers.34.self_attn.q_proj.biases,<br>layers.34.self_attn.q_proj.scales,<br>layers.34.self_attn.q_proj.weight,<br>layers.34.self_attn.v_proj.biases,<br>layers.34.self_attn.v_proj.scales,<br>layers.34.self_attn.v_proj.weight,<br>layers.35.input_layernorm.weight,<br>layers.35.mlp.down_proj.biases,<br>layers.35.mlp.down_proj.scales,<br>layers.35.mlp.down_proj.weight,<br>layers.35.mlp.gate_proj.biases,<br>layers.35.mlp.gate_proj.scales,<br>layers.35.mlp.gate_proj.weight,<br>layers.35.mlp.up_proj.biases,<br>layers.35.mlp.up_proj.scales,<br>layers.35.mlp.up_proj.weight,<br>layers.35.post_attention_layernorm.weight,<br>layers.35.post_feedforward_layernorm.weight,<br>layers.35.pre_feedforward_layernorm.weight,<br>layers.35.self_attn.gate_proj.biases,<br>layers.35.self_attn.gate_proj.scales,<br>layers.35.self_attn.gate_proj.weight,<br>layers.35.self_attn.k_proj.biases,<br>layers.35.self_attn.k_proj.scales,<br>layers.35.self_attn.k_proj.weight,<br>layers.35.self_attn.o_proj.biases,<br>layers.35.self_attn.o_proj.scales,<br>layers.35.self_attn.o_proj.weight,<br>layers.35.self_attn.q_proj.biases,<br>layers.35.self_attn.q_proj.scales,<br>layers.35.self_attn.q_proj.weight,<br>layers.35.self_attn.v_proj.biases,<br>layers.35.self_attn.v_proj.scales,<br>layers.35.self_attn.v_proj.weight,<br>layers.36.input_layernorm.weight,<br>layers.36.mlp.down_proj.biases,<br>layers.36.mlp.down_proj.scales,<br>layers.36.mlp.down_proj.weight,<br>layers.36.mlp.gate_proj.biases,<br>layers.36.mlp.gate_proj.scales,<br>layers.36.mlp.gate_proj.weight,<br>layers.36.mlp.up_proj.biases,<br>layers.36.mlp.up_proj.scales,<br>layers.36.mlp.up_proj.weight,<br>layers.36.post_attention_layernorm.weight,<br>layers.36.post_feedforward_layernorm.weight,<br>layers.36.pre_feedforward_layernorm.weight,<br>layers.36.self_attn.gate_proj.biases,<br>layers.36.self_attn.gate_proj.scales,<br>layers.36.self_attn.gate_proj.weight,<br>layers.36.self_attn.k_proj.biases,<br>layers.36.self_attn.k_proj.scales,<br>layers.36.self_attn.k_proj.weight,<br>layers.36.self_attn.o_proj.biases,<br>layers.36.self_attn.o_proj.scales,<br>layers.36.self_attn.o_proj.weight,<br>layers.36.self_attn.q_proj.biases,<br>layers.36.self_attn.q_proj.scales,<br>layers.36.self_attn.q_proj.weight,<br>layers.36.self_attn.v_proj.biases,<br>layers.36.self_attn.v_proj.scales,<br>layers.36.self_attn.v_proj.weight,<br>layers.37.input_layernorm.weight,<br>layers.37.mlp.down_proj.biases,<br>layers.37.mlp.down_proj.scales,<br>layers.37.mlp.down_proj.weight,<br>layers.37.mlp.gate_proj.biases,<br>layers.37.mlp.gate_proj.scales,<br>layers.37.mlp.gate_proj.weight,<br>layers.37.mlp.up_proj.biases,<br>layers.37.mlp.up_proj.scales,<br>layers.37.mlp.up_proj.weight,<br>layers.37.post_attention_layernorm.weight,<br>layers.37.post_feedforward_layernorm.weight,<br>layers.37.pre_feedforward_layernorm.weight,<br>layers.37.self_attn.gate_proj.biases,<br>layers.37.self_attn.gate_proj.scales,<br>layers.37.self_attn.gate_proj.weight,<br>layers.37.self_attn.k_proj.biases,<br>layers.37.self_attn.k_proj.scales,<br>layers.37.self_attn.k_proj.weight,<br>layers.37.self_attn.o_proj.biases,<br>layers.37.self_attn.o_proj.scales,<br>layers.37.self_attn.o_proj.weight,<br>layers.37.self_attn.q_proj.biases,<br>layers.37.self_attn.q_proj.scales,<br>layers.37.self_attn.q_proj.weight,<br>layers.37.self_attn.v_proj.biases,<br>layers.37.self_attn.v_proj.scales,<br>layers.37.self_attn.v_proj.weight,<br>layers.38.input_layernorm.weight,<br>layers.38.mlp.down_proj.biases,<br>layers.38.mlp.down_proj.scales,<br>layers.38.mlp.down_proj.weight,<br>layers.38.mlp.gate_proj.biases,<br>layers.38.mlp.gate_proj.scales,<br>layers.38.mlp.gate_proj.weight,<br>layers.38.mlp.up_proj.biases,<br>layers.38.mlp.up_proj.scales,<br>layers.38.mlp.up_proj.weight,<br>layers.38.post_attention_layernorm.weight,<br>layers.38.post_feedforward_layernorm.weight,<br>layers.38.pre_feedforward_layernorm.weight,<br>layers.38.self_attn.gate_proj.biases,<br>layers.38.self_attn.gate_proj.scales,<br>layers.38.self_attn.gate_proj.weight,<br>layers.38.self_attn.k_proj.biases,<br>layers.38.self_attn.k_proj.scales,<br>layers.38.self_attn.k_proj.weight,<br>layers.38.self_attn.o_proj.biases,<br>layers.38.self_attn.o_proj.scales,<br>layers.38.self_attn.o_proj.weight,<br>layers.38.self_attn.q_proj.biases,<br>layers.38.self_attn.q_proj.scales,<br>layers.38.self_attn.q_proj.weight,<br>layers.38.self_attn.v_proj.biases,<br>layers.38.self_attn.v_proj.scales,<br>layers.38.self_attn.v_proj.weight,<br>layers.39.input_layernorm.weight,<br>layers.39.mlp.down_proj.biases,<br>layers.39.mlp.down_proj.scales,<br>layers.39.mlp.down_proj.weight,<br>layers.39.mlp.gate_proj.biases,<br>layers.39.mlp.gate_proj.scales,<br>layers.39.mlp.gate_proj.weight,<br>layers.39.mlp.up_proj.biases,<br>layers.39.mlp.up_proj.scales,<br>layers.39.mlp.up_proj.weight,<br>layers.39.post_attention_layernorm.weight,<br>layers.39.post_feedforward_layernorm.weight,<br>layers.39.pre_feedforward_layernorm.weight,<br>layers.39.self_attn.gate_proj.biases,<br>layers.39.self_attn.gate_proj.scales,<br>layers.39.self_attn.gate_proj.weight,<br>layers.39.self_attn.k_proj.biases,<br>layers.39.self_attn.k_proj.scales,<br>layers.39.self_attn.k_proj.weight,<br>layers.39.self_attn.o_proj.biases,<br>layers.39.self_attn.o_proj.scales,<br>layers.39.self_attn.o_proj.weight,<br>layers.39.self_attn.q_proj.biases,<br>layers.39.self_attn.q_proj.scales,<br>layers.39.self_attn.q_proj.weight,<br>layers.39.self_attn.v_proj.biases,<br>layers.39.self_attn.v_proj.scales,<br>layers.39.self_attn.v_proj.weight,<br>layers.4.input_layernorm.weight,<br>layers.4.mlp.down_proj.biases,<br>layers.4.mlp.down_proj.scales,<br>layers.4.mlp.down_proj.weight,<br>layers.4.mlp.gate_proj.biases,<br>layers.4.mlp.gate_proj.scales,<br>layers.4.mlp.gate_proj.weight,<br>layers.4.mlp.up_proj.biases,<br>layers.4.mlp.up_proj.scales,<br>layers.4.mlp.up_proj.weight,<br>layers.4.post_attention_layernorm.weight,<br>layers.4.post_feedforward_layernorm.weight,<br>layers.4.pre_feedforward_layernorm.weight,<br>layers.4.self_attn.gate_proj.biases,<br>layers.4.self_attn.gate_proj.scales,<br>layers.4.self_attn.gate_proj.weight,<br>layers.4.self_attn.k_proj.biases,<br>layers.4.self_attn.k_proj.scales,<br>layers.4.self_attn.k_proj.weight,<br>layers.4.self_attn.o_proj.biases,<br>layers.4.self_attn.o_proj.scales,<br>layers.4.self_attn.o_proj.weight,<br>layers.4.self_attn.q_proj.biases,<br>layers.4.self_attn.q_proj.scales,<br>layers.4.self_attn.q_proj.weight,<br>layers.4.self_attn.v_proj.biases,<br>layers.4.self_attn.v_proj.scales,<br>layers.4.self_attn.v_proj.weight,<br>layers.40.input_layernorm.weight,<br>layers.40.mlp.down_proj.biases,<br>layers.40.mlp.down_proj.scales,<br>layers.40.mlp.down_proj.weight,<br>layers.40.mlp.gate_proj.biases,<br>layers.40.mlp.gate_proj.scales,<br>layers.40.mlp.gate_proj.weight,<br>layers.40.mlp.up_proj.biases,<br>layers.40.mlp.up_proj.scales,<br>layers.40.mlp.up_proj.weight,<br>layers.40.post_attention_layernorm.weight,<br>layers.40.post_feedforward_layernorm.weight,<br>layers.40.pre_feedforward_layernorm.weight,<br>layers.40.self_attn.gate_proj.biases,<br>layers.40.self_attn.gate_proj.scales,<br>layers.40.self_attn.gate_proj.weight,<br>layers.40.self_attn.k_proj.biases,<br>layers.40.self_attn.k_proj.scales,<br>layers.40.self_attn.k_proj.weight,<br>layers.40.self_attn.o_proj.biases,<br>layers.40.self_attn.o_proj.scales,<br>layers.40.self_attn.o_proj.weight,<br>layers.40.self_attn.q_proj.biases,<br>layers.40.self_attn.q_proj.scales,<br>layers.40.self_attn.q_proj.weight,<br>layers.40.self_attn.v_proj.biases,<br>layers.40.self_attn.v_proj.scales,<br>layers.40.self_attn.v_proj.weight,<br>layers.41.input_layernorm.weight,<br>layers.41.mlp.down_proj.biases,<br>layers.41.mlp.down_proj.scales,<br>layers.41.mlp.down_proj.weight,<br>layers.41.mlp.gate_proj.biases,<br>layers.41.mlp.gate_proj.scales,<br>layers.41.mlp.gate_proj.weight,<br>layers.41.mlp.up_proj.biases,<br>layers.41.mlp.up_proj.scales,<br>layers.41.mlp.up_proj.weight,<br>layers.41.post_attention_layernorm.weight,<br>layers.41.post_feedforward_layernorm.weight,<br>layers.41.pre_feedforward_layernorm.weight,<br>layers.41.self_attn.gate_proj.biases,<br>layers.41.self_attn.gate_proj.scales,<br>layers.41.self_attn.gate_proj.weight,<br>layers.41.self_attn.k_proj.biases,<br>layers.41.self_attn.k_proj.scales,<br>layers.41.self_attn.k_proj.weight,<br>layers.41.self_attn.o_proj.biases,<br>layers.41.self_attn.o_proj.scales,<br>layers.41.self_attn.o_proj.weight,<br>layers.41.self_attn.q_proj.biases,<br>layers.41.self_attn.q_proj.scales,<br>layers.41.self_attn.q_proj.weight,<br>layers.41.self_attn.v_proj.biases,<br>layers.41.self_attn.v_proj.scales,<br>layers.41.self_attn.v_proj.weight,<br>layers.42.input_layernorm.weight,<br>layers.42.mlp.down_proj.biases,<br>layers.42.mlp.down_proj.scales,<br>layers.42.mlp.down_proj.weight,<br>layers.42.mlp.gate_proj.biases,<br>layers.42.mlp.gate_proj.scales,<br>layers.42.mlp.gate_proj.weight,<br>layers.42.mlp.up_proj.biases,<br>layers.42.mlp.up_proj.scales,<br>layers.42.mlp.up_proj.weight,<br>layers.42.post_attention_layernorm.weight,<br>layers.42.post_feedforward_layernorm.weight,<br>layers.42.pre_feedforward_layernorm.weight,<br>layers.42.self_attn.gate_proj.biases,<br>layers.42.self_attn.gate_proj.scales,<br>layers.42.self_attn.gate_proj.weight,<br>layers.42.self_attn.k_proj.biases,<br>layers.42.self_attn.k_proj.scales,<br>layers.42.self_attn.k_proj.weight,<br>layers.42.self_attn.o_proj.biases,<br>layers.42.self_attn.o_proj.scales,<br>layers.42.self_attn.o_proj.weight,<br>layers.42.self_attn.q_proj.biases,<br>layers.42.self_attn.q_proj.scales,<br>layers.42.self_attn.q_proj.weight,<br>layers.42.self_attn.v_proj.biases,<br>layers.42.self_attn.v_proj.scales,<br>layers.42.self_attn.v_proj.weight,<br>layers.43.input_layernorm.weight,<br>layers.43.mlp.down_proj.biases,<br>layers.43.mlp.down_proj.scales,<br>layers.43.mlp.down_proj.weight,<br>layers.43.mlp.gate_proj.biases,<br>layers.43.mlp.gate_proj.scales,<br>layers.43.mlp.gate_proj.weight,<br>layers.43.mlp.up_proj.biases,<br>layers.43.mlp.up_proj.scales,<br>layers.43.mlp.up_proj.weight,<br>layers.43.post_attention_layernorm.weight,<br>layers.43.post_feedforward_layernorm.weight,<br>layers.43.pre_feedforward_layernorm.weight,<br>layers.43.self_attn.gate_proj.biases,<br>layers.43.self_attn.gate_proj.scales,<br>layers.43.self_attn.gate_proj.weight,<br>layers.43.self_attn.k_proj.biases,<br>layers.43.self_attn.k_proj.scales,<br>layers.43.self_attn.k_proj.weight,<br>layers.43.self_attn.o_proj.biases,<br>layers.43.self_attn.o_proj.scales,<br>layers.43.self_attn.o_proj.weight,<br>layers.43.self_attn.q_proj.biases,<br>layers.43.self_attn.q_proj.scales,<br>layers.43.self_attn.q_proj.weight,<br>layers.43.self_attn.v_proj.biases,<br>layers.43.self_attn.v_proj.scales,<br>layers.43.self_attn.v_proj.weight,<br>layers.44.input_layernorm.weight,<br>layers.44.mlp.down_proj.biases,<br>layers.44.mlp.down_proj.scales,<br>layers.44.mlp.down_proj.weight,<br>layers.44.mlp.gate_proj.biases,<br>layers.44.mlp.gate_proj.scales,<br>layers.44.mlp.gate_proj.weight,<br>layers.44.mlp.up_proj.biases,<br>layers.44.mlp.up_proj.scales,<br>layers.44.mlp.up_proj.weight,<br>layers.44.post_attention_layernorm.weight,<br>layers.44.post_feedforward_layernorm.weight,<br>layers.44.pre_feedforward_layernorm.weight,<br>layers.44.self_attn.gate_proj.biases,<br>layers.44.self_attn.gate_proj.scales,<br>layers.44.self_attn.gate_proj.weight,<br>layers.44.self_attn.k_proj.biases,<br>layers.44.self_attn.k_proj.scales,<br>layers.44.self_attn.k_proj.weight,<br>layers.44.self_attn.o_proj.biases,<br>layers.44.self_attn.o_proj.scales,<br>layers.44.self_attn.o_proj.weight,<br>layers.44.self_attn.q_proj.biases,<br>layers.44.self_attn.q_proj.scales,<br>layers.44.self_attn.q_proj.weight,<br>layers.44.self_attn.v_proj.biases,<br>layers.44.self_attn.v_proj.scales,<br>layers.44.self_attn.v_proj.weight,<br>layers.45.input_layernorm.weight,<br>layers.45.mlp.down_proj.biases,<br>layers.45.mlp.down_proj.scales,<br>layers.45.mlp.down_proj.weight,<br>layers.45.mlp.gate_proj.biases,<br>layers.45.mlp.gate_proj.scales,<br>layers.45.mlp.gate_proj.weight,<br>layers.45.mlp.up_proj.biases,<br>layers.45.mlp.up_proj.scales,<br>layers.45.mlp.up_proj.weight,<br>layers.45.post_attention_layernorm.weight,<br>layers.45.post_feedforward_layernorm.weight,<br>layers.45.pre_feedforward_layernorm.weight,<br>layers.45.self_attn.gate_proj.biases,<br>layers.45.self_attn.gate_proj.scales,<br>layers.45.self_attn.gate_proj.weight,<br>layers.45.self_attn.k_proj.biases,<br>layers.45.self_attn.k_proj.scales,<br>layers.45.self_attn.k_proj.weight,<br>layers.45.self_attn.o_proj.biases,<br>layers.45.self_attn.o_proj.scales,<br>layers.45.self_attn.o_proj.weight,<br>layers.45.self_attn.q_proj.biases,<br>layers.45.self_attn.q_proj.scales,<br>layers.45.self_attn.q_proj.weight,<br>layers.45.self_attn.v_proj.biases,<br>layers.45.self_attn.v_proj.scales,<br>layers.45.self_attn.v_proj.weight,<br>layers.46.input_layernorm.weight,<br>layers.46.mlp.down_proj.biases,<br>layers.46.mlp.down_proj.scales,<br>layers.46.mlp.down_proj.weight,<br>layers.46.mlp.gate_proj.biases,<br>layers.46.mlp.gate_proj.scales,<br>layers.46.mlp.gate_proj.weight,<br>layers.46.mlp.up_proj.biases,<br>layers.46.mlp.up_proj.scales,<br>layers.46.mlp.up_proj.weight,<br>layers.46.post_attention_layernorm.weight,<br>layers.46.post_feedforward_layernorm.weight,<br>layers.46.pre_feedforward_layernorm.weight,<br>layers.46.self_attn.gate_proj.biases,<br>layers.46.self_attn.gate_proj.scales,<br>layers.46.self_attn.gate_proj.weight,<br>layers.46.self_attn.k_proj.biases,<br>layers.46.self_attn.k_proj.scales,<br>layers.46.self_attn.k_proj.weight,<br>layers.46.self_attn.o_proj.biases,<br>layers.46.self_attn.o_proj.scales,<br>layers.46.self_attn.o_proj.weight,<br>layers.46.self_attn.q_proj.biases,<br>layers.46.self_attn.q_proj.scales,<br>layers.46.self_attn.q_proj.weight,<br>layers.46.self_attn.v_proj.biases,<br>layers.46.self_attn.v_proj.scales,<br>layers.46.self_attn.v_proj.weight,<br>layers.47.input_layernorm.weight,<br>layers.47.mlp.down_proj.biases,<br>layers.47.mlp.down_proj.scales,<br>layers.47.mlp.down_proj.weight,<br>layers.47.mlp.gate_proj.biases,<br>layers.47.mlp.gate_proj.scales,<br>layers.47.mlp.gate_proj.weight,<br>layers.47.mlp.up_proj.biases,<br>layers.47.mlp.up_proj.scales,<br>layers.47.mlp.up_proj.weight,<br>layers.47.post_attention_layernorm.weight,<br>layers.47.post_feedforward_layernorm.weight,<br>layers.47.pre_feedforward_layernorm.weight,<br>layers.47.self_attn.gate_proj.biases,<br>layers.47.self_attn.gate_proj.scales,<br>layers.47.self_attn.gate_proj.weight,<br>layers.47.self_attn.k_proj.biases,<br>layers.47.self_attn.k_proj.scales,<br>layers.47.self_attn.k_proj.weight,<br>layers.47.self_attn.o_proj.biases,<br>layers.47.self_attn.o_proj.scales,<br>layers.47.self_attn.o_proj.weight,<br>layers.47.self_attn.q_proj.biases,<br>layers.47.self_attn.q_proj.scales,<br>layers.47.self_attn.q_proj.weight,<br>layers.47.self_attn.v_proj.biases,<br>layers.47.self_attn.v_proj.scales,<br>layers.47.self_attn.v_proj.weight,<br>layers.48.input_layernorm.weight,<br>layers.48.mlp.down_proj.biases,<br>layers.48.mlp.down_proj.scales,<br>layers.48.mlp.down_proj.weight,<br>layers.48.mlp.gate_proj.biases,<br>layers.48.mlp.gate_proj.scales,<br>layers.48.mlp.gate_proj.weight,<br>layers.48.mlp.up_proj.biases,<br>layers.48.mlp.up_proj.scales,<br>layers.48.mlp.up_proj.weight,<br>layers.48.post_attention_layernorm.weight,<br>layers.48.post_feedforward_layernorm.weight,<br>layers.48.pre_feedforward_layernorm.weight,<br>layers.48.self_attn.gate_proj.biases,<br>layers.48.self_attn.gate_proj.scales,<br>layers.48.self_attn.gate_proj.weight,<br>layers.48.self_attn.k_proj.biases,<br>layers.48.self_attn.k_proj.scales,<br>layers.48.self_attn.k_proj.weight,<br>layers.48.self_attn.o_proj.biases,<br>layers.48.self_attn.o_proj.scales,<br>layers.48.self_attn.o_proj.weight,<br>layers.48.self_attn.q_proj.biases,<br>layers.48.self_attn.q_proj.scales,<br>layers.48.self_attn.q_proj.weight,<br>layers.48.self_attn.v_proj.biases,<br>layers.48.self_attn.v_proj.scales,<br>layers.48.self_attn.v_proj.weight,<br>layers.49.input_layernorm.weight,<br>layers.49.mlp.down_proj.biases,<br>layers.49.mlp.down_proj.scales,<br>layers.49.mlp.down_proj.weight,<br>layers.49.mlp.gate_proj.biases,<br>layers.49.mlp.gate_proj.scales,<br>layers.49.mlp.gate_proj.weight,<br>layers.49.mlp.up_proj.biases,<br>layers.49.mlp.up_proj.scales,<br>layers.49.mlp.up_proj.weight,<br>layers.49.post_attention_layernorm.weight,<br>layers.49.post_feedforward_layernorm.weight,<br>layers.49.pre_feedforward_layernorm.weight,<br>layers.49.self_attn.gate_proj.biases,<br>layers.49.self_attn.gate_proj.scales,<br>layers.49.self_attn.gate_proj.weight,<br>layers.49.self_attn.k_proj.biases,<br>layers.49.self_attn.k_proj.scales,<br>layers.49.self_attn.k_proj.weight,<br>layers.49.self_attn.o_proj.biases,<br>layers.49.self_attn.o_proj.scales,<br>layers.49.self_attn.o_proj.weight,<br>layers.49.self_attn.q_proj.biases,<br>layers.49.self_attn.q_proj.scales,<br>layers.49.self_attn.q_proj.weight,<br>layers.49.self_attn.v_proj.biases,<br>layers.49.self_attn.v_proj.scales,<br>layers.49.self_attn.v_proj.weight,<br>layers.5.input_layernorm.weight,<br>layers.5.mlp.down_proj.biases,<br>layers.5.mlp.down_proj.scales,<br>layers.5.mlp.down_proj.weight,<br>layers.5.mlp.gate_proj.biases,<br>layers.5.mlp.gate_proj.scales,<br>layers.5.mlp.gate_proj.weight,<br>layers.5.mlp.up_proj.biases,<br>layers.5.mlp.up_proj.scales,<br>layers.5.mlp.up_proj.weight,<br>layers.5.post_attention_layernorm.weight,<br>layers.5.post_feedforward_layernorm.weight,<br>layers.5.pre_feedforward_layernorm.weight,<br>layers.5.self_attn.gate_proj.biases,<br>layers.5.self_attn.gate_proj.scales,<br>layers.5.self_attn.gate_proj.weight,<br>layers.5.self_attn.k_proj.biases,<br>layers.5.self_attn.k_proj.scales,<br>layers.5.self_attn.k_proj.weight,<br>layers.5.self_attn.o_proj.biases,<br>layers.5.self_attn.o_proj.scales,<br>layers.5.self_attn.o_proj.weight,<br>layers.5.self_attn.q_proj.biases,<br>layers.5.self_attn.q_proj.scales,<br>layers.5.self_attn.q_proj.weight,<br>layers.5.self_attn.v_proj.biases,<br>layers.5.self_attn.v_proj.scales,<br>layers.5.self_attn.v_proj.weight,<br>layers.50.input_layernorm.weight,<br>layers.50.mlp.down_proj.biases,<br>layers.50.mlp.down_proj.scales,<br>layers.50.mlp.down_proj.weight,<br>layers.50.mlp.gate_proj.biases,<br>layers.50.mlp.gate_proj.scales,<br>layers.50.mlp.gate_proj.weight,<br>layers.50.mlp.up_proj.biases,<br>layers.50.mlp.up_proj.scales,<br>layers.50.mlp.up_proj.weight,<br>layers.50.post_attention_layernorm.weight,<br>layers.50.post_feedforward_layernorm.weight,<br>layers.50.pre_feedforward_layernorm.weight,<br>layers.50.self_attn.gate_proj.biases,<br>layers.50.self_attn.gate_proj.scales,<br>layers.50.self_attn.gate_proj.weight,<br>layers.50.self_attn.k_proj.biases,<br>layers.50.self_attn.k_proj.scales,<br>layers.50.self_attn.k_proj.weight,<br>layers.50.self_attn.o_proj.biases,<br>layers.50.self_attn.o_proj.scales,<br>layers.50.self_attn.o_proj.weight,<br>layers.50.self_attn.q_proj.biases,<br>layers.50.self_attn.q_proj.scales,<br>layers.50.self_attn.q_proj.weight,<br>layers.50.self_attn.v_proj.biases,<br>layers.50.self_attn.v_proj.scales,<br>layers.50.self_attn.v_proj.weight,<br>layers.51.input_layernorm.weight,<br>layers.51.mlp.down_proj.biases,<br>layers.51.mlp.down_proj.scales,<br>layers.51.mlp.down_proj.weight,<br>layers.51.mlp.gate_proj.biases,<br>layers.51.mlp.gate_proj.scales,<br>layers.51.mlp.gate_proj.weight,<br>layers.51.mlp.up_proj.biases,<br>layers.51.mlp.up_proj.scales,<br>layers.51.mlp.up_proj.weight,<br>layers.51.post_attention_layernorm.weight,<br>layers.51.post_feedforward_layernorm.weight,<br>layers.51.pre_feedforward_layernorm.weight,<br>layers.51.self_attn.gate_proj.biases,<br>layers.51.self_attn.gate_proj.scales,<br>layers.51.self_attn.gate_proj.weight,<br>layers.51.self_attn.k_proj.biases,<br>layers.51.self_attn.k_proj.scales,<br>layers.51.self_attn.k_proj.weight,<br>layers.51.self_attn.o_proj.biases,<br>layers.51.self_attn.o_proj.scales,<br>layers.51.self_attn.o_proj.weight,<br>layers.51.self_attn.q_proj.biases,<br>layers.51.self_attn.q_proj.scales,<br>layers.51.self_attn.q_proj.weight,<br>layers.51.self_attn.v_proj.biases,<br>layers.51.self_attn.v_proj.scales,<br>layers.51.self_attn.v_proj.weight,<br>layers.6.input_layernorm.weight,<br>layers.6.mlp.down_proj.biases,<br>layers.6.mlp.down_proj.scales,<br>layers.6.mlp.down_proj.weight,<br>layers.6.mlp.gate_proj.biases,<br>layers.6.mlp.gate_proj.scales,<br>layers.6.mlp.gate_proj.weight,<br>layers.6.mlp.up_proj.biases,<br>layers.6.mlp.up_proj.scales,<br>layers.6.mlp.up_proj.weight,<br>layers.6.post_attention_layernorm.weight,<br>layers.6.post_feedforward_layernorm.weight,<br>layers.6.pre_feedforward_layernorm.weight,<br>layers.6.self_attn.gate_proj.biases,<br>layers.6.self_attn.gate_proj.scales,<br>layers.6.self_attn.gate_proj.weight,<br>layers.6.self_attn.k_proj.biases,<br>layers.6.self_attn.k_proj.scales,<br>layers.6.self_attn.k_proj.weight,<br>layers.6.self_attn.o_proj.biases,<br>layers.6.self_attn.o_proj.scales,<br>layers.6.self_attn.o_proj.weight,<br>layers.6.self_attn.q_proj.biases,<br>layers.6.self_attn.q_proj.scales,<br>layers.6.self_attn.q_proj.weight,<br>layers.6.self_attn.v_proj.biases,<br>layers.6.self_attn.v_proj.scales,<br>layers.6.self_attn.v_proj.weight,<br>layers.7.input_layernorm.weight,<br>layers.7.mlp.down_proj.biases,<br>layers.7.mlp.down_proj.scales,<br>layers.7.mlp.down_proj.weight,<br>layers.7.mlp.gate_proj.biases,<br>layers.7.mlp.gate_proj.scales,<br>layers.7.mlp.gate_proj.weight,<br>layers.7.mlp.up_proj.biases,<br>layers.7.mlp.up_proj.scales,<br>layers.7.mlp.up_proj.weight,<br>layers.7.post_attention_layernorm.weight,<br>layers.7.post_feedforward_layernorm.weight,<br>layers.7.pre_feedforward_layernorm.weight,<br>layers.7.self_attn.gate_proj.biases,<br>layers.7.self_attn.gate_proj.scales,<br>layers.7.self_attn.gate_proj.weight,<br>layers.7.self_attn.k_proj.biases,<br>layers.7.self_attn.k_proj.scales,<br>layers.7.self_attn.k_proj.weight,<br>layers.7.self_attn.o_proj.biases,<br>layers.7.self_attn.o_proj.scales,<br>layers.7.self_attn.o_proj.weight,<br>layers.7.self_attn.q_proj.biases,<br>layers.7.self_attn.q_proj.scales,<br>layers.7.self_attn.q_proj.weight,<br>layers.7.self_attn.v_proj.biases,<br>layers.7.self_attn.v_proj.scales,<br>layers.7.self_attn.v_proj.weight,<br>layers.8.input_layernorm.weight,<br>layers.8.mlp.down_proj.biases,<br>layers.8.mlp.down_proj.scales,<br>layers.8.mlp.down_proj.weight,<br>layers.8.mlp.gate_proj.biases,<br>layers.8.mlp.gate_proj.scales,<br>layers.8.mlp.gate_proj.weight,<br>layers.8.mlp.up_proj.biases,<br>layers.8.mlp.up_proj.scales,<br>layers.8.mlp.up_proj.weight,<br>layers.8.post_attention_layernorm.weight,<br>layers.8.post_feedforward_layernorm.weight,<br>layers.8.pre_feedforward_layernorm.weight,<br>layers.8.self_attn.gate_proj.biases,<br>layers.8.self_attn.gate_proj.scales,<br>layers.8.self_attn.gate_proj.weight,<br>layers.8.self_attn.k_proj.biases,<br>layers.8.self_attn.k_proj.scales,<br>layers.8.self_attn.k_proj.weight,<br>layers.8.self_attn.o_proj.biases,<br>layers.8.self_attn.o_proj.scales,<br>layers.8.self_attn.o_proj.weight,<br>layers.8.self_attn.q_proj.biases,<br>layers.8.self_attn.q_proj.scales,<br>layers.8.self_attn.q_proj.weight,<br>layers.8.self_attn.v_proj.biases,<br>layers.8.self_attn.v_proj.scales,<br>layers.8.self_attn.v_proj.weight,<br>layers.9.input_layernorm.weight,<br>layers.9.mlp.down_proj.biases,<br>layers.9.mlp.down_proj.scales,<br>layers.9.mlp.down_proj.weight,<br>layers.9.mlp.gate_proj.biases,<br>layers.9.mlp.gate_proj.scales,<br>layers.9.mlp.gate_proj.weight,<br>layers.9.mlp.up_proj.biases,<br>layers.9.mlp.up_proj.scales,<br>layers.9.mlp.up_proj.weight,<br>layers.9.post_attention_layernorm.weight,<br>layers.9.post_feedforward_layernorm.weight,<br>layers.9.pre_feedforward_layernorm.weight,<br>layers.9.self_attn.gate_proj.biases,<br>layers.9.self_attn.gate_proj.scales,<br>layers.9.self_attn.gate_proj.weight,<br>layers.9.self_attn.k_proj.biases,<br>layers.9.self_attn.k_proj.scales,<br>layers.9.self_attn.k_proj.weight,<br>layers.9.self_attn.o_proj.biases,<br>layers.9.self_attn.o_proj.scales,<br>layers.9.self_attn.o_proj.weight,<br>layers.9.self_attn.q_proj.biases,<br>layers.9.self_attn.q_proj.scales,<br>layers.9.self_attn.q_proj.weight,<br>layers.9.self_attn.v_proj.biases,<br>layers.9.self_attn.v_proj.scales,<br>layers.9.self_attn.v_proj.weight,<br>norm.weight.
- *Root exception type:* ValueError
- *Root exception module:* builtins
- *Root exception message:* Received 1460 parameters not in model:
  <br>embed_tokens.biases,<br>embed_tokens.scales,<br>embed_tokens.weight,<br>layers.0.input_layernorm.weight,<br>layers.0.mlp.down_proj.biases,<br>layers.0.mlp.down_proj.scales,<br>layers.0.mlp.down_proj.weight,<br>layers.0.mlp.gate_proj.biases,<br>layers.0.mlp.gate_proj.scales,<br>layers.0.mlp.gate_proj.weight,<br>layers.0.mlp.up_proj.biases,<br>layers.0.mlp.up_proj.scales,<br>layers.0.mlp.up_proj.weight,<br>layers.0.post_attention_layernorm.weight,<br>layers.0.post_feedforward_layernorm.weight,<br>layers.0.pre_feedforward_layernorm.weight,<br>layers.0.self_attn.gate_proj.biases,<br>layers.0.self_attn.gate_proj.scales,<br>layers.0.self_attn.gate_proj.weight,<br>layers.0.self_attn.k_proj.biases,<br>layers.0.self_attn.k_proj.scales,<br>layers.0.self_attn.k_proj.weight,<br>layers.0.self_attn.o_proj.biases,<br>layers.0.self_attn.o_proj.scales,<br>layers.0.self_attn.o_proj.weight,<br>layers.0.self_attn.q_proj.biases,<br>layers.0.self_attn.q_proj.scales,<br>layers.0.self_attn.q_proj.weight,<br>layers.0.self_attn.v_proj.biases,<br>layers.0.self_attn.v_proj.scales,<br>layers.0.self_attn.v_proj.weight,<br>layers.1.input_layernorm.weight,<br>layers.1.mlp.down_proj.biases,<br>layers.1.mlp.down_proj.scales,<br>layers.1.mlp.down_proj.weight,<br>layers.1.mlp.gate_proj.biases,<br>layers.1.mlp.gate_proj.scales,<br>layers.1.mlp.gate_proj.weight,<br>layers.1.mlp.up_proj.biases,<br>layers.1.mlp.up_proj.scales,<br>layers.1.mlp.up_proj.weight,<br>layers.1.post_attention_layernorm.weight,<br>layers.1.post_feedforward_layernorm.weight,<br>layers.1.pre_feedforward_layernorm.weight,<br>layers.1.self_attn.gate_proj.biases,<br>layers.1.self_attn.gate_proj.scales,<br>layers.1.self_attn.gate_proj.weight,<br>layers.1.self_attn.k_proj.biases,<br>layers.1.self_attn.k_proj.scales,<br>layers.1.self_attn.k_proj.weight,<br>layers.1.self_attn.o_proj.biases,<br>layers.1.self_attn.o_proj.scales,<br>layers.1.self_attn.o_proj.weight,<br>layers.1.self_attn.q_proj.biases,<br>layers.1.self_attn.q_proj.scales,<br>layers.1.self_attn.q_proj.weight,<br>layers.1.self_attn.v_proj.biases,<br>layers.1.self_attn.v_proj.scales,<br>layers.1.self_attn.v_proj.weight,<br>layers.10.input_layernorm.weight,<br>layers.10.mlp.down_proj.biases,<br>layers.10.mlp.down_proj.scales,<br>layers.10.mlp.down_proj.weight,<br>layers.10.mlp.gate_proj.biases,<br>layers.10.mlp.gate_proj.scales,<br>layers.10.mlp.gate_proj.weight,<br>layers.10.mlp.up_proj.biases,<br>layers.10.mlp.up_proj.scales,<br>layers.10.mlp.up_proj.weight,<br>layers.10.post_attention_layernorm.weight,<br>layers.10.post_feedforward_layernorm.weight,<br>layers.10.pre_feedforward_layernorm.weight,<br>layers.10.self_attn.gate_proj.biases,<br>layers.10.self_attn.gate_proj.scales,<br>layers.10.self_attn.gate_proj.weight,<br>layers.10.self_attn.k_proj.biases,<br>layers.10.self_attn.k_proj.scales,<br>layers.10.self_attn.k_proj.weight,<br>layers.10.self_attn.o_proj.biases,<br>layers.10.self_attn.o_proj.scales,<br>layers.10.self_attn.o_proj.weight,<br>layers.10.self_attn.q_proj.biases,<br>layers.10.self_attn.q_proj.scales,<br>layers.10.self_attn.q_proj.weight,<br>layers.10.self_attn.v_proj.biases,<br>layers.10.self_attn.v_proj.scales,<br>layers.10.self_attn.v_proj.weight,<br>layers.11.input_layernorm.weight,<br>layers.11.mlp.down_proj.biases,<br>layers.11.mlp.down_proj.scales,<br>layers.11.mlp.down_proj.weight,<br>layers.11.mlp.gate_proj.biases,<br>layers.11.mlp.gate_proj.scales,<br>layers.11.mlp.gate_proj.weight,<br>layers.11.mlp.up_proj.biases,<br>layers.11.mlp.up_proj.scales,<br>layers.11.mlp.up_proj.weight,<br>layers.11.post_attention_layernorm.weight,<br>layers.11.post_feedforward_layernorm.weight,<br>layers.11.pre_feedforward_layernorm.weight,<br>layers.11.self_attn.gate_proj.biases,<br>layers.11.self_attn.gate_proj.scales,<br>layers.11.self_attn.gate_proj.weight,<br>layers.11.self_attn.k_proj.biases,<br>layers.11.self_attn.k_proj.scales,<br>layers.11.self_attn.k_proj.weight,<br>layers.11.self_attn.o_proj.biases,<br>layers.11.self_attn.o_proj.scales,<br>layers.11.self_attn.o_proj.weight,<br>layers.11.self_attn.q_proj.biases,<br>layers.11.self_attn.q_proj.scales,<br>layers.11.self_attn.q_proj.weight,<br>layers.11.self_attn.v_proj.biases,<br>layers.11.self_attn.v_proj.scales,<br>layers.11.self_attn.v_proj.weight,<br>layers.12.input_layernorm.weight,<br>layers.12.mlp.down_proj.biases,<br>layers.12.mlp.down_proj.scales,<br>layers.12.mlp.down_proj.weight,<br>layers.12.mlp.gate_proj.biases,<br>layers.12.mlp.gate_proj.scales,<br>layers.12.mlp.gate_proj.weight,<br>layers.12.mlp.up_proj.biases,<br>layers.12.mlp.up_proj.scales,<br>layers.12.mlp.up_proj.weight,<br>layers.12.post_attention_layernorm.weight,<br>layers.12.post_feedforward_layernorm.weight,<br>layers.12.pre_feedforward_layernorm.weight,<br>layers.12.self_attn.gate_proj.biases,<br>layers.12.self_attn.gate_proj.scales,<br>layers.12.self_attn.gate_proj.weight,<br>layers.12.self_attn.k_proj.biases,<br>layers.12.self_attn.k_proj.scales,<br>layers.12.self_attn.k_proj.weight,<br>layers.12.self_attn.o_proj.biases,<br>layers.12.self_attn.o_proj.scales,<br>layers.12.self_attn.o_proj.weight,<br>layers.12.self_attn.q_proj.biases,<br>layers.12.self_attn.q_proj.scales,<br>layers.12.self_attn.q_proj.weight,<br>layers.12.self_attn.v_proj.biases,<br>layers.12.self_attn.v_proj.scales,<br>layers.12.self_attn.v_proj.weight,<br>layers.13.input_layernorm.weight,<br>layers.13.mlp.down_proj.biases,<br>layers.13.mlp.down_proj.scales,<br>layers.13.mlp.down_proj.weight,<br>layers.13.mlp.gate_proj.biases,<br>layers.13.mlp.gate_proj.scales,<br>layers.13.mlp.gate_proj.weight,<br>layers.13.mlp.up_proj.biases,<br>layers.13.mlp.up_proj.scales,<br>layers.13.mlp.up_proj.weight,<br>layers.13.post_attention_layernorm.weight,<br>layers.13.post_feedforward_layernorm.weight,<br>layers.13.pre_feedforward_layernorm.weight,<br>layers.13.self_attn.gate_proj.biases,<br>layers.13.self_attn.gate_proj.scales,<br>layers.13.self_attn.gate_proj.weight,<br>layers.13.self_attn.k_proj.biases,<br>layers.13.self_attn.k_proj.scales,<br>layers.13.self_attn.k_proj.weight,<br>layers.13.self_attn.o_proj.biases,<br>layers.13.self_attn.o_proj.scales,<br>layers.13.self_attn.o_proj.weight,<br>layers.13.self_attn.q_proj.biases,<br>layers.13.self_attn.q_proj.scales,<br>layers.13.self_attn.q_proj.weight,<br>layers.13.self_attn.v_proj.biases,<br>layers.13.self_attn.v_proj.scales,<br>layers.13.self_attn.v_proj.weight,<br>layers.14.input_layernorm.weight,<br>layers.14.mlp.down_proj.biases,<br>layers.14.mlp.down_proj.scales,<br>layers.14.mlp.down_proj.weight,<br>layers.14.mlp.gate_proj.biases,<br>layers.14.mlp.gate_proj.scales,<br>layers.14.mlp.gate_proj.weight,<br>layers.14.mlp.up_proj.biases,<br>layers.14.mlp.up_proj.scales,<br>layers.14.mlp.up_proj.weight,<br>layers.14.post_attention_layernorm.weight,<br>layers.14.post_feedforward_layernorm.weight,<br>layers.14.pre_feedforward_layernorm.weight,<br>layers.14.self_attn.gate_proj.biases,<br>layers.14.self_attn.gate_proj.scales,<br>layers.14.self_attn.gate_proj.weight,<br>layers.14.self_attn.k_proj.biases,<br>layers.14.self_attn.k_proj.scales,<br>layers.14.self_attn.k_proj.weight,<br>layers.14.self_attn.o_proj.biases,<br>layers.14.self_attn.o_proj.scales,<br>layers.14.self_attn.o_proj.weight,<br>layers.14.self_attn.q_proj.biases,<br>layers.14.self_attn.q_proj.scales,<br>layers.14.self_attn.q_proj.weight,<br>layers.14.self_attn.v_proj.biases,<br>layers.14.self_attn.v_proj.scales,<br>layers.14.self_attn.v_proj.weight,<br>layers.15.input_layernorm.weight,<br>layers.15.mlp.down_proj.biases,<br>layers.15.mlp.down_proj.scales,<br>layers.15.mlp.down_proj.weight,<br>layers.15.mlp.gate_proj.biases,<br>layers.15.mlp.gate_proj.scales,<br>layers.15.mlp.gate_proj.weight,<br>layers.15.mlp.up_proj.biases,<br>layers.15.mlp.up_proj.scales,<br>layers.15.mlp.up_proj.weight,<br>layers.15.post_attention_layernorm.weight,<br>layers.15.post_feedforward_layernorm.weight,<br>layers.15.pre_feedforward_layernorm.weight,<br>layers.15.self_attn.gate_proj.biases,<br>layers.15.self_attn.gate_proj.scales,<br>layers.15.self_attn.gate_proj.weight,<br>layers.15.self_attn.k_proj.biases,<br>layers.15.self_attn.k_proj.scales,<br>layers.15.self_attn.k_proj.weight,<br>layers.15.self_attn.o_proj.biases,<br>layers.15.self_attn.o_proj.scales,<br>layers.15.self_attn.o_proj.weight,<br>layers.15.self_attn.q_proj.biases,<br>layers.15.self_attn.q_proj.scales,<br>layers.15.self_attn.q_proj.weight,<br>layers.15.self_attn.v_proj.biases,<br>layers.15.self_attn.v_proj.scales,<br>layers.15.self_attn.v_proj.weight,<br>layers.16.input_layernorm.weight,<br>layers.16.mlp.down_proj.biases,<br>layers.16.mlp.down_proj.scales,<br>layers.16.mlp.down_proj.weight,<br>layers.16.mlp.gate_proj.biases,<br>layers.16.mlp.gate_proj.scales,<br>layers.16.mlp.gate_proj.weight,<br>layers.16.mlp.up_proj.biases,<br>layers.16.mlp.up_proj.scales,<br>layers.16.mlp.up_proj.weight,<br>layers.16.post_attention_layernorm.weight,<br>layers.16.post_feedforward_layernorm.weight,<br>layers.16.pre_feedforward_layernorm.weight,<br>layers.16.self_attn.gate_proj.biases,<br>layers.16.self_attn.gate_proj.scales,<br>layers.16.self_attn.gate_proj.weight,<br>layers.16.self_attn.k_proj.biases,<br>layers.16.self_attn.k_proj.scales,<br>layers.16.self_attn.k_proj.weight,<br>layers.16.self_attn.o_proj.biases,<br>layers.16.self_attn.o_proj.scales,<br>layers.16.self_attn.o_proj.weight,<br>layers.16.self_attn.q_proj.biases,<br>layers.16.self_attn.q_proj.scales,<br>layers.16.self_attn.q_proj.weight,<br>layers.16.self_attn.v_proj.biases,<br>layers.16.self_attn.v_proj.scales,<br>layers.16.self_attn.v_proj.weight,<br>layers.17.input_layernorm.weight,<br>layers.17.mlp.down_proj.biases,<br>layers.17.mlp.down_proj.scales,<br>layers.17.mlp.down_proj.weight,<br>layers.17.mlp.gate_proj.biases,<br>layers.17.mlp.gate_proj.scales,<br>layers.17.mlp.gate_proj.weight,<br>layers.17.mlp.up_proj.biases,<br>layers.17.mlp.up_proj.scales,<br>layers.17.mlp.up_proj.weight,<br>layers.17.post_attention_layernorm.weight,<br>layers.17.post_feedforward_layernorm.weight,<br>layers.17.pre_feedforward_layernorm.weight,<br>layers.17.self_attn.gate_proj.biases,<br>layers.17.self_attn.gate_proj.scales,<br>layers.17.self_attn.gate_proj.weight,<br>layers.17.self_attn.k_proj.biases,<br>layers.17.self_attn.k_proj.scales,<br>layers.17.self_attn.k_proj.weight,<br>layers.17.self_attn.o_proj.biases,<br>layers.17.self_attn.o_proj.scales,<br>layers.17.self_attn.o_proj.weight,<br>layers.17.self_attn.q_proj.biases,<br>layers.17.self_attn.q_proj.scales,<br>layers.17.self_attn.q_proj.weight,<br>layers.17.self_attn.v_proj.biases,<br>layers.17.self_attn.v_proj.scales,<br>layers.17.self_attn.v_proj.weight,<br>layers.18.input_layernorm.weight,<br>layers.18.mlp.down_proj.biases,<br>layers.18.mlp.down_proj.scales,<br>layers.18.mlp.down_proj.weight,<br>layers.18.mlp.gate_proj.biases,<br>layers.18.mlp.gate_proj.scales,<br>layers.18.mlp.gate_proj.weight,<br>layers.18.mlp.up_proj.biases,<br>layers.18.mlp.up_proj.scales,<br>layers.18.mlp.up_proj.weight,<br>layers.18.post_attention_layernorm.weight,<br>layers.18.post_feedforward_layernorm.weight,<br>layers.18.pre_feedforward_layernorm.weight,<br>layers.18.self_attn.gate_proj.biases,<br>layers.18.self_attn.gate_proj.scales,<br>layers.18.self_attn.gate_proj.weight,<br>layers.18.self_attn.k_proj.biases,<br>layers.18.self_attn.k_proj.scales,<br>layers.18.self_attn.k_proj.weight,<br>layers.18.self_attn.o_proj.biases,<br>layers.18.self_attn.o_proj.scales,<br>layers.18.self_attn.o_proj.weight,<br>layers.18.self_attn.q_proj.biases,<br>layers.18.self_attn.q_proj.scales,<br>layers.18.self_attn.q_proj.weight,<br>layers.18.self_attn.v_proj.biases,<br>layers.18.self_attn.v_proj.scales,<br>layers.18.self_attn.v_proj.weight,<br>layers.19.input_layernorm.weight,<br>layers.19.mlp.down_proj.biases,<br>layers.19.mlp.down_proj.scales,<br>layers.19.mlp.down_proj.weight,<br>layers.19.mlp.gate_proj.biases,<br>layers.19.mlp.gate_proj.scales,<br>layers.19.mlp.gate_proj.weight,<br>layers.19.mlp.up_proj.biases,<br>layers.19.mlp.up_proj.scales,<br>layers.19.mlp.up_proj.weight,<br>layers.19.post_attention_layernorm.weight,<br>layers.19.post_feedforward_layernorm.weight,<br>layers.19.pre_feedforward_layernorm.weight,<br>layers.19.self_attn.gate_proj.biases,<br>layers.19.self_attn.gate_proj.scales,<br>layers.19.self_attn.gate_proj.weight,<br>layers.19.self_attn.k_proj.biases,<br>layers.19.self_attn.k_proj.scales,<br>layers.19.self_attn.k_proj.weight,<br>layers.19.self_attn.o_proj.biases,<br>layers.19.self_attn.o_proj.scales,<br>layers.19.self_attn.o_proj.weight,<br>layers.19.self_attn.q_proj.biases,<br>layers.19.self_attn.q_proj.scales,<br>layers.19.self_attn.q_proj.weight,<br>layers.19.self_attn.v_proj.biases,<br>layers.19.self_attn.v_proj.scales,<br>layers.19.self_attn.v_proj.weight,<br>layers.2.input_layernorm.weight,<br>layers.2.mlp.down_proj.biases,<br>layers.2.mlp.down_proj.scales,<br>layers.2.mlp.down_proj.weight,<br>layers.2.mlp.gate_proj.biases,<br>layers.2.mlp.gate_proj.scales,<br>layers.2.mlp.gate_proj.weight,<br>layers.2.mlp.up_proj.biases,<br>layers.2.mlp.up_proj.scales,<br>layers.2.mlp.up_proj.weight,<br>layers.2.post_attention_layernorm.weight,<br>layers.2.post_feedforward_layernorm.weight,<br>layers.2.pre_feedforward_layernorm.weight,<br>layers.2.self_attn.gate_proj.biases,<br>layers.2.self_attn.gate_proj.scales,<br>layers.2.self_attn.gate_proj.weight,<br>layers.2.self_attn.k_proj.biases,<br>layers.2.self_attn.k_proj.scales,<br>layers.2.self_attn.k_proj.weight,<br>layers.2.self_attn.o_proj.biases,<br>layers.2.self_attn.o_proj.scales,<br>layers.2.self_attn.o_proj.weight,<br>layers.2.self_attn.q_proj.biases,<br>layers.2.self_attn.q_proj.scales,<br>layers.2.self_attn.q_proj.weight,<br>layers.2.self_attn.v_proj.biases,<br>layers.2.self_attn.v_proj.scales,<br>layers.2.self_attn.v_proj.weight,<br>layers.20.input_layernorm.weight,<br>layers.20.mlp.down_proj.biases,<br>layers.20.mlp.down_proj.scales,<br>layers.20.mlp.down_proj.weight,<br>layers.20.mlp.gate_proj.biases,<br>layers.20.mlp.gate_proj.scales,<br>layers.20.mlp.gate_proj.weight,<br>layers.20.mlp.up_proj.biases,<br>layers.20.mlp.up_proj.scales,<br>layers.20.mlp.up_proj.weight,<br>layers.20.post_attention_layernorm.weight,<br>layers.20.post_feedforward_layernorm.weight,<br>layers.20.pre_feedforward_layernorm.weight,<br>layers.20.self_attn.gate_proj.biases,<br>layers.20.self_attn.gate_proj.scales,<br>layers.20.self_attn.gate_proj.weight,<br>layers.20.self_attn.k_proj.biases,<br>layers.20.self_attn.k_proj.scales,<br>layers.20.self_attn.k_proj.weight,<br>layers.20.self_attn.o_proj.biases,<br>layers.20.self_attn.o_proj.scales,<br>layers.20.self_attn.o_proj.weight,<br>layers.20.self_attn.q_proj.biases,<br>layers.20.self_attn.q_proj.scales,<br>layers.20.self_attn.q_proj.weight,<br>layers.20.self_attn.v_proj.biases,<br>layers.20.self_attn.v_proj.scales,<br>layers.20.self_attn.v_proj.weight,<br>layers.21.input_layernorm.weight,<br>layers.21.mlp.down_proj.biases,<br>layers.21.mlp.down_proj.scales,<br>layers.21.mlp.down_proj.weight,<br>layers.21.mlp.gate_proj.biases,<br>layers.21.mlp.gate_proj.scales,<br>layers.21.mlp.gate_proj.weight,<br>layers.21.mlp.up_proj.biases,<br>layers.21.mlp.up_proj.scales,<br>layers.21.mlp.up_proj.weight,<br>layers.21.post_attention_layernorm.weight,<br>layers.21.post_feedforward_layernorm.weight,<br>layers.21.pre_feedforward_layernorm.weight,<br>layers.21.self_attn.gate_proj.biases,<br>layers.21.self_attn.gate_proj.scales,<br>layers.21.self_attn.gate_proj.weight,<br>layers.21.self_attn.k_proj.biases,<br>layers.21.self_attn.k_proj.scales,<br>layers.21.self_attn.k_proj.weight,<br>layers.21.self_attn.o_proj.biases,<br>layers.21.self_attn.o_proj.scales,<br>layers.21.self_attn.o_proj.weight,<br>layers.21.self_attn.q_proj.biases,<br>layers.21.self_attn.q_proj.scales,<br>layers.21.self_attn.q_proj.weight,<br>layers.21.self_attn.v_proj.biases,<br>layers.21.self_attn.v_proj.scales,<br>layers.21.self_attn.v_proj.weight,<br>layers.22.input_layernorm.weight,<br>layers.22.mlp.down_proj.biases,<br>layers.22.mlp.down_proj.scales,<br>layers.22.mlp.down_proj.weight,<br>layers.22.mlp.gate_proj.biases,<br>layers.22.mlp.gate_proj.scales,<br>layers.22.mlp.gate_proj.weight,<br>layers.22.mlp.up_proj.biases,<br>layers.22.mlp.up_proj.scales,<br>layers.22.mlp.up_proj.weight,<br>layers.22.post_attention_layernorm.weight,<br>layers.22.post_feedforward_layernorm.weight,<br>layers.22.pre_feedforward_layernorm.weight,<br>layers.22.self_attn.gate_proj.biases,<br>layers.22.self_attn.gate_proj.scales,<br>layers.22.self_attn.gate_proj.weight,<br>layers.22.self_attn.k_proj.biases,<br>layers.22.self_attn.k_proj.scales,<br>layers.22.self_attn.k_proj.weight,<br>layers.22.self_attn.o_proj.biases,<br>layers.22.self_attn.o_proj.scales,<br>layers.22.self_attn.o_proj.weight,<br>layers.22.self_attn.q_proj.biases,<br>layers.22.self_attn.q_proj.scales,<br>layers.22.self_attn.q_proj.weight,<br>layers.22.self_attn.v_proj.biases,<br>layers.22.self_attn.v_proj.scales,<br>layers.22.self_attn.v_proj.weight,<br>layers.23.input_layernorm.weight,<br>layers.23.mlp.down_proj.biases,<br>layers.23.mlp.down_proj.scales,<br>layers.23.mlp.down_proj.weight,<br>layers.23.mlp.gate_proj.biases,<br>layers.23.mlp.gate_proj.scales,<br>layers.23.mlp.gate_proj.weight,<br>layers.23.mlp.up_proj.biases,<br>layers.23.mlp.up_proj.scales,<br>layers.23.mlp.up_proj.weight,<br>layers.23.post_attention_layernorm.weight,<br>layers.23.post_feedforward_layernorm.weight,<br>layers.23.pre_feedforward_layernorm.weight,<br>layers.23.self_attn.gate_proj.biases,<br>layers.23.self_attn.gate_proj.scales,<br>layers.23.self_attn.gate_proj.weight,<br>layers.23.self_attn.k_proj.biases,<br>layers.23.self_attn.k_proj.scales,<br>layers.23.self_attn.k_proj.weight,<br>layers.23.self_attn.o_proj.biases,<br>layers.23.self_attn.o_proj.scales,<br>layers.23.self_attn.o_proj.weight,<br>layers.23.self_attn.q_proj.biases,<br>layers.23.self_attn.q_proj.scales,<br>layers.23.self_attn.q_proj.weight,<br>layers.23.self_attn.v_proj.biases,<br>layers.23.self_attn.v_proj.scales,<br>layers.23.self_attn.v_proj.weight,<br>layers.24.input_layernorm.weight,<br>layers.24.mlp.down_proj.biases,<br>layers.24.mlp.down_proj.scales,<br>layers.24.mlp.down_proj.weight,<br>layers.24.mlp.gate_proj.biases,<br>layers.24.mlp.gate_proj.scales,<br>layers.24.mlp.gate_proj.weight,<br>layers.24.mlp.up_proj.biases,<br>layers.24.mlp.up_proj.scales,<br>layers.24.mlp.up_proj.weight,<br>layers.24.post_attention_layernorm.weight,<br>layers.24.post_feedforward_layernorm.weight,<br>layers.24.pre_feedforward_layernorm.weight,<br>layers.24.self_attn.gate_proj.biases,<br>layers.24.self_attn.gate_proj.scales,<br>layers.24.self_attn.gate_proj.weight,<br>layers.24.self_attn.k_proj.biases,<br>layers.24.self_attn.k_proj.scales,<br>layers.24.self_attn.k_proj.weight,<br>layers.24.self_attn.o_proj.biases,<br>layers.24.self_attn.o_proj.scales,<br>layers.24.self_attn.o_proj.weight,<br>layers.24.self_attn.q_proj.biases,<br>layers.24.self_attn.q_proj.scales,<br>layers.24.self_attn.q_proj.weight,<br>layers.24.self_attn.v_proj.biases,<br>layers.24.self_attn.v_proj.scales,<br>layers.24.self_attn.v_proj.weight,<br>layers.25.input_layernorm.weight,<br>layers.25.mlp.down_proj.biases,<br>layers.25.mlp.down_proj.scales,<br>layers.25.mlp.down_proj.weight,<br>layers.25.mlp.gate_proj.biases,<br>layers.25.mlp.gate_proj.scales,<br>layers.25.mlp.gate_proj.weight,<br>layers.25.mlp.up_proj.biases,<br>layers.25.mlp.up_proj.scales,<br>layers.25.mlp.up_proj.weight,<br>layers.25.post_attention_layernorm.weight,<br>layers.25.post_feedforward_layernorm.weight,<br>layers.25.pre_feedforward_layernorm.weight,<br>layers.25.self_attn.gate_proj.biases,<br>layers.25.self_attn.gate_proj.scales,<br>layers.25.self_attn.gate_proj.weight,<br>layers.25.self_attn.k_proj.biases,<br>layers.25.self_attn.k_proj.scales,<br>layers.25.self_attn.k_proj.weight,<br>layers.25.self_attn.o_proj.biases,<br>layers.25.self_attn.o_proj.scales,<br>layers.25.self_attn.o_proj.weight,<br>layers.25.self_attn.q_proj.biases,<br>layers.25.self_attn.q_proj.scales,<br>layers.25.self_attn.q_proj.weight,<br>layers.25.self_attn.v_proj.biases,<br>layers.25.self_attn.v_proj.scales,<br>layers.25.self_attn.v_proj.weight,<br>layers.26.input_layernorm.weight,<br>layers.26.mlp.down_proj.biases,<br>layers.26.mlp.down_proj.scales,<br>layers.26.mlp.down_proj.weight,<br>layers.26.mlp.gate_proj.biases,<br>layers.26.mlp.gate_proj.scales,<br>layers.26.mlp.gate_proj.weight,<br>layers.26.mlp.up_proj.biases,<br>layers.26.mlp.up_proj.scales,<br>layers.26.mlp.up_proj.weight,<br>layers.26.post_attention_layernorm.weight,<br>layers.26.post_feedforward_layernorm.weight,<br>layers.26.pre_feedforward_layernorm.weight,<br>layers.26.self_attn.gate_proj.biases,<br>layers.26.self_attn.gate_proj.scales,<br>layers.26.self_attn.gate_proj.weight,<br>layers.26.self_attn.k_proj.biases,<br>layers.26.self_attn.k_proj.scales,<br>layers.26.self_attn.k_proj.weight,<br>layers.26.self_attn.o_proj.biases,<br>layers.26.self_attn.o_proj.scales,<br>layers.26.self_attn.o_proj.weight,<br>layers.26.self_attn.q_proj.biases,<br>layers.26.self_attn.q_proj.scales,<br>layers.26.self_attn.q_proj.weight,<br>layers.26.self_attn.v_proj.biases,<br>layers.26.self_attn.v_proj.scales,<br>layers.26.self_attn.v_proj.weight,<br>layers.27.input_layernorm.weight,<br>layers.27.mlp.down_proj.biases,<br>layers.27.mlp.down_proj.scales,<br>layers.27.mlp.down_proj.weight,<br>layers.27.mlp.gate_proj.biases,<br>layers.27.mlp.gate_proj.scales,<br>layers.27.mlp.gate_proj.weight,<br>layers.27.mlp.up_proj.biases,<br>layers.27.mlp.up_proj.scales,<br>layers.27.mlp.up_proj.weight,<br>layers.27.post_attention_layernorm.weight,<br>layers.27.post_feedforward_layernorm.weight,<br>layers.27.pre_feedforward_layernorm.weight,<br>layers.27.self_attn.gate_proj.biases,<br>layers.27.self_attn.gate_proj.scales,<br>layers.27.self_attn.gate_proj.weight,<br>layers.27.self_attn.k_proj.biases,<br>layers.27.self_attn.k_proj.scales,<br>layers.27.self_attn.k_proj.weight,<br>layers.27.self_attn.o_proj.biases,<br>layers.27.self_attn.o_proj.scales,<br>layers.27.self_attn.o_proj.weight,<br>layers.27.self_attn.q_proj.biases,<br>layers.27.self_attn.q_proj.scales,<br>layers.27.self_attn.q_proj.weight,<br>layers.27.self_attn.v_proj.biases,<br>layers.27.self_attn.v_proj.scales,<br>layers.27.self_attn.v_proj.weight,<br>layers.28.input_layernorm.weight,<br>layers.28.mlp.down_proj.biases,<br>layers.28.mlp.down_proj.scales,<br>layers.28.mlp.down_proj.weight,<br>layers.28.mlp.gate_proj.biases,<br>layers.28.mlp.gate_proj.scales,<br>layers.28.mlp.gate_proj.weight,<br>layers.28.mlp.up_proj.biases,<br>layers.28.mlp.up_proj.scales,<br>layers.28.mlp.up_proj.weight,<br>layers.28.post_attention_layernorm.weight,<br>layers.28.post_feedforward_layernorm.weight,<br>layers.28.pre_feedforward_layernorm.weight,<br>layers.28.self_attn.gate_proj.biases,<br>layers.28.self_attn.gate_proj.scales,<br>layers.28.self_attn.gate_proj.weight,<br>layers.28.self_attn.k_proj.biases,<br>layers.28.self_attn.k_proj.scales,<br>layers.28.self_attn.k_proj.weight,<br>layers.28.self_attn.o_proj.biases,<br>layers.28.self_attn.o_proj.scales,<br>layers.28.self_attn.o_proj.weight,<br>layers.28.self_attn.q_proj.biases,<br>layers.28.self_attn.q_proj.scales,<br>layers.28.self_attn.q_proj.weight,<br>layers.28.self_attn.v_proj.biases,<br>layers.28.self_attn.v_proj.scales,<br>layers.28.self_attn.v_proj.weight,<br>layers.29.input_layernorm.weight,<br>layers.29.mlp.down_proj.biases,<br>layers.29.mlp.down_proj.scales,<br>layers.29.mlp.down_proj.weight,<br>layers.29.mlp.gate_proj.biases,<br>layers.29.mlp.gate_proj.scales,<br>layers.29.mlp.gate_proj.weight,<br>layers.29.mlp.up_proj.biases,<br>layers.29.mlp.up_proj.scales,<br>layers.29.mlp.up_proj.weight,<br>layers.29.post_attention_layernorm.weight,<br>layers.29.post_feedforward_layernorm.weight,<br>layers.29.pre_feedforward_layernorm.weight,<br>layers.29.self_attn.gate_proj.biases,<br>layers.29.self_attn.gate_proj.scales,<br>layers.29.self_attn.gate_proj.weight,<br>layers.29.self_attn.k_proj.biases,<br>layers.29.self_attn.k_proj.scales,<br>layers.29.self_attn.k_proj.weight,<br>layers.29.self_attn.o_proj.biases,<br>layers.29.self_attn.o_proj.scales,<br>layers.29.self_attn.o_proj.weight,<br>layers.29.self_attn.q_proj.biases,<br>layers.29.self_attn.q_proj.scales,<br>layers.29.self_attn.q_proj.weight,<br>layers.29.self_attn.v_proj.biases,<br>layers.29.self_attn.v_proj.scales,<br>layers.29.self_attn.v_proj.weight,<br>layers.3.input_layernorm.weight,<br>layers.3.mlp.down_proj.biases,<br>layers.3.mlp.down_proj.scales,<br>layers.3.mlp.down_proj.weight,<br>layers.3.mlp.gate_proj.biases,<br>layers.3.mlp.gate_proj.scales,<br>layers.3.mlp.gate_proj.weight,<br>layers.3.mlp.up_proj.biases,<br>layers.3.mlp.up_proj.scales,<br>layers.3.mlp.up_proj.weight,<br>layers.3.post_attention_layernorm.weight,<br>layers.3.post_feedforward_layernorm.weight,<br>layers.3.pre_feedforward_layernorm.weight,<br>layers.3.self_attn.gate_proj.biases,<br>layers.3.self_attn.gate_proj.scales,<br>layers.3.self_attn.gate_proj.weight,<br>layers.3.self_attn.k_proj.biases,<br>layers.3.self_attn.k_proj.scales,<br>layers.3.self_attn.k_proj.weight,<br>layers.3.self_attn.o_proj.biases,<br>layers.3.self_attn.o_proj.scales,<br>layers.3.self_attn.o_proj.weight,<br>layers.3.self_attn.q_proj.biases,<br>layers.3.self_attn.q_proj.scales,<br>layers.3.self_attn.q_proj.weight,<br>layers.3.self_attn.v_proj.biases,<br>layers.3.self_attn.v_proj.scales,<br>layers.3.self_attn.v_proj.weight,<br>layers.30.input_layernorm.weight,<br>layers.30.mlp.down_proj.biases,<br>layers.30.mlp.down_proj.scales,<br>layers.30.mlp.down_proj.weight,<br>layers.30.mlp.gate_proj.biases,<br>layers.30.mlp.gate_proj.scales,<br>layers.30.mlp.gate_proj.weight,<br>layers.30.mlp.up_proj.biases,<br>layers.30.mlp.up_proj.scales,<br>layers.30.mlp.up_proj.weight,<br>layers.30.post_attention_layernorm.weight,<br>layers.30.post_feedforward_layernorm.weight,<br>layers.30.pre_feedforward_layernorm.weight,<br>layers.30.self_attn.gate_proj.biases,<br>layers.30.self_attn.gate_proj.scales,<br>layers.30.self_attn.gate_proj.weight,<br>layers.30.self_attn.k_proj.biases,<br>layers.30.self_attn.k_proj.scales,<br>layers.30.self_attn.k_proj.weight,<br>layers.30.self_attn.o_proj.biases,<br>layers.30.self_attn.o_proj.scales,<br>layers.30.self_attn.o_proj.weight,<br>layers.30.self_attn.q_proj.biases,<br>layers.30.self_attn.q_proj.scales,<br>layers.30.self_attn.q_proj.weight,<br>layers.30.self_attn.v_proj.biases,<br>layers.30.self_attn.v_proj.scales,<br>layers.30.self_attn.v_proj.weight,<br>layers.31.input_layernorm.weight,<br>layers.31.mlp.down_proj.biases,<br>layers.31.mlp.down_proj.scales,<br>layers.31.mlp.down_proj.weight,<br>layers.31.mlp.gate_proj.biases,<br>layers.31.mlp.gate_proj.scales,<br>layers.31.mlp.gate_proj.weight,<br>layers.31.mlp.up_proj.biases,<br>layers.31.mlp.up_proj.scales,<br>layers.31.mlp.up_proj.weight,<br>layers.31.post_attention_layernorm.weight,<br>layers.31.post_feedforward_layernorm.weight,<br>layers.31.pre_feedforward_layernorm.weight,<br>layers.31.self_attn.gate_proj.biases,<br>layers.31.self_attn.gate_proj.scales,<br>layers.31.self_attn.gate_proj.weight,<br>layers.31.self_attn.k_proj.biases,<br>layers.31.self_attn.k_proj.scales,<br>layers.31.self_attn.k_proj.weight,<br>layers.31.self_attn.o_proj.biases,<br>layers.31.self_attn.o_proj.scales,<br>layers.31.self_attn.o_proj.weight,<br>layers.31.self_attn.q_proj.biases,<br>layers.31.self_attn.q_proj.scales,<br>layers.31.self_attn.q_proj.weight,<br>layers.31.self_attn.v_proj.biases,<br>layers.31.self_attn.v_proj.scales,<br>layers.31.self_attn.v_proj.weight,<br>layers.32.input_layernorm.weight,<br>layers.32.mlp.down_proj.biases,<br>layers.32.mlp.down_proj.scales,<br>layers.32.mlp.down_proj.weight,<br>layers.32.mlp.gate_proj.biases,<br>layers.32.mlp.gate_proj.scales,<br>layers.32.mlp.gate_proj.weight,<br>layers.32.mlp.up_proj.biases,<br>layers.32.mlp.up_proj.scales,<br>layers.32.mlp.up_proj.weight,<br>layers.32.post_attention_layernorm.weight,<br>layers.32.post_feedforward_layernorm.weight,<br>layers.32.pre_feedforward_layernorm.weight,<br>layers.32.self_attn.gate_proj.biases,<br>layers.32.self_attn.gate_proj.scales,<br>layers.32.self_attn.gate_proj.weight,<br>layers.32.self_attn.k_proj.biases,<br>layers.32.self_attn.k_proj.scales,<br>layers.32.self_attn.k_proj.weight,<br>layers.32.self_attn.o_proj.biases,<br>layers.32.self_attn.o_proj.scales,<br>layers.32.self_attn.o_proj.weight,<br>layers.32.self_attn.q_proj.biases,<br>layers.32.self_attn.q_proj.scales,<br>layers.32.self_attn.q_proj.weight,<br>layers.32.self_attn.v_proj.biases,<br>layers.32.self_attn.v_proj.scales,<br>layers.32.self_attn.v_proj.weight,<br>layers.33.input_layernorm.weight,<br>layers.33.mlp.down_proj.biases,<br>layers.33.mlp.down_proj.scales,<br>layers.33.mlp.down_proj.weight,<br>layers.33.mlp.gate_proj.biases,<br>layers.33.mlp.gate_proj.scales,<br>layers.33.mlp.gate_proj.weight,<br>layers.33.mlp.up_proj.biases,<br>layers.33.mlp.up_proj.scales,<br>layers.33.mlp.up_proj.weight,<br>layers.33.post_attention_layernorm.weight,<br>layers.33.post_feedforward_layernorm.weight,<br>layers.33.pre_feedforward_layernorm.weight,<br>layers.33.self_attn.gate_proj.biases,<br>layers.33.self_attn.gate_proj.scales,<br>layers.33.self_attn.gate_proj.weight,<br>layers.33.self_attn.k_proj.biases,<br>layers.33.self_attn.k_proj.scales,<br>layers.33.self_attn.k_proj.weight,<br>layers.33.self_attn.o_proj.biases,<br>layers.33.self_attn.o_proj.scales,<br>layers.33.self_attn.o_proj.weight,<br>layers.33.self_attn.q_proj.biases,<br>layers.33.self_attn.q_proj.scales,<br>layers.33.self_attn.q_proj.weight,<br>layers.33.self_attn.v_proj.biases,<br>layers.33.self_attn.v_proj.scales,<br>layers.33.self_attn.v_proj.weight,<br>layers.34.input_layernorm.weight,<br>layers.34.mlp.down_proj.biases,<br>layers.34.mlp.down_proj.scales,<br>layers.34.mlp.down_proj.weight,<br>layers.34.mlp.gate_proj.biases,<br>layers.34.mlp.gate_proj.scales,<br>layers.34.mlp.gate_proj.weight,<br>layers.34.mlp.up_proj.biases,<br>layers.34.mlp.up_proj.scales,<br>layers.34.mlp.up_proj.weight,<br>layers.34.post_attention_layernorm.weight,<br>layers.34.post_feedforward_layernorm.weight,<br>layers.34.pre_feedforward_layernorm.weight,<br>layers.34.self_attn.gate_proj.biases,<br>layers.34.self_attn.gate_proj.scales,<br>layers.34.self_attn.gate_proj.weight,<br>layers.34.self_attn.k_proj.biases,<br>layers.34.self_attn.k_proj.scales,<br>layers.34.self_attn.k_proj.weight,<br>layers.34.self_attn.o_proj.biases,<br>layers.34.self_attn.o_proj.scales,<br>layers.34.self_attn.o_proj.weight,<br>layers.34.self_attn.q_proj.biases,<br>layers.34.self_attn.q_proj.scales,<br>layers.34.self_attn.q_proj.weight,<br>layers.34.self_attn.v_proj.biases,<br>layers.34.self_attn.v_proj.scales,<br>layers.34.self_attn.v_proj.weight,<br>layers.35.input_layernorm.weight,<br>layers.35.mlp.down_proj.biases,<br>layers.35.mlp.down_proj.scales,<br>layers.35.mlp.down_proj.weight,<br>layers.35.mlp.gate_proj.biases,<br>layers.35.mlp.gate_proj.scales,<br>layers.35.mlp.gate_proj.weight,<br>layers.35.mlp.up_proj.biases,<br>layers.35.mlp.up_proj.scales,<br>layers.35.mlp.up_proj.weight,<br>layers.35.post_attention_layernorm.weight,<br>layers.35.post_feedforward_layernorm.weight,<br>layers.35.pre_feedforward_layernorm.weight,<br>layers.35.self_attn.gate_proj.biases,<br>layers.35.self_attn.gate_proj.scales,<br>layers.35.self_attn.gate_proj.weight,<br>layers.35.self_attn.k_proj.biases,<br>layers.35.self_attn.k_proj.scales,<br>layers.35.self_attn.k_proj.weight,<br>layers.35.self_attn.o_proj.biases,<br>layers.35.self_attn.o_proj.scales,<br>layers.35.self_attn.o_proj.weight,<br>layers.35.self_attn.q_proj.biases,<br>layers.35.self_attn.q_proj.scales,<br>layers.35.self_attn.q_proj.weight,<br>layers.35.self_attn.v_proj.biases,<br>layers.35.self_attn.v_proj.scales,<br>layers.35.self_attn.v_proj.weight,<br>layers.36.input_layernorm.weight,<br>layers.36.mlp.down_proj.biases,<br>layers.36.mlp.down_proj.scales,<br>layers.36.mlp.down_proj.weight,<br>layers.36.mlp.gate_proj.biases,<br>layers.36.mlp.gate_proj.scales,<br>layers.36.mlp.gate_proj.weight,<br>layers.36.mlp.up_proj.biases,<br>layers.36.mlp.up_proj.scales,<br>layers.36.mlp.up_proj.weight,<br>layers.36.post_attention_layernorm.weight,<br>layers.36.post_feedforward_layernorm.weight,<br>layers.36.pre_feedforward_layernorm.weight,<br>layers.36.self_attn.gate_proj.biases,<br>layers.36.self_attn.gate_proj.scales,<br>layers.36.self_attn.gate_proj.weight,<br>layers.36.self_attn.k_proj.biases,<br>layers.36.self_attn.k_proj.scales,<br>layers.36.self_attn.k_proj.weight,<br>layers.36.self_attn.o_proj.biases,<br>layers.36.self_attn.o_proj.scales,<br>layers.36.self_attn.o_proj.weight,<br>layers.36.self_attn.q_proj.biases,<br>layers.36.self_attn.q_proj.scales,<br>layers.36.self_attn.q_proj.weight,<br>layers.36.self_attn.v_proj.biases,<br>layers.36.self_attn.v_proj.scales,<br>layers.36.self_attn.v_proj.weight,<br>layers.37.input_layernorm.weight,<br>layers.37.mlp.down_proj.biases,<br>layers.37.mlp.down_proj.scales,<br>layers.37.mlp.down_proj.weight,<br>layers.37.mlp.gate_proj.biases,<br>layers.37.mlp.gate_proj.scales,<br>layers.37.mlp.gate_proj.weight,<br>layers.37.mlp.up_proj.biases,<br>layers.37.mlp.up_proj.scales,<br>layers.37.mlp.up_proj.weight,<br>layers.37.post_attention_layernorm.weight,<br>layers.37.post_feedforward_layernorm.weight,<br>layers.37.pre_feedforward_layernorm.weight,<br>layers.37.self_attn.gate_proj.biases,<br>layers.37.self_attn.gate_proj.scales,<br>layers.37.self_attn.gate_proj.weight,<br>layers.37.self_attn.k_proj.biases,<br>layers.37.self_attn.k_proj.scales,<br>layers.37.self_attn.k_proj.weight,<br>layers.37.self_attn.o_proj.biases,<br>layers.37.self_attn.o_proj.scales,<br>layers.37.self_attn.o_proj.weight,<br>layers.37.self_attn.q_proj.biases,<br>layers.37.self_attn.q_proj.scales,<br>layers.37.self_attn.q_proj.weight,<br>layers.37.self_attn.v_proj.biases,<br>layers.37.self_attn.v_proj.scales,<br>layers.37.self_attn.v_proj.weight,<br>layers.38.input_layernorm.weight,<br>layers.38.mlp.down_proj.biases,<br>layers.38.mlp.down_proj.scales,<br>layers.38.mlp.down_proj.weight,<br>layers.38.mlp.gate_proj.biases,<br>layers.38.mlp.gate_proj.scales,<br>layers.38.mlp.gate_proj.weight,<br>layers.38.mlp.up_proj.biases,<br>layers.38.mlp.up_proj.scales,<br>layers.38.mlp.up_proj.weight,<br>layers.38.post_attention_layernorm.weight,<br>layers.38.post_feedforward_layernorm.weight,<br>layers.38.pre_feedforward_layernorm.weight,<br>layers.38.self_attn.gate_proj.biases,<br>layers.38.self_attn.gate_proj.scales,<br>layers.38.self_attn.gate_proj.weight,<br>layers.38.self_attn.k_proj.biases,<br>layers.38.self_attn.k_proj.scales,<br>layers.38.self_attn.k_proj.weight,<br>layers.38.self_attn.o_proj.biases,<br>layers.38.self_attn.o_proj.scales,<br>layers.38.self_attn.o_proj.weight,<br>layers.38.self_attn.q_proj.biases,<br>layers.38.self_attn.q_proj.scales,<br>layers.38.self_attn.q_proj.weight,<br>layers.38.self_attn.v_proj.biases,<br>layers.38.self_attn.v_proj.scales,<br>layers.38.self_attn.v_proj.weight,<br>layers.39.input_layernorm.weight,<br>layers.39.mlp.down_proj.biases,<br>layers.39.mlp.down_proj.scales,<br>layers.39.mlp.down_proj.weight,<br>layers.39.mlp.gate_proj.biases,<br>layers.39.mlp.gate_proj.scales,<br>layers.39.mlp.gate_proj.weight,<br>layers.39.mlp.up_proj.biases,<br>layers.39.mlp.up_proj.scales,<br>layers.39.mlp.up_proj.weight,<br>layers.39.post_attention_layernorm.weight,<br>layers.39.post_feedforward_layernorm.weight,<br>layers.39.pre_feedforward_layernorm.weight,<br>layers.39.self_attn.gate_proj.biases,<br>layers.39.self_attn.gate_proj.scales,<br>layers.39.self_attn.gate_proj.weight,<br>layers.39.self_attn.k_proj.biases,<br>layers.39.self_attn.k_proj.scales,<br>layers.39.self_attn.k_proj.weight,<br>layers.39.self_attn.o_proj.biases,<br>layers.39.self_attn.o_proj.scales,<br>layers.39.self_attn.o_proj.weight,<br>layers.39.self_attn.q_proj.biases,<br>layers.39.self_attn.q_proj.scales,<br>layers.39.self_attn.q_proj.weight,<br>layers.39.self_attn.v_proj.biases,<br>layers.39.self_attn.v_proj.scales,<br>layers.39.self_attn.v_proj.weight,<br>layers.4.input_layernorm.weight,<br>layers.4.mlp.down_proj.biases,<br>layers.4.mlp.down_proj.scales,<br>layers.4.mlp.down_proj.weight,<br>layers.4.mlp.gate_proj.biases,<br>layers.4.mlp.gate_proj.scales,<br>layers.4.mlp.gate_proj.weight,<br>layers.4.mlp.up_proj.biases,<br>layers.4.mlp.up_proj.scales,<br>layers.4.mlp.up_proj.weight,<br>layers.4.post_attention_layernorm.weight,<br>layers.4.post_feedforward_layernorm.weight,<br>layers.4.pre_feedforward_layernorm.weight,<br>layers.4.self_attn.gate_proj.biases,<br>layers.4.self_attn.gate_proj.scales,<br>layers.4.self_attn.gate_proj.weight,<br>layers.4.self_attn.k_proj.biases,<br>layers.4.self_attn.k_proj.scales,<br>layers.4.self_attn.k_proj.weight,<br>layers.4.self_attn.o_proj.biases,<br>layers.4.self_attn.o_proj.scales,<br>layers.4.self_attn.o_proj.weight,<br>layers.4.self_attn.q_proj.biases,<br>layers.4.self_attn.q_proj.scales,<br>layers.4.self_attn.q_proj.weight,<br>layers.4.self_attn.v_proj.biases,<br>layers.4.self_attn.v_proj.scales,<br>layers.4.self_attn.v_proj.weight,<br>layers.40.input_layernorm.weight,<br>layers.40.mlp.down_proj.biases,<br>layers.40.mlp.down_proj.scales,<br>layers.40.mlp.down_proj.weight,<br>layers.40.mlp.gate_proj.biases,<br>layers.40.mlp.gate_proj.scales,<br>layers.40.mlp.gate_proj.weight,<br>layers.40.mlp.up_proj.biases,<br>layers.40.mlp.up_proj.scales,<br>layers.40.mlp.up_proj.weight,<br>layers.40.post_attention_layernorm.weight,<br>layers.40.post_feedforward_layernorm.weight,<br>layers.40.pre_feedforward_layernorm.weight,<br>layers.40.self_attn.gate_proj.biases,<br>layers.40.self_attn.gate_proj.scales,<br>layers.40.self_attn.gate_proj.weight,<br>layers.40.self_attn.k_proj.biases,<br>layers.40.self_attn.k_proj.scales,<br>layers.40.self_attn.k_proj.weight,<br>layers.40.self_attn.o_proj.biases,<br>layers.40.self_attn.o_proj.scales,<br>layers.40.self_attn.o_proj.weight,<br>layers.40.self_attn.q_proj.biases,<br>layers.40.self_attn.q_proj.scales,<br>layers.40.self_attn.q_proj.weight,<br>layers.40.self_attn.v_proj.biases,<br>layers.40.self_attn.v_proj.scales,<br>layers.40.self_attn.v_proj.weight,<br>layers.41.input_layernorm.weight,<br>layers.41.mlp.down_proj.biases,<br>layers.41.mlp.down_proj.scales,<br>layers.41.mlp.down_proj.weight,<br>layers.41.mlp.gate_proj.biases,<br>layers.41.mlp.gate_proj.scales,<br>layers.41.mlp.gate_proj.weight,<br>layers.41.mlp.up_proj.biases,<br>layers.41.mlp.up_proj.scales,<br>layers.41.mlp.up_proj.weight,<br>layers.41.post_attention_layernorm.weight,<br>layers.41.post_feedforward_layernorm.weight,<br>layers.41.pre_feedforward_layernorm.weight,<br>layers.41.self_attn.gate_proj.biases,<br>layers.41.self_attn.gate_proj.scales,<br>layers.41.self_attn.gate_proj.weight,<br>layers.41.self_attn.k_proj.biases,<br>layers.41.self_attn.k_proj.scales,<br>layers.41.self_attn.k_proj.weight,<br>layers.41.self_attn.o_proj.biases,<br>layers.41.self_attn.o_proj.scales,<br>layers.41.self_attn.o_proj.weight,<br>layers.41.self_attn.q_proj.biases,<br>layers.41.self_attn.q_proj.scales,<br>layers.41.self_attn.q_proj.weight,<br>layers.41.self_attn.v_proj.biases,<br>layers.41.self_attn.v_proj.scales,<br>layers.41.self_attn.v_proj.weight,<br>layers.42.input_layernorm.weight,<br>layers.42.mlp.down_proj.biases,<br>layers.42.mlp.down_proj.scales,<br>layers.42.mlp.down_proj.weight,<br>layers.42.mlp.gate_proj.biases,<br>layers.42.mlp.gate_proj.scales,<br>layers.42.mlp.gate_proj.weight,<br>layers.42.mlp.up_proj.biases,<br>layers.42.mlp.up_proj.scales,<br>layers.42.mlp.up_proj.weight,<br>layers.42.post_attention_layernorm.weight,<br>layers.42.post_feedforward_layernorm.weight,<br>layers.42.pre_feedforward_layernorm.weight,<br>layers.42.self_attn.gate_proj.biases,<br>layers.42.self_attn.gate_proj.scales,<br>layers.42.self_attn.gate_proj.weight,<br>layers.42.self_attn.k_proj.biases,<br>layers.42.self_attn.k_proj.scales,<br>layers.42.self_attn.k_proj.weight,<br>layers.42.self_attn.o_proj.biases,<br>layers.42.self_attn.o_proj.scales,<br>layers.42.self_attn.o_proj.weight,<br>layers.42.self_attn.q_proj.biases,<br>layers.42.self_attn.q_proj.scales,<br>layers.42.self_attn.q_proj.weight,<br>layers.42.self_attn.v_proj.biases,<br>layers.42.self_attn.v_proj.scales,<br>layers.42.self_attn.v_proj.weight,<br>layers.43.input_layernorm.weight,<br>layers.43.mlp.down_proj.biases,<br>layers.43.mlp.down_proj.scales,<br>layers.43.mlp.down_proj.weight,<br>layers.43.mlp.gate_proj.biases,<br>layers.43.mlp.gate_proj.scales,<br>layers.43.mlp.gate_proj.weight,<br>layers.43.mlp.up_proj.biases,<br>layers.43.mlp.up_proj.scales,<br>layers.43.mlp.up_proj.weight,<br>layers.43.post_attention_layernorm.weight,<br>layers.43.post_feedforward_layernorm.weight,<br>layers.43.pre_feedforward_layernorm.weight,<br>layers.43.self_attn.gate_proj.biases,<br>layers.43.self_attn.gate_proj.scales,<br>layers.43.self_attn.gate_proj.weight,<br>layers.43.self_attn.k_proj.biases,<br>layers.43.self_attn.k_proj.scales,<br>layers.43.self_attn.k_proj.weight,<br>layers.43.self_attn.o_proj.biases,<br>layers.43.self_attn.o_proj.scales,<br>layers.43.self_attn.o_proj.weight,<br>layers.43.self_attn.q_proj.biases,<br>layers.43.self_attn.q_proj.scales,<br>layers.43.self_attn.q_proj.weight,<br>layers.43.self_attn.v_proj.biases,<br>layers.43.self_attn.v_proj.scales,<br>layers.43.self_attn.v_proj.weight,<br>layers.44.input_layernorm.weight,<br>layers.44.mlp.down_proj.biases,<br>layers.44.mlp.down_proj.scales,<br>layers.44.mlp.down_proj.weight,<br>layers.44.mlp.gate_proj.biases,<br>layers.44.mlp.gate_proj.scales,<br>layers.44.mlp.gate_proj.weight,<br>layers.44.mlp.up_proj.biases,<br>layers.44.mlp.up_proj.scales,<br>layers.44.mlp.up_proj.weight,<br>layers.44.post_attention_layernorm.weight,<br>layers.44.post_feedforward_layernorm.weight,<br>layers.44.pre_feedforward_layernorm.weight,<br>layers.44.self_attn.gate_proj.biases,<br>layers.44.self_attn.gate_proj.scales,<br>layers.44.self_attn.gate_proj.weight,<br>layers.44.self_attn.k_proj.biases,<br>layers.44.self_attn.k_proj.scales,<br>layers.44.self_attn.k_proj.weight,<br>layers.44.self_attn.o_proj.biases,<br>layers.44.self_attn.o_proj.scales,<br>layers.44.self_attn.o_proj.weight,<br>layers.44.self_attn.q_proj.biases,<br>layers.44.self_attn.q_proj.scales,<br>layers.44.self_attn.q_proj.weight,<br>layers.44.self_attn.v_proj.biases,<br>layers.44.self_attn.v_proj.scales,<br>layers.44.self_attn.v_proj.weight,<br>layers.45.input_layernorm.weight,<br>layers.45.mlp.down_proj.biases,<br>layers.45.mlp.down_proj.scales,<br>layers.45.mlp.down_proj.weight,<br>layers.45.mlp.gate_proj.biases,<br>layers.45.mlp.gate_proj.scales,<br>layers.45.mlp.gate_proj.weight,<br>layers.45.mlp.up_proj.biases,<br>layers.45.mlp.up_proj.scales,<br>layers.45.mlp.up_proj.weight,<br>layers.45.post_attention_layernorm.weight,<br>layers.45.post_feedforward_layernorm.weight,<br>layers.45.pre_feedforward_layernorm.weight,<br>layers.45.self_attn.gate_proj.biases,<br>layers.45.self_attn.gate_proj.scales,<br>layers.45.self_attn.gate_proj.weight,<br>layers.45.self_attn.k_proj.biases,<br>layers.45.self_attn.k_proj.scales,<br>layers.45.self_attn.k_proj.weight,<br>layers.45.self_attn.o_proj.biases,<br>layers.45.self_attn.o_proj.scales,<br>layers.45.self_attn.o_proj.weight,<br>layers.45.self_attn.q_proj.biases,<br>layers.45.self_attn.q_proj.scales,<br>layers.45.self_attn.q_proj.weight,<br>layers.45.self_attn.v_proj.biases,<br>layers.45.self_attn.v_proj.scales,<br>layers.45.self_attn.v_proj.weight,<br>layers.46.input_layernorm.weight,<br>layers.46.mlp.down_proj.biases,<br>layers.46.mlp.down_proj.scales,<br>layers.46.mlp.down_proj.weight,<br>layers.46.mlp.gate_proj.biases,<br>layers.46.mlp.gate_proj.scales,<br>layers.46.mlp.gate_proj.weight,<br>layers.46.mlp.up_proj.biases,<br>layers.46.mlp.up_proj.scales,<br>layers.46.mlp.up_proj.weight,<br>layers.46.post_attention_layernorm.weight,<br>layers.46.post_feedforward_layernorm.weight,<br>layers.46.pre_feedforward_layernorm.weight,<br>layers.46.self_attn.gate_proj.biases,<br>layers.46.self_attn.gate_proj.scales,<br>layers.46.self_attn.gate_proj.weight,<br>layers.46.self_attn.k_proj.biases,<br>layers.46.self_attn.k_proj.scales,<br>layers.46.self_attn.k_proj.weight,<br>layers.46.self_attn.o_proj.biases,<br>layers.46.self_attn.o_proj.scales,<br>layers.46.self_attn.o_proj.weight,<br>layers.46.self_attn.q_proj.biases,<br>layers.46.self_attn.q_proj.scales,<br>layers.46.self_attn.q_proj.weight,<br>layers.46.self_attn.v_proj.biases,<br>layers.46.self_attn.v_proj.scales,<br>layers.46.self_attn.v_proj.weight,<br>layers.47.input_layernorm.weight,<br>layers.47.mlp.down_proj.biases,<br>layers.47.mlp.down_proj.scales,<br>layers.47.mlp.down_proj.weight,<br>layers.47.mlp.gate_proj.biases,<br>layers.47.mlp.gate_proj.scales,<br>layers.47.mlp.gate_proj.weight,<br>layers.47.mlp.up_proj.biases,<br>layers.47.mlp.up_proj.scales,<br>layers.47.mlp.up_proj.weight,<br>layers.47.post_attention_layernorm.weight,<br>layers.47.post_feedforward_layernorm.weight,<br>layers.47.pre_feedforward_layernorm.weight,<br>layers.47.self_attn.gate_proj.biases,<br>layers.47.self_attn.gate_proj.scales,<br>layers.47.self_attn.gate_proj.weight,<br>layers.47.self_attn.k_proj.biases,<br>layers.47.self_attn.k_proj.scales,<br>layers.47.self_attn.k_proj.weight,<br>layers.47.self_attn.o_proj.biases,<br>layers.47.self_attn.o_proj.scales,<br>layers.47.self_attn.o_proj.weight,<br>layers.47.self_attn.q_proj.biases,<br>layers.47.self_attn.q_proj.scales,<br>layers.47.self_attn.q_proj.weight,<br>layers.47.self_attn.v_proj.biases,<br>layers.47.self_attn.v_proj.scales,<br>layers.47.self_attn.v_proj.weight,<br>layers.48.input_layernorm.weight,<br>layers.48.mlp.down_proj.biases,<br>layers.48.mlp.down_proj.scales,<br>layers.48.mlp.down_proj.weight,<br>layers.48.mlp.gate_proj.biases,<br>layers.48.mlp.gate_proj.scales,<br>layers.48.mlp.gate_proj.weight,<br>layers.48.mlp.up_proj.biases,<br>layers.48.mlp.up_proj.scales,<br>layers.48.mlp.up_proj.weight,<br>layers.48.post_attention_layernorm.weight,<br>layers.48.post_feedforward_layernorm.weight,<br>layers.48.pre_feedforward_layernorm.weight,<br>layers.48.self_attn.gate_proj.biases,<br>layers.48.self_attn.gate_proj.scales,<br>layers.48.self_attn.gate_proj.weight,<br>layers.48.self_attn.k_proj.biases,<br>layers.48.self_attn.k_proj.scales,<br>layers.48.self_attn.k_proj.weight,<br>layers.48.self_attn.o_proj.biases,<br>layers.48.self_attn.o_proj.scales,<br>layers.48.self_attn.o_proj.weight,<br>layers.48.self_attn.q_proj.biases,<br>layers.48.self_attn.q_proj.scales,<br>layers.48.self_attn.q_proj.weight,<br>layers.48.self_attn.v_proj.biases,<br>layers.48.self_attn.v_proj.scales,<br>layers.48.self_attn.v_proj.weight,<br>layers.49.input_layernorm.weight,<br>layers.49.mlp.down_proj.biases,<br>layers.49.mlp.down_proj.scales,<br>layers.49.mlp.down_proj.weight,<br>layers.49.mlp.gate_proj.biases,<br>layers.49.mlp.gate_proj.scales,<br>layers.49.mlp.gate_proj.weight,<br>layers.49.mlp.up_proj.biases,<br>layers.49.mlp.up_proj.scales,<br>layers.49.mlp.up_proj.weight,<br>layers.49.post_attention_layernorm.weight,<br>layers.49.post_feedforward_layernorm.weight,<br>layers.49.pre_feedforward_layernorm.weight,<br>layers.49.self_attn.gate_proj.biases,<br>layers.49.self_attn.gate_proj.scales,<br>layers.49.self_attn.gate_proj.weight,<br>layers.49.self_attn.k_proj.biases,<br>layers.49.self_attn.k_proj.scales,<br>layers.49.self_attn.k_proj.weight,<br>layers.49.self_attn.o_proj.biases,<br>layers.49.self_attn.o_proj.scales,<br>layers.49.self_attn.o_proj.weight,<br>layers.49.self_attn.q_proj.biases,<br>layers.49.self_attn.q_proj.scales,<br>layers.49.self_attn.q_proj.weight,<br>layers.49.self_attn.v_proj.biases,<br>layers.49.self_attn.v_proj.scales,<br>layers.49.self_attn.v_proj.weight,<br>layers.5.input_layernorm.weight,<br>layers.5.mlp.down_proj.biases,<br>layers.5.mlp.down_proj.scales,<br>layers.5.mlp.down_proj.weight,<br>layers.5.mlp.gate_proj.biases,<br>layers.5.mlp.gate_proj.scales,<br>layers.5.mlp.gate_proj.weight,<br>layers.5.mlp.up_proj.biases,<br>layers.5.mlp.up_proj.scales,<br>layers.5.mlp.up_proj.weight,<br>layers.5.post_attention_layernorm.weight,<br>layers.5.post_feedforward_layernorm.weight,<br>layers.5.pre_feedforward_layernorm.weight,<br>layers.5.self_attn.gate_proj.biases,<br>layers.5.self_attn.gate_proj.scales,<br>layers.5.self_attn.gate_proj.weight,<br>layers.5.self_attn.k_proj.biases,<br>layers.5.self_attn.k_proj.scales,<br>layers.5.self_attn.k_proj.weight,<br>layers.5.self_attn.o_proj.biases,<br>layers.5.self_attn.o_proj.scales,<br>layers.5.self_attn.o_proj.weight,<br>layers.5.self_attn.q_proj.biases,<br>layers.5.self_attn.q_proj.scales,<br>layers.5.self_attn.q_proj.weight,<br>layers.5.self_attn.v_proj.biases,<br>layers.5.self_attn.v_proj.scales,<br>layers.5.self_attn.v_proj.weight,<br>layers.50.input_layernorm.weight,<br>layers.50.mlp.down_proj.biases,<br>layers.50.mlp.down_proj.scales,<br>layers.50.mlp.down_proj.weight,<br>layers.50.mlp.gate_proj.biases,<br>layers.50.mlp.gate_proj.scales,<br>layers.50.mlp.gate_proj.weight,<br>layers.50.mlp.up_proj.biases,<br>layers.50.mlp.up_proj.scales,<br>layers.50.mlp.up_proj.weight,<br>layers.50.post_attention_layernorm.weight,<br>layers.50.post_feedforward_layernorm.weight,<br>layers.50.pre_feedforward_layernorm.weight,<br>layers.50.self_attn.gate_proj.biases,<br>layers.50.self_attn.gate_proj.scales,<br>layers.50.self_attn.gate_proj.weight,<br>layers.50.self_attn.k_proj.biases,<br>layers.50.self_attn.k_proj.scales,<br>layers.50.self_attn.k_proj.weight,<br>layers.50.self_attn.o_proj.biases,<br>layers.50.self_attn.o_proj.scales,<br>layers.50.self_attn.o_proj.weight,<br>layers.50.self_attn.q_proj.biases,<br>layers.50.self_attn.q_proj.scales,<br>layers.50.self_attn.q_proj.weight,<br>layers.50.self_attn.v_proj.biases,<br>layers.50.self_attn.v_proj.scales,<br>layers.50.self_attn.v_proj.weight,<br>layers.51.input_layernorm.weight,<br>layers.51.mlp.down_proj.biases,<br>layers.51.mlp.down_proj.scales,<br>layers.51.mlp.down_proj.weight,<br>layers.51.mlp.gate_proj.biases,<br>layers.51.mlp.gate_proj.scales,<br>layers.51.mlp.gate_proj.weight,<br>layers.51.mlp.up_proj.biases,<br>layers.51.mlp.up_proj.scales,<br>layers.51.mlp.up_proj.weight,<br>layers.51.post_attention_layernorm.weight,<br>layers.51.post_feedforward_layernorm.weight,<br>layers.51.pre_feedforward_layernorm.weight,<br>layers.51.self_attn.gate_proj.biases,<br>layers.51.self_attn.gate_proj.scales,<br>layers.51.self_attn.gate_proj.weight,<br>layers.51.self_attn.k_proj.biases,<br>layers.51.self_attn.k_proj.scales,<br>layers.51.self_attn.k_proj.weight,<br>layers.51.self_attn.o_proj.biases,<br>layers.51.self_attn.o_proj.scales,<br>layers.51.self_attn.o_proj.weight,<br>layers.51.self_attn.q_proj.biases,<br>layers.51.self_attn.q_proj.scales,<br>layers.51.self_attn.q_proj.weight,<br>layers.51.self_attn.v_proj.biases,<br>layers.51.self_attn.v_proj.scales,<br>layers.51.self_attn.v_proj.weight,<br>layers.6.input_layernorm.weight,<br>layers.6.mlp.down_proj.biases,<br>layers.6.mlp.down_proj.scales,<br>layers.6.mlp.down_proj.weight,<br>layers.6.mlp.gate_proj.biases,<br>layers.6.mlp.gate_proj.scales,<br>layers.6.mlp.gate_proj.weight,<br>layers.6.mlp.up_proj.biases,<br>layers.6.mlp.up_proj.scales,<br>layers.6.mlp.up_proj.weight,<br>layers.6.post_attention_layernorm.weight,<br>layers.6.post_feedforward_layernorm.weight,<br>layers.6.pre_feedforward_layernorm.weight,<br>layers.6.self_attn.gate_proj.biases,<br>layers.6.self_attn.gate_proj.scales,<br>layers.6.self_attn.gate_proj.weight,<br>layers.6.self_attn.k_proj.biases,<br>layers.6.self_attn.k_proj.scales,<br>layers.6.self_attn.k_proj.weight,<br>layers.6.self_attn.o_proj.biases,<br>layers.6.self_attn.o_proj.scales,<br>layers.6.self_attn.o_proj.weight,<br>layers.6.self_attn.q_proj.biases,<br>layers.6.self_attn.q_proj.scales,<br>layers.6.self_attn.q_proj.weight,<br>layers.6.self_attn.v_proj.biases,<br>layers.6.self_attn.v_proj.scales,<br>layers.6.self_attn.v_proj.weight,<br>layers.7.input_layernorm.weight,<br>layers.7.mlp.down_proj.biases,<br>layers.7.mlp.down_proj.scales,<br>layers.7.mlp.down_proj.weight,<br>layers.7.mlp.gate_proj.biases,<br>layers.7.mlp.gate_proj.scales,<br>layers.7.mlp.gate_proj.weight,<br>layers.7.mlp.up_proj.biases,<br>layers.7.mlp.up_proj.scales,<br>layers.7.mlp.up_proj.weight,<br>layers.7.post_attention_layernorm.weight,<br>layers.7.post_feedforward_layernorm.weight,<br>layers.7.pre_feedforward_layernorm.weight,<br>layers.7.self_attn.gate_proj.biases,<br>layers.7.self_attn.gate_proj.scales,<br>layers.7.self_attn.gate_proj.weight,<br>layers.7.self_attn.k_proj.biases,<br>layers.7.self_attn.k_proj.scales,<br>layers.7.self_attn.k_proj.weight,<br>layers.7.self_attn.o_proj.biases,<br>layers.7.self_attn.o_proj.scales,<br>layers.7.self_attn.o_proj.weight,<br>layers.7.self_attn.q_proj.biases,<br>layers.7.self_attn.q_proj.scales,<br>layers.7.self_attn.q_proj.weight,<br>layers.7.self_attn.v_proj.biases,<br>layers.7.self_attn.v_proj.scales,<br>layers.7.self_attn.v_proj.weight,<br>layers.8.input_layernorm.weight,<br>layers.8.mlp.down_proj.biases,<br>layers.8.mlp.down_proj.scales,<br>layers.8.mlp.down_proj.weight,<br>layers.8.mlp.gate_proj.biases,<br>layers.8.mlp.gate_proj.scales,<br>layers.8.mlp.gate_proj.weight,<br>layers.8.mlp.up_proj.biases,<br>layers.8.mlp.up_proj.scales,<br>layers.8.mlp.up_proj.weight,<br>layers.8.post_attention_layernorm.weight,<br>layers.8.post_feedforward_layernorm.weight,<br>layers.8.pre_feedforward_layernorm.weight,<br>layers.8.self_attn.gate_proj.biases,<br>layers.8.self_attn.gate_proj.scales,<br>layers.8.self_attn.gate_proj.weight,<br>layers.8.self_attn.k_proj.biases,<br>layers.8.self_attn.k_proj.scales,<br>layers.8.self_attn.k_proj.weight,<br>layers.8.self_attn.o_proj.biases,<br>layers.8.self_attn.o_proj.scales,<br>layers.8.self_attn.o_proj.weight,<br>layers.8.self_attn.q_proj.biases,<br>layers.8.self_attn.q_proj.scales,<br>layers.8.self_attn.q_proj.weight,<br>layers.8.self_attn.v_proj.biases,<br>layers.8.self_attn.v_proj.scales,<br>layers.8.self_attn.v_proj.weight,<br>layers.9.input_layernorm.weight,<br>layers.9.mlp.down_proj.biases,<br>layers.9.mlp.down_proj.scales,<br>layers.9.mlp.down_proj.weight,<br>layers.9.mlp.gate_proj.biases,<br>layers.9.mlp.gate_proj.scales,<br>layers.9.mlp.gate_proj.weight,<br>layers.9.mlp.up_proj.biases,<br>layers.9.mlp.up_proj.scales,<br>layers.9.mlp.up_proj.weight,<br>layers.9.post_attention_layernorm.weight,<br>layers.9.post_feedforward_layernorm.weight,<br>layers.9.pre_feedforward_layernorm.weight,<br>layers.9.self_attn.gate_proj.biases,<br>layers.9.self_attn.gate_proj.scales,<br>layers.9.self_attn.gate_proj.weight,<br>layers.9.self_attn.k_proj.biases,<br>layers.9.self_attn.k_proj.scales,<br>layers.9.self_attn.k_proj.weight,<br>layers.9.self_attn.o_proj.biases,<br>layers.9.self_attn.o_proj.scales,<br>layers.9.self_attn.o_proj.weight,<br>layers.9.self_attn.q_proj.biases,<br>layers.9.self_attn.q_proj.scales,<br>layers.9.self_attn.q_proj.weight,<br>layers.9.self_attn.v_proj.biases,<br>layers.9.self_attn.v_proj.scales,<br>layers.9.self_attn.v_proj.weight,<br>norm.weight.
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Model load time:* 0.50s
- *Generation time:* -
- *Total time:* 0.50s
- *Input validation time:* 4.92e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.0729
- *Prompt tokens:* -
- *Generation tokens:* -
- *Total tokens:* -
- *Prompt throughput (raw):* -
- *Generation throughput (raw):* -
- *Peak memory:* -
- *Active memory:* -
- *Cache memory:* -
- *Model-load active memory:* -
- *Post-cleanup active memory:* 0.00598
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
- *Resolved model revision:* b4a74fa6001f1eca3b23eeeb702ffad2773a218f
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-OptiQ-4bit/snapshots/b4a74fa6001f1eca3b23eeeb702ffad2773a218f
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1282, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1147, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 1460 parameters not in model: 
embed_tokens.biases,
embed_tokens.scales,
embed_tokens.weight,
layers.0.input_layernorm.weight,
layers.0.mlp.down_proj.biases,
layers.0.mlp.down_proj.scales,
layers.0.mlp.down_proj.weight,
layers.0.mlp.gate_proj.biases,
layers.0.mlp.gate_proj.scales,
layers.0.mlp.gate_proj.weight,
layers.0.mlp.up_proj.biases,
layers.0.mlp.up_proj.scales,
layers.0.mlp.up_proj.weight,
layers.0.post_attention_layernorm.weight,
layers.0.post_feedforward_layernorm.weight,
layers.0.pre_feedforward_layernorm.weight,
layers.0.self_attn.gate_proj.biases,
layers.0.self_attn.gate_proj.scales,
layers.0.self_attn.gate_proj.weight,
layers.0.self_attn.k_proj.biases,
layers.0.self_attn.k_proj.scales,
layers.0.self_attn.k_proj.weight,
layers.0.self_attn.o_proj.biases,
layers.0.self_attn.o_proj.scales,
layers.0.self_attn.o_proj.weight,
layers.0.self_attn.q_proj.biases,
layers.0.self_attn.q_proj.scales,
layers.0.self_attn.q_proj.weight,
layers.0.self_attn.v_proj.biases,
layers.0.self_attn.v_proj.scales,
layers.0.self_attn.v_proj.weight,
layers.1.input_layernorm.weight,
layers.1.mlp.down_proj.biases,
layers.1.mlp.down_proj.scales,
layers.1.mlp.down_proj.weight,
layers.1.mlp.gate_proj.biases,
layers.1.mlp.gate_proj.scales,
layers.1.mlp.gate_proj.weight,
layers.1.mlp.up_proj.biases,
layers.1.mlp.up_proj.scales,
layers.1.mlp.up_proj.weight,
layers.1.post_attention_layernorm.weight,
layers.1.post_feedforward_layernorm.weight,
layers.1.pre_feedforward_layernorm.weight,
layers.1.self_attn.gate_proj.biases,
layers.1.self_attn.gate_proj.scales,
layers.1.self_attn.gate_proj.weight,
layers.1.self_attn.k_proj.biases,
layers.1.self_attn.k_proj.scales,
layers.1.self_attn.k_proj.weight,
layers.1.self_attn.o_proj.biases,
layers.1.self_attn.o_proj.scales,
layers.1.self_attn.o_proj.weight,
layers.1.self_attn.q_proj.biases,
layers.1.self_attn.q_proj.scales,
layers.1.self_attn.q_proj.weight,
layers.1.self_attn.v_proj.biases,
layers.1.self_attn.v_proj.scales,
layers.1.self_attn.v_proj.weight,
layers.10.input_layernorm.weight,
layers.10.mlp.down_proj.biases,
layers.10.mlp.down_proj.scales,
layers.10.mlp.down_proj.weight,
layers.10.mlp.gate_proj.biases,
layers.10.mlp.gate_proj.scales,
layers.10.mlp.gate_proj.weight,
layers.10.mlp.up_proj.biases,
layers.10.mlp.up_proj.scales,
layers.10.mlp.up_proj.weight,
layers.10.post_attention_layernorm.weight,
layers.10.post_feedforward_layernorm.weight,
layers.10.pre_feedforward_layernorm.weight,
layers.10.self_attn.gate_proj.biases,
layers.10.self_attn.gate_proj.scales,
layers.10.self_attn.gate_proj.weight,
layers.10.self_attn.k_proj.biases,
layers.10.self_attn.k_proj.scales,
layers.10.self_attn.k_proj.weight,
layers.10.self_attn.o_proj.biases,
layers.10.self_attn.o_proj.scales,
layers.10.self_attn.o_proj.weight,
layers.10.self_attn.q_proj.biases,
layers.10.self_attn.q_proj.scales,
layers.10.self_attn.q_proj.weight,
layers.10.self_attn.v_proj.biases,
layers.10.self_attn.v_proj.scales,
layers.10.self_attn.v_proj.weight,
layers.11.input_layernorm.weight,
layers.11.mlp.down_proj.biases,
layers.11.mlp.down_proj.scales,
layers.11.mlp.down_proj.weight,
layers.11.mlp.gate_proj.biases,
layers.11.mlp.gate_proj.scales,
layers.11.mlp.gate_proj.weight,
layers.11.mlp.up_proj.biases,
layers.11.mlp.up_proj.scales,
layers.11.mlp.up_proj.weight,
layers.11.post_attention_layernorm.weight,
layers.11.post_feedforward_layernorm.weight,
layers.11.pre_feedforward_layernorm.weight,
layers.11.self_attn.gate_proj.biases,
layers.11.self_attn.gate_proj.scales,
layers.11.self_attn.gate_proj.weight,
layers.11.self_attn.k_proj.biases,
layers.11.self_attn.k_proj.scales,
layers.11.self_attn.k_proj.weight,
layers.11.self_attn.o_proj.biases,
layers.11.self_attn.o_proj.scales,
layers.11.self_attn.o_proj.weight,
layers.11.self_attn.q_proj.biases,
layers.11.self_attn.q_proj.scales,
layers.11.self_attn.q_proj.weight,
layers.11.self_attn.v_proj.biases,
layers.11.self_attn.v_proj.scales,
layers.11.self_attn.v_proj.weight,
layers.12.input_layernorm.weight,
layers.12.mlp.down_proj.biases,
layers.12.mlp.down_proj.scales,
layers.12.mlp.down_proj.weight,
layers.12.mlp.gate_proj.biases,
layers.12.mlp.gate_proj.scales,
layers.12.mlp.gate_proj.weight,
layers.12.mlp.up_proj.biases,
layers.12.mlp.up_proj.scales,
layers.12.mlp.up_proj.weight,
layers.12.post_attention_layernorm.weight,
layers.12.post_feedforward_layernorm.weight,
layers.12.pre_feedforward_layernorm.weight,
layers.12.self_attn.gate_proj.biases,
layers.12.self_attn.gate_proj.scales,
layers.12.self_attn.gate_proj.weight,
layers.12.self_attn.k_proj.biases,
layers.12.self_attn.k_proj.scales,
layers.12.self_attn.k_proj.weight,
layers.12.self_attn.o_proj.biases,
layers.12.self_attn.o_proj.scales,
layers.12.self_attn.o_proj.weight,
layers.12.self_attn.q_proj.biases,
layers.12.self_attn.q_proj.scales,
layers.12.self_attn.q_proj.weight,
layers.12.self_attn.v_proj.biases,
layers.12.self_attn.v_proj.scales,
layers.12.self_attn.v_proj.weight,
layers.13.input_layernorm.weight,
layers.13.mlp.down_proj.biases,
layers.13.mlp.down_proj.scales,
layers.13.mlp.down_proj.weight,
layers.13.mlp.gate_proj.biases,
layers.13.mlp.gate_proj.scales,
layers.13.mlp.gate_proj.weight,
layers.13.mlp.up_proj.biases,
layers.13.mlp.up_proj.scales,
layers.13.mlp.up_proj.weight,
layers.13.post_attention_layernorm.weight,
layers.13.post_feedforward_layernorm.weight,
layers.13.pre_feedforward_layernorm.weight,
layers.13.self_attn.gate_proj.biases,
layers.13.self_attn.gate_proj.scales,
layers.13.self_attn.gate_proj.weight,
layers.13.self_attn.k_proj.biases,
layers.13.self_attn.k_proj.scales,
layers.13.self_attn.k_proj.weight,
layers.13.self_attn.o_proj.biases,
layers.13.self_attn.o_proj.scales,
layers.13.self_attn.o_proj.weight,
layers.13.self_attn.q_proj.biases,
layers.13.self_attn.q_proj.scales,
layers.13.self_attn.q_proj.weight,
layers.13.self_attn.v_proj.biases,
layers.13.self_attn.v_proj.scales,
layers.13.self_attn.v_proj.weight,
layers.14.input_layernorm.weight,
layers.14.mlp.down_proj.biases,
layers.14.mlp.down_proj.scales,
layers.14.mlp.down_proj.weight,
layers.14.mlp.gate_proj.biases,
layers.14.mlp.gate_proj.scales,
layers.14.mlp.gate_proj.weight,
layers.14.mlp.up_proj.biases,
layers.14.mlp.up_proj.scales,
layers.14.mlp.up_proj.weight,
layers.14.post_attention_layernorm.weight,
layers.14.post_feedforward_layernorm.weight,
layers.14.pre_feedforward_layernorm.weight,
layers.14.self_attn.gate_proj.biases,
layers.14.self_attn.gate_proj.scales,
layers.14.self_attn.gate_proj.weight,
layers.14.self_attn.k_proj.biases,
layers.14.self_attn.k_proj.scales,
layers.14.self_attn.k_proj.weight,
layers.14.self_attn.o_proj.biases,
layers.14.self_attn.o_proj.scales,
layers.14.self_attn.o_proj.weight,
layers.14.self_attn.q_proj.biases,
layers.14.self_attn.q_proj.scales,
layers.14.self_attn.q_proj.weight,
layers.14.self_attn.v_proj.biases,
layers.14.self_attn.v_proj.scales,
layers.14.self_attn.v_proj.weight,
layers.15.input_layernorm.weight,
layers.15.mlp.down_proj.biases,
layers.15.mlp.down_proj.scales,
layers.15.mlp.down_proj.weight,
layers.15.mlp.gate_proj.biases,
layers.15.mlp.gate_proj.scales,
layers.15.mlp.gate_proj.weight,
layers.15.mlp.up_proj.biases,
layers.15.mlp.up_proj.scales,
layers.15.mlp.up_proj.weight,
layers.15.post_attention_layernorm.weight,
layers.15.post_feedforward_layernorm.weight,
layers.15.pre_feedforward_layernorm.weight,
layers.15.self_attn.gate_proj.biases,
layers.15.self_attn.gate_proj.scales,
layers.15.self_attn.gate_proj.weight,
layers.15.self_attn.k_proj.biases,
layers.15.self_attn.k_proj.scales,
layers.15.self_attn.k_proj.weight,
layers.15.self_attn.o_proj.biases,
layers.15.self_attn.o_proj.scales,
layers.15.self_attn.o_proj.weight,
layers.15.self_attn.q_proj.biases,
layers.15.self_attn.q_proj.scales,
layers.15.self_attn.q_proj.weight,
layers.15.self_attn.v_proj.biases,
layers.15.self_attn.v_proj.scales,
layers.15.self_attn.v_proj.weight,
layers.16.input_layernorm.weight,
layers.16.mlp.down_proj.biases,
layers.16.mlp.down_proj.scales,
layers.16.mlp.down_proj.weight,
layers.16.mlp.gate_proj.biases,
layers.16.mlp.gate_proj.scales,
layers.16.mlp.gate_proj.weight,
layers.16.mlp.up_proj.biases,
layers.16.mlp.up_proj.scales,
layers.16.mlp.up_proj.weight,
layers.16.post_attention_layernorm.weight,
layers.16.post_feedforward_layernorm.weight,
layers.16.pre_feedforward_layernorm.weight,
layers.16.self_attn.gate_proj.biases,
layers.16.self_attn.gate_proj.scales,
layers.16.self_attn.gate_proj.weight,
layers.16.self_attn.k_proj.biases,
layers.16.self_attn.k_proj.scales,
layers.16.self_attn.k_proj.weight,
layers.16.self_attn.o_proj.biases,
layers.16.self_attn.o_proj.scales,
layers.16.self_attn.o_proj.weight,
layers.16.self_attn.q_proj.biases,
layers.16.self_attn.q_proj.scales,
layers.16.self_attn.q_proj.weight,
layers.16.self_attn.v_proj.biases,
layers.16.self_attn.v_proj.scales,
layers.16.self_attn.v_proj.weight,
layers.17.input_layernorm.weight,
layers.17.mlp.down_proj.biases,
layers.17.mlp.down_proj.scales,
layers.17.mlp.down_proj.weight,
layers.17.mlp.gate_proj.biases,
layers.17.mlp.gate_proj.scales,
layers.17.mlp.gate_proj.weight,
layers.17.mlp.up_proj.biases,
layers.17.mlp.up_proj.scales,
layers.17.mlp.up_proj.weight,
layers.17.post_attention_layernorm.weight,
layers.17.post_feedforward_layernorm.weight,
layers.17.pre_feedforward_layernorm.weight,
layers.17.self_attn.gate_proj.biases,
layers.17.self_attn.gate_proj.scales,
layers.17.self_attn.gate_proj.weight,
layers.17.self_attn.k_proj.biases,
layers.17.self_attn.k_proj.scales,
layers.17.self_attn.k_proj.weight,
layers.17.self_attn.o_proj.biases,
layers.17.self_attn.o_proj.scales,
layers.17.self_attn.o_proj.weight,
layers.17.self_attn.q_proj.biases,
layers.17.self_attn.q_proj.scales,
layers.17.self_attn.q_proj.weight,
layers.17.self_attn.v_proj.biases,
layers.17.self_attn.v_proj.scales,
layers.17.self_attn.v_proj.weight,
layers.18.input_layernorm.weight,
layers.18.mlp.down_proj.biases,
layers.18.mlp.down_proj.scales,
layers.18.mlp.down_proj.weight,
layers.18.mlp.gate_proj.biases,
layers.18.mlp.gate_proj.scales,
layers.18.mlp.gate_proj.weight,
layers.18.mlp.up_proj.biases,
layers.18.mlp.up_proj.scales,
layers.18.mlp.up_proj.weight,
layers.18.post_attention_layernorm.weight,
layers.18.post_feedforward_layernorm.weight,
layers.18.pre_feedforward_layernorm.weight,
layers.18.self_attn.gate_proj.biases,
layers.18.self_attn.gate_proj.scales,
layers.18.self_attn.gate_proj.weight,
layers.18.self_attn.k_proj.biases,
layers.18.self_attn.k_proj.scales,
layers.18.self_attn.k_proj.weight,
layers.18.self_attn.o_proj.biases,
layers.18.self_attn.o_proj.scales,
layers.18.self_attn.o_proj.weight,
layers.18.self_attn.q_proj.biases,
layers.18.self_attn.q_proj.scales,
layers.18.self_attn.q_proj.weight,
layers.18.self_attn.v_proj.biases,
layers.18.self_attn.v_proj.scales,
layers.18.self_attn.v_proj.weight,
layers.19.input_layernorm.weight,
layers.19.mlp.down_proj.biases,
layers.19.mlp.down_proj.scales,
layers.19.mlp.down_proj.weight,
layers.19.mlp.gate_proj.biases,
layers.19.mlp.gate_proj.scales,
layers.19.mlp.gate_proj.weight,
layers.19.mlp.up_proj.biases,
layers.19.mlp.up_proj.scales,
layers.19.mlp.up_proj.weight,
layers.19.post_attention_layernorm.weight,
layers.19.post_feedforward_layernorm.weight,
layers.19.pre_feedforward_layernorm.weight,
layers.19.self_attn.gate_proj.biases,
layers.19.self_attn.gate_proj.scales,
layers.19.self_attn.gate_proj.weight,
layers.19.self_attn.k_proj.biases,
layers.19.self_attn.k_proj.scales,
layers.19.self_attn.k_proj.weight,
layers.19.self_attn.o_proj.biases,
layers.19.self_attn.o_proj.scales,
layers.19.self_attn.o_proj.weight,
layers.19.self_attn.q_proj.biases,
layers.19.self_attn.q_proj.scales,
layers.19.self_attn.q_proj.weight,
layers.19.self_attn.v_proj.biases,
layers.19.self_attn.v_proj.scales,
layers.19.self_attn.v_proj.weight,
layers.2.input_layernorm.weight,
layers.2.mlp.down_proj.biases,
layers.2.mlp.down_proj.scales,
layers.2.mlp.down_proj.weight,
layers.2.mlp.gate_proj.biases,
layers.2.mlp.gate_proj.scales,
layers.2.mlp.gate_proj.weight,
layers.2.mlp.up_proj.biases,
layers.2.mlp.up_proj.scales,
layers.2.mlp.up_proj.weight,
layers.2.post_attention_layernorm.weight,
layers.2.post_feedforward_layernorm.weight,
layers.2.pre_feedforward_layernorm.weight,
layers.2.self_attn.gate_proj.biases,
layers.2.self_attn.gate_proj.scales,
layers.2.self_attn.gate_proj.weight,
layers.2.self_attn.k_proj.biases,
layers.2.self_attn.k_proj.scales,
layers.2.self_attn.k_proj.weight,
layers.2.self_attn.o_proj.biases,
layers.2.self_attn.o_proj.scales,
layers.2.self_attn.o_proj.weight,
layers.2.self_attn.q_proj.biases,
layers.2.self_attn.q_proj.scales,
layers.2.self_attn.q_proj.weight,
layers.2.self_attn.v_proj.biases,
layers.2.self_attn.v_proj.scales,
layers.2.self_attn.v_proj.weight,
layers.20.input_layernorm.weight,
layers.20.mlp.down_proj.biases,
layers.20.mlp.down_proj.scales,
layers.20.mlp.down_proj.weight,
layers.20.mlp.gate_proj.biases,
layers.20.mlp.gate_proj.scales,
layers.20.mlp.gate_proj.weight,
layers.20.mlp.up_proj.biases,
layers.20.mlp.up_proj.scales,
layers.20.mlp.up_proj.weight,
layers.20.post_attention_layernorm.weight,
layers.20.post_feedforward_layernorm.weight,
layers.20.pre_feedforward_layernorm.weight,
layers.20.self_attn.gate_proj.biases,
layers.20.self_attn.gate_proj.scales,
layers.20.self_attn.gate_proj.weight,
layers.20.self_attn.k_proj.biases,
layers.20.self_attn.k_proj.scales,
layers.20.self_attn.k_proj.weight,
layers.20.self_attn.o_proj.biases,
layers.20.self_attn.o_proj.scales,
layers.20.self_attn.o_proj.weight,
layers.20.self_attn.q_proj.biases,
layers.20.self_attn.q_proj.scales,
layers.20.self_attn.q_proj.weight,
layers.20.self_attn.v_proj.biases,
layers.20.self_attn.v_proj.scales,
layers.20.self_attn.v_proj.weight,
layers.21.input_layernorm.weight,
layers.21.mlp.down_proj.biases,
layers.21.mlp.down_proj.scales,
layers.21.mlp.down_proj.weight,
layers.21.mlp.gate_proj.biases,
layers.21.mlp.gate_proj.scales,
layers.21.mlp.gate_proj.weight,
layers.21.mlp.up_proj.biases,
layers.21.mlp.up_proj.scales,
layers.21.mlp.up_proj.weight,
layers.21.post_attention_layernorm.weight,
layers.21.post_feedforward_layernorm.weight,
layers.21.pre_feedforward_layernorm.weight,
layers.21.self_attn.gate_proj.biases,
layers.21.self_attn.gate_proj.scales,
layers.21.self_attn.gate_proj.weight,
layers.21.self_attn.k_proj.biases,
layers.21.self_attn.k_proj.scales,
layers.21.self_attn.k_proj.weight,
layers.21.self_attn.o_proj.biases,
layers.21.self_attn.o_proj.scales,
layers.21.self_attn.o_proj.weight,
layers.21.self_attn.q_proj.biases,
layers.21.self_attn.q_proj.scales,
layers.21.self_attn.q_proj.weight,
layers.21.self_attn.v_proj.biases,
layers.21.self_attn.v_proj.scales,
layers.21.self_attn.v_proj.weight,
layers.22.input_layernorm.weight,
layers.22.mlp.down_proj.biases,
layers.22.mlp.down_proj.scales,
layers.22.mlp.down_proj.weight,
layers.22.mlp.gate_proj.biases,
layers.22.mlp.gate_proj.scales,
layers.22.mlp.gate_proj.weight,
layers.22.mlp.up_proj.biases,
layers.22.mlp.up_proj.scales,
layers.22.mlp.up_proj.weight,
layers.22.post_attention_layernorm.weight,
layers.22.post_feedforward_layernorm.weight,
layers.22.pre_feedforward_layernorm.weight,
layers.22.self_attn.gate_proj.biases,
layers.22.self_attn.gate_proj.scales,
layers.22.self_attn.gate_proj.weight,
layers.22.self_attn.k_proj.biases,
layers.22.self_attn.k_proj.scales,
layers.22.self_attn.k_proj.weight,
layers.22.self_attn.o_proj.biases,
layers.22.self_attn.o_proj.scales,
layers.22.self_attn.o_proj.weight,
layers.22.self_attn.q_proj.biases,
layers.22.self_attn.q_proj.scales,
layers.22.self_attn.q_proj.weight,
layers.22.self_attn.v_proj.biases,
layers.22.self_attn.v_proj.scales,
layers.22.self_attn.v_proj.weight,
layers.23.input_layernorm.weight,
layers.23.mlp.down_proj.biases,
layers.23.mlp.down_proj.scales,
layers.23.mlp.down_proj.weight,
layers.23.mlp.gate_proj.biases,
layers.23.mlp.gate_proj.scales,
layers.23.mlp.gate_proj.weight,
layers.23.mlp.up_proj.biases,
layers.23.mlp.up_proj.scales,
layers.23.mlp.up_proj.weight,
layers.23.post_attention_layernorm.weight,
layers.23.post_feedforward_layernorm.weight,
layers.23.pre_feedforward_layernorm.weight,
layers.23.self_attn.gate_proj.biases,
layers.23.self_attn.gate_proj.scales,
layers.23.self_attn.gate_proj.weight,
layers.23.self_attn.k_proj.biases,
layers.23.self_attn.k_proj.scales,
layers.23.self_attn.k_proj.weight,
layers.23.self_attn.o_proj.biases,
layers.23.self_attn.o_proj.scales,
layers.23.self_attn.o_proj.weight,
layers.23.self_attn.q_proj.biases,
layers.23.self_attn.q_proj.scales,
layers.23.self_attn.q_proj.weight,
layers.23.self_attn.v_proj.biases,
layers.23.self_attn.v_proj.scales,
layers.23.self_attn.v_proj.weight,
layers.24.input_layernorm.weight,
layers.24.mlp.down_proj.biases,
layers.24.mlp.down_proj.scales,
layers.24.mlp.down_proj.weight,
layers.24.mlp.gate_proj.biases,
layers.24.mlp.gate_proj.scales,
layers.24.mlp.gate_proj.weight,
layers.24.mlp.up_proj.biases,
layers.24.mlp.up_proj.scales,
layers.24.mlp.up_proj.weight,
layers.24.post_attention_layernorm.weight,
layers.24.post_feedforward_layernorm.weight,
layers.24.pre_feedforward_layernorm.weight,
layers.24.self_attn.gate_proj.biases,
layers.24.self_attn.gate_proj.scales,
layers.24.self_attn.gate_proj.weight,
layers.24.self_attn.k_proj.biases,
layers.24.self_attn.k_proj.scales,
layers.24.self_attn.k_proj.weight,
layers.24.self_attn.o_proj.biases,
layers.24.self_attn.o_proj.scales,
layers.24.self_attn.o_proj.weight,
layers.24.self_attn.q_proj.biases,
layers.24.self_attn.q_proj.scales,
layers.24.self_attn.q_proj.weight,
layers.24.self_attn.v_proj.biases,
layers.24.self_attn.v_proj.scales,
layers.24.self_attn.v_proj.weight,
layers.25.input_layernorm.weight,
layers.25.mlp.down_proj.biases,
layers.25.mlp.down_proj.scales,
layers.25.mlp.down_proj.weight,
layers.25.mlp.gate_proj.biases,
layers.25.mlp.gate_proj.scales,
layers.25.mlp.gate_proj.weight,
layers.25.mlp.up_proj.biases,
layers.25.mlp.up_proj.scales,
layers.25.mlp.up_proj.weight,
layers.25.post_attention_layernorm.weight,
layers.25.post_feedforward_layernorm.weight,
layers.25.pre_feedforward_layernorm.weight,
layers.25.self_attn.gate_proj.biases,
layers.25.self_attn.gate_proj.scales,
layers.25.self_attn.gate_proj.weight,
layers.25.self_attn.k_proj.biases,
layers.25.self_attn.k_proj.scales,
layers.25.self_attn.k_proj.weight,
layers.25.self_attn.o_proj.biases,
layers.25.self_attn.o_proj.scales,
layers.25.self_attn.o_proj.weight,
layers.25.self_attn.q_proj.biases,
layers.25.self_attn.q_proj.scales,
layers.25.self_attn.q_proj.weight,
layers.25.self_attn.v_proj.biases,
layers.25.self_attn.v_proj.scales,
layers.25.self_attn.v_proj.weight,
layers.26.input_layernorm.weight,
layers.26.mlp.down_proj.biases,
layers.26.mlp.down_proj.scales,
layers.26.mlp.down_proj.weight,
layers.26.mlp.gate_proj.biases,
layers.26.mlp.gate_proj.scales,
layers.26.mlp.gate_proj.weight,
layers.26.mlp.up_proj.biases,
layers.26.mlp.up_proj.scales,
layers.26.mlp.up_proj.weight,
layers.26.post_attention_layernorm.weight,
layers.26.post_feedforward_layernorm.weight,
layers.26.pre_feedforward_layernorm.weight,
layers.26.self_attn.gate_proj.biases,
layers.26.self_attn.gate_proj.scales,
layers.26.self_attn.gate_proj.weight,
layers.26.self_attn.k_proj.biases,
layers.26.self_attn.k_proj.scales,
layers.26.self_attn.k_proj.weight,
layers.26.self_attn.o_proj.biases,
layers.26.self_attn.o_proj.scales,
layers.26.self_attn.o_proj.weight,
layers.26.self_attn.q_proj.biases,
layers.26.self_attn.q_proj.scales,
layers.26.self_attn.q_proj.weight,
layers.26.self_attn.v_proj.biases,
layers.26.self_attn.v_proj.scales,
layers.26.self_attn.v_proj.weight,
layers.27.input_layernorm.weight,
layers.27.mlp.down_proj.biases,
layers.27.mlp.down_proj.scales,
layers.27.mlp.down_proj.weight,
layers.27.mlp.gate_proj.biases,
layers.27.mlp.gate_proj.scales,
layers.27.mlp.gate_proj.weight,
layers.27.mlp.up_proj.biases,
layers.27.mlp.up_proj.scales,
layers.27.mlp.up_proj.weight,
layers.27.post_attention_layernorm.weight,
layers.27.post_feedforward_layernorm.weight,
layers.27.pre_feedforward_layernorm.weight,
layers.27.self_attn.gate_proj.biases,
layers.27.self_attn.gate_proj.scales,
layers.27.self_attn.gate_proj.weight,
layers.27.self_attn.k_proj.biases,
layers.27.self_attn.k_proj.scales,
layers.27.self_attn.k_proj.weight,
layers.27.self_attn.o_proj.biases,
layers.27.self_attn.o_proj.scales,
layers.27.self_attn.o_proj.weight,
layers.27.self_attn.q_proj.biases,
layers.27.self_attn.q_proj.scales,
layers.27.self_attn.q_proj.weight,
layers.27.self_attn.v_proj.biases,
layers.27.self_attn.v_proj.scales,
layers.27.self_attn.v_proj.weight,
layers.28.input_layernorm.weight,
layers.28.mlp.down_proj.biases,
layers.28.mlp.down_proj.scales,
layers.28.mlp.down_proj.weight,
layers.28.mlp.gate_proj.biases,
layers.28.mlp.gate_proj.scales,
layers.28.mlp.gate_proj.weight,
layers.28.mlp.up_proj.biases,
layers.28.mlp.up_proj.scales,
layers.28.mlp.up_proj.weight,
layers.28.post_attention_layernorm.weight,
layers.28.post_feedforward_layernorm.weight,
layers.28.pre_feedforward_layernorm.weight,
layers.28.self_attn.gate_proj.biases,
layers.28.self_attn.gate_proj.scales,
layers.28.self_attn.gate_proj.weight,
layers.28.self_attn.k_proj.biases,
layers.28.self_attn.k_proj.scales,
layers.28.self_attn.k_proj.weight,
layers.28.self_attn.o_proj.biases,
layers.28.self_attn.o_proj.scales,
layers.28.self_attn.o_proj.weight,
layers.28.self_attn.q_proj.biases,
layers.28.self_attn.q_proj.scales,
layers.28.self_attn.q_proj.weight,
layers.28.self_attn.v_proj.biases,
layers.28.self_attn.v_proj.scales,
layers.28.self_attn.v_proj.weight,
layers.29.input_layernorm.weight,
layers.29.mlp.down_proj.biases,
layers.29.mlp.down_proj.scales,
layers.29.mlp.down_proj.weight,
layers.29.mlp.gate_proj.biases,
layers.29.mlp.gate_proj.scales,
layers.29.mlp.gate_proj.weight,
layers.29.mlp.up_proj.biases,
layers.29.mlp.up_proj.scales,
layers.29.mlp.up_proj.weight,
layers.29.post_attention_layernorm.weight,
layers.29.post_feedforward_layernorm.weight,
layers.29.pre_feedforward_layernorm.weight,
layers.29.self_attn.gate_proj.biases,
layers.29.self_attn.gate_proj.scales,
layers.29.self_attn.gate_proj.weight,
layers.29.self_attn.k_proj.biases,
layers.29.self_attn.k_proj.scales,
layers.29.self_attn.k_proj.weight,
layers.29.self_attn.o_proj.biases,
layers.29.self_attn.o_proj.scales,
layers.29.self_attn.o_proj.weight,
layers.29.self_attn.q_proj.biases,
layers.29.self_attn.q_proj.scales,
layers.29.self_attn.q_proj.weight,
layers.29.self_attn.v_proj.biases,
layers.29.self_attn.v_proj.scales,
layers.29.self_attn.v_proj.weight,
layers.3.input_layernorm.weight,
layers.3.mlp.down_proj.biases,
layers.3.mlp.down_proj.scales,
layers.3.mlp.down_proj.weight,
layers.3.mlp.gate_proj.biases,
layers.3.mlp.gate_proj.scales,
layers.3.mlp.gate_proj.weight,
layers.3.mlp.up_proj.biases,
layers.3.mlp.up_proj.scales,
layers.3.mlp.up_proj.weight,
layers.3.post_attention_layernorm.weight,
layers.3.post_feedforward_layernorm.weight,
layers.3.pre_feedforward_layernorm.weight,
layers.3.self_attn.gate_proj.biases,
layers.3.self_attn.gate_proj.scales,
layers.3.self_attn.gate_proj.weight,
layers.3.self_attn.k_proj.biases,
layers.3.self_attn.k_proj.scales,
layers.3.self_attn.k_proj.weight,
layers.3.self_attn.o_proj.biases,
layers.3.self_attn.o_proj.scales,
layers.3.self_attn.o_proj.weight,
layers.3.self_attn.q_proj.biases,
layers.3.self_attn.q_proj.scales,
layers.3.self_attn.q_proj.weight,
layers.3.self_attn.v_proj.biases,
layers.3.self_attn.v_proj.scales,
layers.3.self_attn.v_proj.weight,
layers.30.input_layernorm.weight,
layers.30.mlp.down_proj.biases,
layers.30.mlp.down_proj.scales,
layers.30.mlp.down_proj.weight,
layers.30.mlp.gate_proj.biases,
layers.30.mlp.gate_proj.scales,
layers.30.mlp.gate_proj.weight,
layers.30.mlp.up_proj.biases,
layers.30.mlp.up_proj.scales,
layers.30.mlp.up_proj.weight,
layers.30.post_attention_layernorm.weight,
layers.30.post_feedforward_layernorm.weight,
layers.30.pre_feedforward_layernorm.weight,
layers.30.self_attn.gate_proj.biases,
layers.30.self_attn.gate_proj.scales,
layers.30.self_attn.gate_proj.weight,
layers.30.self_attn.k_proj.biases,
layers.30.self_attn.k_proj.scales,
layers.30.self_attn.k_proj.weight,
layers.30.self_attn.o_proj.biases,
layers.30.self_attn.o_proj.scales,
layers.30.self_attn.o_proj.weight,
layers.30.self_attn.q_proj.biases,
layers.30.self_attn.q_proj.scales,
layers.30.self_attn.q_proj.weight,
layers.30.self_attn.v_proj.biases,
layers.30.self_attn.v_proj.scales,
layers.30.self_attn.v_proj.weight,
layers.31.input_layernorm.weight,
layers.31.mlp.down_proj.biases,
layers.31.mlp.down_proj.scales,
layers.31.mlp.down_proj.weight,
layers.31.mlp.gate_proj.biases,
layers.31.mlp.gate_proj.scales,
layers.31.mlp.gate_proj.weight,
layers.31.mlp.up_proj.biases,
layers.31.mlp.up_proj.scales,
layers.31.mlp.up_proj.weight,
layers.31.post_attention_layernorm.weight,
layers.31.post_feedforward_layernorm.weight,
layers.31.pre_feedforward_layernorm.weight,
layers.31.self_attn.gate_proj.biases,
layers.31.self_attn.gate_proj.scales,
layers.31.self_attn.gate_proj.weight,
layers.31.self_attn.k_proj.biases,
layers.31.self_attn.k_proj.scales,
layers.31.self_attn.k_proj.weight,
layers.31.self_attn.o_proj.biases,
layers.31.self_attn.o_proj.scales,
layers.31.self_attn.o_proj.weight,
layers.31.self_attn.q_proj.biases,
layers.31.self_attn.q_proj.scales,
layers.31.self_attn.q_proj.weight,
layers.31.self_attn.v_proj.biases,
layers.31.self_attn.v_proj.scales,
layers.31.self_attn.v_proj.weight,
layers.32.input_layernorm.weight,
layers.32.mlp.down_proj.biases,
layers.32.mlp.down_proj.scales,
layers.32.mlp.down_proj.weight,
layers.32.mlp.gate_proj.biases,
layers.32.mlp.gate_proj.scales,
layers.32.mlp.gate_proj.weight,
layers.32.mlp.up_proj.biases,
layers.32.mlp.up_proj.scales,
layers.32.mlp.up_proj.weight,
layers.32.post_attention_layernorm.weight,
layers.32.post_feedforward_layernorm.weight,
layers.32.pre_feedforward_layernorm.weight,
layers.32.self_attn.gate_proj.biases,
layers.32.self_attn.gate_proj.scales,
layers.32.self_attn.gate_proj.weight,
layers.32.self_attn.k_proj.biases,
layers.32.self_attn.k_proj.scales,
layers.32.self_attn.k_proj.weight,
layers.32.self_attn.o_proj.biases,
layers.32.self_attn.o_proj.scales,
layers.32.self_attn.o_proj.weight,
layers.32.self_attn.q_proj.biases,
layers.32.self_attn.q_proj.scales,
layers.32.self_attn.q_proj.weight,
layers.32.self_attn.v_proj.biases,
layers.32.self_attn.v_proj.scales,
layers.32.self_attn.v_proj.weight,
layers.33.input_layernorm.weight,
layers.33.mlp.down_proj.biases,
layers.33.mlp.down_proj.scales,
layers.33.mlp.down_proj.weight,
layers.33.mlp.gate_proj.biases,
layers.33.mlp.gate_proj.scales,
layers.33.mlp.gate_proj.weight,
layers.33.mlp.up_proj.biases,
layers.33.mlp.up_proj.scales,
layers.33.mlp.up_proj.weight,
layers.33.post_attention_layernorm.weight,
layers.33.post_feedforward_layernorm.weight,
layers.33.pre_feedforward_layernorm.weight,
layers.33.self_attn.gate_proj.biases,
layers.33.self_attn.gate_proj.scales,
layers.33.self_attn.gate_proj.weight,
layers.33.self_attn.k_proj.biases,
layers.33.self_attn.k_proj.scales,
layers.33.self_attn.k_proj.weight,
layers.33.self_attn.o_proj.biases,
layers.33.self_attn.o_proj.scales,
layers.33.self_attn.o_proj.weight,
layers.33.self_attn.q_proj.biases,
layers.33.self_attn.q_proj.scales,
layers.33.self_attn.q_proj.weight,
layers.33.self_attn.v_proj.biases,
layers.33.self_attn.v_proj.scales,
layers.33.self_attn.v_proj.weight,
layers.34.input_layernorm.weight,
layers.34.mlp.down_proj.biases,
layers.34.mlp.down_proj.scales,
layers.34.mlp.down_proj.weight,
layers.34.mlp.gate_proj.biases,
layers.34.mlp.gate_proj.scales,
layers.34.mlp.gate_proj.weight,
layers.34.mlp.up_proj.biases,
layers.34.mlp.up_proj.scales,
layers.34.mlp.up_proj.weight,
layers.34.post_attention_layernorm.weight,
layers.34.post_feedforward_layernorm.weight,
layers.34.pre_feedforward_layernorm.weight,
layers.34.self_attn.gate_proj.biases,
layers.34.self_attn.gate_proj.scales,
layers.34.self_attn.gate_proj.weight,
layers.34.self_attn.k_proj.biases,
layers.34.self_attn.k_proj.scales,
layers.34.self_attn.k_proj.weight,
layers.34.self_attn.o_proj.biases,
layers.34.self_attn.o_proj.scales,
layers.34.self_attn.o_proj.weight,
layers.34.self_attn.q_proj.biases,
layers.34.self_attn.q_proj.scales,
layers.34.self_attn.q_proj.weight,
layers.34.self_attn.v_proj.biases,
layers.34.self_attn.v_proj.scales,
layers.34.self_attn.v_proj.weight,
layers.35.input_layernorm.weight,
layers.35.mlp.down_proj.biases,
layers.35.mlp.down_proj.scales,
layers.35.mlp.down_proj.weight,
layers.35.mlp.gate_proj.biases,
layers.35.mlp.gate_proj.scales,
layers.35.mlp.gate_proj.weight,
layers.35.mlp.up_proj.biases,
layers.35.mlp.up_proj.scales,
layers.35.mlp.up_proj.weight,
layers.35.post_attention_layernorm.weight,
layers.35.post_feedforward_layernorm.weight,
layers.35.pre_feedforward_layernorm.weight,
layers.35.self_attn.gate_proj.biases,
layers.35.self_attn.gate_proj.scales,
layers.35.self_attn.gate_proj.weight,
layers.35.self_attn.k_proj.biases,
layers.35.self_attn.k_proj.scales,
layers.35.self_attn.k_proj.weight,
layers.35.self_attn.o_proj.biases,
layers.35.self_attn.o_proj.scales,
layers.35.self_attn.o_proj.weight,
layers.35.self_attn.q_proj.biases,
layers.35.self_attn.q_proj.scales,
layers.35.self_attn.q_proj.weight,
layers.35.self_attn.v_proj.biases,
layers.35.self_attn.v_proj.scales,
layers.35.self_attn.v_proj.weight,
layers.36.input_layernorm.weight,
layers.36.mlp.down_proj.biases,
layers.36.mlp.down_proj.scales,
layers.36.mlp.down_proj.weight,
layers.36.mlp.gate_proj.biases,
layers.36.mlp.gate_proj.scales,
layers.36.mlp.gate_proj.weight,
layers.36.mlp.up_proj.biases,
layers.36.mlp.up_proj.scales,
layers.36.mlp.up_proj.weight,
layers.36.post_attention_layernorm.weight,
layers.36.post_feedforward_layernorm.weight,
layers.36.pre_feedforward_layernorm.weight,
layers.36.self_attn.gate_proj.biases,
layers.36.self_attn.gate_proj.scales,
layers.36.self_attn.gate_proj.weight,
layers.36.self_attn.k_proj.biases,
layers.36.self_attn.k_proj.scales,
layers.36.self_attn.k_proj.weight,
layers.36.self_attn.o_proj.biases,
layers.36.self_attn.o_proj.scales,
layers.36.self_attn.o_proj.weight,
layers.36.self_attn.q_proj.biases,
layers.36.self_attn.q_proj.scales,
layers.36.self_attn.q_proj.weight,
layers.36.self_attn.v_proj.biases,
layers.36.self_attn.v_proj.scales,
layers.36.self_attn.v_proj.weight,
layers.37.input_layernorm.weight,
layers.37.mlp.down_proj.biases,
layers.37.mlp.down_proj.scales,
layers.37.mlp.down_proj.weight,
layers.37.mlp.gate_proj.biases,
layers.37.mlp.gate_proj.scales,
layers.37.mlp.gate_proj.weight,
layers.37.mlp.up_proj.biases,
layers.37.mlp.up_proj.scales,
layers.37.mlp.up_proj.weight,
layers.37.post_attention_layernorm.weight,
layers.37.post_feedforward_layernorm.weight,
layers.37.pre_feedforward_layernorm.weight,
layers.37.self_attn.gate_proj.biases,
layers.37.self_attn.gate_proj.scales,
layers.37.self_attn.gate_proj.weight,
layers.37.self_attn.k_proj.biases,
layers.37.self_attn.k_proj.scales,
layers.37.self_attn.k_proj.weight,
layers.37.self_attn.o_proj.biases,
layers.37.self_attn.o_proj.scales,
layers.37.self_attn.o_proj.weight,
layers.37.self_attn.q_proj.biases,
layers.37.self_attn.q_proj.scales,
layers.37.self_attn.q_proj.weight,
layers.37.self_attn.v_proj.biases,
layers.37.self_attn.v_proj.scales,
layers.37.self_attn.v_proj.weight,
layers.38.input_layernorm.weight,
layers.38.mlp.down_proj.biases,
layers.38.mlp.down_proj.scales,
layers.38.mlp.down_proj.weight,
layers.38.mlp.gate_proj.biases,
layers.38.mlp.gate_proj.scales,
layers.38.mlp.gate_proj.weight,
layers.38.mlp.up_proj.biases,
layers.38.mlp.up_proj.scales,
layers.38.mlp.up_proj.weight,
layers.38.post_attention_layernorm.weight,
layers.38.post_feedforward_layernorm.weight,
layers.38.pre_feedforward_layernorm.weight,
layers.38.self_attn.gate_proj.biases,
layers.38.self_attn.gate_proj.scales,
layers.38.self_attn.gate_proj.weight,
layers.38.self_attn.k_proj.biases,
layers.38.self_attn.k_proj.scales,
layers.38.self_attn.k_proj.weight,
layers.38.self_attn.o_proj.biases,
layers.38.self_attn.o_proj.scales,
layers.38.self_attn.o_proj.weight,
layers.38.self_attn.q_proj.biases,
layers.38.self_attn.q_proj.scales,
layers.38.self_attn.q_proj.weight,
layers.38.self_attn.v_proj.biases,
layers.38.self_attn.v_proj.scales,
layers.38.self_attn.v_proj.weight,
layers.39.input_layernorm.weight,
layers.39.mlp.down_proj.biases,
layers.39.mlp.down_proj.scales,
layers.39.mlp.down_proj.weight,
layers.39.mlp.gate_proj.biases,
layers.39.mlp.gate_proj.scales,
layers.39.mlp.gate_proj.weight,
layers.39.mlp.up_proj.biases,
layers.39.mlp.up_proj.scales,
layers.39.mlp.up_proj.weight,
layers.39.post_attention_layernorm.weight,
layers.39.post_feedforward_layernorm.weight,
layers.39.pre_feedforward_layernorm.weight,
layers.39.self_attn.gate_proj.biases,
layers.39.self_attn.gate_proj.scales,
layers.39.self_attn.gate_proj.weight,
layers.39.self_attn.k_proj.biases,
layers.39.self_attn.k_proj.scales,
layers.39.self_attn.k_proj.weight,
layers.39.self_attn.o_proj.biases,
layers.39.self_attn.o_proj.scales,
layers.39.self_attn.o_proj.weight,
layers.39.self_attn.q_proj.biases,
layers.39.self_attn.q_proj.scales,
layers.39.self_attn.q_proj.weight,
layers.39.self_attn.v_proj.biases,
layers.39.self_attn.v_proj.scales,
layers.39.self_attn.v_proj.weight,
layers.4.input_layernorm.weight,
layers.4.mlp.down_proj.biases,
layers.4.mlp.down_proj.scales,
layers.4.mlp.down_proj.weight,
layers.4.mlp.gate_proj.biases,
layers.4.mlp.gate_proj.scales,
layers.4.mlp.gate_proj.weight,
layers.4.mlp.up_proj.biases,
layers.4.mlp.up_proj.scales,
layers.4.mlp.up_proj.weight,
layers.4.post_attention_layernorm.weight,
layers.4.post_feedforward_layernorm.weight,
layers.4.pre_feedforward_layernorm.weight,
layers.4.self_attn.gate_proj.biases,
layers.4.self_attn.gate_proj.scales,
layers.4.self_attn.gate_proj.weight,
layers.4.self_attn.k_proj.biases,
layers.4.self_attn.k_proj.scales,
layers.4.self_attn.k_proj.weight,
layers.4.self_attn.o_proj.biases,
layers.4.self_attn.o_proj.scales,
layers.4.self_attn.o_proj.weight,
layers.4.self_attn.q_proj.biases,
layers.4.self_attn.q_proj.scales,
layers.4.self_attn.q_proj.weight,
layers.4.self_attn.v_proj.biases,
layers.4.self_attn.v_proj.scales,
layers.4.self_attn.v_proj.weight,
layers.40.input_layernorm.weight,
layers.40.mlp.down_proj.biases,
layers.40.mlp.down_proj.scales,
layers.40.mlp.down_proj.weight,
layers.40.mlp.gate_proj.biases,
layers.40.mlp.gate_proj.scales,
layers.40.mlp.gate_proj.weight,
layers.40.mlp.up_proj.biases,
layers.40.mlp.up_proj.scales,
layers.40.mlp.up_proj.weight,
layers.40.post_attention_layernorm.weight,
layers.40.post_feedforward_layernorm.weight,
layers.40.pre_feedforward_layernorm.weight,
layers.40.self_attn.gate_proj.biases,
layers.40.self_attn.gate_proj.scales,
layers.40.self_attn.gate_proj.weight,
layers.40.self_attn.k_proj.biases,
layers.40.self_attn.k_proj.scales,
layers.40.self_attn.k_proj.weight,
layers.40.self_attn.o_proj.biases,
layers.40.self_attn.o_proj.scales,
layers.40.self_attn.o_proj.weight,
layers.40.self_attn.q_proj.biases,
layers.40.self_attn.q_proj.scales,
layers.40.self_attn.q_proj.weight,
layers.40.self_attn.v_proj.biases,
layers.40.self_attn.v_proj.scales,
layers.40.self_attn.v_proj.weight,
layers.41.input_layernorm.weight,
layers.41.mlp.down_proj.biases,
layers.41.mlp.down_proj.scales,
layers.41.mlp.down_proj.weight,
layers.41.mlp.gate_proj.biases,
layers.41.mlp.gate_proj.scales,
layers.41.mlp.gate_proj.weight,
layers.41.mlp.up_proj.biases,
layers.41.mlp.up_proj.scales,
layers.41.mlp.up_proj.weight,
layers.41.post_attention_layernorm.weight,
layers.41.post_feedforward_layernorm.weight,
layers.41.pre_feedforward_layernorm.weight,
layers.41.self_attn.gate_proj.biases,
layers.41.self_attn.gate_proj.scales,
layers.41.self_attn.gate_proj.weight,
layers.41.self_attn.k_proj.biases,
layers.41.self_attn.k_proj.scales,
layers.41.self_attn.k_proj.weight,
layers.41.self_attn.o_proj.biases,
layers.41.self_attn.o_proj.scales,
layers.41.self_attn.o_proj.weight,
layers.41.self_attn.q_proj.biases,
layers.41.self_attn.q_proj.scales,
layers.41.self_attn.q_proj.weight,
layers.41.self_attn.v_proj.biases,
layers.41.self_attn.v_proj.scales,
layers.41.self_attn.v_proj.weight,
layers.42.input_layernorm.weight,
layers.42.mlp.down_proj.biases,
layers.42.mlp.down_proj.scales,
layers.42.mlp.down_proj.weight,
layers.42.mlp.gate_proj.biases,
layers.42.mlp.gate_proj.scales,
layers.42.mlp.gate_proj.weight,
layers.42.mlp.up_proj.biases,
layers.42.mlp.up_proj.scales,
layers.42.mlp.up_proj.weight,
layers.42.post_attention_layernorm.weight,
layers.42.post_feedforward_layernorm.weight,
layers.42.pre_feedforward_layernorm.weight,
layers.42.self_attn.gate_proj.biases,
layers.42.self_attn.gate_proj.scales,
layers.42.self_attn.gate_proj.weight,
layers.42.self_attn.k_proj.biases,
layers.42.self_attn.k_proj.scales,
layers.42.self_attn.k_proj.weight,
layers.42.self_attn.o_proj.biases,
layers.42.self_attn.o_proj.scales,
layers.42.self_attn.o_proj.weight,
layers.42.self_attn.q_proj.biases,
layers.42.self_attn.q_proj.scales,
layers.42.self_attn.q_proj.weight,
layers.42.self_attn.v_proj.biases,
layers.42.self_attn.v_proj.scales,
layers.42.self_attn.v_proj.weight,
layers.43.input_layernorm.weight,
layers.43.mlp.down_proj.biases,
layers.43.mlp.down_proj.scales,
layers.43.mlp.down_proj.weight,
layers.43.mlp.gate_proj.biases,
layers.43.mlp.gate_proj.scales,
layers.43.mlp.gate_proj.weight,
layers.43.mlp.up_proj.biases,
layers.43.mlp.up_proj.scales,
layers.43.mlp.up_proj.weight,
layers.43.post_attention_layernorm.weight,
layers.43.post_feedforward_layernorm.weight,
layers.43.pre_feedforward_layernorm.weight,
layers.43.self_attn.gate_proj.biases,
layers.43.self_attn.gate_proj.scales,
layers.43.self_attn.gate_proj.weight,
layers.43.self_attn.k_proj.biases,
layers.43.self_attn.k_proj.scales,
layers.43.self_attn.k_proj.weight,
layers.43.self_attn.o_proj.biases,
layers.43.self_attn.o_proj.scales,
layers.43.self_attn.o_proj.weight,
layers.43.self_attn.q_proj.biases,
layers.43.self_attn.q_proj.scales,
layers.43.self_attn.q_proj.weight,
layers.43.self_attn.v_proj.biases,
layers.43.self_attn.v_proj.scales,
layers.43.self_attn.v_proj.weight,
layers.44.input_layernorm.weight,
layers.44.mlp.down_proj.biases,
layers.44.mlp.down_proj.scales,
layers.44.mlp.down_proj.weight,
layers.44.mlp.gate_proj.biases,
layers.44.mlp.gate_proj.scales,
layers.44.mlp.gate_proj.weight,
layers.44.mlp.up_proj.biases,
layers.44.mlp.up_proj.scales,
layers.44.mlp.up_proj.weight,
layers.44.post_attention_layernorm.weight,
layers.44.post_feedforward_layernorm.weight,
layers.44.pre_feedforward_layernorm.weight,
layers.44.self_attn.gate_proj.biases,
layers.44.self_attn.gate_proj.scales,
layers.44.self_attn.gate_proj.weight,
layers.44.self_attn.k_proj.biases,
layers.44.self_attn.k_proj.scales,
layers.44.self_attn.k_proj.weight,
layers.44.self_attn.o_proj.biases,
layers.44.self_attn.o_proj.scales,
layers.44.self_attn.o_proj.weight,
layers.44.self_attn.q_proj.biases,
layers.44.self_attn.q_proj.scales,
layers.44.self_attn.q_proj.weight,
layers.44.self_attn.v_proj.biases,
layers.44.self_attn.v_proj.scales,
layers.44.self_attn.v_proj.weight,
layers.45.input_layernorm.weight,
layers.45.mlp.down_proj.biases,
layers.45.mlp.down_proj.scales,
layers.45.mlp.down_proj.weight,
layers.45.mlp.gate_proj.biases,
layers.45.mlp.gate_proj.scales,
layers.45.mlp.gate_proj.weight,
layers.45.mlp.up_proj.biases,
layers.45.mlp.up_proj.scales,
layers.45.mlp.up_proj.weight,
layers.45.post_attention_layernorm.weight,
layers.45.post_feedforward_layernorm.weight,
layers.45.pre_feedforward_layernorm.weight,
layers.45.self_attn.gate_proj.biases,
layers.45.self_attn.gate_proj.scales,
layers.45.self_attn.gate_proj.weight,
layers.45.self_attn.k_proj.biases,
layers.45.self_attn.k_proj.scales,
layers.45.self_attn.k_proj.weight,
layers.45.self_attn.o_proj.biases,
layers.45.self_attn.o_proj.scales,
layers.45.self_attn.o_proj.weight,
layers.45.self_attn.q_proj.biases,
layers.45.self_attn.q_proj.scales,
layers.45.self_attn.q_proj.weight,
layers.45.self_attn.v_proj.biases,
layers.45.self_attn.v_proj.scales,
layers.45.self_attn.v_proj.weight,
layers.46.input_layernorm.weight,
layers.46.mlp.down_proj.biases,
layers.46.mlp.down_proj.scales,
layers.46.mlp.down_proj.weight,
layers.46.mlp.gate_proj.biases,
layers.46.mlp.gate_proj.scales,
layers.46.mlp.gate_proj.weight,
layers.46.mlp.up_proj.biases,
layers.46.mlp.up_proj.scales,
layers.46.mlp.up_proj.weight,
layers.46.post_attention_layernorm.weight,
layers.46.post_feedforward_layernorm.weight,
layers.46.pre_feedforward_layernorm.weight,
layers.46.self_attn.gate_proj.biases,
layers.46.self_attn.gate_proj.scales,
layers.46.self_attn.gate_proj.weight,
layers.46.self_attn.k_proj.biases,
layers.46.self_attn.k_proj.scales,
layers.46.self_attn.k_proj.weight,
layers.46.self_attn.o_proj.biases,
layers.46.self_attn.o_proj.scales,
layers.46.self_attn.o_proj.weight,
layers.46.self_attn.q_proj.biases,
layers.46.self_attn.q_proj.scales,
layers.46.self_attn.q_proj.weight,
layers.46.self_attn.v_proj.biases,
layers.46.self_attn.v_proj.scales,
layers.46.self_attn.v_proj.weight,
layers.47.input_layernorm.weight,
layers.47.mlp.down_proj.biases,
layers.47.mlp.down_proj.scales,
layers.47.mlp.down_proj.weight,
layers.47.mlp.gate_proj.biases,
layers.47.mlp.gate_proj.scales,
layers.47.mlp.gate_proj.weight,
layers.47.mlp.up_proj.biases,
layers.47.mlp.up_proj.scales,
layers.47.mlp.up_proj.weight,
layers.47.post_attention_layernorm.weight,
layers.47.post_feedforward_layernorm.weight,
layers.47.pre_feedforward_layernorm.weight,
layers.47.self_attn.gate_proj.biases,
layers.47.self_attn.gate_proj.scales,
layers.47.self_attn.gate_proj.weight,
layers.47.self_attn.k_proj.biases,
layers.47.self_attn.k_proj.scales,
layers.47.self_attn.k_proj.weight,
layers.47.self_attn.o_proj.biases,
layers.47.self_attn.o_proj.scales,
layers.47.self_attn.o_proj.weight,
layers.47.self_attn.q_proj.biases,
layers.47.self_attn.q_proj.scales,
layers.47.self_attn.q_proj.weight,
layers.47.self_attn.v_proj.biases,
layers.47.self_attn.v_proj.scales,
layers.47.self_attn.v_proj.weight,
layers.48.input_layernorm.weight,
layers.48.mlp.down_proj.biases,
layers.48.mlp.down_proj.scales,
layers.48.mlp.down_proj.weight,
layers.48.mlp.gate_proj.biases,
layers.48.mlp.gate_proj.scales,
layers.48.mlp.gate_proj.weight,
layers.48.mlp.up_proj.biases,
layers.48.mlp.up_proj.scales,
layers.48.mlp.up_proj.weight,
layers.48.post_attention_layernorm.weight,
layers.48.post_feedforward_layernorm.weight,
layers.48.pre_feedforward_layernorm.weight,
layers.48.self_attn.gate_proj.biases,
layers.48.self_attn.gate_proj.scales,
layers.48.self_attn.gate_proj.weight,
layers.48.self_attn.k_proj.biases,
layers.48.self_attn.k_proj.scales,
layers.48.self_attn.k_proj.weight,
layers.48.self_attn.o_proj.biases,
layers.48.self_attn.o_proj.scales,
layers.48.self_attn.o_proj.weight,
layers.48.self_attn.q_proj.biases,
layers.48.self_attn.q_proj.scales,
layers.48.self_attn.q_proj.weight,
layers.48.self_attn.v_proj.biases,
layers.48.self_attn.v_proj.scales,
layers.48.self_attn.v_proj.weight,
layers.49.input_layernorm.weight,
layers.49.mlp.down_proj.biases,
layers.49.mlp.down_proj.scales,
layers.49.mlp.down_proj.weight,
layers.49.mlp.gate_proj.biases,
layers.49.mlp.gate_proj.scales,
layers.49.mlp.gate_proj.weight,
layers.49.mlp.up_proj.biases,
layers.49.mlp.up_proj.scales,
layers.49.mlp.up_proj.weight,
layers.49.post_attention_layernorm.weight,
layers.49.post_feedforward_layernorm.weight,
layers.49.pre_feedforward_layernorm.weight,
layers.49.self_attn.gate_proj.biases,
layers.49.self_attn.gate_proj.scales,
layers.49.self_attn.gate_proj.weight,
layers.49.self_attn.k_proj.biases,
layers.49.self_attn.k_proj.scales,
layers.49.self_attn.k_proj.weight,
layers.49.self_attn.o_proj.biases,
layers.49.self_attn.o_proj.scales,
layers.49.self_attn.o_proj.weight,
layers.49.self_attn.q_proj.biases,
layers.49.self_attn.q_proj.scales,
layers.49.self_attn.q_proj.weight,
layers.49.self_attn.v_proj.biases,
layers.49.self_attn.v_proj.scales,
layers.49.self_attn.v_proj.weight,
layers.5.input_layernorm.weight,
layers.5.mlp.down_proj.biases,
layers.5.mlp.down_proj.scales,
layers.5.mlp.down_proj.weight,
layers.5.mlp.gate_proj.biases,
layers.5.mlp.gate_proj.scales,
layers.5.mlp.gate_proj.weight,
layers.5.mlp.up_proj.biases,
layers.5.mlp.up_proj.scales,
layers.5.mlp.up_proj.weight,
layers.5.post_attention_layernorm.weight,
layers.5.post_feedforward_layernorm.weight,
layers.5.pre_feedforward_layernorm.weight,
layers.5.self_attn.gate_proj.biases,
layers.5.self_attn.gate_proj.scales,
layers.5.self_attn.gate_proj.weight,
layers.5.self_attn.k_proj.biases,
layers.5.self_attn.k_proj.scales,
layers.5.self_attn.k_proj.weight,
layers.5.self_attn.o_proj.biases,
layers.5.self_attn.o_proj.scales,
layers.5.self_attn.o_proj.weight,
layers.5.self_attn.q_proj.biases,
layers.5.self_attn.q_proj.scales,
layers.5.self_attn.q_proj.weight,
layers.5.self_attn.v_proj.biases,
layers.5.self_attn.v_proj.scales,
layers.5.self_attn.v_proj.weight,
layers.50.input_layernorm.weight,
layers.50.mlp.down_proj.biases,
layers.50.mlp.down_proj.scales,
layers.50.mlp.down_proj.weight,
layers.50.mlp.gate_proj.biases,
layers.50.mlp.gate_proj.scales,
layers.50.mlp.gate_proj.weight,
layers.50.mlp.up_proj.biases,
layers.50.mlp.up_proj.scales,
layers.50.mlp.up_proj.weight,
layers.50.post_attention_layernorm.weight,
layers.50.post_feedforward_layernorm.weight,
layers.50.pre_feedforward_layernorm.weight,
layers.50.self_attn.gate_proj.biases,
layers.50.self_attn.gate_proj.scales,
layers.50.self_attn.gate_proj.weight,
layers.50.self_attn.k_proj.biases,
layers.50.self_attn.k_proj.scales,
layers.50.self_attn.k_proj.weight,
layers.50.self_attn.o_proj.biases,
layers.50.self_attn.o_proj.scales,
layers.50.self_attn.o_proj.weight,
layers.50.self_attn.q_proj.biases,
layers.50.self_attn.q_proj.scales,
layers.50.self_attn.q_proj.weight,
layers.50.self_attn.v_proj.biases,
layers.50.self_attn.v_proj.scales,
layers.50.self_attn.v_proj.weight,
layers.51.input_layernorm.weight,
layers.51.mlp.down_proj.biases,
layers.51.mlp.down_proj.scales,
layers.51.mlp.down_proj.weight,
layers.51.mlp.gate_proj.biases,
layers.51.mlp.gate_proj.scales,
layers.51.mlp.gate_proj.weight,
layers.51.mlp.up_proj.biases,
layers.51.mlp.up_proj.scales,
layers.51.mlp.up_proj.weight,
layers.51.post_attention_layernorm.weight,
layers.51.post_feedforward_layernorm.weight,
layers.51.pre_feedforward_layernorm.weight,
layers.51.self_attn.gate_proj.biases,
layers.51.self_attn.gate_proj.scales,
layers.51.self_attn.gate_proj.weight,
layers.51.self_attn.k_proj.biases,
layers.51.self_attn.k_proj.scales,
layers.51.self_attn.k_proj.weight,
layers.51.self_attn.o_proj.biases,
layers.51.self_attn.o_proj.scales,
layers.51.self_attn.o_proj.weight,
layers.51.self_attn.q_proj.biases,
layers.51.self_attn.q_proj.scales,
layers.51.self_attn.q_proj.weight,
layers.51.self_attn.v_proj.biases,
layers.51.self_attn.v_proj.scales,
layers.51.self_attn.v_proj.weight,
layers.6.input_layernorm.weight,
layers.6.mlp.down_proj.biases,
layers.6.mlp.down_proj.scales,
layers.6.mlp.down_proj.weight,
layers.6.mlp.gate_proj.biases,
layers.6.mlp.gate_proj.scales,
layers.6.mlp.gate_proj.weight,
layers.6.mlp.up_proj.biases,
layers.6.mlp.up_proj.scales,
layers.6.mlp.up_proj.weight,
layers.6.post_attention_layernorm.weight,
layers.6.post_feedforward_layernorm.weight,
layers.6.pre_feedforward_layernorm.weight,
layers.6.self_attn.gate_proj.biases,
layers.6.self_attn.gate_proj.scales,
layers.6.self_attn.gate_proj.weight,
layers.6.self_attn.k_proj.biases,
layers.6.self_attn.k_proj.scales,
layers.6.self_attn.k_proj.weight,
layers.6.self_attn.o_proj.biases,
layers.6.self_attn.o_proj.scales,
layers.6.self_attn.o_proj.weight,
layers.6.self_attn.q_proj.biases,
layers.6.self_attn.q_proj.scales,
layers.6.self_attn.q_proj.weight,
layers.6.self_attn.v_proj.biases,
layers.6.self_attn.v_proj.scales,
layers.6.self_attn.v_proj.weight,
layers.7.input_layernorm.weight,
layers.7.mlp.down_proj.biases,
layers.7.mlp.down_proj.scales,
layers.7.mlp.down_proj.weight,
layers.7.mlp.gate_proj.biases,
layers.7.mlp.gate_proj.scales,
layers.7.mlp.gate_proj.weight,
layers.7.mlp.up_proj.biases,
layers.7.mlp.up_proj.scales,
layers.7.mlp.up_proj.weight,
layers.7.post_attention_layernorm.weight,
layers.7.post_feedforward_layernorm.weight,
layers.7.pre_feedforward_layernorm.weight,
layers.7.self_attn.gate_proj.biases,
layers.7.self_attn.gate_proj.scales,
layers.7.self_attn.gate_proj.weight,
layers.7.self_attn.k_proj.biases,
layers.7.self_attn.k_proj.scales,
layers.7.self_attn.k_proj.weight,
layers.7.self_attn.o_proj.biases,
layers.7.self_attn.o_proj.scales,
layers.7.self_attn.o_proj.weight,
layers.7.self_attn.q_proj.biases,
layers.7.self_attn.q_proj.scales,
layers.7.self_attn.q_proj.weight,
layers.7.self_attn.v_proj.biases,
layers.7.self_attn.v_proj.scales,
layers.7.self_attn.v_proj.weight,
layers.8.input_layernorm.weight,
layers.8.mlp.down_proj.biases,
layers.8.mlp.down_proj.scales,
layers.8.mlp.down_proj.weight,
layers.8.mlp.gate_proj.biases,
layers.8.mlp.gate_proj.scales,
layers.8.mlp.gate_proj.weight,
layers.8.mlp.up_proj.biases,
layers.8.mlp.up_proj.scales,
layers.8.mlp.up_proj.weight,
layers.8.post_attention_layernorm.weight,
layers.8.post_feedforward_layernorm.weight,
layers.8.pre_feedforward_layernorm.weight,
layers.8.self_attn.gate_proj.biases,
layers.8.self_attn.gate_proj.scales,
layers.8.self_attn.gate_proj.weight,
layers.8.self_attn.k_proj.biases,
layers.8.self_attn.k_proj.scales,
layers.8.self_attn.k_proj.weight,
layers.8.self_attn.o_proj.biases,
layers.8.self_attn.o_proj.scales,
layers.8.self_attn.o_proj.weight,
layers.8.self_attn.q_proj.biases,
layers.8.self_attn.q_proj.scales,
layers.8.self_attn.q_proj.weight,
layers.8.self_attn.v_proj.biases,
layers.8.self_attn.v_proj.scales,
layers.8.self_attn.v_proj.weight,
layers.9.input_layernorm.weight,
layers.9.mlp.down_proj.biases,
layers.9.mlp.down_proj.scales,
layers.9.mlp.down_proj.weight,
layers.9.mlp.gate_proj.biases,
layers.9.mlp.gate_proj.scales,
layers.9.mlp.gate_proj.weight,
layers.9.mlp.up_proj.biases,
layers.9.mlp.up_proj.scales,
layers.9.mlp.up_proj.weight,
layers.9.post_attention_layernorm.weight,
layers.9.post_feedforward_layernorm.weight,
layers.9.pre_feedforward_layernorm.weight,
layers.9.self_attn.gate_proj.biases,
layers.9.self_attn.gate_proj.scales,
layers.9.self_attn.gate_proj.weight,
layers.9.self_attn.k_proj.biases,
layers.9.self_attn.k_proj.scales,
layers.9.self_attn.k_proj.weight,
layers.9.self_attn.o_proj.biases,
layers.9.self_attn.o_proj.scales,
layers.9.self_attn.o_proj.weight,
layers.9.self_attn.q_proj.biases,
layers.9.self_attn.q_proj.scales,
layers.9.self_attn.q_proj.weight,
layers.9.self_attn.v_proj.biases,
layers.9.self_attn.v_proj.scales,
layers.9.self_attn.v_proj.weight,
norm.weight.

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
ValueError: Model loading failed: Received 1460 parameters not in model: 
embed_tokens.biases,
embed_tokens.scales,
embed_tokens.weight,
layers.0.input_layernorm.weight,
layers.0.mlp.down_proj.biases,
layers.0.mlp.down_proj.scales,
layers.0.mlp.down_proj.weight,
layers.0.mlp.gate_proj.biases,
layers.0.mlp.gate_proj.scales,
layers.0.mlp.gate_proj.weight,
layers.0.mlp.up_proj.biases,
layers.0.mlp.up_proj.scales,
layers.0.mlp.up_proj.weight,
layers.0.post_attention_layernorm.weight,
layers.0.post_feedforward_layernorm.weight,
layers.0.pre_feedforward_layernorm.weight,
layers.0.self_attn.gate_proj.biases,
layers.0.self_attn.gate_proj.scales,
layers.0.self_attn.gate_proj.weight,
layers.0.self_attn.k_proj.biases,
layers.0.self_attn.k_proj.scales,
layers.0.self_attn.k_proj.weight,
layers.0.self_attn.o_proj.biases,
layers.0.self_attn.o_proj.scales,
layers.0.self_attn.o_proj.weight,
layers.0.self_attn.q_proj.biases,
layers.0.self_attn.q_proj.scales,
layers.0.self_attn.q_proj.weight,
layers.0.self_attn.v_proj.biases,
layers.0.self_attn.v_proj.scales,
layers.0.self_attn.v_proj.weight,
layers.1.input_layernorm.weight,
layers.1.mlp.down_proj.biases,
layers.1.mlp.down_proj.scales,
layers.1.mlp.down_proj.weight,
layers.1.mlp.gate_proj.biases,
layers.1.mlp.gate_proj.scales,
layers.1.mlp.gate_proj.weight,
layers.1.mlp.up_proj.biases,
layers.1.mlp.up_proj.scales,
layers.1.mlp.up_proj.weight,
layers.1.post_attention_layernorm.weight,
layers.1.post_feedforward_layernorm.weight,
layers.1.pre_feedforward_layernorm.weight,
layers.1.self_attn.gate_proj.biases,
layers.1.self_attn.gate_proj.scales,
layers.1.self_attn.gate_proj.weight,
layers.1.self_attn.k_proj.biases,
layers.1.self_attn.k_proj.scales,
layers.1.self_attn.k_proj.weight,
layers.1.self_attn.o_proj.biases,
layers.1.self_attn.o_proj.scales,
layers.1.self_attn.o_proj.weight,
layers.1.self_attn.q_proj.biases,
layers.1.self_attn.q_proj.scales,
layers.1.self_attn.q_proj.weight,
layers.1.self_attn.v_proj.biases,
layers.1.self_attn.v_proj.scales,
layers.1.self_attn.v_proj.weight,
layers.10.input_layernorm.weight,
layers.10.mlp.down_proj.biases,
layers.10.mlp.down_proj.scales,
layers.10.mlp.down_proj.weight,
layers.10.mlp.gate_proj.biases,
layers.10.mlp.gate_proj.scales,
layers.10.mlp.gate_proj.weight,
layers.10.mlp.up_proj.biases,
layers.10.mlp.up_proj.scales,
layers.10.mlp.up_proj.weight,
layers.10.post_attention_layernorm.weight,
layers.10.post_feedforward_layernorm.weight,
layers.10.pre_feedforward_layernorm.weight,
layers.10.self_attn.gate_proj.biases,
layers.10.self_attn.gate_proj.scales,
layers.10.self_attn.gate_proj.weight,
layers.10.self_attn.k_proj.biases,
layers.10.self_attn.k_proj.scales,
layers.10.self_attn.k_proj.weight,
layers.10.self_attn.o_proj.biases,
layers.10.self_attn.o_proj.scales,
layers.10.self_attn.o_proj.weight,
layers.10.self_attn.q_proj.biases,
layers.10.self_attn.q_proj.scales,
layers.10.self_attn.q_proj.weight,
layers.10.self_attn.v_proj.biases,
layers.10.self_attn.v_proj.scales,
layers.10.self_attn.v_proj.weight,
layers.11.input_layernorm.weight,
layers.11.mlp.down_proj.biases,
layers.11.mlp.down_proj.scales,
layers.11.mlp.down_proj.weight,
layers.11.mlp.gate_proj.biases,
layers.11.mlp.gate_proj.scales,
layers.11.mlp.gate_proj.weight,
layers.11.mlp.up_proj.biases,
layers.11.mlp.up_proj.scales,
layers.11.mlp.up_proj.weight,
layers.11.post_attention_layernorm.weight,
layers.11.post_feedforward_layernorm.weight,
layers.11.pre_feedforward_layernorm.weight,
layers.11.self_attn.gate_proj.biases,
layers.11.self_attn.gate_proj.scales,
layers.11.self_attn.gate_proj.weight,
layers.11.self_attn.k_proj.biases,
layers.11.self_attn.k_proj.scales,
layers.11.self_attn.k_proj.weight,
layers.11.self_attn.o_proj.biases,
layers.11.self_attn.o_proj.scales,
layers.11.self_attn.o_proj.weight,
layers.11.self_attn.q_proj.biases,
layers.11.self_attn.q_proj.scales,
layers.11.self_attn.q_proj.weight,
layers.11.self_attn.v_proj.biases,
layers.11.self_attn.v_proj.scales,
layers.11.self_attn.v_proj.weight,
layers.12.input_layernorm.weight,
layers.12.mlp.down_proj.biases,
layers.12.mlp.down_proj.scales,
layers.12.mlp.down_proj.weight,
layers.12.mlp.gate_proj.biases,
layers.12.mlp.gate_proj.scales,
layers.12.mlp.gate_proj.weight,
layers.12.mlp.up_proj.biases,
layers.12.mlp.up_proj.scales,
layers.12.mlp.up_proj.weight,
layers.12.post_attention_layernorm.weight,
layers.12.post_feedforward_layernorm.weight,
layers.12.pre_feedforward_layernorm.weight,
layers.12.self_attn.gate_proj.biases,
layers.12.self_attn.gate_proj.scales,
layers.12.self_attn.gate_proj.weight,
layers.12.self_attn.k_proj.biases,
layers.12.self_attn.k_proj.scales,
layers.12.self_attn.k_proj.weight,
layers.12.self_attn.o_proj.biases,
layers.12.self_attn.o_proj.scales,
layers.12.self_attn.o_proj.weight,
layers.12.self_attn.q_proj.biases,
layers.12.self_attn.q_proj.scales,
layers.12.self_attn.q_proj.weight,
layers.12.self_attn.v_proj.biases,
layers.12.self_attn.v_proj.scales,
layers.12.self_attn.v_proj.weight,
layers.13.input_layernorm.weight,
layers.13.mlp.down_proj.biases,
layers.13.mlp.down_proj.scales,
layers.13.mlp.down_proj.weight,
layers.13.mlp.gate_proj.biases,
layers.13.mlp.gate_proj.scales,
layers.13.mlp.gate_proj.weight,
layers.13.mlp.up_proj.biases,
layers.13.mlp.up_proj.scales,
layers.13.mlp.up_proj.weight,
layers.13.post_attention_layernorm.weight,
layers.13.post_feedforward_layernorm.weight,
layers.13.pre_feedforward_layernorm.weight,
layers.13.self_attn.gate_proj.biases,
layers.13.self_attn.gate_proj.scales,
layers.13.self_attn.gate_proj.weight,
layers.13.self_attn.k_proj.biases,
layers.13.self_attn.k_proj.scales,
layers.13.self_attn.k_proj.weight,
layers.13.self_attn.o_proj.biases,
layers.13.self_attn.o_proj.scales,
layers.13.self_attn.o_proj.weight,
layers.13.self_attn.q_proj.biases,
layers.13.self_attn.q_proj.scales,
layers.13.self_attn.q_proj.weight,
layers.13.self_attn.v_proj.biases,
layers.13.self_attn.v_proj.scales,
layers.13.self_attn.v_proj.weight,
layers.14.input_layernorm.weight,
layers.14.mlp.down_proj.biases,
layers.14.mlp.down_proj.scales,
layers.14.mlp.down_proj.weight,
layers.14.mlp.gate_proj.biases,
layers.14.mlp.gate_proj.scales,
layers.14.mlp.gate_proj.weight,
layers.14.mlp.up_proj.biases,
layers.14.mlp.up_proj.scales,
layers.14.mlp.up_proj.weight,
layers.14.post_attention_layernorm.weight,
layers.14.post_feedforward_layernorm.weight,
layers.14.pre_feedforward_layernorm.weight,
layers.14.self_attn.gate_proj.biases,
layers.14.self_attn.gate_proj.scales,
layers.14.self_attn.gate_proj.weight,
layers.14.self_attn.k_proj.biases,
layers.14.self_attn.k_proj.scales,
layers.14.self_attn.k_proj.weight,
layers.14.self_attn.o_proj.biases,
layers.14.self_attn.o_proj.scales,
layers.14.self_attn.o_proj.weight,
layers.14.self_attn.q_proj.biases,
layers.14.self_attn.q_proj.scales,
layers.14.self_attn.q_proj.weight,
layers.14.self_attn.v_proj.biases,
layers.14.self_attn.v_proj.scales,
layers.14.self_attn.v_proj.weight,
layers.15.input_layernorm.weight,
layers.15.mlp.down_proj.biases,
layers.15.mlp.down_proj.scales,
layers.15.mlp.down_proj.weight,
layers.15.mlp.gate_proj.biases,
layers.15.mlp.gate_proj.scales,
layers.15.mlp.gate_proj.weight,
layers.15.mlp.up_proj.biases,
layers.15.mlp.up_proj.scales,
layers.15.mlp.up_proj.weight,
layers.15.post_attention_layernorm.weight,
layers.15.post_feedforward_layernorm.weight,
layers.15.pre_feedforward_layernorm.weight,
layers.15.self_attn.gate_proj.biases,
layers.15.self_attn.gate_proj.scales,
layers.15.self_attn.gate_proj.weight,
layers.15.self_attn.k_proj.biases,
layers.15.self_attn.k_proj.scales,
layers.15.self_attn.k_proj.weight,
layers.15.self_attn.o_proj.biases,
layers.15.self_attn.o_proj.scales,
layers.15.self_attn.o_proj.weight,
layers.15.self_attn.q_proj.biases,
layers.15.self_attn.q_proj.scales,
layers.15.self_attn.q_proj.weight,
layers.15.self_attn.v_proj.biases,
layers.15.self_attn.v_proj.scales,
layers.15.self_attn.v_proj.weight,
layers.16.input_layernorm.weight,
layers.16.mlp.down_proj.biases,
layers.16.mlp.down_proj.scales,
layers.16.mlp.down_proj.weight,
layers.16.mlp.gate_proj.biases,
layers.16.mlp.gate_proj.scales,
layers.16.mlp.gate_proj.weight,
layers.16.mlp.up_proj.biases,
layers.16.mlp.up_proj.scales,
layers.16.mlp.up_proj.weight,
layers.16.post_attention_layernorm.weight,
layers.16.post_feedforward_layernorm.weight,
layers.16.pre_feedforward_layernorm.weight,
layers.16.self_attn.gate_proj.biases,
layers.16.self_attn.gate_proj.scales,
layers.16.self_attn.gate_proj.weight,
layers.16.self_attn.k_proj.biases,
layers.16.self_attn.k_proj.scales,
layers.16.self_attn.k_proj.weight,
layers.16.self_attn.o_proj.biases,
layers.16.self_attn.o_proj.scales,
layers.16.self_attn.o_proj.weight,
layers.16.self_attn.q_proj.biases,
layers.16.self_attn.q_proj.scales,
layers.16.self_attn.q_proj.weight,
layers.16.self_attn.v_proj.biases,
layers.16.self_attn.v_proj.scales,
layers.16.self_attn.v_proj.weight,
layers.17.input_layernorm.weight,
layers.17.mlp.down_proj.biases,
layers.17.mlp.down_proj.scales,
layers.17.mlp.down_proj.weight,
layers.17.mlp.gate_proj.biases,
layers.17.mlp.gate_proj.scales,
layers.17.mlp.gate_proj.weight,
layers.17.mlp.up_proj.biases,
layers.17.mlp.up_proj.scales,
layers.17.mlp.up_proj.weight,
layers.17.post_attention_layernorm.weight,
layers.17.post_feedforward_layernorm.weight,
layers.17.pre_feedforward_layernorm.weight,
layers.17.self_attn.gate_proj.biases,
layers.17.self_attn.gate_proj.scales,
layers.17.self_attn.gate_proj.weight,
layers.17.self_attn.k_proj.biases,
layers.17.self_attn.k_proj.scales,
layers.17.self_attn.k_proj.weight,
layers.17.self_attn.o_proj.biases,
layers.17.self_attn.o_proj.scales,
layers.17.self_attn.o_proj.weight,
layers.17.self_attn.q_proj.biases,
layers.17.self_attn.q_proj.scales,
layers.17.self_attn.q_proj.weight,
layers.17.self_attn.v_proj.biases,
layers.17.self_attn.v_proj.scales,
layers.17.self_attn.v_proj.weight,
layers.18.input_layernorm.weight,
layers.18.mlp.down_proj.biases,
layers.18.mlp.down_proj.scales,
layers.18.mlp.down_proj.weight,
layers.18.mlp.gate_proj.biases,
layers.18.mlp.gate_proj.scales,
layers.18.mlp.gate_proj.weight,
layers.18.mlp.up_proj.biases,
layers.18.mlp.up_proj.scales,
layers.18.mlp.up_proj.weight,
layers.18.post_attention_layernorm.weight,
layers.18.post_feedforward_layernorm.weight,
layers.18.pre_feedforward_layernorm.weight,
layers.18.self_attn.gate_proj.biases,
layers.18.self_attn.gate_proj.scales,
layers.18.self_attn.gate_proj.weight,
layers.18.self_attn.k_proj.biases,
layers.18.self_attn.k_proj.scales,
layers.18.self_attn.k_proj.weight,
layers.18.self_attn.o_proj.biases,
layers.18.self_attn.o_proj.scales,
layers.18.self_attn.o_proj.weight,
layers.18.self_attn.q_proj.biases,
layers.18.self_attn.q_proj.scales,
layers.18.self_attn.q_proj.weight,
layers.18.self_attn.v_proj.biases,
layers.18.self_attn.v_proj.scales,
layers.18.self_attn.v_proj.weight,
layers.19.input_layernorm.weight,
layers.19.mlp.down_proj.biases,
layers.19.mlp.down_proj.scales,
layers.19.mlp.down_proj.weight,
layers.19.mlp.gate_proj.biases,
layers.19.mlp.gate_proj.scales,
layers.19.mlp.gate_proj.weight,
layers.19.mlp.up_proj.biases,
layers.19.mlp.up_proj.scales,
layers.19.mlp.up_proj.weight,
layers.19.post_attention_layernorm.weight,
layers.19.post_feedforward_layernorm.weight,
layers.19.pre_feedforward_layernorm.weight,
layers.19.self_attn.gate_proj.biases,
layers.19.self_attn.gate_proj.scales,
layers.19.self_attn.gate_proj.weight,
layers.19.self_attn.k_proj.biases,
layers.19.self_attn.k_proj.scales,
layers.19.self_attn.k_proj.weight,
layers.19.self_attn.o_proj.biases,
layers.19.self_attn.o_proj.scales,
layers.19.self_attn.o_proj.weight,
layers.19.self_attn.q_proj.biases,
layers.19.self_attn.q_proj.scales,
layers.19.self_attn.q_proj.weight,
layers.19.self_attn.v_proj.biases,
layers.19.self_attn.v_proj.scales,
layers.19.self_attn.v_proj.weight,
layers.2.input_layernorm.weight,
layers.2.mlp.down_proj.biases,
layers.2.mlp.down_proj.scales,
layers.2.mlp.down_proj.weight,
layers.2.mlp.gate_proj.biases,
layers.2.mlp.gate_proj.scales,
layers.2.mlp.gate_proj.weight,
layers.2.mlp.up_proj.biases,
layers.2.mlp.up_proj.scales,
layers.2.mlp.up_proj.weight,
layers.2.post_attention_layernorm.weight,
layers.2.post_feedforward_layernorm.weight,
layers.2.pre_feedforward_layernorm.weight,
layers.2.self_attn.gate_proj.biases,
layers.2.self_attn.gate_proj.scales,
layers.2.self_attn.gate_proj.weight,
layers.2.self_attn.k_proj.biases,
layers.2.self_attn.k_proj.scales,
layers.2.self_attn.k_proj.weight,
layers.2.self_attn.o_proj.biases,
layers.2.self_attn.o_proj.scales,
layers.2.self_attn.o_proj.weight,
layers.2.self_attn.q_proj.biases,
layers.2.self_attn.q_proj.scales,
layers.2.self_attn.q_proj.weight,
layers.2.self_attn.v_proj.biases,
layers.2.self_attn.v_proj.scales,
layers.2.self_attn.v_proj.weight,
layers.20.input_layernorm.weight,
layers.20.mlp.down_proj.biases,
layers.20.mlp.down_proj.scales,
layers.20.mlp.down_proj.weight,
layers.20.mlp.gate_proj.biases,
layers.20.mlp.gate_proj.scales,
layers.20.mlp.gate_proj.weight,
layers.20.mlp.up_proj.biases,
layers.20.mlp.up_proj.scales,
layers.20.mlp.up_proj.weight,
layers.20.post_attention_layernorm.weight,
layers.20.post_feedforward_layernorm.weight,
layers.20.pre_feedforward_layernorm.weight,
layers.20.self_attn.gate_proj.biases,
layers.20.self_attn.gate_proj.scales,
layers.20.self_attn.gate_proj.weight,
layers.20.self_attn.k_proj.biases,
layers.20.self_attn.k_proj.scales,
layers.20.self_attn.k_proj.weight,
layers.20.self_attn.o_proj.biases,
layers.20.self_attn.o_proj.scales,
layers.20.self_attn.o_proj.weight,
layers.20.self_attn.q_proj.biases,
layers.20.self_attn.q_proj.scales,
layers.20.self_attn.q_proj.weight,
layers.20.self_attn.v_proj.biases,
layers.20.self_attn.v_proj.scales,
layers.20.self_attn.v_proj.weight,
layers.21.input_layernorm.weight,
layers.21.mlp.down_proj.biases,
layers.21.mlp.down_proj.scales,
layers.21.mlp.down_proj.weight,
layers.21.mlp.gate_proj.biases,
layers.21.mlp.gate_proj.scales,
layers.21.mlp.gate_proj.weight,
layers.21.mlp.up_proj.biases,
layers.21.mlp.up_proj.scales,
layers.21.mlp.up_proj.weight,
layers.21.post_attention_layernorm.weight,
layers.21.post_feedforward_layernorm.weight,
layers.21.pre_feedforward_layernorm.weight,
layers.21.self_attn.gate_proj.biases,
layers.21.self_attn.gate_proj.scales,
layers.21.self_attn.gate_proj.weight,
layers.21.self_attn.k_proj.biases,
layers.21.self_attn.k_proj.scales,
layers.21.self_attn.k_proj.weight,
layers.21.self_attn.o_proj.biases,
layers.21.self_attn.o_proj.scales,
layers.21.self_attn.o_proj.weight,
layers.21.self_attn.q_proj.biases,
layers.21.self_attn.q_proj.scales,
layers.21.self_attn.q_proj.weight,
layers.21.self_attn.v_proj.biases,
layers.21.self_attn.v_proj.scales,
layers.21.self_attn.v_proj.weight,
layers.22.input_layernorm.weight,
layers.22.mlp.down_proj.biases,
layers.22.mlp.down_proj.scales,
layers.22.mlp.down_proj.weight,
layers.22.mlp.gate_proj.biases,
layers.22.mlp.gate_proj.scales,
layers.22.mlp.gate_proj.weight,
layers.22.mlp.up_proj.biases,
layers.22.mlp.up_proj.scales,
layers.22.mlp.up_proj.weight,
layers.22.post_attention_layernorm.weight,
layers.22.post_feedforward_layernorm.weight,
layers.22.pre_feedforward_layernorm.weight,
layers.22.self_attn.gate_proj.biases,
layers.22.self_attn.gate_proj.scales,
layers.22.self_attn.gate_proj.weight,
layers.22.self_attn.k_proj.biases,
layers.22.self_attn.k_proj.scales,
layers.22.self_attn.k_proj.weight,
layers.22.self_attn.o_proj.biases,
layers.22.self_attn.o_proj.scales,
layers.22.self_attn.o_proj.weight,
layers.22.self_attn.q_proj.biases,
layers.22.self_attn.q_proj.scales,
layers.22.self_attn.q_proj.weight,
layers.22.self_attn.v_proj.biases,
layers.22.self_attn.v_proj.scales,
layers.22.self_attn.v_proj.weight,
layers.23.input_layernorm.weight,
layers.23.mlp.down_proj.biases,
layers.23.mlp.down_proj.scales,
layers.23.mlp.down_proj.weight,
layers.23.mlp.gate_proj.biases,
layers.23.mlp.gate_proj.scales,
layers.23.mlp.gate_proj.weight,
layers.23.mlp.up_proj.biases,
layers.23.mlp.up_proj.scales,
layers.23.mlp.up_proj.weight,
layers.23.post_attention_layernorm.weight,
layers.23.post_feedforward_layernorm.weight,
layers.23.pre_feedforward_layernorm.weight,
layers.23.self_attn.gate_proj.biases,
layers.23.self_attn.gate_proj.scales,
layers.23.self_attn.gate_proj.weight,
layers.23.self_attn.k_proj.biases,
layers.23.self_attn.k_proj.scales,
layers.23.self_attn.k_proj.weight,
layers.23.self_attn.o_proj.biases,
layers.23.self_attn.o_proj.scales,
layers.23.self_attn.o_proj.weight,
layers.23.self_attn.q_proj.biases,
layers.23.self_attn.q_proj.scales,
layers.23.self_attn.q_proj.weight,
layers.23.self_attn.v_proj.biases,
layers.23.self_attn.v_proj.scales,
layers.23.self_attn.v_proj.weight,
layers.24.input_layernorm.weight,
layers.24.mlp.down_proj.biases,
layers.24.mlp.down_proj.scales,
layers.24.mlp.down_proj.weight,
layers.24.mlp.gate_proj.biases,
layers.24.mlp.gate_proj.scales,
layers.24.mlp.gate_proj.weight,
layers.24.mlp.up_proj.biases,
layers.24.mlp.up_proj.scales,
layers.24.mlp.up_proj.weight,
layers.24.post_attention_layernorm.weight,
layers.24.post_feedforward_layernorm.weight,
layers.24.pre_feedforward_layernorm.weight,
layers.24.self_attn.gate_proj.biases,
layers.24.self_attn.gate_proj.scales,
layers.24.self_attn.gate_proj.weight,
layers.24.self_attn.k_proj.biases,
layers.24.self_attn.k_proj.scales,
layers.24.self_attn.k_proj.weight,
layers.24.self_attn.o_proj.biases,
layers.24.self_attn.o_proj.scales,
layers.24.self_attn.o_proj.weight,
layers.24.self_attn.q_proj.biases,
layers.24.self_attn.q_proj.scales,
layers.24.self_attn.q_proj.weight,
layers.24.self_attn.v_proj.biases,
layers.24.self_attn.v_proj.scales,
layers.24.self_attn.v_proj.weight,
layers.25.input_layernorm.weight,
layers.25.mlp.down_proj.biases,
layers.25.mlp.down_proj.scales,
layers.25.mlp.down_proj.weight,
layers.25.mlp.gate_proj.biases,
layers.25.mlp.gate_proj.scales,
layers.25.mlp.gate_proj.weight,
layers.25.mlp.up_proj.biases,
layers.25.mlp.up_proj.scales,
layers.25.mlp.up_proj.weight,
layers.25.post_attention_layernorm.weight,
layers.25.post_feedforward_layernorm.weight,
layers.25.pre_feedforward_layernorm.weight,
layers.25.self_attn.gate_proj.biases,
layers.25.self_attn.gate_proj.scales,
layers.25.self_attn.gate_proj.weight,
layers.25.self_attn.k_proj.biases,
layers.25.self_attn.k_proj.scales,
layers.25.self_attn.k_proj.weight,
layers.25.self_attn.o_proj.biases,
layers.25.self_attn.o_proj.scales,
layers.25.self_attn.o_proj.weight,
layers.25.self_attn.q_proj.biases,
layers.25.self_attn.q_proj.scales,
layers.25.self_attn.q_proj.weight,
layers.25.self_attn.v_proj.biases,
layers.25.self_attn.v_proj.scales,
layers.25.self_attn.v_proj.weight,
layers.26.input_layernorm.weight,
layers.26.mlp.down_proj.biases,
layers.26.mlp.down_proj.scales,
layers.26.mlp.down_proj.weight,
layers.26.mlp.gate_proj.biases,
layers.26.mlp.gate_proj.scales,
layers.26.mlp.gate_proj.weight,
layers.26.mlp.up_proj.biases,
layers.26.mlp.up_proj.scales,
layers.26.mlp.up_proj.weight,
layers.26.post_attention_layernorm.weight,
layers.26.post_feedforward_layernorm.weight,
layers.26.pre_feedforward_layernorm.weight,
layers.26.self_attn.gate_proj.biases,
layers.26.self_attn.gate_proj.scales,
layers.26.self_attn.gate_proj.weight,
layers.26.self_attn.k_proj.biases,
layers.26.self_attn.k_proj.scales,
layers.26.self_attn.k_proj.weight,
layers.26.self_attn.o_proj.biases,
layers.26.self_attn.o_proj.scales,
layers.26.self_attn.o_proj.weight,
layers.26.self_attn.q_proj.biases,
layers.26.self_attn.q_proj.scales,
layers.26.self_attn.q_proj.weight,
layers.26.self_attn.v_proj.biases,
layers.26.self_attn.v_proj.scales,
layers.26.self_attn.v_proj.weight,
layers.27.input_layernorm.weight,
layers.27.mlp.down_proj.biases,
layers.27.mlp.down_proj.scales,
layers.27.mlp.down_proj.weight,
layers.27.mlp.gate_proj.biases,
layers.27.mlp.gate_proj.scales,
layers.27.mlp.gate_proj.weight,
layers.27.mlp.up_proj.biases,
layers.27.mlp.up_proj.scales,
layers.27.mlp.up_proj.weight,
layers.27.post_attention_layernorm.weight,
layers.27.post_feedforward_layernorm.weight,
layers.27.pre_feedforward_layernorm.weight,
layers.27.self_attn.gate_proj.biases,
layers.27.self_attn.gate_proj.scales,
layers.27.self_attn.gate_proj.weight,
layers.27.self_attn.k_proj.biases,
layers.27.self_attn.k_proj.scales,
layers.27.self_attn.k_proj.weight,
layers.27.self_attn.o_proj.biases,
layers.27.self_attn.o_proj.scales,
layers.27.self_attn.o_proj.weight,
layers.27.self_attn.q_proj.biases,
layers.27.self_attn.q_proj.scales,
layers.27.self_attn.q_proj.weight,
layers.27.self_attn.v_proj.biases,
layers.27.self_attn.v_proj.scales,
layers.27.self_attn.v_proj.weight,
layers.28.input_layernorm.weight,
layers.28.mlp.down_proj.biases,
layers.28.mlp.down_proj.scales,
layers.28.mlp.down_proj.weight,
layers.28.mlp.gate_proj.biases,
layers.28.mlp.gate_proj.scales,
layers.28.mlp.gate_proj.weight,
layers.28.mlp.up_proj.biases,
layers.28.mlp.up_proj.scales,
layers.28.mlp.up_proj.weight,
layers.28.post_attention_layernorm.weight,
layers.28.post_feedforward_layernorm.weight,
layers.28.pre_feedforward_layernorm.weight,
layers.28.self_attn.gate_proj.biases,
layers.28.self_attn.gate_proj.scales,
layers.28.self_attn.gate_proj.weight,
layers.28.self_attn.k_proj.biases,
layers.28.self_attn.k_proj.scales,
layers.28.self_attn.k_proj.weight,
layers.28.self_attn.o_proj.biases,
layers.28.self_attn.o_proj.scales,
layers.28.self_attn.o_proj.weight,
layers.28.self_attn.q_proj.biases,
layers.28.self_attn.q_proj.scales,
layers.28.self_attn.q_proj.weight,
layers.28.self_attn.v_proj.biases,
layers.28.self_attn.v_proj.scales,
layers.28.self_attn.v_proj.weight,
layers.29.input_layernorm.weight,
layers.29.mlp.down_proj.biases,
layers.29.mlp.down_proj.scales,
layers.29.mlp.down_proj.weight,
layers.29.mlp.gate_proj.biases,
layers.29.mlp.gate_proj.scales,
layers.29.mlp.gate_proj.weight,
layers.29.mlp.up_proj.biases,
layers.29.mlp.up_proj.scales,
layers.29.mlp.up_proj.weight,
layers.29.post_attention_layernorm.weight,
layers.29.post_feedforward_layernorm.weight,
layers.29.pre_feedforward_layernorm.weight,
layers.29.self_attn.gate_proj.biases,
layers.29.self_attn.gate_proj.scales,
layers.29.self_attn.gate_proj.weight,
layers.29.self_attn.k_proj.biases,
layers.29.self_attn.k_proj.scales,
layers.29.self_attn.k_proj.weight,
layers.29.self_attn.o_proj.biases,
layers.29.self_attn.o_proj.scales,
layers.29.self_attn.o_proj.weight,
layers.29.self_attn.q_proj.biases,
layers.29.self_attn.q_proj.scales,
layers.29.self_attn.q_proj.weight,
layers.29.self_attn.v_proj.biases,
layers.29.self_attn.v_proj.scales,
layers.29.self_attn.v_proj.weight,
layers.3.input_layernorm.weight,
layers.3.mlp.down_proj.biases,
layers.3.mlp.down_proj.scales,
layers.3.mlp.down_proj.weight,
layers.3.mlp.gate_proj.biases,
layers.3.mlp.gate_proj.scales,
layers.3.mlp.gate_proj.weight,
layers.3.mlp.up_proj.biases,
layers.3.mlp.up_proj.scales,
layers.3.mlp.up_proj.weight,
layers.3.post_attention_layernorm.weight,
layers.3.post_feedforward_layernorm.weight,
layers.3.pre_feedforward_layernorm.weight,
layers.3.self_attn.gate_proj.biases,
layers.3.self_attn.gate_proj.scales,
layers.3.self_attn.gate_proj.weight,
layers.3.self_attn.k_proj.biases,
layers.3.self_attn.k_proj.scales,
layers.3.self_attn.k_proj.weight,
layers.3.self_attn.o_proj.biases,
layers.3.self_attn.o_proj.scales,
layers.3.self_attn.o_proj.weight,
layers.3.self_attn.q_proj.biases,
layers.3.self_attn.q_proj.scales,
layers.3.self_attn.q_proj.weight,
layers.3.self_attn.v_proj.biases,
layers.3.self_attn.v_proj.scales,
layers.3.self_attn.v_proj.weight,
layers.30.input_layernorm.weight,
layers.30.mlp.down_proj.biases,
layers.30.mlp.down_proj.scales,
layers.30.mlp.down_proj.weight,
layers.30.mlp.gate_proj.biases,
layers.30.mlp.gate_proj.scales,
layers.30.mlp.gate_proj.weight,
layers.30.mlp.up_proj.biases,
layers.30.mlp.up_proj.scales,
layers.30.mlp.up_proj.weight,
layers.30.post_attention_layernorm.weight,
layers.30.post_feedforward_layernorm.weight,
layers.30.pre_feedforward_layernorm.weight,
layers.30.self_attn.gate_proj.biases,
layers.30.self_attn.gate_proj.scales,
layers.30.self_attn.gate_proj.weight,
layers.30.self_attn.k_proj.biases,
layers.30.self_attn.k_proj.scales,
layers.30.self_attn.k_proj.weight,
layers.30.self_attn.o_proj.biases,
layers.30.self_attn.o_proj.scales,
layers.30.self_attn.o_proj.weight,
layers.30.self_attn.q_proj.biases,
layers.30.self_attn.q_proj.scales,
layers.30.self_attn.q_proj.weight,
layers.30.self_attn.v_proj.biases,
layers.30.self_attn.v_proj.scales,
layers.30.self_attn.v_proj.weight,
layers.31.input_layernorm.weight,
layers.31.mlp.down_proj.biases,
layers.31.mlp.down_proj.scales,
layers.31.mlp.down_proj.weight,
layers.31.mlp.gate_proj.biases,
layers.31.mlp.gate_proj.scales,
layers.31.mlp.gate_proj.weight,
layers.31.mlp.up_proj.biases,
layers.31.mlp.up_proj.scales,
layers.31.mlp.up_proj.weight,
layers.31.post_attention_layernorm.weight,
layers.31.post_feedforward_layernorm.weight,
layers.31.pre_feedforward_layernorm.weight,
layers.31.self_attn.gate_proj.biases,
layers.31.self_attn.gate_proj.scales,
layers.31.self_attn.gate_proj.weight,
layers.31.self_attn.k_proj.biases,
layers.31.self_attn.k_proj.scales,
layers.31.self_attn.k_proj.weight,
layers.31.self_attn.o_proj.biases,
layers.31.self_attn.o_proj.scales,
layers.31.self_attn.o_proj.weight,
layers.31.self_attn.q_proj.biases,
layers.31.self_attn.q_proj.scales,
layers.31.self_attn.q_proj.weight,
layers.31.self_attn.v_proj.biases,
layers.31.self_attn.v_proj.scales,
layers.31.self_attn.v_proj.weight,
layers.32.input_layernorm.weight,
layers.32.mlp.down_proj.biases,
layers.32.mlp.down_proj.scales,
layers.32.mlp.down_proj.weight,
layers.32.mlp.gate_proj.biases,
layers.32.mlp.gate_proj.scales,
layers.32.mlp.gate_proj.weight,
layers.32.mlp.up_proj.biases,
layers.32.mlp.up_proj.scales,
layers.32.mlp.up_proj.weight,
layers.32.post_attention_layernorm.weight,
layers.32.post_feedforward_layernorm.weight,
layers.32.pre_feedforward_layernorm.weight,
layers.32.self_attn.gate_proj.biases,
layers.32.self_attn.gate_proj.scales,
layers.32.self_attn.gate_proj.weight,
layers.32.self_attn.k_proj.biases,
layers.32.self_attn.k_proj.scales,
layers.32.self_attn.k_proj.weight,
layers.32.self_attn.o_proj.biases,
layers.32.self_attn.o_proj.scales,
layers.32.self_attn.o_proj.weight,
layers.32.self_attn.q_proj.biases,
layers.32.self_attn.q_proj.scales,
layers.32.self_attn.q_proj.weight,
layers.32.self_attn.v_proj.biases,
layers.32.self_attn.v_proj.scales,
layers.32.self_attn.v_proj.weight,
layers.33.input_layernorm.weight,
layers.33.mlp.down_proj.biases,
layers.33.mlp.down_proj.scales,
layers.33.mlp.down_proj.weight,
layers.33.mlp.gate_proj.biases,
layers.33.mlp.gate_proj.scales,
layers.33.mlp.gate_proj.weight,
layers.33.mlp.up_proj.biases,
layers.33.mlp.up_proj.scales,
layers.33.mlp.up_proj.weight,
layers.33.post_attention_layernorm.weight,
layers.33.post_feedforward_layernorm.weight,
layers.33.pre_feedforward_layernorm.weight,
layers.33.self_attn.gate_proj.biases,
layers.33.self_attn.gate_proj.scales,
layers.33.self_attn.gate_proj.weight,
layers.33.self_attn.k_proj.biases,
layers.33.self_attn.k_proj.scales,
layers.33.self_attn.k_proj.weight,
layers.33.self_attn.o_proj.biases,
layers.33.self_attn.o_proj.scales,
layers.33.self_attn.o_proj.weight,
layers.33.self_attn.q_proj.biases,
layers.33.self_attn.q_proj.scales,
layers.33.self_attn.q_proj.weight,
layers.33.self_attn.v_proj.biases,
layers.33.self_attn.v_proj.scales,
layers.33.self_attn.v_proj.weight,
layers.34.input_layernorm.weight,
layers.34.mlp.down_proj.biases,
layers.34.mlp.down_proj.scales,
layers.34.mlp.down_proj.weight,
layers.34.mlp.gate_proj.biases,
layers.34.mlp.gate_proj.scales,
layers.34.mlp.gate_proj.weight,
layers.34.mlp.up_proj.biases,
layers.34.mlp.up_proj.scales,
layers.34.mlp.up_proj.weight,
layers.34.post_attention_layernorm.weight,
layers.34.post_feedforward_layernorm.weight,
layers.34.pre_feedforward_layernorm.weight,
layers.34.self_attn.gate_proj.biases,
layers.34.self_attn.gate_proj.scales,
layers.34.self_attn.gate_proj.weight,
layers.34.self_attn.k_proj.biases,
layers.34.self_attn.k_proj.scales,
layers.34.self_attn.k_proj.weight,
layers.34.self_attn.o_proj.biases,
layers.34.self_attn.o_proj.scales,
layers.34.self_attn.o_proj.weight,
layers.34.self_attn.q_proj.biases,
layers.34.self_attn.q_proj.scales,
layers.34.self_attn.q_proj.weight,
layers.34.self_attn.v_proj.biases,
layers.34.self_attn.v_proj.scales,
layers.34.self_attn.v_proj.weight,
layers.35.input_layernorm.weight,
layers.35.mlp.down_proj.biases,
layers.35.mlp.down_proj.scales,
layers.35.mlp.down_proj.weight,
layers.35.mlp.gate_proj.biases,
layers.35.mlp.gate_proj.scales,
layers.35.mlp.gate_proj.weight,
layers.35.mlp.up_proj.biases,
layers.35.mlp.up_proj.scales,
layers.35.mlp.up_proj.weight,
layers.35.post_attention_layernorm.weight,
layers.35.post_feedforward_layernorm.weight,
layers.35.pre_feedforward_layernorm.weight,
layers.35.self_attn.gate_proj.biases,
layers.35.self_attn.gate_proj.scales,
layers.35.self_attn.gate_proj.weight,
layers.35.self_attn.k_proj.biases,
layers.35.self_attn.k_proj.scales,
layers.35.self_attn.k_proj.weight,
layers.35.self_attn.o_proj.biases,
layers.35.self_attn.o_proj.scales,
layers.35.self_attn.o_proj.weight,
layers.35.self_attn.q_proj.biases,
layers.35.self_attn.q_proj.scales,
layers.35.self_attn.q_proj.weight,
layers.35.self_attn.v_proj.biases,
layers.35.self_attn.v_proj.scales,
layers.35.self_attn.v_proj.weight,
layers.36.input_layernorm.weight,
layers.36.mlp.down_proj.biases,
layers.36.mlp.down_proj.scales,
layers.36.mlp.down_proj.weight,
layers.36.mlp.gate_proj.biases,
layers.36.mlp.gate_proj.scales,
layers.36.mlp.gate_proj.weight,
layers.36.mlp.up_proj.biases,
layers.36.mlp.up_proj.scales,
layers.36.mlp.up_proj.weight,
layers.36.post_attention_layernorm.weight,
layers.36.post_feedforward_layernorm.weight,
layers.36.pre_feedforward_layernorm.weight,
layers.36.self_attn.gate_proj.biases,
layers.36.self_attn.gate_proj.scales,
layers.36.self_attn.gate_proj.weight,
layers.36.self_attn.k_proj.biases,
layers.36.self_attn.k_proj.scales,
layers.36.self_attn.k_proj.weight,
layers.36.self_attn.o_proj.biases,
layers.36.self_attn.o_proj.scales,
layers.36.self_attn.o_proj.weight,
layers.36.self_attn.q_proj.biases,
layers.36.self_attn.q_proj.scales,
layers.36.self_attn.q_proj.weight,
layers.36.self_attn.v_proj.biases,
layers.36.self_attn.v_proj.scales,
layers.36.self_attn.v_proj.weight,
layers.37.input_layernorm.weight,
layers.37.mlp.down_proj.biases,
layers.37.mlp.down_proj.scales,
layers.37.mlp.down_proj.weight,
layers.37.mlp.gate_proj.biases,
layers.37.mlp.gate_proj.scales,
layers.37.mlp.gate_proj.weight,
layers.37.mlp.up_proj.biases,
layers.37.mlp.up_proj.scales,
layers.37.mlp.up_proj.weight,
layers.37.post_attention_layernorm.weight,
layers.37.post_feedforward_layernorm.weight,
layers.37.pre_feedforward_layernorm.weight,
layers.37.self_attn.gate_proj.biases,
layers.37.self_attn.gate_proj.scales,
layers.37.self_attn.gate_proj.weight,
layers.37.self_attn.k_proj.biases,
layers.37.self_attn.k_proj.scales,
layers.37.self_attn.k_proj.weight,
layers.37.self_attn.o_proj.biases,
layers.37.self_attn.o_proj.scales,
layers.37.self_attn.o_proj.weight,
layers.37.self_attn.q_proj.biases,
layers.37.self_attn.q_proj.scales,
layers.37.self_attn.q_proj.weight,
layers.37.self_attn.v_proj.biases,
layers.37.self_attn.v_proj.scales,
layers.37.self_attn.v_proj.weight,
layers.38.input_layernorm.weight,
layers.38.mlp.down_proj.biases,
layers.38.mlp.down_proj.scales,
layers.38.mlp.down_proj.weight,
layers.38.mlp.gate_proj.biases,
layers.38.mlp.gate_proj.scales,
layers.38.mlp.gate_proj.weight,
layers.38.mlp.up_proj.biases,
layers.38.mlp.up_proj.scales,
layers.38.mlp.up_proj.weight,
layers.38.post_attention_layernorm.weight,
layers.38.post_feedforward_layernorm.weight,
layers.38.pre_feedforward_layernorm.weight,
layers.38.self_attn.gate_proj.biases,
layers.38.self_attn.gate_proj.scales,
layers.38.self_attn.gate_proj.weight,
layers.38.self_attn.k_proj.biases,
layers.38.self_attn.k_proj.scales,
layers.38.self_attn.k_proj.weight,
layers.38.self_attn.o_proj.biases,
layers.38.self_attn.o_proj.scales,
layers.38.self_attn.o_proj.weight,
layers.38.self_attn.q_proj.biases,
layers.38.self_attn.q_proj.scales,
layers.38.self_attn.q_proj.weight,
layers.38.self_attn.v_proj.biases,
layers.38.self_attn.v_proj.scales,
layers.38.self_attn.v_proj.weight,
layers.39.input_layernorm.weight,
layers.39.mlp.down_proj.biases,
layers.39.mlp.down_proj.scales,
layers.39.mlp.down_proj.weight,
layers.39.mlp.gate_proj.biases,
layers.39.mlp.gate_proj.scales,
layers.39.mlp.gate_proj.weight,
layers.39.mlp.up_proj.biases,
layers.39.mlp.up_proj.scales,
layers.39.mlp.up_proj.weight,
layers.39.post_attention_layernorm.weight,
layers.39.post_feedforward_layernorm.weight,
layers.39.pre_feedforward_layernorm.weight,
layers.39.self_attn.gate_proj.biases,
layers.39.self_attn.gate_proj.scales,
layers.39.self_attn.gate_proj.weight,
layers.39.self_attn.k_proj.biases,
layers.39.self_attn.k_proj.scales,
layers.39.self_attn.k_proj.weight,
layers.39.self_attn.o_proj.biases,
layers.39.self_attn.o_proj.scales,
layers.39.self_attn.o_proj.weight,
layers.39.self_attn.q_proj.biases,
layers.39.self_attn.q_proj.scales,
layers.39.self_attn.q_proj.weight,
layers.39.self_attn.v_proj.biases,
layers.39.self_attn.v_proj.scales,
layers.39.self_attn.v_proj.weight,
layers.4.input_layernorm.weight,
layers.4.mlp.down_proj.biases,
layers.4.mlp.down_proj.scales,
layers.4.mlp.down_proj.weight,
layers.4.mlp.gate_proj.biases,
layers.4.mlp.gate_proj.scales,
layers.4.mlp.gate_proj.weight,
layers.4.mlp.up_proj.biases,
layers.4.mlp.up_proj.scales,
layers.4.mlp.up_proj.weight,
layers.4.post_attention_layernorm.weight,
layers.4.post_feedforward_layernorm.weight,
layers.4.pre_feedforward_layernorm.weight,
layers.4.self_attn.gate_proj.biases,
layers.4.self_attn.gate_proj.scales,
layers.4.self_attn.gate_proj.weight,
layers.4.self_attn.k_proj.biases,
layers.4.self_attn.k_proj.scales,
layers.4.self_attn.k_proj.weight,
layers.4.self_attn.o_proj.biases,
layers.4.self_attn.o_proj.scales,
layers.4.self_attn.o_proj.weight,
layers.4.self_attn.q_proj.biases,
layers.4.self_attn.q_proj.scales,
layers.4.self_attn.q_proj.weight,
layers.4.self_attn.v_proj.biases,
layers.4.self_attn.v_proj.scales,
layers.4.self_attn.v_proj.weight,
layers.40.input_layernorm.weight,
layers.40.mlp.down_proj.biases,
layers.40.mlp.down_proj.scales,
layers.40.mlp.down_proj.weight,
layers.40.mlp.gate_proj.biases,
layers.40.mlp.gate_proj.scales,
layers.40.mlp.gate_proj.weight,
layers.40.mlp.up_proj.biases,
layers.40.mlp.up_proj.scales,
layers.40.mlp.up_proj.weight,
layers.40.post_attention_layernorm.weight,
layers.40.post_feedforward_layernorm.weight,
layers.40.pre_feedforward_layernorm.weight,
layers.40.self_attn.gate_proj.biases,
layers.40.self_attn.gate_proj.scales,
layers.40.self_attn.gate_proj.weight,
layers.40.self_attn.k_proj.biases,
layers.40.self_attn.k_proj.scales,
layers.40.self_attn.k_proj.weight,
layers.40.self_attn.o_proj.biases,
layers.40.self_attn.o_proj.scales,
layers.40.self_attn.o_proj.weight,
layers.40.self_attn.q_proj.biases,
layers.40.self_attn.q_proj.scales,
layers.40.self_attn.q_proj.weight,
layers.40.self_attn.v_proj.biases,
layers.40.self_attn.v_proj.scales,
layers.40.self_attn.v_proj.weight,
layers.41.input_layernorm.weight,
layers.41.mlp.down_proj.biases,
layers.41.mlp.down_proj.scales,
layers.41.mlp.down_proj.weight,
layers.41.mlp.gate_proj.biases,
layers.41.mlp.gate_proj.scales,
layers.41.mlp.gate_proj.weight,
layers.41.mlp.up_proj.biases,
layers.41.mlp.up_proj.scales,
layers.41.mlp.up_proj.weight,
layers.41.post_attention_layernorm.weight,
layers.41.post_feedforward_layernorm.weight,
layers.41.pre_feedforward_layernorm.weight,
layers.41.self_attn.gate_proj.biases,
layers.41.self_attn.gate_proj.scales,
layers.41.self_attn.gate_proj.weight,
layers.41.self_attn.k_proj.biases,
layers.41.self_attn.k_proj.scales,
layers.41.self_attn.k_proj.weight,
layers.41.self_attn.o_proj.biases,
layers.41.self_attn.o_proj.scales,
layers.41.self_attn.o_proj.weight,
layers.41.self_attn.q_proj.biases,
layers.41.self_attn.q_proj.scales,
layers.41.self_attn.q_proj.weight,
layers.41.self_attn.v_proj.biases,
layers.41.self_attn.v_proj.scales,
layers.41.self_attn.v_proj.weight,
layers.42.input_layernorm.weight,
layers.42.mlp.down_proj.biases,
layers.42.mlp.down_proj.scales,
layers.42.mlp.down_proj.weight,
layers.42.mlp.gate_proj.biases,
layers.42.mlp.gate_proj.scales,
layers.42.mlp.gate_proj.weight,
layers.42.mlp.up_proj.biases,
layers.42.mlp.up_proj.scales,
layers.42.mlp.up_proj.weight,
layers.42.post_attention_layernorm.weight,
layers.42.post_feedforward_layernorm.weight,
layers.42.pre_feedforward_layernorm.weight,
layers.42.self_attn.gate_proj.biases,
layers.42.self_attn.gate_proj.scales,
layers.42.self_attn.gate_proj.weight,
layers.42.self_attn.k_proj.biases,
layers.42.self_attn.k_proj.scales,
layers.42.self_attn.k_proj.weight,
layers.42.self_attn.o_proj.biases,
layers.42.self_attn.o_proj.scales,
layers.42.self_attn.o_proj.weight,
layers.42.self_attn.q_proj.biases,
layers.42.self_attn.q_proj.scales,
layers.42.self_attn.q_proj.weight,
layers.42.self_attn.v_proj.biases,
layers.42.self_attn.v_proj.scales,
layers.42.self_attn.v_proj.weight,
layers.43.input_layernorm.weight,
layers.43.mlp.down_proj.biases,
layers.43.mlp.down_proj.scales,
layers.43.mlp.down_proj.weight,
layers.43.mlp.gate_proj.biases,
layers.43.mlp.gate_proj.scales,
layers.43.mlp.gate_proj.weight,
layers.43.mlp.up_proj.biases,
layers.43.mlp.up_proj.scales,
layers.43.mlp.up_proj.weight,
layers.43.post_attention_layernorm.weight,
layers.43.post_feedforward_layernorm.weight,
layers.43.pre_feedforward_layernorm.weight,
layers.43.self_attn.gate_proj.biases,
layers.43.self_attn.gate_proj.scales,
layers.43.self_attn.gate_proj.weight,
layers.43.self_attn.k_proj.biases,
layers.43.self_attn.k_proj.scales,
layers.43.self_attn.k_proj.weight,
layers.43.self_attn.o_proj.biases,
layers.43.self_attn.o_proj.scales,
layers.43.self_attn.o_proj.weight,
layers.43.self_attn.q_proj.biases,
layers.43.self_attn.q_proj.scales,
layers.43.self_attn.q_proj.weight,
layers.43.self_attn.v_proj.biases,
layers.43.self_attn.v_proj.scales,
layers.43.self_attn.v_proj.weight,
layers.44.input_layernorm.weight,
layers.44.mlp.down_proj.biases,
layers.44.mlp.down_proj.scales,
layers.44.mlp.down_proj.weight,
layers.44.mlp.gate_proj.biases,
layers.44.mlp.gate_proj.scales,
layers.44.mlp.gate_proj.weight,
layers.44.mlp.up_proj.biases,
layers.44.mlp.up_proj.scales,
layers.44.mlp.up_proj.weight,
layers.44.post_attention_layernorm.weight,
layers.44.post_feedforward_layernorm.weight,
layers.44.pre_feedforward_layernorm.weight,
layers.44.self_attn.gate_proj.biases,
layers.44.self_attn.gate_proj.scales,
layers.44.self_attn.gate_proj.weight,
layers.44.self_attn.k_proj.biases,
layers.44.self_attn.k_proj.scales,
layers.44.self_attn.k_proj.weight,
layers.44.self_attn.o_proj.biases,
layers.44.self_attn.o_proj.scales,
layers.44.self_attn.o_proj.weight,
layers.44.self_attn.q_proj.biases,
layers.44.self_attn.q_proj.scales,
layers.44.self_attn.q_proj.weight,
layers.44.self_attn.v_proj.biases,
layers.44.self_attn.v_proj.scales,
layers.44.self_attn.v_proj.weight,
layers.45.input_layernorm.weight,
layers.45.mlp.down_proj.biases,
layers.45.mlp.down_proj.scales,
layers.45.mlp.down_proj.weight,
layers.45.mlp.gate_proj.biases,
layers.45.mlp.gate_proj.scales,
layers.45.mlp.gate_proj.weight,
layers.45.mlp.up_proj.biases,
layers.45.mlp.up_proj.scales,
layers.45.mlp.up_proj.weight,
layers.45.post_attention_layernorm.weight,
layers.45.post_feedforward_layernorm.weight,
layers.45.pre_feedforward_layernorm.weight,
layers.45.self_attn.gate_proj.biases,
layers.45.self_attn.gate_proj.scales,
layers.45.self_attn.gate_proj.weight,
layers.45.self_attn.k_proj.biases,
layers.45.self_attn.k_proj.scales,
layers.45.self_attn.k_proj.weight,
layers.45.self_attn.o_proj.biases,
layers.45.self_attn.o_proj.scales,
layers.45.self_attn.o_proj.weight,
layers.45.self_attn.q_proj.biases,
layers.45.self_attn.q_proj.scales,
layers.45.self_attn.q_proj.weight,
layers.45.self_attn.v_proj.biases,
layers.45.self_attn.v_proj.scales,
layers.45.self_attn.v_proj.weight,
layers.46.input_layernorm.weight,
layers.46.mlp.down_proj.biases,
layers.46.mlp.down_proj.scales,
layers.46.mlp.down_proj.weight,
layers.46.mlp.gate_proj.biases,
layers.46.mlp.gate_proj.scales,
layers.46.mlp.gate_proj.weight,
layers.46.mlp.up_proj.biases,
layers.46.mlp.up_proj.scales,
layers.46.mlp.up_proj.weight,
layers.46.post_attention_layernorm.weight,
layers.46.post_feedforward_layernorm.weight,
layers.46.pre_feedforward_layernorm.weight,
layers.46.self_attn.gate_proj.biases,
layers.46.self_attn.gate_proj.scales,
layers.46.self_attn.gate_proj.weight,
layers.46.self_attn.k_proj.biases,
layers.46.self_attn.k_proj.scales,
layers.46.self_attn.k_proj.weight,
layers.46.self_attn.o_proj.biases,
layers.46.self_attn.o_proj.scales,
layers.46.self_attn.o_proj.weight,
layers.46.self_attn.q_proj.biases,
layers.46.self_attn.q_proj.scales,
layers.46.self_attn.q_proj.weight,
layers.46.self_attn.v_proj.biases,
layers.46.self_attn.v_proj.scales,
layers.46.self_attn.v_proj.weight,
layers.47.input_layernorm.weight,
layers.47.mlp.down_proj.biases,
layers.47.mlp.down_proj.scales,
layers.47.mlp.down_proj.weight,
layers.47.mlp.gate_proj.biases,
layers.47.mlp.gate_proj.scales,
layers.47.mlp.gate_proj.weight,
layers.47.mlp.up_proj.biases,
layers.47.mlp.up_proj.scales,
layers.47.mlp.up_proj.weight,
layers.47.post_attention_layernorm.weight,
layers.47.post_feedforward_layernorm.weight,
layers.47.pre_feedforward_layernorm.weight,
layers.47.self_attn.gate_proj.biases,
layers.47.self_attn.gate_proj.scales,
layers.47.self_attn.gate_proj.weight,
layers.47.self_attn.k_proj.biases,
layers.47.self_attn.k_proj.scales,
layers.47.self_attn.k_proj.weight,
layers.47.self_attn.o_proj.biases,
layers.47.self_attn.o_proj.scales,
layers.47.self_attn.o_proj.weight,
layers.47.self_attn.q_proj.biases,
layers.47.self_attn.q_proj.scales,
layers.47.self_attn.q_proj.weight,
layers.47.self_attn.v_proj.biases,
layers.47.self_attn.v_proj.scales,
layers.47.self_attn.v_proj.weight,
layers.48.input_layernorm.weight,
layers.48.mlp.down_proj.biases,
layers.48.mlp.down_proj.scales,
layers.48.mlp.down_proj.weight,
layers.48.mlp.gate_proj.biases,
layers.48.mlp.gate_proj.scales,
layers.48.mlp.gate_proj.weight,
layers.48.mlp.up_proj.biases,
layers.48.mlp.up_proj.scales,
layers.48.mlp.up_proj.weight,
layers.48.post_attention_layernorm.weight,
layers.48.post_feedforward_layernorm.weight,
layers.48.pre_feedforward_layernorm.weight,
layers.48.self_attn.gate_proj.biases,
layers.48.self_attn.gate_proj.scales,
layers.48.self_attn.gate_proj.weight,
layers.48.self_attn.k_proj.biases,
layers.48.self_attn.k_proj.scales,
layers.48.self_attn.k_proj.weight,
layers.48.self_attn.o_proj.biases,
layers.48.self_attn.o_proj.scales,
layers.48.self_attn.o_proj.weight,
layers.48.self_attn.q_proj.biases,
layers.48.self_attn.q_proj.scales,
layers.48.self_attn.q_proj.weight,
layers.48.self_attn.v_proj.biases,
layers.48.self_attn.v_proj.scales,
layers.48.self_attn.v_proj.weight,
layers.49.input_layernorm.weight,
layers.49.mlp.down_proj.biases,
layers.49.mlp.down_proj.scales,
layers.49.mlp.down_proj.weight,
layers.49.mlp.gate_proj.biases,
layers.49.mlp.gate_proj.scales,
layers.49.mlp.gate_proj.weight,
layers.49.mlp.up_proj.biases,
layers.49.mlp.up_proj.scales,
layers.49.mlp.up_proj.weight,
layers.49.post_attention_layernorm.weight,
layers.49.post_feedforward_layernorm.weight,
layers.49.pre_feedforward_layernorm.weight,
layers.49.self_attn.gate_proj.biases,
layers.49.self_attn.gate_proj.scales,
layers.49.self_attn.gate_proj.weight,
layers.49.self_attn.k_proj.biases,
layers.49.self_attn.k_proj.scales,
layers.49.self_attn.k_proj.weight,
layers.49.self_attn.o_proj.biases,
layers.49.self_attn.o_proj.scales,
layers.49.self_attn.o_proj.weight,
layers.49.self_attn.q_proj.biases,
layers.49.self_attn.q_proj.scales,
layers.49.self_attn.q_proj.weight,
layers.49.self_attn.v_proj.biases,
layers.49.self_attn.v_proj.scales,
layers.49.self_attn.v_proj.weight,
layers.5.input_layernorm.weight,
layers.5.mlp.down_proj.biases,
layers.5.mlp.down_proj.scales,
layers.5.mlp.down_proj.weight,
layers.5.mlp.gate_proj.biases,
layers.5.mlp.gate_proj.scales,
layers.5.mlp.gate_proj.weight,
layers.5.mlp.up_proj.biases,
layers.5.mlp.up_proj.scales,
layers.5.mlp.up_proj.weight,
layers.5.post_attention_layernorm.weight,
layers.5.post_feedforward_layernorm.weight,
layers.5.pre_feedforward_layernorm.weight,
layers.5.self_attn.gate_proj.biases,
layers.5.self_attn.gate_proj.scales,
layers.5.self_attn.gate_proj.weight,
layers.5.self_attn.k_proj.biases,
layers.5.self_attn.k_proj.scales,
layers.5.self_attn.k_proj.weight,
layers.5.self_attn.o_proj.biases,
layers.5.self_attn.o_proj.scales,
layers.5.self_attn.o_proj.weight,
layers.5.self_attn.q_proj.biases,
layers.5.self_attn.q_proj.scales,
layers.5.self_attn.q_proj.weight,
layers.5.self_attn.v_proj.biases,
layers.5.self_attn.v_proj.scales,
layers.5.self_attn.v_proj.weight,
layers.50.input_layernorm.weight,
layers.50.mlp.down_proj.biases,
layers.50.mlp.down_proj.scales,
layers.50.mlp.down_proj.weight,
layers.50.mlp.gate_proj.biases,
layers.50.mlp.gate_proj.scales,
layers.50.mlp.gate_proj.weight,
layers.50.mlp.up_proj.biases,
layers.50.mlp.up_proj.scales,
layers.50.mlp.up_proj.weight,
layers.50.post_attention_layernorm.weight,
layers.50.post_feedforward_layernorm.weight,
layers.50.pre_feedforward_layernorm.weight,
layers.50.self_attn.gate_proj.biases,
layers.50.self_attn.gate_proj.scales,
layers.50.self_attn.gate_proj.weight,
layers.50.self_attn.k_proj.biases,
layers.50.self_attn.k_proj.scales,
layers.50.self_attn.k_proj.weight,
layers.50.self_attn.o_proj.biases,
layers.50.self_attn.o_proj.scales,
layers.50.self_attn.o_proj.weight,
layers.50.self_attn.q_proj.biases,
layers.50.self_attn.q_proj.scales,
layers.50.self_attn.q_proj.weight,
layers.50.self_attn.v_proj.biases,
layers.50.self_attn.v_proj.scales,
layers.50.self_attn.v_proj.weight,
layers.51.input_layernorm.weight,
layers.51.mlp.down_proj.biases,
layers.51.mlp.down_proj.scales,
layers.51.mlp.down_proj.weight,
layers.51.mlp.gate_proj.biases,
layers.51.mlp.gate_proj.scales,
layers.51.mlp.gate_proj.weight,
layers.51.mlp.up_proj.biases,
layers.51.mlp.up_proj.scales,
layers.51.mlp.up_proj.weight,
layers.51.post_attention_layernorm.weight,
layers.51.post_feedforward_layernorm.weight,
layers.51.pre_feedforward_layernorm.weight,
layers.51.self_attn.gate_proj.biases,
layers.51.self_attn.gate_proj.scales,
layers.51.self_attn.gate_proj.weight,
layers.51.self_attn.k_proj.biases,
layers.51.self_attn.k_proj.scales,
layers.51.self_attn.k_proj.weight,
layers.51.self_attn.o_proj.biases,
layers.51.self_attn.o_proj.scales,
layers.51.self_attn.o_proj.weight,
layers.51.self_attn.q_proj.biases,
layers.51.self_attn.q_proj.scales,
layers.51.self_attn.q_proj.weight,
layers.51.self_attn.v_proj.biases,
layers.51.self_attn.v_proj.scales,
layers.51.self_attn.v_proj.weight,
layers.6.input_layernorm.weight,
layers.6.mlp.down_proj.biases,
layers.6.mlp.down_proj.scales,
layers.6.mlp.down_proj.weight,
layers.6.mlp.gate_proj.biases,
layers.6.mlp.gate_proj.scales,
layers.6.mlp.gate_proj.weight,
layers.6.mlp.up_proj.biases,
layers.6.mlp.up_proj.scales,
layers.6.mlp.up_proj.weight,
layers.6.post_attention_layernorm.weight,
layers.6.post_feedforward_layernorm.weight,
layers.6.pre_feedforward_layernorm.weight,
layers.6.self_attn.gate_proj.biases,
layers.6.self_attn.gate_proj.scales,
layers.6.self_attn.gate_proj.weight,
layers.6.self_attn.k_proj.biases,
layers.6.self_attn.k_proj.scales,
layers.6.self_attn.k_proj.weight,
layers.6.self_attn.o_proj.biases,
layers.6.self_attn.o_proj.scales,
layers.6.self_attn.o_proj.weight,
layers.6.self_attn.q_proj.biases,
layers.6.self_attn.q_proj.scales,
layers.6.self_attn.q_proj.weight,
layers.6.self_attn.v_proj.biases,
layers.6.self_attn.v_proj.scales,
layers.6.self_attn.v_proj.weight,
layers.7.input_layernorm.weight,
layers.7.mlp.down_proj.biases,
layers.7.mlp.down_proj.scales,
layers.7.mlp.down_proj.weight,
layers.7.mlp.gate_proj.biases,
layers.7.mlp.gate_proj.scales,
layers.7.mlp.gate_proj.weight,
layers.7.mlp.up_proj.biases,
layers.7.mlp.up_proj.scales,
layers.7.mlp.up_proj.weight,
layers.7.post_attention_layernorm.weight,
layers.7.post_feedforward_layernorm.weight,
layers.7.pre_feedforward_layernorm.weight,
layers.7.self_attn.gate_proj.biases,
layers.7.self_attn.gate_proj.scales,
layers.7.self_attn.gate_proj.weight,
layers.7.self_attn.k_proj.biases,
layers.7.self_attn.k_proj.scales,
layers.7.self_attn.k_proj.weight,
layers.7.self_attn.o_proj.biases,
layers.7.self_attn.o_proj.scales,
layers.7.self_attn.o_proj.weight,
layers.7.self_attn.q_proj.biases,
layers.7.self_attn.q_proj.scales,
layers.7.self_attn.q_proj.weight,
layers.7.self_attn.v_proj.biases,
layers.7.self_attn.v_proj.scales,
layers.7.self_attn.v_proj.weight,
layers.8.input_layernorm.weight,
layers.8.mlp.down_proj.biases,
layers.8.mlp.down_proj.scales,
layers.8.mlp.down_proj.weight,
layers.8.mlp.gate_proj.biases,
layers.8.mlp.gate_proj.scales,
layers.8.mlp.gate_proj.weight,
layers.8.mlp.up_proj.biases,
layers.8.mlp.up_proj.scales,
layers.8.mlp.up_proj.weight,
layers.8.post_attention_layernorm.weight,
layers.8.post_feedforward_layernorm.weight,
layers.8.pre_feedforward_layernorm.weight,
layers.8.self_attn.gate_proj.biases,
layers.8.self_attn.gate_proj.scales,
layers.8.self_attn.gate_proj.weight,
layers.8.self_attn.k_proj.biases,
layers.8.self_attn.k_proj.scales,
layers.8.self_attn.k_proj.weight,
layers.8.self_attn.o_proj.biases,
layers.8.self_attn.o_proj.scales,
layers.8.self_attn.o_proj.weight,
layers.8.self_attn.q_proj.biases,
layers.8.self_attn.q_proj.scales,
layers.8.self_attn.q_proj.weight,
layers.8.self_attn.v_proj.biases,
layers.8.self_attn.v_proj.scales,
layers.8.self_attn.v_proj.weight,
layers.9.input_layernorm.weight,
layers.9.mlp.down_proj.biases,
layers.9.mlp.down_proj.scales,
layers.9.mlp.down_proj.weight,
layers.9.mlp.gate_proj.biases,
layers.9.mlp.gate_proj.scales,
layers.9.mlp.gate_proj.weight,
layers.9.mlp.up_proj.biases,
layers.9.mlp.up_proj.scales,
layers.9.mlp.up_proj.weight,
layers.9.post_attention_layernorm.weight,
layers.9.post_feedforward_layernorm.weight,
layers.9.pre_feedforward_layernorm.weight,
layers.9.self_attn.gate_proj.biases,
layers.9.self_attn.gate_proj.scales,
layers.9.self_attn.gate_proj.weight,
layers.9.self_attn.k_proj.biases,
layers.9.self_attn.k_proj.scales,
layers.9.self_attn.k_proj.weight,
layers.9.self_attn.o_proj.biases,
layers.9.self_attn.o_proj.scales,
layers.9.self_attn.o_proj.weight,
layers.9.self_attn.q_proj.biases,
layers.9.self_attn.q_proj.scales,
layers.9.self_attn.q_proj.weight,
layers.9.self_attn.v_proj.biases,
layers.9.self_attn.v_proj.scales,
layers.9.self_attn.v_proj.weight,
norm.weight.

```

#### Captured upstream output

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]
Fetching 13 files: 100%|##########| 13/13 [00:00<00:00, 2893.24it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:51:27] DEBUG    HF Cache Info for mlx-community/Muse-Glimmer-30B-OptiQ-4bit: size=22770.3 MB, files=15
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
- *Model load time:* 1.17s
- *Generation time:* -
- *Total time:* 1.17s
- *Input validation time:* 5.17e-05
- *Prompt preparation time:* -
- *First-token latency:* -
- *Cleanup time:* 0.142
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1292, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1437, in load_processor
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
  [Previous line repeated 10 more times]
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
Fetching 19 files: 100%|##########| 19/19 [00:00<00:00, 2703.52it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:59:14] DEBUG    HF Cache Info for tencent/Youtu-VL-4B-Instruct: size=10229.6 MB, files=27
```

</details>

---
