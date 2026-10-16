# Model Output Gallery

Generated on: 2026-10-16 23:21:25 BST

- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB

This run records model responses to one shared image and prompt (evaluation
lane: blind). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

Complete per-model evidence artifact with image metadata, the source prompt, a
facts-only chooser, and full generated or crash output for every attempted
model.

## Reference Image

![Reference image](assets/source-image-c4aeb8d18432656f.jpg)

## Current-run Chooser

Mechanical observations and captured resource facts for this run only. No concerns detected does not mean the response fulfilled an arbitrary prompt or described the image accurately. Consult the assessment scope above. Total time is end-to-end; throughput covers generation only and requires at least 16 generated tokens. Prefill/first is the measured time to first token (input preparation, prefill and the first decode step) when captured, else upstream's model-loop first-token time; Prompt tok is the full rendered prompt including image tokens, which drives prefill cost. For cross-attention architectures the token count reflects the tokenised text burden only, not total vision prefill compute.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                               | Mechanical checks               | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|---------|---------------------|-----------------|---------|------------|---------|---------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected`          | 0.58s   | 521 tok/s           | 0.06            | 1.2     | 350        | 47      | none                                                                |
| [`TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`](#model-technobaptist-ternary-bonsai-2-27b-mlx-2bit)                                 | `no concerns detected`          | 4.76s   | 40.9 tok/s          | 1.21            | 11      | 421        | 94      | none                                                                |
| [`mlx-community/AREX-2-4bit`](#model-mlx-community-arex-2-4bit)                                                                     | `no concerns detected`          | 5.16s   | 33.3 tok/s          | 0.68            | 17      | 421        | 80      | none                                                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `no concerns detected`          | 6.38s   | 109 tok/s           | 0.34            | 18      | 545        | 478     | none                                                                |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected`          | 2.54s   | 86.5 tok/s          | 0.33            | 8.0     | 499        | 49      | none                                                                |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                                | `no concerns detected`          | 12.08s  | 49.9 tok/s          | 3.25            | 63      | 499        | 85      | none                                                                |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected`          | 4.96s   | 56.1 tok/s          | 2.17            | 10      | 3,435      | 90      | none                                                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                               | `no concerns detected`          | 9.61s   | 67.8 tok/s          | 0.72            | 20      | 489        | 435     | none                                                                |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected`          | 1.69s   | 214 tok/s           | 0.13            | 3.6     | 344        | 74      | none                                                                |
| [`mlx-community/MiniCPM-V-4.6-4bit`](#model-mlx-community-minicpm-v-46-4bit)                                                        | `no concerns detected`          | 3.01s   | 255 tok/s           | 0.14            | 3.1     | 330        | 478     | none                                                                |
| [`mlx-community/MiniCPM-V-4.7-1B-4bit`](#model-mlx-community-minicpm-v-47-1b-4bit)                                                  | `no concerns detected`          | 1.35s   | 261 tok/s           | 0.14            | 3.6     | 332        | 64      | none                                                                |
| [`mlx-community/MiniCPM-V-4.7-35B-A3B-4bit`](#model-mlx-community-minicpm-v-47-35b-a3b-4bit)                                        | `no concerns detected`          | 4.15s   | 78.9 tok/s          | 0.39            | 24      | 332        | 86      | none                                                                |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `no concerns detected`          | 2.27s   | 103 tok/s           | 0.20            | 7.0     | 177        | 103     | none                                                                |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected`          | 3.31s   | 68.7 tok/s          | 0.70            | 9.8     | 1,070      | 72      | none                                                                |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected`          | 1.92s   | 198 tok/s           | 0.27            | 4.5     | 1,069      | 100     | none                                                                |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `no concerns detected`          | 3.05s   | 71.3 tok/s          | 0.51            | 8.5     | 873        | 95      | none                                                                |
| [`mlx-community/MolmoPoint-8B-4bit`](#model-mlx-community-molmopoint-8b-4bit)                                                       | `no concerns detected`          | 4.50s   | 32.0 tok/s          | 0.71            | 12      | 860        | 70      | none                                                                |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `no concerns detected`          | 44.80s  | 23.1 tok/s          | 0.99            | 25      | 545        | 937     | none                                                                |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected`          | 1.70s   | 164 tok/s           | 0.18            | 3.5     | 418        | 78      | none                                                                |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected`          | 2.78s   | 59.4 tok/s          | 0.27            | 9.1     | 883        | 94      | none                                                                |
| [`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit)                           | `no concerns detected`          | 3.80s   | 84.5 tok/s          | 0.37            | 23      | 409        | 87      | none                                                                |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected`          | 3.28s   | 88.4 tok/s          | 0.34            | 19      | 409        | 83      | none                                                                |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected`          | 1.80s   | 102 tok/s           | 0.28            | 6.8     | 409        | 60      | none                                                                |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected`          | 4.11s   | 79.3 tok/s          | 0.37            | 21      | 421        | 102     | none                                                                |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected`          | 6.49s   | 30.5 tok/s          | 0.69            | 17      | 421        | 113     | none                                                                |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected`          | 1.37s   | 131 tok/s           | 0.41            | 5.6     | 1,204      | 31      | none                                                                |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected`          | 4.09s   | 100 tok/s           | 1.54            | 6.5     | 2,909      | 82      | none                                                                |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected`          | 5.08s   | 54.0 tok/s          | 1.74            | 28      | 386        | 67      | none                                                                |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                             | `no concerns detected`          | 5.94s   | 31.5 tok/s          | 0.71            | 17      | 379        | 89      | none                                                                |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                                   | `no concerns detected`          | 2.89s   | 72.3 tok/s          | 0.29            | 7.2     | 378        | 63      | none                                                                |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `no concerns detected`          | 3.33s   | 62.6 tok/s          | 0.28            | 7.5     | 390        | 77      | none                                                                |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected`          | 3.64s   | 115 tok/s           | 0.39            | 16      | 390        | 81      | none                                                                |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected`          | 2.42s   | 125 tok/s           | 0.19            | 5.9     | 386        | 62      | none                                                                |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected`          | 1.33s   | 179 tok/s           | 0.30            | 4.7     | 734        | 66      | none                                                                |
| [`mlx-community/granite-vision-3.2-2b-nvfp4`](#model-mlx-community-granite-vision-32-2b-nvfp4)                                      | `no concerns detected`          | 2.22s   | 174 tok/s           | 1.08            | 3.9     | 3,095      | 119     | none                                                                |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                           | `no concerns detected`          | 4.80s   | 41.1 tok/s          | 0.80            | 15      | 1,335      | 89      | none                                                                |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected`          | 1.60s   | 149 tok/s           | 0.16            | 4.9     | 420        | 89      | none                                                                |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected`          | 5.03s   | 37.3 tok/s          | 0.76            | 17      | 538        | 87      | none                                                                |
| [`nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit`](#model-nativ-community-nvidia-nemotron-35-bolt-74b-a9b-4bit)              | `no concerns detected`          | 6.89s   | 61.4 tok/s          | 0.75            | 46      | 420        | 105     | none                                                                |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected`          | 3.25s   | 141 tok/s           | 0.33            | 21      | 421        | 65      | none                                                                |
| [`sahilchachra/LensVLM-9B-MXFP4`](#model-sahilchachra-lensvlm-9b-mxfp4)                                                             | `no concerns detected`          | 2.49s   | 105 tok/s           | 0.27            | 6.8     | 421        | 111     | none                                                                |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns: answer format` | 0.88s   | 356 tok/s           | 0.12            | 2.2     | 120        | 32      | labelled fields not detected                                        |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `major concerns: generation`    | 4.09s   | 319 tok/s           | 0.37            | 2.3     | 3,438      | 1,000   | repeated text; cut off at token limit; duplicate keywords           |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)                        | `major concerns: generation`    | 56.01s  | 19.1 tok/s          | 1.51            | 15      | 107        | 1,000   | repeated text; cut off at token limit; duplicate keywords           |
| [`mlx-community/Qwen2-VL-2B-mlx`](#model-mlx-community-qwen2-vl-2b-mlx)                                                             | `major concerns: generation`    | 8.11s   | 141 tok/s           | 0.17            | 5.4     | 511        | 1,000   | repeated text; labelled fields not detected; cut off at token limit |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                                       | `major concerns: answer format` | 0.64s   | insufficient sample | 0.21            | 1.1     | 983        | 8       | labelled fields not detected                                        |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                                      | `major concerns: answer format` | 32.15s  | 50.5 tok/s          | 13.10           | 87      | 286        | 236     | answer emitted twice; control tokens visible                        |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                                       | `major concerns: generation`    | 5.79s   | 64.9 tok/s          | 0.35            | 10      | 511        | 275     | repeated text; stopped early: repeating; duplicate keywords         |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns: answer format` | 2.66s   | insufficient sample | 1.37            | 7.0     | 3,512      | 14      | control tokens visible; labelled fields not detected                |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns: answer format` | 1.05s   | 224 tok/s           | 0.10            | 1.6     | 116        | 97      | labelled fields not detected                                        |
| [`vikhyatk/moondream2`](#model-vikhyatk-moondream2)                                                                                 | `major concerns: answer format` | 1.41s   | 177 tok/s           | 0.15            | 4.5     | 833        | 34      | labelled fields not detected                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 0.58s

Lowest peak memory among completions without detected concerns: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.2 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Major Mechanical Concerns in This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                        | Mechanical checks               | Observations                                                        |
|--------------------------------------------------------------------------------------------------------------|---------------------------------|---------------------------------------------------------------------|
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                   | `major concerns: answer format` | labelled fields not detected                                        |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                               | `major concerns: generation`    | repeated text; cut off at token limit; duplicate keywords           |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit) | `major concerns: generation`    | repeated text; cut off at token limit; duplicate keywords           |
| [`mlx-community/Qwen2-VL-2B-mlx`](#model-mlx-community-qwen2-vl-2b-mlx)                                      | `major concerns: generation`    | repeated text; labelled fields not detected; cut off at token limit |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                | `major concerns: answer format` | labelled fields not detected                                        |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                               | `major concerns: answer format` | answer emitted twice; control tokens visible                        |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                | `major concerns: generation`    | repeated text; stopped early: repeating; duplicate keywords         |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | `major concerns: answer format` | control tokens visible; labelled fields not detected                |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                 | `major concerns: answer format` | labelled fields not detected                                        |
| [`vikhyatk/moondream2`](#model-vikhyatk-moondream2)                                                          | `major concerns: answer format` | labelled fields not detected                                        |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

A compact preview of each model's final answer (or failure evidence for crashes), in chooser order. Where the requested catalogue fields were detected, the preview shows a little of each: the title, the start of the description, and the first keywords with their count, so the weakest field is not hidden behind a long description. Otherwise it is the first 280 characters. A closed reasoning trace is left out of the preview and reported as an omitted-character count; the complete output, trace included, is in the model's evidence block below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                               | Mechanical checks      | Output preview                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected` | Title: Cats on a Pink Couch \| Description: Two tabby cats are lying on a bright pink couch, appearing relaxed and comfortable. \| Keywords (6): cats, pink couch, relaxation, tabby, comfortable, cozy                                                                                                                                                                               |
| [`TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`](#model-technobaptist-ternary-bonsai-2-27b-mlx-2bit)                                 | `no concerns detected` | Title: Two tabby cats sleeping on pink sofa \| Description: Two tabby cats lie on their sides on a bright pink sofa cushion, with two white remote controls resting ne... \| Keywords (14): tabby cats, sleeping, pink sofa, remote controls, striped fur, relaxed, cushion, indoor, pets, resting, ...                                                                               |
| [`mlx-community/AREX-2-4bit`](#model-mlx-community-arex-2-4bit)                                                                     | `no concerns detected` | Title: Two Tabby Cats Sleeping on Pink Couch \| Description: Two tabby cats lie sleeping on a bright pink couch, each positioned beside a white remote control. The... \| Keywords (8): tabby cats, sleeping, pink couch, remote controls, striped fur, indoor pets, relaxed posture, domestic setting                                                                                |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `no concerns detected` | Title: Two cats on a sofa \| Description: Two cats, one with a green collar, rest on a sofa with a remote control nearby. The scene is indoors with soft lighting. \| Keywords (17): cat, cats, pet, domestic cat, animal, two cats, lying down, resting, sofa, couch, feline, green collar, ...[1,410 characters of reasoning omitted; complete output in the evidence block]        |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected` | Title: Two tabby cats on pink fabric. \| Description: Two tabby cats lie on a pink fabric surface, with remote controls nearby. \| Keywords (5): tabby cats, pink fabric, remote controls, domestic cats, feline.                                                                                                                                                                     |
| [`mlx-community/GLM-4.6V-nvfp4`](#model-mlx-community-glm-46v-nvfp4)                                                                | `no concerns detected` | Title: Two cats on a pink couch \| Description: Two cats lie on a pink couch, each near a remote control, with one cat’s head resting on the fabric and the other st... \| Keywords (17): cats, couch, pink, remote controls, sleeping, relaxed, feline, domestic, indoor, pets, lounging, striped, ...                                                                               |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected` | Title: Cats Sleeping on Sofa with Remote Controls \| Description: Two cats are sleeping on a pink sofa, each with a remote control nearby. The lighting is soft,... \| Keywords (18): cats, sleeping, sofa, pink, remote controls, striped, relaxed, home, pets, cozy, domestic, resting, comfort, ...                                                                                |
| [`mlx-community/Kimi-VL-A3B-Thinking-2506-8bit`](#model-mlx-community-kimi-vl-a3b-thinking-2506-8bit)                               | `no concerns detected` | Title: Two tabby cats asleep on pink sofa with remote controls \| Description: Two tabby cats lie asleep on a pink fabric surface with remote controls nearby, showca... \| Keywords (11): tabby cats, pink sofa, asleep, remote controls, fabric surface, two cats, relaxed posture, blue remotes, ...[1,505 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected` | Title: Two cats sleeping on a pink couch. \| Description: Two tabby cats are lying on a pink couch with remote controls nearby. The lighting is bright, highlighting... \| Keywords (10): cats, pink couch, tabby, sleeping, remote controls, bright lighting, fur patterns, feline, relaxation, indoor                                                                               |
| [`mlx-community/MiniCPM-V-4.6-4bit`](#model-mlx-community-minicpm-v-46-4bit)                                                        | `no concerns detected` | Title: Two Sleeping Cats on Pink Couch \| Description: Two tabby cats are resting and sleeping on a pink fabric, with remote controls visible. \| Keywords (9): cats, sleeping, tabby, pink, couch, remote, relaxation, indoor, sleep[1,668 characters of reasoning omitted; complete output in the evidence block]                                                                   |
| [`mlx-community/MiniCPM-V-4.7-1B-4bit`](#model-mlx-community-minicpm-v-47-1b-4bit)                                                  | `no concerns detected` | Title: Sleeping Cats on Pink Couch \| Description: Two tabby cats rest on a pink fabric couch, one near a remote, the other beside it, under soft indoor lighting. \| Keywords (12): tabby, cat, sleeping, pink, couch, remote, indoor, light, fur, stretch, curled, relaxed                                                                                                          |
| [`mlx-community/MiniCPM-V-4.7-35B-A3B-4bit`](#model-mlx-community-minicpm-v-47-35b-a3b-4bit)                                        | `no concerns detected` | Title: Two Tabby Cats Sleeping on Pink Sofa \| Description: Two tabby cats are lying on a bright pink sofa cover, each positioned near a white remote control. The sc... \| Keywords (15): cats, tabby, sleeping, pink, sofa, remote control, furniture, domestic, animal, resting, striped, cover, ...                                                                               |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `no concerns detected` | Title: Two tabby cats sleeping on red couch \| Description: Two tabby cats lie side by side on a red upholstered couch, partially covered by a pink blanket. A re... \| Keywords (11): tabby cats, sleeping, red couch, pink blanket, remote controls, indoor, feline, rest, domestic animals, pet, ...                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected` | Title: Two Cats Relaxing on Pink Cushion \| Description: Two cats lie stretched out on a pink cushion, positioned near remote controls. The setting appears indo... \| Keywords (10): cats, pink cushion, remote controls, indoor setting, relaxed posture, soft lighting, resting, domestic scene, ...                                                                               |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected` | Title: Two Kitten Friends on a Pink Couch \| Description: Two striped kittens lie relaxed on a bright pink cushion, each with a remote control beside them. The larger k... \| Keywords (10): kitten, striped-pattern, pink-cushion, remote-control, cozy-interior, resting-pose, domestic-setting, ...                                                                               |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `no concerns detected` | Title: Two Calico Cats Sleeping on Pink Bedding \| Description: Two calico cats are sleeping side by side on pink bedding. They are positioned with their heads facin... \| Keywords (10): calico cats, sleeping, pink bedding, white remotes, black and brown fur, white cats, striped cats, cozy, ...                                                                               |
| [`mlx-community/MolmoPoint-8B-4bit`](#model-mlx-community-molmopoint-8b-4bit)                                                       | `no concerns detected` | Title: Two Cats Sleeping on Pink Blanket with Remote Controls \| Description: Two cats are sleeping on a pink blanket. A white remote control is visible behind them. Th... \| Keywords (10): cats, pink blanket, remote controls, sleeping, white remote, comfortable, soft surface, feline, rest, ...                                                                               |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `no concerns detected` | Title: Two Tabby Cats Resting On Pink Blanket \| Description: Two tabby cats lie on a bright pink blanket on a red cushioned surface. One cat appears asleep w... \| Keywords (15): cats, tabby, pink blanket, red sofa, remote controls, resting, sleeping, indoor, pets, fabric, cushion, collar, ...                                                                               |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected` | Title: "Cats Resting on a Pink Couch" \| Description: "Two cats are comfortably sprawled out on a vibrant pink couch, basking in the soft light of the day. Their relaxed postu... \| Keywords (12): cats, couch, resting, lounging, pink, soft light, relaxed, tranquil, basking, cozy, serene, scene.                                                                               |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected` | Title: Two Cats Sleeping on Pink Sofa \| Description: Two cats are sleeping on a pink sofa with remote controls beside them. \| Keywords (19): cats, sleeping, pink sofa, remote controls, relaxed, comfortable, indoor, domestic, fur, striped, resting, ...                                                                                                                         |
| [`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit)                           | `no concerns detected` | Title: Two tabby cats sleeping on sofa \| Description: Two tabby cats are sleeping on a bright pink sofa. The sofa has two remote controls placed near the cats. \| Keywords (18): tabby cats, sleeping, sofa, pink, remote controls, domestic cats, resting, animal, pet, living room, indoor, ...                                                                                   |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected` | Title: Two cats resting on a pink surface \| Description: Two cats are lying on a bright pink fabric surface. Two remote controls are visible near the cats. \| Keywords (19): cats, pink fabric, remote controls, lying down, resting, domestic cats, patterned fur, light source, ...                                                                                               |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected` | Title: Two Cats Sleeping on Pink Sofa \| Description: Two tabby cats rest peacefully on a bright pink sofa. Remote controls lie beside them. Natural daylight illuminates the scene. \| Keywords (10): cats, sleeping, pink, sofa, remote, tabby, rest, daylight, couch, feline                                                                                                       |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected` | Title: Two Tabby Cats Sleeping On Pink Sofa \| Description: Two striped tabby cats lie asleep on a bright pink sofa, each near a remote control. Their relaxed p... \| Keywords (19): cats, tabby, sleeping, sofa, pink, remotes, domestic pets, striped fur, cozy setting, animal rest, furniture, ...                                                                               |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected` | Title: Two Tabby Cats Reclining on Pink Sofa Cushion \| Description: Two striped tabby cats recline on a bright pink fabric surface with two white television r... \| Keywords (14): tabby cats, pink cushion, reclining posture, striped coats, remote controls, domestic setting, fabric texture, ...                                                                               |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected` | Title: Two Cats on a Couch \| Description: Two cats are sleeping on a pink couch. \| Keywords (4): cats, couch, sleeping, pink                                                                                                                                                                                                                                                        |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected` | Title: Two Cats Relaxing Together \| Description: Two tabby cats are curled up and sleeping side by side on a pink blanket, with a remote control placed nearby on the sofa. The sce... \| Keywords (10): cats, sleeping, pink blanket, sofa, remote control, tabby, relaxed, home, indoor, comfortable                                                                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected` | Title: Two tabby cats sleeping on a pink blanket \| Description: Two tabby cats are lying down on a pink fabric surface next to two remote controls. \| Keywords (15): cats, tabby, sleeping, pink, blanket, couch, remote, controls, indoor, domestic, resting, fur, striped, feline, ...                                                                                            |
| [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)                                             | `no concerns detected` | Title: Two Tabby Cats Resting on Pink Fabric \| Description: Two tabby cats lie prone on a pink fabric surface, each with a remote control nearby. Both cats are... \| Keywords (18): cats, tabby, pink, fabric, remote controls, sleeping, domestic animals, indoors, lying down, striped, feline, ...                                                                               |
| [`mlx-community/gemma-3n-E4B-it-4bit`](#model-mlx-community-gemma-3n-e4b-it-4bit)                                                   | `no concerns detected` | Title: Cozy Cat Nap \| Description: Two tabby cats are peacefully asleep on a pink couch, each nestled beside a remote control. \| Keywords (15): cats, sleeping, couch, remote, tabby, nap, pets, cozy, relaxation, comfort, indoors, rest, feline, peaceful, ...                                                                                                                    |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `no concerns detected` | Title: Two Tabby Cats on a Pink Sofa \| Description: Two tabby cats lie on a pink velvet sofa, with one resting its head on a remote control. The scene is illumina... \| Keywords (15): cats, tabby, sofa, couch, pet, sleeping, reclining, domestic, animal, grey, striped, fur, furniture, pink, ...                                                                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected` | Title: Two tabby cats lying on a pink surface \| Description: Two tabby cats lie on a pink fabric surface next to two remote controls. The cats are positioned w... \| Keywords (12): tabby cats, pink fabric, remote controls, lying down, domestic cats, striped fur, indoor, two cats, close-up, ...                                                                               |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected` | Title: Two cats rest upon a pink surface \| Description: Two cats lie on a bright pink surface. One cat's owner's remote control is visible nearby. \| Keywords (12): cats, feline, resting, pink, surface, animals, tortoiseshell, tabby, indoors, slumber, domestic, companion                                                                                                      |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected` | Title: "Cats at Rest on a Vibrant Red Sofa" \| Description: Two cats are peacefully sleeping on a bright red sofa, with a remote control nearby, suggesting a relaxed home environment. \| Keywords (10): cats, rest, red sofa, remote control, home, comfort, sleep, striped, domestic, tranquility                                                                                  |
| [`mlx-community/granite-vision-3.2-2b-nvfp4`](#model-mlx-community-granite-vision-32-2b-nvfp4)                                      | `no concerns detected` | Title: "Whisker Waves" \| Description: A close-up photograph of a cat's whiskers in motion, captured in a blurred motion blur against a pink background. The cat's fur is a mix of... \| Keywords (7): Cat, Whiskers, Motion Blur, Pink Background, Soft Lighting, Fur Pattern, Motion Blur Photography                                                                               |
| [`mlx-community/pixtral-12b-8bit`](#model-mlx-community-pixtral-12b-8bit)                                                           | `no concerns detected` | Title: Two Cats Lounging on Red Couch \| Description: Two cats, one lying on its back and the other on its side, relax on a red couch. Remote controls are plac... \| Keywords (17): cats, lounging, red couch, remote controls, relaxed, indoor, comfort, feline, domestic, paws, stretch, lounge, ...                                                                               |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected` | Title: Two cats sleeping on a red couch with remote controls nearby. \| Description: Two cats are sleeping on a red couch, with one cat lying on its side and the other cat... \| Keywords (10): cats, sleeping, red couch, remote controls, living room, natural light, side, stomach, remote, living                                                                                |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected` | Title: Two Cats Resting on a Pink Blanket \| Description: Two tabby cats are lying on a pink blanket, one stretched out and the other curled up. A remote contr... \| Keywords (18): cats, tabby, pink blanket, resting, remote control, fur, relaxed, indoor, couch, paws, stripes, sleeping, pet, ...                                                                               |
| [`nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit`](#model-nativ-community-nvidia-nemotron-35-bolt-74b-a9b-4bit)              | `no concerns detected` | Title: Two tabby cats sleeping on a pink couch \| Description: Two domestic cats with mackerel tabby markings are lying asleep on a bright pink fabric surface,... \| Keywords (16): cats, sleeping, tabby, mackerel, pink, couch, sofa, remote control, pet, domestic animal, rest, fur, striped, ...                                                                                |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected` | Title: Two cats sleeping on a sofa \| Description: Two tabby cats are sleeping on a pink sofa, one curled up and the other stretched out, with remote controls nearby. \| Keywords (10): cat, tabby, pink sofa, remote control, sleeping, curled, stretched, pink blanket, pink cushion, cat furniture                                                                                |
| [`sahilchachra/LensVLM-9B-MXFP4`](#model-sahilchachra-lensvlm-9b-mxfp4)                                                             | `no concerns detected` | Title: Two tabby cats sleeping on a pink sofa \| Description: Two tabby cats are curled up asleep on a bright pink fabric surface, likely a sofa or blanket. One cat lies on its side near the left edg... \| Keywords (10): cats, sleeping, tabby, pink, sofa, remote, rest, curled, striped, domestic                                                                               |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns`       | A serene moment of feline tranquility on a vibrant pink couch, with two cats in peaceful slumber, their fur patterns contrasting against the soft fabric.                                                                                                                                                                                                                             |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `major concerns`       | Title: Two cats sleeping on a pink couch \| Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a r... \| Keywords (144): cats, sleeping, pink couch, remote control, striped, British English, British, British-English, ...                                                                               |
| [`mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`](#model-mlx-community-llama-32-11b-vision-instruct-8bit)                        | `major concerns`       | Title: Two Cats on a Pink Blanket \| Description: Two tabby cats are sleeping on a pink blanket, with two remote controls nearby. The cats are lying on their backs, w... \| Keywords (141): Tabby cats, pink blanket, remote controls, sleeping, feline, domestic, household, comfort, relaxation, ...                                                                               |
| [`mlx-community/Qwen2-VL-2B-mlx`](#model-mlx-community-qwen2-vl-2b-mlx)                                                             | `major concerns`       | Title: Two cats sleeping on a pink couch \| Description: Two cats, one with a green collar, are sleeping on a pink couch. One cat is lying on its side, while the other cat is lying on its back. Both cats are striped and have their eyes closed. There are two remote... \| Keywords: (not detected)                                                                               |
| [`mlx-community/SmolVLM-256M-Instruct-4bit`](#model-mlx-community-smolvlm-256m-instruct-4bit)                                       | `major concerns`       | The main subject is a cat.                                                                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/Step-3.7-Flash-oQ3e`](#model-mlx-community-step-37-flash-oq3e)                                                      | `major concerns`       | Title: Two cats sleeping on a pink blanket \| Description: Two tabby cats are lying on a bright pink blanket on a red sofa. Both cats appear to be asleep, with one c... \| Keywords (19): cats, sleeping, pink blanket, red sofa, tabby cats, remote controls, green collar, large cat, small cat, ...                                                                               |
| [`mlx-community/X-Reasoner-7B-8bit`](#model-mlx-community-x-reasoner-7b-8bit)                                                       | `major concerns`       | Title: Cats Sleeping on Pink Couch with Remotes \| Description: Two tabby cats are curled up on a pink couch, one slightly larger than the other, both appeari... \| Keywords (54): cats, sleeping, pink couch, tabby, remote control, relaxed, striped fur, natural light, cozy, domestic, feline, ...                                                                               |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns`       | <\|channel\|> analysis<\|message\|> The two cats are sleeping on a pink couch.                                                                                                                                                                                                                                                                                                        |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns`       | Title: "A Study in Stripes" \| Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting... \| Keywords: (not detected)                                                                                |
| [`vikhyatk/moondream2`](#model-vikhyatk-moondream2)                                                                                 | `major concerns`       | Two tabby cats sleep on a pink blanket on a red couch. One cat is sleeping with its eyes closed, while the other is sleeping with its eyes open. A remote control is visible near the cats.                                                                                                                                                                                           |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.7.8`
- `mlx`: `0.32.5.dev20261016+b6642e604`
- `transformers`: `5.20.1`
- `tokenizers`: `0.23.3`
- `huggingface-hub`: `2.2.0`
- *Python Version:* 3.14.7
- *OS:* Darwin 27.0.0
- *macOS Version:* 27.0.1
- *GPU/Chip:* Apple M5 Max
- *MLX Device:* Apple M5 Max
- *GPU Architecture:* applegpu_g17s
- *RAM:* 128.0 GB
- *Recommended Working Set:* 115 GB
- *Fused Attention:* Available

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Create British-English catalogue metadata using only clearly visible facts.
> Omit uncertain details and unsupported identity, location, event, brand,
> species, period, or intent.
>
> &#8203;Write:
> &#45; a concrete 5-10-word title;
> &#45; a 1-2-sentence factual description of the main subject, setting, action,
> lighting, and distinctive details;
> &#45; 10-18 unique, comma-separated keywords.
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
- *Model load time:* 0.40s
- *Generation time:* 0.15s
- *Total time:* 0.58s
- *Prompt preparation time:* 0.014
- *First-token latency:* 0.0522
- *Cleanup time:* 0.0686
- *Prompt tokens:* 350
- *Generation tokens:* 47
- *Total tokens:* 397
- *Prompt throughput (raw):* 6,702 tok/s
- *Generation throughput (raw):* 521 tok/s
- *Peak memory (GB):* 1.2
- *Active memory (GB):* 0.90
- *Cache memory (GB):* 0.02
- *Model-load active memory (GB):* 0.897
- *Post-cleanup active memory (GB):* 0.000133
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 523
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
Cats on a Pink Couch

Description:
Two tabby cats are lying on a bright pink couch, appearing relaxed and comfortable.

Keywords:
cats, pink couch, relaxation, tabby, comfortable, cozy
</pre>

</details>

---

<a id="model-technobaptist-ternary-bonsai-2-27b-mlx-2bit"></a>

### TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit

<details>
<summary>Complete evidence: TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type
  prism_hadamard_qwen35)
- *Model load time:* 1.21s
- *Generation time:* 3.51s
- *Total time:* 4.76s
- *Prompt preparation time:* 0.0239
- *First-token latency:* 1.15
- *Cleanup time:* 0.114
- *Prompt tokens:* 421
- *Generation tokens:* 94
- *Total tokens:* 515
- *Prompt throughput (raw):* 365 tok/s
- *Generation throughput (raw):* 40.9 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 8.6
- *Cache memory (GB):* 0.59
- *Model-load active memory (GB):* 8.6
- *Post-cleanup active memory (GB):* 0.00113
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 498775b03b55d038f716fea2568bf6ae6b8e0b5e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--TechnoBaptist--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/498775b03b55d038f716fea2568bf6ae6b8e0b5e
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two tabby cats sleeping on pink sofa
Description: Two tabby cats lie on their sides on a bright pink sofa cushion, with two white remote controls resting nearby. The cats are positioned head-to-head, appearing relaxed and asleep in soft, even lighting.
Keywords: tabby cats, sleeping, pink sofa, remote controls, striped fur, relaxed, cushion, indoor, pets, resting, two cats, white remotes, fabric, lounging
</pre>

</details>

---

<a id="model-mlx-community-arex-2-4bit"></a>

### mlx-community/AREX-2-4bit

<details>
<summary>Complete evidence: mlx-community/AREX-2-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 2.04s
- *Generation time:* 3.09s
- *Total time:* 5.16s
- *Prompt preparation time:* 0.00973
- *First-token latency:* 0.671
- *Cleanup time:* 0.134
- *Prompt tokens:* 421
- *Generation tokens:* 80
- *Total tokens:* 501
- *Prompt throughput (raw):* 627 tok/s
- *Generation throughput (raw):* 33.3 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.00164
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 02551ee54839c7541ea5adb44b83c9aa86aeb4f5
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--AREX-2-4bit/snapshots/02551ee54839c7541ea5adb44b83c9aa86aeb4f5
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Sleeping on Pink Couch

Description: Two tabby cats lie sleeping on a bright pink couch, each positioned beside a white remote control. The scene is lit by even indoor light, highlighting the cats' striped coats and relaxed postures.

Keywords: tabby cats, sleeping, pink couch, remote controls, striped fur, indoor pets, relaxed posture, domestic setting
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
- *Model load time:* 1.63s
- *Generation time:* 4.72s
- *Total time:* 6.38s
- *Prompt preparation time:* 0.0195
- *First-token latency:* 0.329
- *Cleanup time:* 0.101
- *Prompt tokens:* 545
- *Generation tokens:* 478
- *Total tokens:* 1,023
- *Prompt throughput (raw):* 1,659 tok/s
- *Generation throughput (raw):* 109 tok/s
- *Peak memory (GB):* 18
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 17.4
- *Post-cleanup active memory (GB):* 0.00185
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 662
- *Processor:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 846ea5576854468f25af3767d769b0805b1b08b6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--ERNIE-4.5-VL-28B-A3B-Thinking-4bit/snapshots/846ea5576854468f25af3767d769b0805b1b08b6
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 0.6, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_p": 0.95}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get this done. I need to create some metadata for an image, and the instructions are pretty clear. First things first: the title. I need something short and sweet. "Two cats on a sofa" seems about right, at five words. Now for the description. It's about the main subject, which is the cats. They're lying on a sofa, probably indoors. The lighting is soft, and I see a couple of distinct features: one cat has a green collar, and there's a remote control nearby. Gotta keep it factual. "Two cats, one with a green collar, rest on a sofa with a remote control nearby." That's two sentences, hits all the points.

Now for the keywords. I need to be specific, and only use what's *clearly* visible. Let's see... "cat", "cats", "pet", "domestic cat", "animal", "two cats", "lying down", "resting", "sofa", "couch", "feline", "green collar", "remote control", "television remote", "indoor", "pet cat", "pet", "home", "feline", "animal", "sleeping". I'll trim it down to a few more, but that's the core. I think it's a good selection. Let's go with that.

So, final check. Title: "Two cats on a sofa." Description: "Two cats, one with a green collar, rest on a sofa with a remote control nearby." Keywords: "cat, cats, pet, domestic cat, animal, two cats, lying down, resting, sofa, couch, feline, green collar, remote control, television remote, indoor, pet cat, home". Yep. That's it. Done.
&lt;/think&gt;

Title: Two cats on a sofa

Description: Two cats, one with a green collar, rest on a sofa with a remote control nearby. The scene is indoors with soft lighting.

Keywords: cat, cats, pet, domestic cat, animal, two cats, lying down, resting, sofa, couch, feline, green collar, remote control, television remote, indoor, pet cat, home
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get this done. I need to create some metadata for an image, and the instructions are pretty clear. First things first: the title. I need something short and sweet. "Two cats on a sofa" seems about right, at five words. Now for the description. It's about the main subject, which is the cats. They're lying on a sofa, probably indoors. The lighting is soft, and I see a couple of distinct features: one cat has a green collar, and there's a remote control nearby. Gotta keep it factual. "Two cats, one with a green collar, rest on a sofa with a remote control nearby." That's two sentences, hits all the points.

Now for the keywords. I need to be specific, and only use what's *clearly* visible. Let's see... "cat", "cats", "pet", "domestic cat", "animal", "two cats", "lying down", "resting", "sofa", "couch", "feline", "green collar", "remote control", "television remote", "indoor", "pet cat", "pet", "home", "feline", "animal", "sleeping". I'll trim it down to a few more, but that's the core. I think it's a good selection. Let's go with that.

So, final check. Title: "Two cats on a sofa." Description: "Two cats, one with a green collar, rest on a sofa with a remote control nearby." Keywords: "cat, cats, pet, domestic cat, animal, two cats, lying down, resting, sofa, couch, feline, green collar, remote control, television remote, indoor, pet cat, home". Yep. That's it. Done.
</think>

Title: Two cats on a sofa

Description: Two cats, one with a green collar, rest on a sofa with a remote control nearby. The scene is indoors with soft lighting.

Keywords: cat, cats, pet, domestic cat, animal, two cats, lying down, resting, sofa, couch, feline, green collar, remote control, television remote, indoor, pet cat, home
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
- *Model load time:* 1.61s
- *Generation time:* 0.90s
- *Total time:* 2.54s
- *Prompt preparation time:* 0.0146
- *First-token latency:* 0.327
- *Cleanup time:* 0.0981
- *Prompt tokens:* 499
- *Generation tokens:* 49
- *Total tokens:* 548
- *Prompt throughput (raw):* 1,525 tok/s
- *Generation throughput (raw):* 86.5 tok/s
- *Peak memory (GB):* 8.0
- *Active memory (GB):* 7.1
- *Cache memory (GB):* 0.06
- *Model-load active memory (GB):* 7.08
- *Post-cleanup active memory (GB):* 0.00248
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 553
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
Title: Two tabby cats on pink fabric.
Description: Two tabby cats lie on a pink fabric surface, with remote controls nearby.
Keywords: tabby cats, pink fabric, remote controls, domestic cats, feline.
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
- *Model load time:* 7.09s
- *Generation time:* 4.96s
- *Total time:* 12.08s
- *Prompt preparation time:* 0.0177
- *First-token latency:* 2.18
- *Cleanup time:* 0.217
- *Prompt tokens:* 499
- *Generation tokens:* 85
- *Total tokens:* 584
- *Prompt throughput (raw):* 228 tok/s
- *Generation throughput (raw):* 49.9 tok/s
- *Peak memory (GB):* 63
- *Active memory (GB):* 62
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 61.9
- *Post-cleanup active memory (GB):* 0.00279
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 553
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
Title: Two cats on a pink couch
Description: Two cats lie on a pink couch, each near a remote control, with one cat’s head resting on the fabric and the other stretched out with eyes closed.
Keywords: cats, couch, pink, remote controls, sleeping, relaxed, feline, domestic, indoor, pets, lounging, striped, tabby, cozy, comfortable, household, leisure
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
- *Generation time:* 3.77s
- *Total time:* 4.96s
- *Prompt preparation time:* 0.0275
- *First-token latency:* 2.08
- *Cleanup time:* 0.119
- *Prompt tokens:* 3,435
- *Generation tokens:* 90
- *Total tokens:* 3,525
- *Prompt throughput (raw):* 1,648 tok/s
- *Generation throughput (raw):* 56.1 tok/s
- *Peak memory (GB):* 10
- *Active memory (GB):* 8.9
- *Cache memory (GB):* 0.79
- *Model-load active memory (GB):* 8.94
- *Post-cleanup active memory (GB):* 0.0031
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 477
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
Title: Cats Sleeping on Sofa with Remote Controls

Description:
Two cats are sleeping on a pink sofa, each with a remote control nearby. The lighting is soft, highlighting the relaxed posture of the cats and the vibrant pink fabric of the sofa.

Keywords:
cats, sleeping, sofa, pink, remote controls, striped, relaxed, home, pets, cozy, domestic, resting, comfort, household, feline, nap, indoors, tranquility
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
- *Model load time:* 2.42s
- *Generation time:* 7.14s
- *Total time:* 9.61s
- *Prompt preparation time:* 0.0389
- *First-token latency:* 0.593
- *Cleanup time:* 0.13
- *Prompt tokens:* 489
- *Generation tokens:* 435
- *Total tokens:* 924
- *Prompt throughput (raw):* 825 tok/s
- *Generation throughput (raw):* 67.8 tok/s
- *Peak memory (GB):* 20
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.76
- *Model-load active memory (GB):* 17.8
- *Post-cleanup active memory (GB):* 0.00407
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 659
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
◁think▷Okay, let's tackle this task. First, I need to create a British-English catalogue metadata based on the image. The user wants only visible facts, no uncertain details. Let's start by looking at the image.

The main subject is two cats sleeping on a pink fabric surface. There's a remote control next to one cat. The lighting is bright, and the colors are vivid, maybe enhanced. The cats are tabby, so I should note that. The setting is a couch, so "sofa" or "cushion" could be better than "couch" for British English. The action is sleeping, so "asleep" instead of just "sleeping". Distinctive details: the remote controls are blue, the fabric is pink, the cats have distinct tabby patterns.

Now, the title needs to be concise, 5-10 words. Maybe "Two tabby cats asleep on pink sofa with remote controls." That's 10 words. Next, the description: one sentence. Mention the setting (pink sofa), the cats (two tabby, sleeping), the remote controls. So: "Two tabby cats lie asleep on a pink fabric surface with remote controls nearby." That's 20 words, stays within the limit.

Keywords: need 10-18 unique, comma-separated. Keywords: tabby cats, pink sofa, asleep, remote controls, fabric surface, two cats, relaxed posture, blue remotes, indoor setting, soft lighting, relaxed. Let me check if these are all visible. Yes, the remotes are blue, the fabric is pink, they're on a sofa, they're asleep. No unsupported details like age, gender, or brands. Make sure to avoid those. Keep it concise.◁/think▷Title: Two tabby cats asleep on pink sofa with remote controls
Description: Two tabby cats lie asleep on a pink fabric surface with remote controls nearby, showcasing relaxed postures.
Keywords: tabby cats, pink sofa, asleep, remote controls, fabric surface, two cats, relaxed posture, blue remotes, indoor setting, soft lighting, resting cats
</pre>

<details>
<summary>Exact raw output</summary>

```text
◁think▷Okay, let's tackle this task. First, I need to create a British-English catalogue metadata based on the image. The user wants only visible facts, no uncertain details. Let's start by looking at the image.

The main subject is two cats sleeping on a pink fabric surface. There's a remote control next to one cat. The lighting is bright, and the colors are vivid, maybe enhanced. The cats are tabby, so I should note that. The setting is a couch, so "sofa" or "cushion" could be better than "couch" for British English. The action is sleeping, so "asleep" instead of just "sleeping". Distinctive details: the remote controls are blue, the fabric is pink, the cats have distinct tabby patterns.

Now, the title needs to be concise, 5-10 words. Maybe "Two tabby cats asleep on pink sofa with remote controls." That's 10 words. Next, the description: one sentence. Mention the setting (pink sofa), the cats (two tabby, sleeping), the remote controls. So: "Two tabby cats lie asleep on a pink fabric surface with remote controls nearby." That's 20 words, stays within the limit.

Keywords: need 10-18 unique, comma-separated. Keywords: tabby cats, pink sofa, asleep, remote controls, fabric surface, two cats, relaxed posture, blue remotes, indoor setting, soft lighting, relaxed. Let me check if these are all visible. Yes, the remotes are blue, the fabric is pink, they're on a sofa, they're asleep. No unsupported details like age, gender, or brands. Make sure to avoid those. Keep it concise.◁/think▷Title: Two tabby cats asleep on pink sofa with remote controls  
Description: Two tabby cats lie asleep on a pink fabric surface with remote controls nearby, showcasing relaxed postures.  
Keywords: tabby cats, pink sofa, asleep, remote controls, fabric surface, two cats, relaxed posture, blue remotes, indoor setting, soft lighting, resting cats
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
- *Generation time:* 0.48s
- *Total time:* 1.69s
- *Prompt preparation time:* 0.0167
- *First-token latency:* 0.124
- *Cleanup time:* 0.0941
- *Prompt tokens:* 344
- *Generation tokens:* 74
- *Total tokens:* 418
- *Prompt throughput (raw):* 2,776 tok/s
- *Generation throughput (raw):* 214 tok/s
- *Peak memory (GB):* 3.6
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.02
- *Model-load active memory (GB):* 2.82
- *Post-cleanup active memory (GB):* 0.00433
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 523
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
Title: Two cats sleeping on a pink couch.
Description: Two tabby cats are lying on a pink couch with remote controls nearby. The lighting is bright, highlighting the cats' fur patterns and the vibrant color of the couch.
Keywords: cats, pink couch, tabby, sleeping, remote controls, bright lighting, fur patterns, feline, relaxation, indoor
</pre>

</details>

---

<a id="model-mlx-community-minicpm-v-46-4bit"></a>

### mlx-community/MiniCPM-V-4.6-4bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.6-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Model load time:* 0.96s
- *Generation time:* 2.02s
- *Total time:* 3.01s
- *Prompt preparation time:* 0.0241
- *First-token latency:* 0.113
- *Cleanup time:* 0.111
- *Prompt tokens:* 330
- *Generation tokens:* 478
- *Total tokens:* 808
- *Prompt throughput (raw):* 2,922 tok/s
- *Generation throughput (raw):* 255 tok/s
- *Peak memory (GB):* 3.1
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 2.16
- *Post-cleanup active memory (GB):* 0.00536
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 540
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 86cd463d33a946e4481b77e3c10fc63121b60a19
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.6-4bit/snapshots/86cd463d33a946e4481b77e3c10fc63121b60a19
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.7, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;
First, I need to create a title. It should be 5-10 words. Looking at the image, it's about two cats sleeping on a pink fabric. So, I'll think of a title like "Two Sleeping Cats on Pink Couch". But I need to ensure it's 5-10 words. Let me count: "Two Sleeping Cats on Pink Couch" is 6 words. That works.

Now, the description: I need to describe the main subject, setting, action, lighting, and distinctive details. The cats are lying down, sleeping, with remote controls nearby. Lighting seems to be indoor and soft. Distinctive details include the pink blanket and the remote controls.

Keywords should be 10-18 unique, comma-separated. I need to list keywords that capture the essence of the image. For example, cats, sleeping, pink, remote, cat, cushion, etc.

Let me outline:

Title: Two Sleeping Cats on Pink Couch

Description: Two tabby cats are resting and sleeping on a pink fabric, with remote controls visible.

Keywords: cats, sleeping, tabby, pink, couch, remote, cat, relaxation, indoor, sleep.

I need to ensure they are unique and cover the main points. Let's list:

1. cats

2. sleeping

3. tabby (since the cats have striped)

4. pink

5. couch

6. remote

7. cat

8. relaxation

9. indoor

10. sleep

11. cushion

12. fur

But I should make them concise and relevant. The requirement is 10-18 unique keywords, so I'll select the most important ones.

Final selection:

Title: Two Sleeping Cats on Pink Couch

Description: Two tabby cats are resting and sleeping on a pink fabric, with remote controls nearby.

Keywords: cats, sleeping, tabby, pink, couch, remote, relaxation, indoor, sleep

Now, I'll format it exactly as specified.
&lt;/think&gt;

Title: Two Sleeping Cats on Pink Couch
Description: Two tabby cats are resting and sleeping on a pink fabric, with remote controls visible.
Keywords: cats, sleeping, tabby, pink, couch, remote, relaxation, indoor, sleep
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>
First, I need to create a title. It should be 5-10 words. Looking at the image, it's about two cats sleeping on a pink fabric. So, I'll think of a title like "Two Sleeping Cats on Pink Couch". But I need to ensure it's 5-10 words. Let me count: "Two Sleeping Cats on Pink Couch" is 6 words. That works.

Now, the description: I need to describe the main subject, setting, action, lighting, and distinctive details. The cats are lying down, sleeping, with remote controls nearby. Lighting seems to be indoor and soft. Distinctive details include the pink blanket and the remote controls.

Keywords should be 10-18 unique, comma-separated. I need to list keywords that capture the essence of the image. For example, cats, sleeping, pink, remote, cat, cushion, etc.

Let me outline:

Title: Two Sleeping Cats on Pink Couch

Description: Two tabby cats are resting and sleeping on a pink fabric, with remote controls visible.

Keywords: cats, sleeping, tabby, pink, couch, remote, cat, relaxation, indoor, sleep.

I need to ensure they are unique and cover the main points. Let's list:

1. cats

2. sleeping

3. tabby (since the cats have striped)

4. pink

5. couch

6. remote

7. cat

8. relaxation

9. indoor

10. sleep

11. cushion

12. fur

But I should make them concise and relevant. The requirement is 10-18 unique keywords, so I'll select the most important ones.

Final selection:

Title: Two Sleeping Cats on Pink Couch

Description: Two tabby cats are resting and sleeping on a pink fabric, with remote controls nearby.

Keywords: cats, sleeping, tabby, pink, couch, remote, relaxation, indoor, sleep

Now, I'll format it exactly as specified.
</think>

Title: Two Sleeping Cats on Pink Couch
Description: Two tabby cats are resting and sleeping on a pink fabric, with remote controls visible.
Keywords: cats, sleeping, tabby, pink, couch, remote, relaxation, indoor, sleep
```

</details>

</details>

---

<a id="model-mlx-community-minicpm-v-47-1b-4bit"></a>

### mlx-community/MiniCPM-V-4.7-1B-4bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.7-1B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_7)
- *Model load time:* 0.93s
- *Generation time:* 0.39s
- *Total time:* 1.35s
- *Prompt preparation time:* 0.0237
- *First-token latency:* 0.108
- *Cleanup time:* 0.12
- *Prompt tokens:* 332
- *Generation tokens:* 64
- *Total tokens:* 396
- *Prompt throughput (raw):* 3,065 tok/s
- *Generation throughput (raw):* 261 tok/s
- *Peak memory (GB):* 3.6
- *Active memory (GB):* 2.7
- *Cache memory (GB):* 0.04
- *Model-load active memory (GB):* 2.67
- *Post-cleanup active memory (GB):* 0.00587
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 534
- *Processor:* mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 13caa569f70e4b68b0ad7dd9574f15a241b34a7d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.7-1B-4bit/snapshots/13caa569f70e4b68b0ad7dd9574f15a241b34a7d
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sleeping Cats on Pink Couch
Description: Two tabby cats rest on a pink fabric couch, one near a remote, the other beside it, under soft indoor lighting.
Keywords: tabby, cat, sleeping, pink, couch, remote, indoor, light, fur, stretch, curled, relaxed
</pre>

</details>

---

<a id="model-mlx-community-minicpm-v-47-35b-a3b-4bit"></a>

### mlx-community/MiniCPM-V-4.7-35B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-V-4.7-35B-A3B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_7)
- *Model load time:* 2.63s
- *Generation time:* 1.49s
- *Total time:* 4.15s
- *Prompt preparation time:* 0.0152
- *First-token latency:* 0.377
- *Cleanup time:* 0.143
- *Prompt tokens:* 332
- *Generation tokens:* 86
- *Total tokens:* 418
- *Prompt throughput (raw):* 881 tok/s
- *Generation throughput (raw):* 78.9 tok/s
- *Peak memory (GB):* 24
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 23.5
- *Post-cleanup active memory (GB):* 0.00638
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 534
- *Processor:* mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0e3b706c1599ad95294122c67c77ce662de18794
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.7-35B-A3B-4bit/snapshots/0e3b706c1599ad95294122c67c77ce662de18794
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Sleeping on Pink Sofa

Description: Two tabby cats are lying on a bright pink sofa cover, each positioned near a white remote control. The scene is lit by indoor lighting, highlighting the striped fur patterns of the cats.

Keywords: cats, tabby, sleeping, pink, sofa, remote control, furniture, domestic, animal, resting, striped, cover, indoor, pet, comfortable
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
- *Generation time:* 1.20s
- *Total time:* 2.27s
- *Prompt preparation time:* 0.0135
- *First-token latency:* 0.144
- *Cleanup time:* 0.113
- *Prompt tokens:* 177
- *Generation tokens:* 103
- *Total tokens:* 280
- *Prompt throughput (raw):* 1,225 tok/s
- *Generation throughput (raw):* 103 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 6.2
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 6.15
- *Post-cleanup active memory (GB):* 0.00669
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 527
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
Title:
Two tabby cats sleeping on red couch

Description:
Two tabby cats lie side by side on a red upholstered couch, partially covered by a pink blanket. A remote control is visible near each cat. Lighting is even and appears to be indoor ambient. One cat is stretched out, the other curled slightly, both appear to be resting peacefully.

Keywords:
tabby cats, sleeping, red couch, pink blanket, remote controls, indoor, feline, rest, domestic animals, pet, upholstery
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
- *Model load time:* 1.53s
- *Generation time:* 1.75s
- *Total time:* 3.31s
- *Prompt preparation time:* 0.0175
- *First-token latency:* 0.689
- *Cleanup time:* 0.107
- *Prompt tokens:* 1,070
- *Generation tokens:* 72
- *Total tokens:* 1,142
- *Prompt throughput (raw):* 1,553 tok/s
- *Generation throughput (raw):* 68.7 tok/s
- *Peak memory (GB):* 9.8
- *Active memory (GB):* 8.0
- *Cache memory (GB):* 0.23
- *Model-load active memory (GB):* 8.01
- *Post-cleanup active memory (GB):* 0.00695
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 2910
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
Two Cats Relaxing on Pink Cushion

Description:
Two cats lie stretched out on a pink cushion, positioned near remote controls. The setting appears indoors with soft, natural lighting.

Keywords:
cats, pink cushion, remote controls, indoor setting, relaxed posture, soft lighting, resting, domestic scene, furry pets, cushioned surface
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
- *Model load time:* 1.11s
- *Generation time:* 0.78s
- *Total time:* 1.92s
- *Prompt preparation time:* 0.0172
- *First-token latency:* 0.247
- *Cleanup time:* 0.0946
- *Prompt tokens:* 1,069
- *Generation tokens:* 100
- *Total tokens:* 1,169
- *Prompt throughput (raw):* 4,333 tok/s
- *Generation throughput (raw):* 198 tok/s
- *Peak memory (GB):* 4.5
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.15
- *Model-load active memory (GB):* 2.75
- *Post-cleanup active memory (GB):* 0.00721
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 2909
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
Two Kitten Friends on a Pink Couch

Description:
Two striped kittens lie relaxed on a bright pink cushion, each with a remote control beside them. The larger kitten is curled, while the smaller one stretches out, bathed in soft indoor lighting.

Keywords:
kitten, striped-pattern, pink-cushion, remote-control, cozy-interior, resting-pose, domestic-setting, playful-ambiance, pet-photo, soft-lighting
</pre>

</details>

---

<a id="model-mlx-community-molmo2-8b-4bit"></a>

### mlx-community/Molmo2-8B-4bit

<details>
<summary>Complete evidence: mlx-community/Molmo2-8B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Model load time:* 1.16s
- *Generation time:* 1.85s
- *Total time:* 3.05s
- *Prompt preparation time:* 0.0205
- *First-token latency:* 0.498
- *Cleanup time:* 0.126
- *Prompt tokens:* 873
- *Generation tokens:* 95
- *Total tokens:* 968
- *Prompt throughput (raw):* 1,753 tok/s
- *Generation throughput (raw):* 71.3 tok/s
- *Peak memory (GB):* 8.5
- *Active memory (GB):* 6.5
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 6.45
- *Post-cleanup active memory (GB):* 0.00752
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 510
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
Title: Two Calico Cats Sleeping on Pink Bedding

Description: Two calico cats are sleeping side by side on pink bedding. They are positioned with their heads facing each other and their tails touching. The cats have white, black, and brown fur. Two white remote controls are visible behind the cats.

Keywords: calico cats, sleeping, pink bedding, white remotes, black and brown fur, white cats, striped cats, cozy, relaxed, feline
</pre>

</details>

---

<a id="model-mlx-community-molmopoint-8b-4bit"></a>

### mlx-community/MolmoPoint-8B-4bit

<details>
<summary>Complete evidence: mlx-community/MolmoPoint-8B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type molmo_point)
- *Model load time:* 1.55s
- *Generation time:* 2.91s
- *Total time:* 4.50s
- *Prompt preparation time:* 0.0201
- *First-token latency:* 0.642
- *Cleanup time:* 0.106
- *Prompt tokens:* 860
- *Generation tokens:* 70
- *Total tokens:* 930
- *Prompt throughput (raw):* 1,341 tok/s
- *Generation throughput (raw):* 32.0 tok/s
- *Peak memory (GB):* 12
- *Active memory (GB):* 8.5
- *Cache memory (GB):* 3.1
- *Model-load active memory (GB):* 7.23
- *Post-cleanup active memory (GB):* 0.00815
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 510
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 9bab196f867ceedecdb1eab071395bfedcd62538
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MolmoPoint-8B-4bit/snapshots/9bab196f867ceedecdb1eab071395bfedcd62538
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Cats Sleeping on Pink Blanket with Remote Controls

Description: Two cats are sleeping on a pink blanket. A white remote control is visible behind them. The cats are resting comfortably on the soft surface.

Keywords: cats, pink blanket, remote controls, sleeping, white remote, comfortable, soft surface, feline, rest, relaxation
</pre>

</details>

---

<a id="model-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Muse-Glimmer-30B-OptiQ-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Model load time:* 3.23s
- *Generation time:* 41.52s
- *Total time:* 44.80s
- *Prompt preparation time:* 0.0302
- *First-token latency:* 0.98
- *Cleanup time:* 0.21
- *Prompt tokens:* 545
- *Generation tokens:* 937
- *Total tokens:* 1,482
- *Prompt throughput (raw):* 556 tok/s
- *Generation throughput (raw):* 23.1 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 23.9
- *Post-cleanup active memory (GB):* 0.00855
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 710
- *Processor:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 865ee8de65799cc4ee406587e4b607063708f358
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Muse-Glimmer-30B-OptiQ-4bit/snapshots/865ee8de65799cc4ee406587e4b607063708f358
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 64, "top_p": 0.95}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Resting On Pink Blanket
Description: Two tabby cats lie on a bright pink blanket on a red cushioned surface. One cat appears asleep with eyes closed and the other is positioned head down, with two remote controls and a green collar visible.
Keywords: cats, tabby, pink blanket, red sofa, remote controls, resting, sleeping, indoor, pets, fabric, cushion, collar, domestic, feline, blanket
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
- *Model load time:* 1.01s
- *Generation time:* 0.66s
- *Total time:* 1.70s
- *Prompt preparation time:* 0.0204
- *First-token latency:* 0.153
- *Cleanup time:* 0.0982
- *Prompt tokens:* 418
- *Generation tokens:* 78
- *Total tokens:* 496
- *Prompt throughput (raw):* 2,738 tok/s
- *Generation throughput (raw):* 164 tok/s
- *Peak memory (GB):* 3.5
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.10
- *Model-load active memory (GB):* 2.18
- *Post-cleanup active memory (GB):* 0.00908
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 603
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
Title: "Cats Resting on a Pink Couch"

Description: "Two cats are comfortably sprawled out on a vibrant pink couch, basking in the soft light of the day. Their relaxed postures and the cozy setting create a serene and tranquil scene."

Keywords: cats, couch, resting, lounging, pink, soft light, relaxed, tranquil, basking, cozy, serene, scene.
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
- *Model load time:* 0.90s
- *Generation time:* 1.86s
- *Total time:* 2.78s
- *Prompt preparation time:* 0.0177
- *First-token latency:* 0.213
- *Cleanup time:* 0.0894
- *Prompt tokens:* 883
- *Generation tokens:* 94
- *Total tokens:* 977
- *Prompt throughput (raw):* 4,137 tok/s
- *Generation throughput (raw):* 59.4 tok/s
- *Peak memory (GB):* 9.1
- *Active memory (GB):* 8.3
- *Cache memory (GB):* 0.42
- *Model-load active memory (GB):* 8.3
- *Post-cleanup active memory (GB):* 0.00914
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 493
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
Title: Two Cats Sleeping on Pink Sofa

Description: Two cats are sleeping on a pink sofa with remote controls beside them.

Keywords: cats, sleeping, pink sofa, remote controls, relaxed, comfortable, indoor, domestic, fur, striped, resting, tranquil, domesticated, pet, animal, relaxation, cozy, home, companionship
</pre>

</details>

---

<a id="model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit"></a>

### mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_omni_moe)
- *Model load time:* 2.36s
- *Generation time:* 1.40s
- *Total time:* 3.80s
- *Prompt preparation time:* 0.0249
- *First-token latency:* 0.357
- *Cleanup time:* 0.144
- *Prompt tokens:* 409
- *Generation tokens:* 87
- *Total tokens:* 496
- *Prompt throughput (raw):* 1,147 tok/s
- *Generation throughput (raw):* 84.5 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 22
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 21.8
- *Post-cleanup active memory (GB):* 0.00977
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 544
- *Processor:* mlx_vlm.models.qwen3_omni_moe.processing_qwen3_omni_moe.Qwen3OmniMoeProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 93b3cbddd65ed4babff8f22fba491cdba7a21778
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-Omni-30B-A3B-Instruct-4bit/snapshots/93b3cbddd65ed4babff8f22fba491cdba7a21778
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two tabby cats sleeping on sofa

Description:
Two tabby cats are sleeping on a bright pink sofa. The sofa has two remote controls placed near the cats.

Keywords:
tabby cats, sleeping, sofa, pink, remote controls, domestic cats, resting, animal, pet, living room, indoor, animal resting, animal sleep, animal comfort, animal relaxation, animal rest, animal cuddling, animal companionship
</pre>

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
- *Model load time:* 1.97s
- *Generation time:* 1.28s
- *Total time:* 3.28s
- *Prompt preparation time:* 0.0205
- *First-token latency:* 0.331
- *Cleanup time:* 0.136
- *Prompt tokens:* 409
- *Generation tokens:* 83
- *Total tokens:* 492
- *Prompt throughput (raw):* 1,236 tok/s
- *Generation throughput (raw):* 88.4 tok/s
- *Peak memory (GB):* 19
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 18.3
- *Post-cleanup active memory (GB):* 0.0101
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 544
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0555d34cb1ed80c0e61a5635194c70027b4c2ff3
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-30B-A3B-Instruct-4bit/snapshots/0555d34cb1ed80c0e61a5635194c70027b4c2ff3
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two cats resting on a pink surface

Description: Two cats are lying on a bright pink fabric surface. Two remote controls are visible near the cats.

Keywords: cats, pink fabric, remote controls, lying down, resting, domestic cats, patterned fur, light source, indoor setting, soft material, animals, pets, domestic animals, household items, objects, lying, sleeping, calm, relaxed
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
- *Model load time:* 0.90s
- *Generation time:* 0.87s
- *Total time:* 1.80s
- *Prompt preparation time:* 0.0105
- *First-token latency:* 0.228
- *Cleanup time:* 0.115
- *Prompt tokens:* 409
- *Generation tokens:* 60
- *Total tokens:* 469
- *Prompt throughput (raw):* 1,790 tok/s
- *Generation throughput (raw):* 102 tok/s
- *Peak memory (GB):* 6.8
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 0.11
- *Model-load active memory (GB):* 5.77
- *Post-cleanup active memory (GB):* 0.0104
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 544
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* defcdea7cc7a4b0858fea563cbbce171d328e457
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3-VL-8B-Instruct-4bit/snapshots/defcdea7cc7a4b0858fea563cbbce171d328e457
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.7, "top_k": 20, "top_p": 0.8}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Cats Sleeping on Pink Sofa

Description: Two tabby cats rest peacefully on a bright pink sofa. Remote controls lie beside them. Natural daylight illuminates the scene.

Keywords: cats, sleeping, pink, sofa, remote, tabby, rest, daylight, couch, feline
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
- *Model load time:* 2.41s
- *Generation time:* 1.66s
- *Total time:* 4.11s
- *Prompt preparation time:* 0.0248
- *First-token latency:* 0.363
- *Cleanup time:* 0.141
- *Prompt tokens:* 421
- *Generation tokens:* 102
- *Total tokens:* 523
- *Prompt throughput (raw):* 1,160 tok/s
- *Generation throughput (raw):* 79.3 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.11
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.0109
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Sleeping On Pink Sofa

Description: Two striped tabby cats lie asleep on a bright pink sofa, each near a remote control. Their relaxed postures and soft fur contrast with the vivid upholstery under even indoor lighting.

Keywords: cats, tabby, sleeping, sofa, pink, remotes, domestic pets, striped fur, cozy setting, animal rest, furniture, household, calm, striped pattern, relaxation, pet care, soft fabric, indoor scene, feline behavior
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Two Tabby Cats Sleeping On Pink Sofa  

Description: Two striped tabby cats lie asleep on a bright pink sofa, each near a remote control. Their relaxed postures and soft fur contrast with the vivid upholstery under even indoor lighting.  

Keywords: cats, tabby, sleeping, sofa, pink, remotes, domestic pets, striped fur, cozy setting, animal rest, furniture, household, calm, striped pattern, relaxation, pet care, soft fabric, indoor scene, feline behavior
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
- *Model load time:* 2.07s
- *Generation time:* 4.40s
- *Total time:* 6.49s
- *Prompt preparation time:* 0.0105
- *First-token latency:* 0.683
- *Cleanup time:* 0.141
- *Prompt tokens:* 421
- *Generation tokens:* 113
- *Total tokens:* 534
- *Prompt throughput (raw):* 617 tok/s
- *Generation throughput (raw):* 30.5 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.25
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.0114
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Reclining on Pink Sofa Cushion

Description: Two striped tabby cats recline on a bright pink fabric surface with two white television remote controls placed nearby. The cats lie in opposing orientations with their limbs relaxed, and natural indoor lighting illuminates their patterned coats and the surrounding fabric.

Keywords: tabby cats, pink cushion, reclining posture, striped coats, remote controls, domestic setting, fabric texture, relaxed animals, twin composition, indoor scene, feline fur, sleeping position, textile backdrop, contrasting colours
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
- *Generation time:* 0.65s
- *Total time:* 1.37s
- *Prompt preparation time:* 0.0616
- *First-token latency:* 0.355
- *Cleanup time:* 0.092
- *Prompt tokens:* 1,204
- *Generation tokens:* 31
- *Total tokens:* 1,235
- *Prompt throughput (raw):* 3,389 tok/s
- *Generation throughput (raw):* 131 tok/s
- *Peak memory (GB):* 5.6
- *Active memory (GB):* 4.5
- *Cache memory (GB):* 0.30
- *Model-load active memory (GB):* 4.51
- *Post-cleanup active memory (GB):* 0.0116
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 504
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
 Title: Two Cats on a Couch
Description: Two cats are sleeping on a pink couch.
Keywords: cats, couch, sleeping, pink
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
- *Model load time:* 1.12s
- *Generation time:* 2.36s
- *Total time:* 4.09s
- *Prompt preparation time:* 0.593
- *First-token latency:* 1.03
- *Cleanup time:* 0.114
- *Prompt tokens:* 2,909
- *Generation tokens:* 82
- *Total tokens:* 2,991
- *Prompt throughput (raw):* 2,823 tok/s
- *Generation throughput (raw):* 100 tok/s
- *Peak memory (GB):* 6.5
- *Active memory (GB):* 5.5
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 5.46
- *Post-cleanup active memory (GB):* 0.0127
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 3410
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
Title: Two Cats Relaxing Together
Description: Two tabby cats are curled up and sleeping side by side on a pink blanket, with a remote control placed nearby on the sofa. The scene is well-lit, capturing the peaceful atmosphere of a cozy living room.
Keywords: cats, sleeping, pink blanket, sofa, remote control, tabby, relaxed, home, indoor, comfortable
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
- *Model load time:* 3.28s
- *Generation time:* 1.74s
- *Total time:* 5.08s
- *Prompt preparation time:* 0.0308
- *First-token latency:* 0.286
- *Cleanup time:* 0.146
- *Prompt tokens:* 386
- *Generation tokens:* 67
- *Total tokens:* 453
- *Prompt throughput (raw):* 1,350 tok/s
- *Generation throughput (raw):* 54.0 tok/s
- *Peak memory (GB):* 28
- *Active memory (GB):* 27
- *Cache memory (GB):* 0.01
- *Model-load active memory (GB):* 27.2
- *Post-cleanup active memory (GB):* 0.0127
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 498
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
Title: Two tabby cats sleeping on a pink blanket
Description: Two tabby cats are lying down on a pink fabric surface next to two remote controls.
Keywords: cats, tabby, sleeping, pink, blanket, couch, remote, controls, indoor, domestic, resting, fur, striped, feline, lounging
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
- *Model load time:* 2.34s
- *Generation time:* 3.55s
- *Total time:* 5.94s
- *Prompt preparation time:* 0.0213
- *First-token latency:* 0.698
- *Cleanup time:* 0.145
- *Prompt tokens:* 379
- *Generation tokens:* 89
- *Total tokens:* 468
- *Prompt throughput (raw):* 543 tok/s
- *Generation throughput (raw):* 31.5 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.33
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.0133
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 527
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
Title: Two Tabby Cats Resting on Pink Fabric

Description: Two tabby cats lie prone on a pink fabric surface, each with a remote control nearby. Both cats are asleep, exhibiting striped fur patterns and relaxed postures.

Keywords: cats, tabby, pink, fabric, remote controls, sleeping, domestic animals, indoors, lying down, striped, feline, pets, resting, horizontal, two animals, close-up, daylight, cushions
</pre>

</details>

---

<a id="model-mlx-community-gemma-3n-e4b-it-4bit"></a>

### mlx-community/gemma-3n-E4B-it-4bit

<details>
<summary>Complete evidence: mlx-community/gemma-3n-E4B-it-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Model load time:* 1.68s
- *Generation time:* 1.17s
- *Total time:* 2.89s
- *Prompt preparation time:* 0.0134
- *First-token latency:* 0.284
- *Cleanup time:* 0.125
- *Prompt tokens:* 378
- *Generation tokens:* 63
- *Total tokens:* 441
- *Prompt throughput (raw):* 1,330 tok/s
- *Generation throughput (raw):* 72.3 tok/s
- *Peak memory (GB):* 7.2
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 0.04
- *Model-load active memory (GB):* 5.83
- *Post-cleanup active memory (GB):* 0.0138
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 529
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
**Title:** Cozy Cat Nap

**Description:** Two tabby cats are peacefully asleep on a pink couch, each nestled beside a remote control.

**Keywords:** cats, sleeping, couch, remote, tabby, nap, pets, cozy, relaxation, comfort, indoors, rest, feline, peaceful, home.
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
- *Model load time:* 1.76s
- *Generation time:* 1.52s
- *Total time:* 3.33s
- *Prompt preparation time:* 0.0321
- *First-token latency:* 0.274
- *Cleanup time:* 0.129
- *Prompt tokens:* 390
- *Generation tokens:* 77
- *Total tokens:* 467
- *Prompt throughput (raw):* 1,422 tok/s
- *Generation throughput (raw):* 62.6 tok/s
- *Peak memory (GB):* 7.5
- *Active memory (GB):* 6.8
- *Cache memory (GB):* 0.25
- *Model-load active memory (GB):* 6.76
- *Post-cleanup active memory (GB):* 0.0143
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 526
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
Title: Two Tabby Cats on a Pink Sofa

Description: Two tabby cats lie on a pink velvet sofa, with one resting its head on a remote control. The scene is illuminated by bright, even lighting.

Keywords: cats, tabby, sofa, couch, pet, sleeping, reclining, domestic, animal, grey, striped, fur, furniture, pink, remote control
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
- *Model load time:* 2.50s
- *Generation time:* 1.10s
- *Total time:* 3.64s
- *Prompt preparation time:* 0.00986
- *First-token latency:* 0.384
- *Cleanup time:* 0.143
- *Prompt tokens:* 390
- *Generation tokens:* 81
- *Total tokens:* 471
- *Prompt throughput (raw):* 1,016 tok/s
- *Generation throughput (raw):* 115 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 15.4
- *Post-cleanup active memory (GB):* 0.0149
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 526
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
Title: Two tabby cats lying on a pink surface

Description: Two tabby cats lie on a pink fabric surface next to two remote controls. The cats are positioned with their bodies facing different directions under bright lighting.

Keywords: tabby cats, pink fabric, remote controls, lying down, domestic cats, striped fur, indoor, two cats, close-up, bright lighting, patterned fur, soft surface
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
- *Generation time:* 0.69s
- *Total time:* 2.42s
- *Prompt preparation time:* 0.00949
- *First-token latency:* 0.177
- *Cleanup time:* 0.116
- *Prompt tokens:* 386
- *Generation tokens:* 62
- *Total tokens:* 448
- *Prompt throughput (raw):* 2,183 tok/s
- *Generation throughput (raw):* 125 tok/s
- *Peak memory (GB):* 5.9
- *Active memory (GB):* 5.2
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 5.16
- *Post-cleanup active memory (GB):* 0.0154
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 498
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
Title: Two cats rest upon a pink surface
Description: Two cats lie on a bright pink surface. One cat's owner's remote control is visible nearby.
Keywords: cats, feline, resting, pink, surface, animals, tortoiseshell, tabby, indoors, slumber, domestic, companion
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
- *Generation time:* 0.67s
- *Total time:* 1.33s
- *Prompt preparation time:* 0.0323
- *First-token latency:* 0.256
- *Cleanup time:* 0.101
- *Prompt tokens:* 734
- *Generation tokens:* 66
- *Total tokens:* 800
- *Prompt throughput (raw):* 2,870 tok/s
- *Generation throughput (raw):* 179 tok/s
- *Peak memory (GB):* 4.7
- *Active memory (GB):* 3.0
- *Cache memory (GB):* 0.13
- *Model-load active memory (GB):* 3.03
- *Post-cleanup active memory (GB):* 0.0156
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 696
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
Title: "Cats at Rest on a Vibrant Red Sofa"
Description: Two cats are peacefully sleeping on a bright red sofa, with a remote control nearby, suggesting a relaxed home environment.
Keywords: cats, rest, red sofa, remote control, home, comfort, sleep, striped, domestic, tranquility
</pre>

</details>

---

<a id="model-mlx-community-granite-vision-32-2b-nvfp4"></a>

### mlx-community/granite-vision-3.2-2b-nvfp4

<details>
<summary>Complete evidence: mlx-community/granite-vision-3.2-2b-nvfp4</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type granite_vision)
- *Model load time:* 0.42s
- *Generation time:* 1.77s
- *Total time:* 2.22s
- *Prompt preparation time:* 0.0267
- *First-token latency:* 1.04
- *Cleanup time:* 0.103
- *Prompt tokens:* 3,095
- *Generation tokens:* 119
- *Total tokens:* 3,214
- *Prompt throughput (raw):* 2,976 tok/s
- *Generation throughput (raw):* 174 tok/s
- *Peak memory (GB):* 3.9
- *Active memory (GB):* 2.4
- *Cache memory (GB):* 0.68
- *Model-load active memory (GB):* 2.36
- *Post-cleanup active memory (GB):* 0.0157
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 649
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
Title: "Whisker Waves"
Description: A close-up photograph of a cat's whiskers in motion, captured in a blurred motion blur against a pink background. The cat's fur is a mix of white and brown, with a distinctive pattern of stripes. The lighting is soft and diffused, creating a gentle contrast between the cat's fur and the background.
Keywords: Cat, Whiskers, Motion Blur, Pink Background, Soft Lighting, Fur Pattern, Motion Blur Photography
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
- *Model load time:* 1.81s
- *Generation time:* 2.97s
- *Total time:* 4.80s
- *Prompt preparation time:* 0.011
- *First-token latency:* 0.789
- *Cleanup time:* 0.119
- *Prompt tokens:* 1,335
- *Generation tokens:* 89
- *Total tokens:* 1,424
- *Prompt throughput (raw):* 1,692 tok/s
- *Generation throughput (raw):* 41.1 tok/s
- *Peak memory (GB):* 15
- *Active memory (GB):* 13
- *Cache memory (GB):* 0.27
- *Model-load active memory (GB):* 13.5
- *Post-cleanup active memory (GB):* 0.017
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 472
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
Two Cats Lounging on Red Couch

### Description:
Two cats, one lying on its back and the other on its side, relax on a red couch. Remote controls are placed nearby.

### Keywords:
cats, lounging, red couch, remote controls, relaxed, indoor, comfort, feline, domestic, paws, stretch, lounge, resting, home, soft, purring, peaceful
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
- *Model load time:* 0.82s
- *Generation time:* 0.76s
- *Total time:* 1.60s
- *Prompt preparation time:* 0.0103
- *First-token latency:* 0.129
- *Cleanup time:* 0.108
- *Prompt tokens:* 420
- *Generation tokens:* 89
- *Total tokens:* 509
- *Prompt throughput (raw):* 3,255 tok/s
- *Generation throughput (raw):* 149 tok/s
- *Peak memory (GB):* 4.9
- *Active memory (GB):* 3.9
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 3.93
- *Post-cleanup active memory (GB):* 0.0173
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 602
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
Title:
Two cats sleeping on a red couch with remote controls nearby.

Description:
Two cats are sleeping on a red couch, with one cat lying on its side and the other cat lying on its stomach. The cats are surrounded by remote controls, indicating a living room setting. The lighting is natural, suggesting daytime.

Keywords:
cats, sleeping, red couch, remote controls, living room, natural light, side, stomach, remote, living
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
- *Model load time:* 1.90s
- *Generation time:* 3.10s
- *Total time:* 5.03s
- *Prompt preparation time:* 0.012
- *First-token latency:* 0.753
- *Cleanup time:* 0.124
- *Prompt tokens:* 538
- *Generation tokens:* 87
- *Total tokens:* 625
- *Prompt throughput (raw):* 715 tok/s
- *Generation throughput (raw):* 37.3 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.14
- *Model-load active memory (GB):* 15.1
- *Post-cleanup active memory (GB):* 0.0176
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 472
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
Two Cats Resting on a Pink Blanket

Description:
Two tabby cats are lying on a pink blanket, one stretched out and the other curled up. A remote control is visible near the cats.

Keywords:
cats, tabby, pink blanket, resting, remote control, fur, relaxed, indoor, couch, paws, stripes, sleeping, pet, animal, lounge, cushion, relaxed posture, domestic
</pre>

</details>

---

<a id="model-nativ-community-nvidia-nemotron-35-bolt-74b-a9b-4bit"></a>

### nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit

<details>
<summary>Complete evidence: nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type nemotron_h_omni via
  nemotron_h_nano_omni)
- *Model load time:* 4.39s
- *Generation time:* 2.46s
- *Total time:* 6.89s
- *Prompt preparation time:* 0.0232
- *First-token latency:* 0.741
- *Cleanup time:* 0.159
- *Prompt tokens:* 420
- *Generation tokens:* 105
- *Total tokens:* 525
- *Prompt throughput (raw):* 567 tok/s
- *Generation throughput (raw):* 61.4 tok/s
- *Peak memory (GB):* 46
- *Active memory (GB):* 43
- *Cache memory (GB):* 0.20
- *Model-load active memory (GB):* 43.2
- *Post-cleanup active memory (GB):* 0.0178
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 553
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.nemotron_h_nano_omni.processing_nemotron_h_nano_omni.NemotronHNanoOmniProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 4bc0a247d78e3621a1dc8929a3d0172ace572ada
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nativ-community--NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit/snapshots/4bc0a247d78e3621a1dc8929a3d0172ace572ada
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two tabby cats sleeping on a pink couch

Description: Two domestic cats with mackerel tabby markings are lying asleep on a bright pink fabric surface, likely a sofa. Each cat has a separate remote control placed nearby on the surface, and both animals are in a relaxed, curled sleeping position.

Keywords: cats, sleeping, tabby, mackerel, pink, couch, sofa, remote control, pet, domestic animal, rest, fur, striped, indoor, two animals, lying down
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
- *Model load time:* 2.41s
- *Generation time:* 0.79s
- *Total time:* 3.25s
- *Prompt preparation time:* 0.0232
- *First-token latency:* 0.322
- *Cleanup time:* 0.134
- *Prompt tokens:* 421
- *Generation tokens:* 65
- *Total tokens:* 486
- *Prompt throughput (raw):* 1,309 tok/s
- *Generation throughput (raw):* 141 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 19.7
- *Post-cleanup active memory (GB):* 0.0181
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 554
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.nemotron_h_nano_omni.processing_nemotron_h_nano_omni.NemotronHNanoOmniProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 75c89904e1c2a045098d9111e7cb250a4d8adcec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nativ-community--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit/snapshots/75c89904e1c2a045098d9111e7cb250a4d8adcec
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.6, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">

Title: Two cats sleeping on a sofa
Description: Two tabby cats are sleeping on a pink sofa, one curled up and the other stretched out, with remote controls nearby.
Keywords: cat, tabby, pink sofa, remote control, sleeping, curled, stretched, pink blanket, pink cushion, cat furniture
</pre>

</details>

---

<a id="model-sahilchachra-lensvlm-9b-mxfp4"></a>

### sahilchachra/LensVLM-9B-MXFP4

<details>
<summary>Complete evidence: sahilchachra/LensVLM-9B-MXFP4</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Model load time:* 1.13s
- *Generation time:* 1.34s
- *Total time:* 2.49s
- *Prompt preparation time:* 0.0101
- *First-token latency:* 0.223
- *Cleanup time:* 0.126
- *Prompt tokens:* 421
- *Generation tokens:* 111
- *Total tokens:* 532
- *Prompt throughput (raw):* 1,885 tok/s
- *Generation throughput (raw):* 105 tok/s
- *Peak memory (GB):* 6.8
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 5.69
- *Post-cleanup active memory (GB):* 0.0186
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 23ae80ae9a7d0b8ce975f5a409ba6f017699af6c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--sahilchachra--LensVLM-9B-MXFP4/snapshots/23ae80ae9a7d0b8ce975f5a409ba6f017699af6c
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two tabby cats sleeping on a pink sofa

Description:
Two tabby cats are curled up asleep on a bright pink fabric surface, likely a sofa or blanket. One cat lies on its side near the left edge, while the other rests on its back towards the right, with a white remote control visible between them. The scene is evenly lit, highlighting the striped pattern of their coats against the vivid background.

Keywords:
cats, sleeping, tabby, pink, sofa, remote, rest, curled, striped, domestic
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
- *Model load time:* 0.64s
- *Generation time:* 0.21s
- *Total time:* 0.88s
- *Prompt preparation time:* 0.0224
- *First-token latency:* 0.0915
- *Cleanup time:* 0.085
- *Prompt tokens:* 120
- *Generation tokens:* 32
- *Total tokens:* 152
- *Prompt throughput (raw):* 1,311 tok/s
- *Generation throughput (raw):* 356 tok/s
- *Peak memory (GB):* 2.2
- *Active memory (GB):* 1.2
- *Cache memory (GB):* 0.02
- *Model-load active memory (GB):* 1.25
- *Post-cleanup active memory (GB):* 0.00216
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 567
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
A serene moment of feline tranquility on a vibrant pink couch, with two cats in peaceful slumber, their fur patterns contrasting against the soft fabric.
</pre>

</details>

---

<a id="model-mlx-community-internvl35-1b-4bit"></a>

### mlx-community/InternVL3_5-1B-4bit

<details>
<summary>Complete evidence: mlx-community/InternVL3_5-1B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Duplicate keywords: british english, british english
  examples of metadata, british english examples of details, british english
  examples of context, british english examples of information, british
  english examples of data, british english examples of content, british
  english examples of examples
- *Arch supported by installed mlx-vlm:* yes (model_type internvl)
- *Model load time:* 0.55s
- *Generation time:* 3.51s
- *Total time:* 4.09s
- *Prompt preparation time:* 0.0274
- *First-token latency:* 0.34
- *Cleanup time:* 0.0919
- *Prompt tokens:* 3,438
- *Generation tokens:* 1,000
- *Total tokens:* 4,438
- *Prompt throughput (raw):* 10,110 tok/s
- *Generation throughput (raw):* 319 tok/s
- *Peak memory (GB):* 2.3
- *Active memory (GB):* 1.1
- *Cache memory (GB):* 0.56
- *Model-load active memory (GB):* 1.08
- *Post-cleanup active memory (GB):* 0.00341
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 515
- *Processor:* mlx_vlm.models.internvl.processing_internvl.InternVLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--InternVL3_5-1B-4bit/snapshots/f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two cats sleeping on a pink couch

Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a remote control.

Keywords: cats, sleeping, pink couch, remote control, striped, British English, British, British-English, British-English catalogue, British-English metadata, British-English details, British-English context, British-English information, British-English data, British-English content, British-English examples, British-English examples of metadata, British-English examples of keywords, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata
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
- *Maintainer status:* observation needs reproduction
- *Observations:* Response repeats the same text; Response appears cut off at
  the token limit; Duplicate keywords: feline household domesticated animals,
  feline household domesticated companions, feline household domesticated
  friends, feline household domesticated comfort, feline household
  domesticated relaxation, feline household domesticated comfort and
  relaxation
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Model load time:* 2.20s
- *Generation time:* 53.78s
- *Total time:* 56.01s
- *Prompt preparation time:* 0.0122
- *First-token latency:* 1.51
- *Cleanup time:* 0.112
- *Prompt tokens:* 107
- *Generation tokens:* 1,000
- *Total tokens:* 1,107
- *Prompt throughput (raw):* 71.1 tok/s
- *Generation throughput (raw):* 19.1 tok/s
- *Peak memory (GB):* 15
- *Active memory (GB):* 11
- *Cache memory (GB):* 0.33
- *Model-load active memory (GB):* 11.4
- *Post-cleanup active memory (GB):* 0.00485
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 576
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
Here is the metadata for the image:

**Title:** Two Cats on a Pink Blanket

**Description:** Two tabby cats are sleeping on a pink blanket, with two remote controls nearby. The cats are lying on their backs, with their paws stretched out and their eyes closed. The blanket is a deep pink color and appears to be made of a soft, plush material.

**Keywords:** Tabby cats, pink blanket, remote controls, sleeping, feline, domestic, household, comfort, relaxation, domesticated, household pets, feline companions, household items, comfort and relaxation, domesticated animals, household companions, feline friends, household comfort, feline relaxation, household felines, feline household, household feline, feline household items, feline household comfort, feline household relaxation, feline household companions, feline household friends, feline household comfort and relaxation, feline household domesticated, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation,
</pre>

</details>

---

<a id="model-mlx-community-qwen2-vl-2b-mlx"></a>

### mlx-community/Qwen2-VL-2B-mlx

<details>
<summary>Complete evidence: mlx-community/Qwen2-VL-2B-mlx</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* Response repeats the same text; Required labelled fields not
  detected: keywords; Response appears cut off at the token limit
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Model load time:* 0.81s
- *Generation time:* 7.28s
- *Total time:* 8.11s
- *Prompt preparation time:* 0.013
- *First-token latency:* 0.136
- *Cleanup time:* 0.0969
- *Prompt tokens:* 511
- *Generation tokens:* 1,000
- *Total tokens:* 1,511
- *Prompt throughput (raw):* 3,750 tok/s
- *Generation throughput (raw):* 141 tok/s
- *Peak memory (GB):* 5.4
- *Active memory (GB):* 4.4
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 4.43
- *Post-cleanup active memory (GB):* 0.00946
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 602
- *Processor:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* d8c7c767e2e2c62cda8a51943276458ea6ad43bc
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-2B-mlx/snapshots/d8c7c767e2e2c62cda8a51943276458ea6ad43bc
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.1, "top_k": 1, "top_p": 0.001}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two cats sleeping on a pink couch

Description: Two cats, one with a green collar, are sleeping on a pink couch. One cat is lying on its side, while the other cat is lying on its back. Both cats are striped and have their eyes closed. There are two remote controls on the couch next to the cats. The cats are lying on a pink blanket that covers the couch. The cats are not interacting with each other. The cats are not looking at the camera. The cats are not wearing any other accessories. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry.
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
- *Model load time:* 0.31s
- *Generation time:* 0.23s
- *Total time:* 0.64s
- *Prompt preparation time:* 0.0952
- *First-token latency:* 0.117
- *Cleanup time:* 0.0795
- *Prompt tokens:* 983
- *Generation tokens:* 8
- *Total tokens:* 991
- *Prompt throughput (raw):* 8,412 tok/s
- *Generation throughput (raw):* 393 tok/s
- *Peak memory (GB):* 1.1
- *Active memory (GB):* 0.16
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 0.161
- *Post-cleanup active memory (GB):* 0.0115
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 504
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
 The main subject is a cat.
</pre>

</details>

---

<a id="model-mlx-community-step-37-flash-oq3e"></a>

### mlx-community/Step-3.7-Flash-oQ3e

<details>
<summary>Complete evidence: mlx-community/Step-3.7-Flash-oQ3e</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* Final answer emitted twice, around &lt;/think&gt;;
  Unrecognised model control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Model load time:* 14.27s
- *Generation time:* 17.78s
- *Total time:* 32.15s
- *Prompt preparation time:* 0.0766
- *First-token latency:* 11.4
- *Cleanup time:* 0.29
- *Prompt tokens:* 286
- *Generation tokens:* 236
- *Total tokens:* 522
- *Prompt throughput (raw):* 25.0 tok/s
- *Generation throughput (raw):* 50.5 tok/s
- *Peak memory (GB):* 87
- *Active memory (GB):* 85
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 85
- *Post-cleanup active memory (GB):* 0.0119
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 550
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
Two cats sleeping on a pink blanket

Description:
Two tabby cats are lying on a bright pink blanket on a red sofa. Both cats appear to be asleep, with one cat on the left having a green collar and the other cat on the right being larger. Two remote controls are placed near the cats on the blanket.

Keywords:
cats, sleeping, pink blanket, red sofa, tabby cats, remote controls, green collar, large cat, small cat, relaxed, pets, home interior, soft fabric, lying down, close-up, daytime, comfortable, cozy
&lt;/think&gt;
Title:
Two cats sleeping on a pink blanket

Description:
Two tabby cats are lying on a bright pink blanket on a red sofa. Both cats appear to be asleep, with one cat on the left having a green collar and the other cat on the right being larger. Two remote controls are placed near the cats on the blanket.

Keywords:
cats, sleeping, pink blanket, red sofa, tabby cats, remote controls, green collar, large cat, small cat, relaxed, pets, home interior, soft fabric, lying down, close-up, daytime, comfortable, cozy
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title:
Two cats sleeping on a pink blanket

Description:
Two tabby cats are lying on a bright pink blanket on a red sofa. Both cats appear to be asleep, with one cat on the left having a green collar and the other cat on the right being larger. Two remote controls are placed near the cats on the blanket.

Keywords:
cats, sleeping, pink blanket, red sofa, tabby cats, remote controls, green collar, large cat, small cat, relaxed, pets, home interior, soft fabric, lying down, close-up, daytime, comfortable, cozy
</think>
Title:
Two cats sleeping on a pink blanket

Description:
Two tabby cats are lying on a bright pink blanket on a red sofa. Both cats appear to be asleep, with one cat on the left having a green collar and the other cat on the right being larger. Two remote controls are placed near the cats on the blanket.

Keywords:
cats, sleeping, pink blanket, red sofa, tabby cats, remote controls, green collar, large cat, small cat, relaxed, pets, home interior, soft fabric, lying down, close-up, daytime, comfortable, cozy
```

</details>

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
- *Maintainer status:* observation needs reproduction
- *Observations:* Response repeats the same text; Generation was stopped early
  after sustained repeated output; Duplicate keywords: feline behavior, feline
  companions, feline comfort, feline companionship, feline lifestyle, feline
  rest, feline home
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Model load time:* 1.18s
- *Generation time:* 4.59s
- *Total time:* 5.79s
- *Prompt preparation time:* 0.0104
- *First-token latency:* 0.335
- *Cleanup time:* 0.132
- *Prompt tokens:* 511
- *Generation tokens:* 275
- *Total tokens:* 786
- *Prompt throughput (raw):* 1,524 tok/s
- *Generation throughput (raw):* 64.9 tok/s
- *Peak memory (GB):* 10
- *Active memory (GB):* 9.5
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 9.46
- *Post-cleanup active memory (GB):* 0.0122
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* repetition_abort
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 602
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
Title: Cats Sleeping on Pink Couch with Remotes

Description: Two tabby cats are curled up on a pink couch, one slightly larger than the other, both appearing relaxed. A remote control rests between them, adding a casual, lived-in feel to the scene. The lighting is soft and natural, highlighting the cats' striped fur and the vibrant pink fabric.

Keywords: cats, sleeping, pink couch, tabby, remote control, relaxed, striped fur, natural light, cozy, domestic, feline, comfort, home setting, feline behavior, rest, domestic animals, feline companions, feline friends, feline relaxation, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companions, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior,
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
- *Maintainer status:* observation needs reproduction
- *Observations:* Unrecognised model control tokens remain visible; Required
  labelled fields not detected: title, description, keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Model load time:* 1.11s
- *Generation time:* 1.51s
- *Total time:* 2.66s
- *Prompt preparation time:* 0.0392
- *First-token latency:* 1.33
- *Cleanup time:* 0.174
- *Prompt tokens:* 3,512
- *Generation tokens:* 14
- *Total tokens:* 3,526
- *Prompt throughput (raw):* 2,634 tok/s
- *Generation throughput (raw):* 105 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.55
- *Model-load active memory (GB):* 5.7
- *Post-cleanup active memory (GB):* 0.0161
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 731
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
&lt;|channel|&gt; analysis&lt;|message|&gt; The two cats are sleeping on a pink couch.
</pre>

<details>
<summary>Exact raw output</summary>

```text
<|channel|> analysis<|message|> The two cats are sleeping on a pink couch.
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
- *Observations:* Required labelled fields not detected: keywords
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Model load time:* 0.50s
- *Generation time:* 0.54s
- *Total time:* 1.05s
- *Prompt preparation time:* 0.00922
- *First-token latency:* 0.0896
- *Cleanup time:* 0.104
- *Prompt tokens:* 116
- *Generation tokens:* 97
- *Total tokens:* 213
- *Prompt throughput (raw):* 1,295 tok/s
- *Generation throughput (raw):* 224 tok/s
- *Peak memory (GB):* 1.6
- *Active memory (GB):* 0.62
- *Cache memory (GB):* 0.23
- *Model-load active memory (GB):* 0.615
- *Post-cleanup active memory (GB):* 0.0167
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 558
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
Title: "A Study in Stripes"
Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting is soft, and the person is facing away from the camera, giving a sense of intimacy. The image is a straightforward representation of the subject, with no additional details or context provided.
</pre>

</details>

---

<a id="model-vikhyatk-moondream2"></a>

### vikhyatk/moondream2

<details>
<summary>Complete evidence: vikhyatk/moondream2</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: title, description,
  keywords
- *Arch supported by installed mlx-vlm:* yes (model_type moondream1 via
  moondream2)
- *Model load time:* 1.05s
- *Generation time:* 0.35s
- *Total time:* 1.41s
- *Prompt preparation time:* 0.00791
- *First-token latency:* 0.148
- *Cleanup time:* 0.0909
- *Prompt tokens:* 833
- *Generation tokens:* 34
- *Total tokens:* 867
- *Prompt throughput (raw):* 5,615 tok/s
- *Generation throughput (raw):* 177 tok/s
- *Peak memory (GB):* 4.5
- *Active memory (GB):* 3.8
- *Cache memory (GB):* 0.21
- *Model-load active memory (GB):* 3.75
- *Post-cleanup active memory (GB):* 0.0187
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 451
- *Processor:* mlx_vlm.models.moondream2.processing_moondream2.Moondream2Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 5d6c926f44e26b07957b0dd315bbedcb4c17a5fe
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--vikhyatk--moondream2/snapshots/5d6c926f44e26b07957b0dd315bbedcb4c17a5fe
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_p": 1.0}
- *EOS token:* <\|endoftext\|>

*Readable output:*

<pre class="model-output-readable">
Two tabby cats sleep on a pink blanket on a red couch. One cat is sleeping with its eyes closed, while the other is sleeping with its eyes open. A remote control is visible near the cats.
</pre>

</details>

---
