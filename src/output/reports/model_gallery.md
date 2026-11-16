# Model Output Gallery

Generated on: 2026-11-16 14:23:23 GMT

- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords; length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Human accuracy notes:* 1 of 41 outputs reviewed by a person for this exact image, model revision and output (accuracy_notes.yaml); the others are unreviewed

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

| Model                                                                                                                               | Mechanical checks               | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                              |
|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|---------|---------------------|-----------------|---------|------------|---------|-----------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected`          | 0.79s   | 513 tok/s           | 0.06            | 1.3     | 350        | 47      | none                                                      |
| [`TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`](#model-technobaptist-ternary-bonsai-2-27b-mlx-2bit)                                 | `no concerns detected`          | 4.42s   | 42.1 tok/s          | 0.77            | 11      | 421        | 94      | none                                                      |
| [`ibm-granite/granite-vision-4.1-4b`](#model-ibm-granite-granite-vision-41-4b)                                                      | `no concerns detected`          | 2.34s   | 61.8 tok/s          | 0.32            | 9.7     | 721        | 55      | none                                                      |
| [`mlx-community/AREX-2-4bit`](#model-mlx-community-arex-2-4bit)                                                                     | `no concerns detected`          | 5.41s   | 32.2 tok/s          | 0.65            | 17      | 421        | 80      | none                                                      |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected`          | 2.21s   | 84.5 tok/s          | 0.32            | 8.0     | 499        | 49      | none                                                      |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected`          | 4.98s   | 55.2 tok/s          | 2.02            | 10      | 3,435      | 90      | none                                                      |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected`          | 1.51s   | 202 tok/s           | 0.12            | 3.7     | 344        | 73      | none                                                      |
| [`mlx-community/MiniCPM-V-4.7-1B-4bit`](#model-mlx-community-minicpm-v-47-1b-4bit)                                                  | `no concerns detected`          | 1.61s   | 310 tok/s           | 0.13            | 3.6     | 332        | 64      | none                                                      |
| [`mlx-community/MiniCPM-V-4.7-35B-A3B-4bit`](#model-mlx-community-minicpm-v-47-35b-a3b-4bit)                                        | `no concerns detected`          | 4.35s   | 86.7 tok/s          | 0.41            | 24      | 332        | 93      | none                                                      |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `no concerns detected`          | 2.38s   | 104 tok/s           | 0.17            | 7.0     | 177        | 99      | none                                                      |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected`          | 3.24s   | 68.4 tok/s          | 0.65            | 8.9     | 1,070      | 81      | none                                                      |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected`          | 1.73s   | 193 tok/s           | 0.24            | 3.7     | 1,069      | 102     | none                                                      |
| [`mlx-community/Mistral-Small-4-119B-2603-4bit`](#model-mlx-community-mistral-small-4-119b-2603-4bit)                               | `no concerns detected`          | 20.25s  | 86.8 tok/s          | 7.18            | 69      | 549        | 82      | none                                                      |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `no concerns detected`          | 3.23s   | 69.7 tok/s          | 0.48            | 8.2     | 873        | 95      | none                                                      |
| [`mlx-community/MolmoPoint-8B-4bit`](#model-mlx-community-molmopoint-8b-4bit)                                                       | `no concerns detected`          | 4.36s   | 32.0 tok/s          | 0.69            | 12      | 860        | 70      | none                                                      |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `no concerns detected`          | 43.71s  | 22.8 tok/s          | 0.88            | 25      | 545        | 908     | none                                                      |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected`          | 2.34s   | 210 tok/s           | 0.19            | 3.5     | 418        | 78      | none                                                      |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected`          | 3.28s   | 51.2 tok/s          | 0.32            | 9.1     | 883        | 94      | none                                                      |
| [`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit)                           | `no concerns detected`          | 4.17s   | 83.5 tok/s          | 0.42            | 23      | 409        | 81      | none                                                      |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected`          | 3.31s   | 105 tok/s           | 0.35            | 19      | 409        | 83      | none                                                      |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected`          | 2.23s   | 91.2 tok/s          | 0.28            | 6.8     | 409        | 60      | none                                                      |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected`          | 4.08s   | 98.0 tok/s          | 0.38            | 21      | 421        | 88      | none                                                      |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected`          | 6.73s   | 28.9 tok/s          | 0.76            | 17      | 421        | 109     | none                                                      |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected`          | 1.77s   | 110 tok/s           | 0.48            | 5.6     | 1,204      | 31      | none                                                      |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected`          | 4.67s   | 87.5 tok/s          | 1.56            | 6.5     | 2,909      | 82      | none                                                      |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected`          | 5.31s   | 59.3 tok/s          | 1.65            | 28      | 386        | 66      | none                                                      |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `no concerns detected`          | 4.23s   | 56.1 tok/s          | 0.31            | 7.5     | 390        | 88      | none                                                      |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected`          | 3.89s   | 107 tok/s           | 0.39            | 16      | 390        | 75      | none                                                      |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected`          | 2.82s   | 108 tok/s           | 0.19            | 5.9     | 386        | 67      | none                                                      |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected`          | 1.53s   | 158 tok/s           | 0.31            | 4.7     | 734        | 57      | none                                                      |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected`          | 1.93s   | 130 tok/s           | 0.18            | 5.3     | 420        | 88      | none                                                      |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected`          | 5.44s   | 31.4 tok/s          | 0.78            | 16      | 538        | 87      | none                                                      |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected`          | 3.54s   | 128 tok/s           | 0.39            | 21      | 421        | 65      | none                                                      |
| [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](#model-nvidia-nvidia-nemotron-35-super-vl-120b-a12b-nvfp4)                  | `no concerns detected`          | 38.68s  | 25.7 tok/s          | 17.93           | 81      | 420        | 89      | none                                                      |
| [`sahilchachra/LensVLM-9B-MXFP4`](#model-sahilchachra-lensvlm-9b-mxfp4)                                                             | `no concerns detected`          | 3.51s   | 88.8 tok/s          | 0.28            | 6.8     | 421        | 111     | none                                                      |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `concerns detected`             | 6.50s   | 133 tok/s           | 0.36            | 19      | 545        | 540     | duplicate keywords                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns: answer format` | 1.16s   | 370 tok/s           | 0.12            | 2.1     | 120        | 32      | labelled fields not detected                              |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `major concerns: generation`    | 4.11s   | 346 tok/s           | 0.37            | 2.3     | 3,438      | 1,000   | repeated text; cut off at token limit; duplicate keywords |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)                                                    | `major concerns: answer format` | 4.23s   | 96.8 tok/s          | 0.40            | 21      | 421        | 96      | labelled fields not detected                              |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns: answer format` | 2.95s   | insufficient sample | 1.57            | 7.0     | 3,512      | 14      | control tokens visible; labelled fields not detected      |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns: answer format` | 1.41s   | 279 tok/s           | 0.11            | 1.6     | 116        | 97      | labelled fields not detected                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 0.79s

Lowest peak memory among completions without detected concerns: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.3 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Major Mechanical Concerns in This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                   | Mechanical checks               | Observations                                              |
|-----------------------------------------------------------------------------------------|---------------------------------|-----------------------------------------------------------|
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)              | `major concerns: answer format` | labelled fields not detected                              |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)          | `major concerns: generation`    | repeated text; cut off at token limit; duplicate keywords |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)        | `major concerns: answer format` | labelled fields not detected                              |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit) | `major concerns: answer format` | control tokens visible; labelled fields not detected      |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)            | `major concerns: answer format` | labelled fields not detected                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

A compact preview of each model's final answer (or failure evidence for crashes), in chooser order. Where the requested catalogue fields were detected, the preview shows a little of each: the title, the start of the description, and the first keywords with their count, so the weakest field is not hidden behind a long description. Otherwise it is the first 280 characters. A closed reasoning trace is left out of the preview and reported as an omitted-character count; the complete output, trace included, is in the model's evidence block below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                               | Mechanical checks      | Output preview                                                                                                                                                                                                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected` | Title: Cats on a Pink Couch \| Description: Two tabby cats are lying on a bright pink couch, appearing relaxed and comfortable. \| Keywords (6): cats, pink couch, relaxation, tabby, comfortable, cozy                                                                                                                                                                               |
| [`TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`](#model-technobaptist-ternary-bonsai-2-27b-mlx-2bit)                                 | `no concerns detected` | Title: Two tabby cats sleeping on pink sofa \| Description: Two tabby cats lie on their sides on a bright pink sofa cushion, with two white remote controls resting ne... \| Keywords (14): tabby cats, sleeping, pink sofa, remote controls, striped fur, relaxed, cushion, indoor, pets, resting, ...                                                                               |
| [`ibm-granite/granite-vision-4.1-4b`](#model-ibm-granite-granite-vision-41-4b)                                                      | `no concerns detected` | Title: "Cats at Rest on a Pink Sofa" \| Description: Two cats are peacefully sleeping on a pink sofa, with remote controls nearby. \| Keywords (9): cats, sleeping, pink sofa, remote controls, domestic, comfort, tranquility, home, leisure.                                                                                                                                        |
| [`mlx-community/AREX-2-4bit`](#model-mlx-community-arex-2-4bit)                                                                     | `no concerns detected` | Title: Two Tabby Cats Sleeping on Pink Couch \| Description: Two tabby cats lie sleeping on a bright pink couch, each positioned beside a white remote control. The... \| Keywords (8): tabby cats, sleeping, pink couch, remote controls, striped fur, indoor pets, relaxed posture, domestic setting                                                                                |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected` | Title: Two tabby cats on pink fabric. \| Description: Two tabby cats lie on a pink fabric surface, with remote controls nearby. \| Keywords (5): tabby cats, pink fabric, remote controls, domestic cats, feline.                                                                                                                                                                     |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected` | Title: Cats Sleeping on Sofa with Remote Controls \| Description: Two cats are sleeping on a pink sofa, each with a remote control nearby. The lighting is soft,... \| Keywords (18): cats, sleeping, sofa, pink, remote controls, striped, relaxed, home, pets, cozy, domestic, resting, comfort, ...                                                                                |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected` | Title: Two cats sleeping on a pink couch. \| Description: Two tabby cats are lying on a pink couch with remote controls nearby. The lighting is bright, highlighting th... \| Keywords (10): cats, pink couch, tabby, sleeping, remote controls, bright lighting, fur patterns, relaxation, indoor, ...                                                                               |
| [`mlx-community/MiniCPM-V-4.7-1B-4bit`](#model-mlx-community-minicpm-v-47-1b-4bit)                                                  | `no concerns detected` | Title: Sleeping Cats on Pink Couch \| Description: Two tabby cats rest on a pink fabric couch, one near a remote, the other beside it, under soft indoor lighting. \| Keywords (12): tabby, cat, sleeping, pink, couch, remote, indoor, light, fur, body, stretch, relaxation                                                                                                         |
| [`mlx-community/MiniCPM-V-4.7-35B-A3B-4bit`](#model-mlx-community-minicpm-v-47-35b-a3b-4bit)                                        | `no concerns detected` | Title: Two Tabby Cats Sleeping on Pink Sofa \| Description: Two tabby cats are lying on a bright pink sofa cover, each positioned near a white remote control. T... \| Keywords (18): cat, tabby, sleeping, sofa, pink, remote control, furniture, pet, resting, striped fur, domestic, relaxation, ...                                                                               |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `no concerns detected` | Title: Two tabby cats sleeping on red couch \| Description: Two tabby cats lie side by side on a red upholstered couch, partially covered by a pink blanket. A remote co... \| Keywords (11): cats, tabby, sleeping, red couch, pink blanket, remote controls, indoor, animals, fur, relaxed, household                                                                               |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected` | Title: Two Cats Relaxing on Pink Cushioned Surface \| Description: Two cats lie stretched out on a pink cushioned surface, positioned near remote controls. The... \| Keywords (10): cats, pink cushioned surface, remote controls, relaxed posture, indoor setting, natural lighting, striped fur, ...                                                                               |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected` | Title: Two resting tabby cats on a pink sofa \| Description: Two adult tabby cats lie relaxed on a bright pink sofa cushion, each with remote controls beside them. Soft, na... \| Keywords (12): tabby-cats, pink-sofa, remote-controls, indoor-resting, striped-fur, cozy-setting, domestic-pets, ...                                                                               |
| [`mlx-community/Mistral-Small-4-119B-2603-4bit`](#model-mlx-community-mistral-small-4-119b-2603-4bit)                               | `no concerns detected` | Title: Two cats resting on a pink sofa \| Description: Two cats are lying on a pink sofa with a remote control nearby. Soft lighting highlights their relaxed poses and... \| Keywords (18): cats, sofa, pink, remote control, resting, indoor, pet, cushion, domestic, relaxation, soft lighting, ...                                                                                |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `no concerns detected` | Title: Two Calico Cats Sleeping on Pink Bedding \| Description: Two calico cats are sleeping side by side on pink bedding. They are positioned with their heads facin... \| Keywords (10): calico cats, sleeping, pink bedding, white remotes, black and brown fur, white cats, striped cats, cozy, ...                                                                               |
| [`mlx-community/MolmoPoint-8B-4bit`](#model-mlx-community-molmopoint-8b-4bit)                                                       | `no concerns detected` | Title: Two Cats Sleeping on Pink Blanket with Remote Controls \| Description: Two cats are sleeping on a pink blanket. A white remote control is visible behind them. Th... \| Keywords (10): cats, pink blanket, remote controls, sleeping, white remote, comfortable, soft surface, feline, rest, ...                                                                               |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `no concerns detected` | Title: Two Tabby Cats Resting on Pink Blanket \| Description: Two tabby cats lie on a bright pink blanket on red cushions, one with its head resting down and the othe... \| Keywords (14): cats, tabby, pink blanket, red cushions, remote controls, resting, indoor, fabric, striped fur, collar, ...                                                                               |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected` | Title: "Cats Resting on a Pink Couch" \| Description: "Two cats are comfortably sprawled out on a vibrant pink couch, basking in the soft light of the day. Their relaxed postu... \| Keywords (12): cats, couch, resting, lounging, pink, soft light, relaxed, tranquil, basking, cozy, serene, scene.                                                                               |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected` | Title: Two Cats Sleeping on Pink Sofa \| Description: Two cats are sleeping on a pink sofa with remote controls beside them. \| Keywords (19): cats, sleeping, pink sofa, remote controls, relaxed, comfortable, indoor, domestic, fur, striped, resting, ...                                                                                                                         |
| [`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit)                           | `no concerns detected` | Title: Two tabby cats sleeping on sofa \| Description: Two tabby cats are sleeping on a bright pink sofa. The sofa has two remote controls placed near the cats. \| Keywords (18): tabby cats, sleeping, sofa, pink, remote controls, domestic cats, resting, indoor, animal, pet, living room, ...                                                                                   |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected` | Title: Two cats resting on a pink surface \| Description: Two cats are lying on a bright pink fabric surface. One cat is curled up with its eyes closed, while... \| Keywords (10): cats, remote controls, pink fabric, lying down, curled up, eyes closed, striped pattern, light, indoor setting, ...                                                                               |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected` | Title: Two Cats Sleeping on Pink Sofa \| Description: Two tabby cats rest peacefully on a bright pink sofa. Remote controls lie beside them. Natural daylight illuminates the scene. \| Keywords (10): cats, sleeping, pink, sofa, remote, tabby, rest, daylight, couch, feline                                                                                                       |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected` | Title: Two Tabby Cats Reclining on Pink Sofa \| Description: Two tabby cats lie stretched on a bright pink sofa; one wears a green collar, and two remote controls re... \| Keywords (10): tabby cat, pink sofa, cat collar, remote control, feline relaxation, domestic interior, pet photography, ...                                                                               |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected` | Title: Two Tabby Cats Reclining on Pink Sofa Cushion \| Description: Two striped tabby cats recline on a bright pink fabric surface with two white television remote con... \| Keywords (13): tabby cats, pink cushion, reclining, striped coats, remote controls, indoor setting, relaxed posture, ...                                                                               |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected` | Title: Two Cats on a Couch \| Description: Two cats are sleeping on a pink couch. \| Keywords (4): cats, couch, sleeping, pink                                                                                                                                                                                                                                                        |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected` | Title: Two Cats Relaxing Together \| Description: Two tabby cats are curled up and sleeping side by side on a pink blanket, with a remote control placed nearby on the sofa. The sce... \| Keywords (10): cats, sleeping, pink blanket, sofa, remote control, tabby, relaxed, home, indoor, comfortable                                                                               |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected` | Title: Two tabby cats sleeping on a pink blanket \| Description: Two striped tabby cats lie resting on a pink fabric surface next to two remote controls. \| Keywords (14): cats, tabby, sleeping, pink, blanket, sofa, indoor, remote controls, lounging, resting, striped, fur, fabric, ...                                                                                         |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `no concerns detected` | Title: Two Tabby Cats on a Pink Sofa \| Description: Two tabby cats lie on a pink fabric sofa, with one resting horizontally and the other positioned verticall... \| Keywords (18): cats, tabby, sofa, couch, pet, sleeping, reclining, domestic animals, pink fabric, remote controls, furniture, ...                                                                               |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected` | Title: Two tabby cats sleeping on a pink surface \| Description: Two tabby cats lie on a pink fabric surface next to two remote controls. The cats are positioned... \| Keywords (15): tabby, cat, sleeping, pink, fabric, remote, electronics, domestic, animal, lying, indoor, bright, two, cats, ...                                                                               |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected` | Title: Two cats rest upon bright pink fabric \| Description: Two cats lie on a vividly coloured pink surface under soft lighting. They are accompanied by a remote... \| Keywords (14): cats, tabby, pile, fabric, pink, sleeping, reptiles, appendages, lounged, indoors, tortoiseshell, domestic, ...                                                                               |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected` | Title: "Feline Serenity on a Red Couch" \| Description: Two striped cats lounging on a red blanket, one with a remote control nearby. \| Keywords (10): cats, striped, red, blanket, lounging, remote control, comfort, rest, domestic, leisure                                                                                                                                       |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected` | Title: Two Cats Resting on a Pink Couch with Remotes \| Description: Two tabby cats are lying on a pink couch, with a remote control placed between them. The cats appear to be... \| Keywords (10): cats, resting, pink couch, remote control, tabby, sleeping, indoor, bright lighting, tail, visible                                                                               |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected` | Title: Two Cats Resting on a Pink Blanket \| Description: Two tabby cats are lying on a pink blanket, one stretched out and the other curled up. A remote contr... \| Keywords (18): cats, tabby, pink blanket, resting, remote control, fur, relaxed, indoor, couch, paws, stripes, sleeping, pet, ...                                                                               |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected` | Title: Two cats sleeping on a sofa \| Description: Two tabby cats are sleeping on a pink sofa, one curled up and the other stretched out, with remote controls nearby. \| Keywords (10): cat, tabby, pink sofa, remote control, sleeping, curled, stretched, pink blanket, pink cushion, cat furniture                                                                                |
| [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](#model-nvidia-nvidia-nemotron-35-super-vl-120b-a12b-nvfp4)                  | `no concerns detected` | Title: Two tabby cats sleeping on a pink sofa \| Description: Two domestic cats rest side by side on a bright pink cushioned surface, each with a striped coat. A white... \| Keywords (13): cats, tabby, pink, sofa, cushion, sleeping, remote controls, striped, animals, domestic, indoor, pets, ...                                                                               |
| [`sahilchachra/LensVLM-9B-MXFP4`](#model-sahilchachra-lensvlm-9b-mxfp4)                                                             | `no concerns detected` | Title: Two tabby cats sleeping on a pink sofa \| Description: Two tabby cats are curled up asleep on a bright pink fabric surface, likely a sofa or blanket. One cat lies on its side near the left edg... \| Keywords (10): cats, sleeping, tabby, pink, sofa, remote, rest, curled, striped, domestic                                                                               |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `concerns detected`    | Title: Two cats sleeping on a red sofa \| Description: A close-up image of two cats, one with calico fur and the other with tiger-striped fur, sleeping together on... \| Keywords (16): cat, cats, sleeping, sofa, red, pet, domestic, kitten, calico, tiger-striped, tuxedo, fur, lying, petting, ...[1,607 characters of reasoning omitted; complete output in the evidence block] |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns`       | A serene moment of feline tranquility on a vibrant pink couch, with two cats in peaceful slumber, their fur patterns contrasting against the soft fabric.                                                                                                                                                                                                                             |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `major concerns`       | Title: Two cats sleeping on a pink couch \| Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a r... \| Keywords (144): cats, sleeping, pink couch, remote control, striped, British English, British, British-English, ...                                                                               |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)                                                    | `major concerns`       | Title: Two Tabby Cats Sleeping on Pink Sofa \| Description: (not detected) \| Keywords (20): tabby cats, pink sofa, white remote controls, sleeping animals, domestic cats, striped fur, relaxed pose, ...                                                                                                                                                                            |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns`       | <\|channel\|> analysis<\|message\|> The two cats are sleeping on a pink couch.                                                                                                                                                                                                                                                                                                        |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns`       | Title: "A Study in Stripes" \| Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting... \| Keywords: (not detected)                                                                                |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.8.0`
- `mlx`: `0.32.5.dev20261107+fc344c986`
- `transformers`: `5.23.0`
- `tokenizers`: `0.23.3`
- `huggingface-hub`: `2.4.0`
- *Python Version:* 3.14.7
- *OS:* Darwin 27.2.0
- *macOS Version:* 27.2.1
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
- *Model load time:* 0.55s
- *Generation time:* 0.15s
- *Total time:* 0.79s
- *Prompt preparation time:* 0.0128
- *First-token latency:* 0.0501
- *Cleanup time:* 0.0656
- *Prompt tokens:* 350
- *Generation tokens:* 47
- *Total tokens:* 397
- *Prompt throughput (raw):* 6,982 tok/s
- *Generation throughput (raw):* 513 tok/s
- *Peak memory (GB):* 1.3
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 1.28s
- *Generation time:* 3.01s
- *Total time:* 4.42s
- *Prompt preparation time:* 0.0228
- *First-token latency:* 0.747
- *Cleanup time:* 0.0977
- *Prompt tokens:* 421
- *Generation tokens:* 94
- *Total tokens:* 515
- *Prompt throughput (raw):* 563 tok/s
- *Generation throughput (raw):* 42.1 tok/s
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two tabby cats sleeping on pink sofa
Description: Two tabby cats lie on their sides on a bright pink sofa cushion, with two white remote controls resting nearby. The cats are positioned head-to-head, appearing relaxed and asleep in soft, even lighting.
Keywords: tabby cats, sleeping, pink sofa, remote controls, striped fur, relaxed, cushion, indoor, pets, resting, two cats, white remotes, fabric, lounging
</pre>

</details>

---

<a id="model-ibm-granite-granite-vision-41-4b"></a>

### ibm-granite/granite-vision-4.1-4b

<details>
<summary>Complete evidence: ibm-granite/granite-vision-4.1-4b</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type granite4_vision)
- *Model load time:* 1.00s
- *Generation time:* 1.22s
- *Total time:* 2.34s
- *Prompt preparation time:* 0.031
- *First-token latency:* 0.302
- *Cleanup time:* 0.0814
- *Prompt tokens:* 721
- *Generation tokens:* 55
- *Total tokens:* 776
- *Prompt throughput (raw):* 2,389 tok/s
- *Generation throughput (raw):* 61.8 tok/s
- *Peak memory (GB):* 9.7
- *Active memory (GB):* 8.0
- *Cache memory (GB):* 0.13
- *Model-load active memory (GB):* 8
- *Post-cleanup active memory (GB):* 0.00135
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 696
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.granite4_vision.processing_granite4_vision.Granite4VisionProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 37d591f06319e8f1638b5adcf58bdf50e0f84f7a
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--ibm-granite--granite-vision-4.1-4b/snapshots/37d591f06319e8f1638b5adcf58bdf50e0f84f7a
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Cats at Rest on a Pink Sofa"
Description: Two cats are peacefully sleeping on a pink sofa, with remote controls nearby.
Keywords: cats, sleeping, pink sofa, remote controls, domestic, comfort, tranquility, home, leisure.
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
- *Model load time:* 2.14s
- *Generation time:* 3.14s
- *Total time:* 5.41s
- *Prompt preparation time:* 0.00924
- *First-token latency:* 0.641
- *Cleanup time:* 0.113
- *Prompt tokens:* 421
- *Generation tokens:* 80
- *Total tokens:* 501
- *Prompt throughput (raw):* 657 tok/s
- *Generation throughput (raw):* 32.2 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.22
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.00185
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Sleeping on Pink Couch

Description: Two tabby cats lie sleeping on a bright pink couch, each positioned beside a white remote control. The scene is lit by even indoor light, highlighting the cats' striped coats and relaxed postures.

Keywords: tabby cats, sleeping, pink couch, remote controls, striped fur, indoor pets, relaxed posture, domestic setting
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
- *Model load time:* 1.18s
- *Generation time:* 0.91s
- *Total time:* 2.21s
- *Prompt preparation time:* 0.0213
- *First-token latency:* 0.316
- *Cleanup time:* 0.0957
- *Prompt tokens:* 499
- *Generation tokens:* 49
- *Total tokens:* 548
- *Prompt throughput (raw):* 1,579 tok/s
- *Generation throughput (raw):* 84.5 tok/s
- *Peak memory (GB):* 8.0
- *Active memory (GB):* 7.1
- *Cache memory (GB):* 0.06
- *Model-load active memory (GB):* 7.08
- *Post-cleanup active memory (GB):* 0.00269
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
- *Model load time:* 1.19s
- *Generation time:* 3.65s
- *Total time:* 4.98s
- *Prompt preparation time:* 0.0266
- *First-token latency:* 1.95
- *Cleanup time:* 0.105
- *Prompt tokens:* 3,435
- *Generation tokens:* 90
- *Total tokens:* 3,525
- *Prompt throughput (raw):* 1,763 tok/s
- *Generation throughput (raw):* 55.2 tok/s
- *Peak memory (GB):* 10
- *Active memory (GB):* 8.9
- *Cache memory (GB):* 0.79
- *Model-load active memory (GB):* 8.94
- *Post-cleanup active memory (GB):* 0.003
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 0.90s
- *Generation time:* 0.49s
- *Total time:* 1.51s
- *Prompt preparation time:* 0.0169
- *First-token latency:* 0.114
- *Cleanup time:* 0.0949
- *Prompt tokens:* 344
- *Generation tokens:* 73
- *Total tokens:* 417
- *Prompt throughput (raw):* 3,017 tok/s
- *Generation throughput (raw):* 202 tok/s
- *Peak memory (GB):* 3.7
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.02
- *Model-load active memory (GB):* 2.82
- *Post-cleanup active memory (GB):* 0.00357
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 523
- *Image placeholders:* 1
- *Processor:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* b6311011d95c8894a57289bba8b916e6fbae9a61
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--LFM2.5-VL-3B-OptiQ-4bit/snapshots/b6311011d95c8894a57289bba8b916e6fbae9a61
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two cats sleeping on a pink couch.
Description: Two tabby cats are lying on a pink couch with remote controls nearby. The lighting is bright, highlighting the cats' fur patterns and the vibrant color of the couch.
Keywords: cats, pink couch, tabby, sleeping, remote controls, bright lighting, fur patterns, relaxation, indoor, domestic
</pre>

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
- *Model load time:* 1.13s
- *Generation time:* 0.34s
- *Total time:* 1.61s
- *Prompt preparation time:* 0.0224
- *First-token latency:* 0.106
- *Cleanup time:* 0.0953
- *Prompt tokens:* 332
- *Generation tokens:* 64
- *Total tokens:* 396
- *Prompt throughput (raw):* 3,128 tok/s
- *Generation throughput (raw):* 310 tok/s
- *Peak memory (GB):* 3.6
- *Active memory (GB):* 2.7
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 2.67
- *Post-cleanup active memory (GB):* 0.00408
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Sleeping Cats on Pink Couch
Description: Two tabby cats rest on a pink fabric couch, one near a remote, the other beside it, under soft indoor lighting.
Keywords: tabby, cat, sleeping, pink, couch, remote, indoor, light, fur, body, stretch, relaxation
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
- *Model load time:* 2.70s
- *Generation time:* 1.49s
- *Total time:* 4.35s
- *Prompt preparation time:* 0.0149
- *First-token latency:* 0.387
- *Cleanup time:* 0.133
- *Prompt tokens:* 332
- *Generation tokens:* 93
- *Total tokens:* 425
- *Prompt throughput (raw):* 858 tok/s
- *Generation throughput (raw):* 86.7 tok/s
- *Peak memory (GB):* 24
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.10
- *Model-load active memory (GB):* 23.5
- *Post-cleanup active memory (GB):* 0.00459
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

Keywords: cat, tabby, sleeping, sofa, pink, remote control, furniture, pet, resting, striped fur, domestic, relaxation, cozy, indoors, animal, leisure, cushion, household
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
- *Model load time:* 1.12s
- *Generation time:* 1.13s
- *Total time:* 2.38s
- *Prompt preparation time:* 0.0125
- *First-token latency:* 0.128
- *Cleanup time:* 0.104
- *Prompt tokens:* 177
- *Generation tokens:* 99
- *Total tokens:* 276
- *Prompt throughput (raw):* 1,379 tok/s
- *Generation throughput (raw):* 104 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 6.1
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 6.15
- *Post-cleanup active memory (GB):* 0.0049
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
Two tabby cats lie side by side on a red upholstered couch, partially covered by a pink blanket. A remote control is visible near each cat. Lighting is even and appears to be indoor ambient. One cat is stretched, the other is curled slightly, both appear to be resting.

Keywords:
cats, tabby, sleeping, red couch, pink blanket, remote controls, indoor, animals, fur, relaxed, household
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
- *Model load time:* 1.28s
- *Generation time:* 1.84s
- *Total time:* 3.24s
- *Prompt preparation time:* 0.0164
- *First-token latency:* 0.638
- *Cleanup time:* 0.0943
- *Prompt tokens:* 1,070
- *Generation tokens:* 81
- *Total tokens:* 1,151
- *Prompt throughput (raw):* 1,678 tok/s
- *Generation throughput (raw):* 68.4 tok/s
- *Peak memory (GB):* 8.9
- *Active memory (GB):* 8.0
- *Cache memory (GB):* 0.23
- *Model-load active memory (GB):* 8.01
- *Post-cleanup active memory (GB):* 0.00516
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two Cats Relaxing on Pink Cushioned Surface

Description:
Two cats lie stretched out on a pink cushioned surface, positioned near remote controls. The lighting is natural, and the cats appear calm and at ease.

Keywords:
cats, pink cushioned surface, remote controls, relaxed posture, indoor setting, natural lighting, striped fur, resting, cushioned fabric, domestic scene
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
- *Model load time:* 0.83s
- *Generation time:* 0.77s
- *Total time:* 1.73s
- *Prompt preparation time:* 0.0161
- *First-token latency:* 0.211
- *Cleanup time:* 0.094
- *Prompt tokens:* 1,069
- *Generation tokens:* 102
- *Total tokens:* 1,171
- *Prompt throughput (raw):* 5,057 tok/s
- *Generation throughput (raw):* 193 tok/s
- *Peak memory (GB):* 3.7
- *Active memory (GB):* 2.8
- *Cache memory (GB):* 0.16
- *Model-load active memory (GB):* 2.75
- *Post-cleanup active memory (GB):* 0.00543
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two resting tabby cats on a pink sofa

Description:
Two adult tabby cats lie relaxed on a bright pink sofa cushion, each with remote controls beside them. Soft, natural lighting highlights their striped fur patterns and relaxed postures.

Keywords:
tabby-cats, pink-sofa, remote-controls, indoor-resting, striped-fur, cozy-setting, domestic-pets, household-items, resting-scene, pet-photography, domestic-interior, soft-lighting
</pre>

</details>

---

<a id="model-mlx-community-mistral-small-4-119b-2603-4bit"></a>

### mlx-community/Mistral-Small-4-119B-2603-4bit

<details>
<summary>Complete evidence: mlx-community/Mistral-Small-4-119B-2603-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Model load time:* 11.77s
- *Generation time:* 8.13s
- *Total time:* 20.25s
- *Prompt preparation time:* 0.0264
- *First-token latency:* 5.57
- *Cleanup time:* 0.311
- *Prompt tokens:* 549
- *Generation tokens:* 82
- *Total tokens:* 631
- *Prompt throughput (raw):* 98.6 tok/s
- *Generation throughput (raw):* 86.8 tok/s
- *Peak memory (GB):* 69
- *Active memory (GB):* 68
- *Cache memory (GB):* 0.46
- *Model-load active memory (GB):* 67.8
- *Post-cleanup active memory (GB):* 0.00569
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 533
- *Processor:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 87b1249207734f029ec9dcefab80eabedfa6b812
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Mistral-Small-4-119B-2603-4bit/snapshots/87b1249207734f029ec9dcefab80eabedfa6b812
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Title:
Two cats resting on a pink sofa

Description:
Two cats are lying on a pink sofa with a remote control nearby. Soft lighting highlights their relaxed poses and the sofa's colour.

Keywords:
cats, sofa, pink, remote control, resting, indoor, pet, cushion, domestic, relaxation, soft lighting, living room, animal, comfort, leisure, home decor, sofa cushion, pet accessories
</pre>

</details>

---

<a id="model-mlx-community-molmo2-8b-4bit"></a>

### mlx-community/Molmo2-8B-4bit

#### Human accuracy note

Reviewed by john on 2026-10-25 for this exact image, model revision and
output. This is a person's judgement, not one of the mechanical checks.

Describes the pictured tabby cats as calico cats and gives incorrect
positional details.

<details>
<summary>Complete evidence: mlx-community/Molmo2-8B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Model load time:* 1.26s
- *Generation time:* 1.85s
- *Total time:* 3.23s
- *Prompt preparation time:* 0.0181
- *First-token latency:* 0.466
- *Cleanup time:* 0.0933
- *Prompt tokens:* 873
- *Generation tokens:* 95
- *Total tokens:* 968
- *Prompt throughput (raw):* 1,873 tok/s
- *Generation throughput (raw):* 69.7 tok/s
- *Peak memory (GB):* 8.2
- *Active memory (GB):* 6.5
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 6.45
- *Post-cleanup active memory (GB):* 0.006
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 1.34s
- *Generation time:* 2.89s
- *Total time:* 4.36s
- *Prompt preparation time:* 0.0182
- *First-token latency:* 0.632
- *Cleanup time:* 0.0978
- *Prompt tokens:* 860
- *Generation tokens:* 70
- *Total tokens:* 930
- *Prompt throughput (raw):* 1,361 tok/s
- *Generation throughput (raw):* 32.0 tok/s
- *Peak memory (GB):* 12
- *Active memory (GB):* 8.5
- *Cache memory (GB):* 3.1
- *Model-load active memory (GB):* 7.23
- *Post-cleanup active memory (GB):* 0.00662
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 2.77s
- *Generation time:* 40.75s
- *Total time:* 43.71s
- *Prompt preparation time:* 0.0294
- *First-token latency:* 0.874
- *Cleanup time:* 0.14
- *Prompt tokens:* 545
- *Generation tokens:* 908
- *Total tokens:* 1,453
- *Prompt throughput (raw):* 624 tok/s
- *Generation throughput (raw):* 22.8 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 23.9
- *Post-cleanup active memory (GB):* 0.00703
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
Title:
Two Tabby Cats Resting on Pink Blanket

Description:
Two tabby cats lie on a bright pink blanket on red cushions, one with its head resting down and the other facing downwards. Two white remote controls lie on the blanket beside them in bright light.

Keywords:
cats, tabby, pink blanket, red cushions, remote controls, resting, indoor, fabric, striped fur, collar, white remotes, sofa, pets, animals
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
- *Model load time:* 1.63s
- *Generation time:* 0.56s
- *Total time:* 2.34s
- *Prompt preparation time:* 0.0171
- *First-token latency:* 0.161
- *Cleanup time:* 0.105
- *Prompt tokens:* 418
- *Generation tokens:* 78
- *Total tokens:* 496
- *Prompt throughput (raw):* 2,593 tok/s
- *Generation throughput (raw):* 210 tok/s
- *Peak memory (GB):* 3.5
- *Active memory (GB):* 2.2
- *Cache memory (GB):* 0.11
- *Model-load active memory (GB):* 2.18
- *Post-cleanup active memory (GB):* 0.00756
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
- *Model load time:* 0.98s
- *Generation time:* 2.17s
- *Total time:* 3.28s
- *Prompt preparation time:* 0.0225
- *First-token latency:* 0.256
- *Cleanup time:* 0.105
- *Prompt tokens:* 883
- *Generation tokens:* 94
- *Total tokens:* 977
- *Prompt throughput (raw):* 3,444 tok/s
- *Generation throughput (raw):* 51.2 tok/s
- *Peak memory (GB):* 9.1
- *Active memory (GB):* 8.3
- *Cache memory (GB):* 0.42
- *Model-load active memory (GB):* 8.3
- *Post-cleanup active memory (GB):* 0.00762
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 2.56s
- *Generation time:* 1.39s
- *Total time:* 4.17s
- *Prompt preparation time:* 0.0275
- *First-token latency:* 0.412
- *Cleanup time:* 0.162
- *Prompt tokens:* 409
- *Generation tokens:* 81
- *Total tokens:* 490
- *Prompt throughput (raw):* 993 tok/s
- *Generation throughput (raw):* 83.5 tok/s
- *Peak memory (GB):* 23
- *Active memory (GB):* 22
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 21.8
- *Post-cleanup active memory (GB):* 0.00793
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title:
Two tabby cats sleeping on sofa

Description:
Two tabby cats are sleeping on a bright pink sofa. The sofa has two remote controls placed near the cats.

Keywords:
tabby cats, sleeping, sofa, pink, remote controls, domestic cats, resting, indoor, animal, pet, living room, comfort, relaxation, animal companions, domestic animals, household, feline, rest
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
- *Model load time:* 1.99s
- *Generation time:* 1.14s
- *Total time:* 3.31s
- *Prompt preparation time:* 0.0222
- *First-token latency:* 0.341
- *Cleanup time:* 0.145
- *Prompt tokens:* 409
- *Generation tokens:* 83
- *Total tokens:* 492
- *Prompt throughput (raw):* 1,200 tok/s
- *Generation throughput (raw):* 105 tok/s
- *Peak memory (GB):* 19
- *Active memory (GB):* 18
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 18.3
- *Post-cleanup active memory (GB):* 0.00824
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

Description: Two cats are lying on a bright pink fabric surface. One cat is curled up with its eyes closed, while the other lies on its side. Two remote controls are visible on the surface near the cats.

Keywords: cats, remote controls, pink fabric, lying down, curled up, eyes closed, striped pattern, light, indoor setting, domestic animals
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
- *Model load time:* 1.13s
- *Generation time:* 0.95s
- *Total time:* 2.23s
- *Prompt preparation time:* 0.0111
- *First-token latency:* 0.233
- *Cleanup time:* 0.121
- *Prompt tokens:* 409
- *Generation tokens:* 60
- *Total tokens:* 469
- *Prompt throughput (raw):* 1,754 tok/s
- *Generation throughput (raw):* 91.2 tok/s
- *Peak memory (GB):* 6.8
- *Active memory (GB):* 5.8
- *Cache memory (GB):* 0.11
- *Model-load active memory (GB):* 5.77
- *Post-cleanup active memory (GB):* 0.00855
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
- *Model load time:* 2.60s
- *Generation time:* 1.28s
- *Total time:* 4.08s
- *Prompt preparation time:* 0.0253
- *First-token latency:* 0.373
- *Cleanup time:* 0.152
- *Prompt tokens:* 421
- *Generation tokens:* 88
- *Total tokens:* 509
- *Prompt throughput (raw):* 1,130 tok/s
- *Generation throughput (raw):* 98.0 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.10
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.00906
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
Title: Two Tabby Cats Reclining on Pink Sofa
Description: Two tabby cats lie stretched on a bright pink sofa; one wears a green collar, and two remote controls rest nearby. Lighting appears even and indoor, with no shadows or motion blur.
Keywords: tabby cat, pink sofa, cat collar, remote control, feline relaxation, domestic interior, pet photography, animal rest, fabric texture, dual subjects
</pre>

<details>
<summary>Exact raw output</summary>

```text
Title: Two Tabby Cats Reclining on Pink Sofa  
Description: Two tabby cats lie stretched on a bright pink sofa; one wears a green collar, and two remote controls rest nearby. Lighting appears even and indoor, with no shadows or motion blur.  
Keywords: tabby cat, pink sofa, cat collar, remote control, feline relaxation, domestic interior, pet photography, animal rest, fabric texture, dual subjects
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
- *Model load time:* 2.02s
- *Generation time:* 4.54s
- *Total time:* 6.73s
- *Prompt preparation time:* 0.0103
- *First-token latency:* 0.747
- *Cleanup time:* 0.147
- *Prompt tokens:* 421
- *Generation tokens:* 109
- *Total tokens:* 530
- *Prompt throughput (raw):* 564 tok/s
- *Generation throughput (raw):* 28.9 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.25
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.0101
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* f9cb6495054684578f19bf751882716f87542a96
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/f9cb6495054684578f19bf751882716f87542a96
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Reclining on Pink Sofa Cushion

Description: Two striped tabby cats recline on a bright pink fabric surface with two white television remote controls placed nearby. The cats lie in opposing orientations with their limbs relaxed, and even indoor lighting illuminates their patterned coats and the vivid upholstery.

Keywords: tabby cats, pink cushion, reclining, striped coats, remote controls, indoor setting, relaxed posture, fabric texture, domestic scene, striped patterns, cat tails, feline rest, two cats
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
- *Model load time:* 0.82s
- *Generation time:* 0.77s
- *Total time:* 1.77s
- *Prompt preparation time:* 0.0649
- *First-token latency:* 0.429
- *Cleanup time:* 0.107
- *Prompt tokens:* 1,204
- *Generation tokens:* 31
- *Total tokens:* 1,235
- *Prompt throughput (raw):* 2,806 tok/s
- *Generation throughput (raw):* 110 tok/s
- *Peak memory (GB):* 5.6
- *Active memory (GB):* 4.5
- *Cache memory (GB):* 0.30
- *Model-load active memory (GB):* 4.5
- *Post-cleanup active memory (GB):* 0.0102
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 1.49s
- *Generation time:* 2.50s
- *Total time:* 4.67s
- *Prompt preparation time:* 0.555
- *First-token latency:* 1.07
- *Cleanup time:* 0.116
- *Prompt tokens:* 2,909
- *Generation tokens:* 82
- *Total tokens:* 2,991
- *Prompt throughput (raw):* 2,713 tok/s
- *Generation throughput (raw):* 87.5 tok/s
- *Peak memory (GB):* 6.5
- *Active memory (GB):* 5.5
- *Cache memory (GB):* 0.47
- *Model-load active memory (GB):* 5.46
- *Post-cleanup active memory (GB):* 0.0107
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 3.45s
- *Generation time:* 1.66s
- *Total time:* 5.31s
- *Prompt preparation time:* 0.0296
- *First-token latency:* 0.278
- *Cleanup time:* 0.141
- *Prompt tokens:* 386
- *Generation tokens:* 66
- *Total tokens:* 452
- *Prompt throughput (raw):* 1,391 tok/s
- *Generation throughput (raw):* 59.3 tok/s
- *Peak memory (GB):* 28
- *Active memory (GB):* 27
- *Cache memory (GB):* 0.01
- *Model-load active memory (GB):* 27.2
- *Post-cleanup active memory (GB):* 0.0107
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* &lt;eos&gt;

*Readable output:*

<pre class="model-output-readable">
Title: Two tabby cats sleeping on a pink blanket
Description: Two striped tabby cats lie resting on a pink fabric surface next to two remote controls.
Keywords: cats, tabby, sleeping, pink, blanket, sofa, indoor, remote controls, lounging, resting, striped, fur, fabric, domestic
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
- *Model load time:* 2.15s
- *Generation time:* 1.89s
- *Total time:* 4.23s
- *Prompt preparation time:* 0.0323
- *First-token latency:* 0.299
- *Cleanup time:* 0.131
- *Prompt tokens:* 390
- *Generation tokens:* 88
- *Total tokens:* 478
- *Prompt throughput (raw):* 1,306 tok/s
- *Generation throughput (raw):* 56.1 tok/s
- *Peak memory (GB):* 7.5
- *Active memory (GB):* 6.8
- *Cache memory (GB):* 0.25
- *Model-load active memory (GB):* 6.75
- *Post-cleanup active memory (GB):* 0.0112
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

Description: Two tabby cats lie on a pink fabric sofa, with one resting horizontally and the other positioned vertically. Two remote controls are also visible near the cats.

Keywords: cats, tabby, sofa, couch, pet, sleeping, reclining, domestic animals, pink fabric, remote controls, furniture, tabby cat, resting, indoor, grey and brown, stripes, paws, tail
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
- *Model load time:* 2.60s
- *Generation time:* 1.10s
- *Total time:* 3.89s
- *Prompt preparation time:* 0.0106
- *First-token latency:* 0.383
- *Cleanup time:* 0.156
- *Prompt tokens:* 390
- *Generation tokens:* 75
- *Total tokens:* 465
- *Prompt throughput (raw):* 1,018 tok/s
- *Generation throughput (raw):* 107 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.17
- *Model-load active memory (GB):* 15.4
- *Post-cleanup active memory (GB):* 0.0118
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
Title: Two tabby cats sleeping on a pink surface

Description: Two tabby cats lie on a pink fabric surface next to two remote controls. The cats are positioned with their bodies facing different directions under bright lighting.

Keywords: tabby, cat, sleeping, pink, fabric, remote, electronics, domestic, animal, lying, indoor, bright, two, cats, striped
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
- *Model load time:* 1.84s
- *Generation time:* 0.81s
- *Total time:* 2.82s
- *Prompt preparation time:* 0.0108
- *First-token latency:* 0.177
- *Cleanup time:* 0.124
- *Prompt tokens:* 386
- *Generation tokens:* 67
- *Total tokens:* 453
- *Prompt throughput (raw):* 2,184 tok/s
- *Generation throughput (raw):* 108 tok/s
- *Peak memory (GB):* 5.9
- *Active memory (GB):* 5.2
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 5.16
- *Post-cleanup active memory (GB):* 0.0123
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
Title: Two cats rest upon bright pink fabric
Description: Two cats lie on a vividly coloured pink surface under soft lighting. They are accompanied by a remote control device.
Keywords: cats, tabby, pile, fabric, pink, sleeping, reptiles, appendages, lounged, indoors, tortoiseshell, domestic, creature, fluffy
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
- *Model load time:* 0.68s
- *Generation time:* 0.67s
- *Total time:* 1.53s
- *Prompt preparation time:* 0.0354
- *First-token latency:* 0.283
- *Cleanup time:* 0.121
- *Prompt tokens:* 734
- *Generation tokens:* 57
- *Total tokens:* 791
- *Prompt throughput (raw):* 2,592 tok/s
- *Generation throughput (raw):* 158 tok/s
- *Peak memory (GB):* 4.7
- *Active memory (GB):* 3.0
- *Cache memory (GB):* 0.13
- *Model-load active memory (GB):* 3.03
- *Post-cleanup active memory (GB):* 0.0125
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Feline Serenity on a Red Couch"
Description: Two striped cats lounging on a red blanket, one with a remote control nearby.
Keywords: cats, striped, red, blanket, lounging, remote control, comfort, rest, domestic, leisure
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
- *Model load time:* 0.94s
- *Generation time:* 0.86s
- *Total time:* 1.93s
- *Prompt preparation time:* 0.0127
- *First-token latency:* 0.143
- *Cleanup time:* 0.116
- *Prompt tokens:* 420
- *Generation tokens:* 88
- *Total tokens:* 508
- *Prompt throughput (raw):* 2,930 tok/s
- *Generation throughput (raw):* 130 tok/s
- *Peak memory (GB):* 5.3
- *Active memory (GB):* 3.9
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 3.93
- *Post-cleanup active memory (GB):* 0.0138
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Cats Resting on a Pink Couch with Remotes
Description: Two tabby cats are lying on a pink couch, with a remote control placed between them. The cats appear to be sleeping or resting, with one cat's tail visible. The lighting is bright, and the setting is indoors.
Keywords: cats, resting, pink couch, remote control, tabby, sleeping, indoor, bright lighting, tail, visible
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
- *Model load time:* 1.73s
- *Generation time:* 3.56s
- *Total time:* 5.44s
- *Prompt preparation time:* 0.0128
- *First-token latency:* 0.766
- *Cleanup time:* 0.132
- *Prompt tokens:* 538
- *Generation tokens:* 87
- *Total tokens:* 625
- *Prompt throughput (raw):* 702 tok/s
- *Generation throughput (raw):* 31.4 tok/s
- *Peak memory (GB):* 16
- *Active memory (GB):* 15
- *Cache memory (GB):* 0.14
- *Model-load active memory (GB):* 15.1
- *Post-cleanup active memory (GB):* 0.0141
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
  "seed": 0, "temperature": 0.15, "top_k": 0, "top_p": 1.0}
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
- *Generation time:* 0.90s
- *Total time:* 3.54s
- *Prompt preparation time:* 0.0247
- *First-token latency:* 0.382
- *Cleanup time:* 0.148
- *Prompt tokens:* 421
- *Generation tokens:* 65
- *Total tokens:* 486
- *Prompt throughput (raw):* 1,101 tok/s
- *Generation throughput (raw):* 128 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.07
- *Model-load active memory (GB):* 19.7
- *Post-cleanup active memory (GB):* 0.0144
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
  "seed": 0, "temperature": 0.6, "top_k": 0, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">

Title: Two cats sleeping on a sofa
Description: Two tabby cats are sleeping on a pink sofa, one curled up and the other stretched out, with remote controls nearby.
Keywords: cat, tabby, pink sofa, remote control, sleeping, curled, stretched, pink blanket, pink cushion, cat furniture
</pre>

</details>

---

<a id="model-nvidia-nvidia-nemotron-35-super-vl-120b-a12b-nvfp4"></a>

### nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4

<details>
<summary>Complete evidence: nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type nemotron_h_omni via
  nemotron_h_nano_omni)
- *Model load time:* 16.88s
- *Generation time:* 21.41s
- *Total time:* 38.68s
- *Prompt preparation time:* 0.0282
- *First-token latency:* 16.3
- *Cleanup time:* 0.32
- *Prompt tokens:* 420
- *Generation tokens:* 89
- *Total tokens:* 509
- *Prompt throughput (raw):* 25.7 tok/s
- *Generation throughput (raw):* 25.7 tok/s
- *Peak memory (GB):* 81
- *Active memory (GB):* 78
- *Cache memory (GB):* 0.20
- *Model-load active memory (GB):* 78.4
- *Post-cleanup active memory (GB):* 0.0146
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 553
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.nemotron_h_nano_omni.processing_nemotron_h_nano_omni.NemotronHNanoOmniProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 3acb09951cff50cda7be0200ebbad2c15a3e63ba
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/3acb09951cff50cda7be0200ebbad2c15a3e63ba
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 0, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two tabby cats sleeping on a pink sofa

Description: Two domestic cats rest side by side on a bright pink cushioned surface, each with a striped coat. A white remote control lies beside each animal, with the larger cat stretched out and the smaller one curled.

Keywords: cats, tabby, pink, sofa, cushion, sleeping, remote controls, striped, animals, domestic, indoor, pets, furnishings
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
- *Model load time:* 1.82s
- *Generation time:* 1.54s
- *Total time:* 3.51s
- *Prompt preparation time:* 0.0103
- *First-token latency:* 0.273
- *Cleanup time:* 0.127
- *Prompt tokens:* 421
- *Generation tokens:* 111
- *Total tokens:* 532
- *Prompt throughput (raw):* 1,544 tok/s
- *Generation throughput (raw):* 88.8 tok/s
- *Peak memory (GB):* 6.8
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 5.69
- *Post-cleanup active memory (GB):* 0.0151
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit"></a>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit

<details>
<summary>Complete evidence: mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Duplicate keywords: sofa
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Model load time:* 1.94s
- *Generation time:* 4.42s
- *Total time:* 6.50s
- *Prompt preparation time:* 0.0197
- *First-token latency:* 0.352
- *Cleanup time:* 0.105
- *Prompt tokens:* 545
- *Generation tokens:* 540
- *Total tokens:* 1,085
- *Prompt throughput (raw):* 1,550 tok/s
- *Generation throughput (raw):* 133 tok/s
- *Peak memory (GB):* 19
- *Active memory (GB):* 17
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 17.4
- *Post-cleanup active memory (GB):* 0.00207
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
  "&lt;think&gt;", "top_k": 0, "top_p": 0.95}
- *EOS token:* &lt;/s&gt;

*Readable output:*

<pre class="model-output-readable">
Alright, let's get this done. I need to create some metadata for an image, and it's got to be spot-on.

First, the image itself. I see two cats, definitely on a red sofa, and they're *sleeping*. One's a calico or tortoiseshell, and the other is a tiger-striped or tuxedo. The lighting is soft, probably natural. The image quality looks good.

Okay, the title. It needs to be short and sweet, and in British English. I think "Two cats sleeping on a red sofa" is perfect. Concise and accurate.

Now, the description. I'll cover the main subject (two cats), the setting (red sofa), the action (sleeping), and the lighting (soft, natural). I'll also mention the distinct details, like the different colors and patterns of the cats. I'll make sure to be objective and stick to what I *see*.

Finally, the keywords. I need to be thorough here. I'll include terms like "cat," "cats," "kittens," "sleeping," "sofa," "red," and some breed-related terms. I'll also add words like "pet" and "domestic" to be as specific as possible. I'll make sure the keywords are distinct and relevant to the image.

Let's get this right.

**Title:** Two cats sleeping on a red sofa

**Description:** A close-up image of two cats, one with calico fur and the other with tiger-striped fur, sleeping together on a red sofa. Soft, natural light illuminates the scene.

**Keywords:** cat, cats, sleeping, sofa, red, pet, domestic, kitten, calico, tiger-striped, tuxedo, fur, lying, petting, sunlight, sofa.

That should cover it. It's all about clear, factual observations. Nothing speculative, just what's right in front of me.
&lt;/think&gt;

Title: Two cats sleeping on a red sofa

Description: A close-up image of two cats, one with calico fur and the other with tiger-striped fur, sleeping together on a red sofa. Soft, natural light illuminates the scene.

Keywords: cat, cats, sleeping, sofa, red, pet, domestic, kitten, calico, tiger-striped, tuxedo, fur, lying, petting, sunlight, sofa.
</pre>

<details>
<summary>Exact raw output</summary>

```text
Alright, let's get this done. I need to create some metadata for an image, and it's got to be spot-on.

First, the image itself. I see two cats, definitely on a red sofa, and they're *sleeping*. One's a calico or tortoiseshell, and the other is a tiger-striped or tuxedo. The lighting is soft, probably natural. The image quality looks good.

Okay, the title. It needs to be short and sweet, and in British English. I think "Two cats sleeping on a red sofa" is perfect. Concise and accurate.

Now, the description. I'll cover the main subject (two cats), the setting (red sofa), the action (sleeping), and the lighting (soft, natural). I'll also mention the distinct details, like the different colors and patterns of the cats. I'll make sure to be objective and stick to what I *see*.

Finally, the keywords. I need to be thorough here. I'll include terms like "cat," "cats," "kittens," "sleeping," "sofa," "red," and some breed-related terms. I'll also add words like "pet" and "domestic" to be as specific as possible. I'll make sure the keywords are distinct and relevant to the image.

Let's get this right.

**Title:** Two cats sleeping on a red sofa

**Description:** A close-up image of two cats, one with calico fur and the other with tiger-striped fur, sleeping together on a red sofa. Soft, natural light illuminates the scene.

**Keywords:** cat, cats, sleeping, sofa, red, pet, domestic, kitten, calico, tiger-striped, tuxedo, fur, lying, petting, sunlight, sofa.

That should cover it. It's all about clear, factual observations. Nothing speculative, just what's right in front of me.
</think>

Title: Two cats sleeping on a red sofa

Description: A close-up image of two cats, one with calico fur and the other with tiger-striped fur, sleeping together on a red sofa. Soft, natural light illuminates the scene.

Keywords: cat, cats, sleeping, sofa, red, pet, domestic, kitten, calico, tiger-striped, tuxedo, fur, lying, petting, sunlight, sofa.
```

</details>

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
- *Model load time:* 0.83s
- *Generation time:* 0.21s
- *Total time:* 1.16s
- *Prompt preparation time:* 0.0231
- *First-token latency:* 0.102
- *Cleanup time:* 0.0809
- *Prompt tokens:* 120
- *Generation tokens:* 32
- *Total tokens:* 152
- *Prompt throughput (raw):* 1,180 tok/s
- *Generation throughput (raw):* 370 tok/s
- *Peak memory (GB):* 2.1
- *Active memory (GB):* 1.2
- *Cache memory (GB):* 0.02
- *Model-load active memory (GB):* 1.25
- *Post-cleanup active memory (GB):* 0.00238
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 0.72s
- *Generation time:* 3.27s
- *Total time:* 4.11s
- *Prompt preparation time:* 0.028
- *First-token latency:* 0.348
- *Cleanup time:* 0.0859
- *Prompt tokens:* 3,438
- *Generation tokens:* 1,000
- *Total tokens:* 4,438
- *Prompt throughput (raw):* 9,868 tok/s
- *Generation throughput (raw):* 346 tok/s
- *Peak memory (GB):* 2.3
- *Active memory (GB):* 1.1
- *Cache memory (GB):* 0.56
- *Model-load active memory (GB):* 1.08
- *Post-cleanup active memory (GB):* 0.00331
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two cats sleeping on a pink couch

Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a remote control.

Keywords: cats, sleeping, pink couch, remote control, striped, British English, British, British-English, British-English catalogue, British-English metadata, British-English details, British-English context, British-English information, British-English data, British-English content, British-English examples, British-English examples of metadata, British-English examples of keywords, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata
</pre>

</details>

---

<a id="model-mlx-community-qwen36-35b-a3b-4bit"></a>

### mlx-community/Qwen3.6-35B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-35B-A3B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Required labelled fields not detected: description
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.64s
- *Generation time:* 1.40s
- *Total time:* 4.23s
- *Prompt preparation time:* 0.0266
- *First-token latency:* 0.394
- *Cleanup time:* 0.151
- *Prompt tokens:* 421
- *Generation tokens:* 96
- *Total tokens:* 517
- *Prompt throughput (raw):* 1,068 tok/s
- *Generation throughput (raw):* 96.8 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.11
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.00957
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 563
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 38740b847e4cb78f352aba30aa41c76e08e6eb46
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: Two Tabby Cats Sleeping on Pink Sofa

Two tabby cats lie sleeping on a bright pink sofa, positioned next to white remote controls.

Keywords: tabby cats, pink sofa, white remote controls, sleeping animals, domestic cats, striped fur, relaxed pose, indoor setting, living room, pet furniture, household items, soft lighting, comfortable space, feline companions, textile pattern, home decor, pet lifestyle, cozy environment, striped pattern, electronic device
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
- *Model load time:* 0.99s
- *Generation time:* 1.73s
- *Total time:* 2.95s
- *Prompt preparation time:* 0.0421
- *First-token latency:* 1.54
- *Cleanup time:* 0.178
- *Prompt tokens:* 3,512
- *Generation tokens:* 14
- *Total tokens:* 3,526
- *Prompt throughput (raw):* 2,284 tok/s
- *Generation throughput (raw):* 91.9 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.55
- *Model-load active memory (GB):* 5.69
- *Post-cleanup active memory (GB):* 0.0129
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
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
- *Model load time:* 0.81s
- *Generation time:* 0.45s
- *Total time:* 1.41s
- *Prompt preparation time:* 0.0163
- *First-token latency:* 0.0903
- *Cleanup time:* 0.122
- *Prompt tokens:* 116
- *Generation tokens:* 97
- *Total tokens:* 213
- *Prompt throughput (raw):* 1,285 tok/s
- *Generation throughput (raw):* 279 tok/s
- *Peak memory (GB):* 1.6
- *Active memory (GB):* 0.61
- *Cache memory (GB):* 0.24
- *Model-load active memory (GB):* 0.612
- *Post-cleanup active memory (GB):* 0.0135
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
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

*Readable output:*

<pre class="model-output-readable">
Title: "A Study in Stripes"
Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting is soft, and the person is facing away from the camera, giving a sense of intimacy. The image is a straightforward representation of the subject, with no additional details or context provided.
</pre>

</details>

---
