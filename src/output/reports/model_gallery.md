# Model Output Gallery

Generated on: 2026-11-11 18:47:27 GMT

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

| Model                                                                                                                               | Mechanical checks               | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Observations                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|---------|---------------------|-----------------|---------|------------|---------|---------------------------------------------------------------------------------------------------|
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected`          | 0.56s   | 516 tok/s           | 0.07            | 1.3     | 350        | 47      | none                                                                                              |
| [`TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`](#model-technobaptist-ternary-bonsai-2-27b-mlx-2bit)                                 | `no concerns detected`          | 4.73s   | 43.4 tok/s          | 1.18            | 11      | 421        | 94      | none                                                                                              |
| [`ibm-granite/granite-vision-4.1-4b`](#model-ibm-granite-granite-vision-41-4b)                                                      | `no concerns detected`          | 2.44s   | 65.4 tok/s          | 0.53            | 9.7     | 721        | 55      | none                                                                                              |
| [`mlx-community/AREX-2-4bit`](#model-mlx-community-arex-2-4bit)                                                                     | `no concerns detected`          | 5.17s   | 33.0 tok/s          | 0.60            | 17      | 421        | 80      | none                                                                                              |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected`          | 2.15s   | 85.7 tok/s          | 0.31            | 8.0     | 499        | 49      | none                                                                                              |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected`          | 4.85s   | 55.4 tok/s          | 1.96            | 10      | 3,435      | 90      | none                                                                                              |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected`          | 1.46s   | 214 tok/s           | 0.22            | 3.7     | 344        | 73      | none                                                                                              |
| [`mlx-community/MiniCPM-V-4.7-1B-4bit`](#model-mlx-community-minicpm-v-47-1b-4bit)                                                  | `no concerns detected`          | 1.42s   | 314 tok/s           | 0.13            | 3.6     | 332        | 64      | none                                                                                              |
| [`mlx-community/MiniCPM-V-4.7-35B-A3B-4bit`](#model-mlx-community-minicpm-v-47-35b-a3b-4bit)                                        | `no concerns detected`          | 4.19s   | 86.8 tok/s          | 0.34            | 24      | 332        | 93      | none                                                                                              |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `no concerns detected`          | 2.33s   | 106 tok/s           | 0.16            | 7.0     | 177        | 99      | none                                                                                              |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected`          | 3.35s   | 68.7 tok/s          | 0.68            | 8.9     | 1,070      | 81      | none                                                                                              |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected`          | 1.63s   | 201 tok/s           | 0.23            | 3.7     | 1,069      | 102     | none                                                                                              |
| [`mlx-community/Mistral-Small-4-119B-2603-4bit`](#model-mlx-community-mistral-small-4-119b-2603-4bit)                               | `no concerns detected`          | 12.56s  | 90.2 tok/s          | 3.72            | 69      | 549        | 82      | none                                                                                              |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `no concerns detected`          | 2.98s   | 73.2 tok/s          | 0.48            | 8.2     | 873        | 95      | none                                                                                              |
| [`mlx-community/MolmoPoint-8B-4bit`](#model-mlx-community-molmopoint-8b-4bit)                                                       | `no concerns detected`          | 4.34s   | 32.6 tok/s          | 0.77            | 12      | 860        | 70      | none                                                                                              |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected`          | 1.65s   | 246 tok/s           | 0.17            | 3.5     | 418        | 78      | none                                                                                              |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected`          | 2.92s   | 59.8 tok/s          | 0.29            | 9.1     | 883        | 94      | none                                                                                              |
| [`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit)                           | `no concerns detected`          | 3.76s   | 94.6 tok/s          | 0.35            | 23      | 409        | 81      | none                                                                                              |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected`          | 3.11s   | 129 tok/s           | 0.32            | 19      | 409        | 83      | none                                                                                              |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected`          | 1.83s   | 103 tok/s           | 0.25            | 6.8     | 409        | 60      | none                                                                                              |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected`          | 4.02s   | 113 tok/s           | 0.33            | 21      | 421        | 88      | none                                                                                              |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected`          | 7.10s   | 33.0 tok/s          | 0.67            | 17      | 421        | 109     | none                                                                                              |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected`          | 1.44s   | 125 tok/s           | 0.41            | 5.6     | 1,204      | 31      | none                                                                                              |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected`          | 3.98s   | 100 tok/s           | 1.41            | 6.5     | 2,909      | 82      | none                                                                                              |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected`          | 4.89s   | 66.9 tok/s          | 1.44            | 28      | 386        | 66      | none                                                                                              |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `no concerns detected`          | 3.55s   | 62.8 tok/s          | 0.26            | 7.5     | 390        | 88      | none                                                                                              |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected`          | 3.42s   | 126 tok/s           | 0.32            | 16      | 390        | 75      | none                                                                                              |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected`          | 2.39s   | 125 tok/s           | 0.17            | 5.9     | 386        | 67      | none                                                                                              |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected`          | 1.35s   | 181 tok/s           | 0.28            | 4.7     | 734        | 57      | none                                                                                              |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected`          | 1.70s   | 150 tok/s           | 0.16            | 5.3     | 420        | 88      | none                                                                                              |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected`          | 4.86s   | 36.3 tok/s          | 0.65            | 16      | 538        | 87      | none                                                                                              |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected`          | 3.05s   | 159 tok/s           | 0.33            | 21      | 421        | 65      | none                                                                                              |
| [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](#model-nvidia-nvidia-nemotron-35-super-vl-120b-a12b-nvfp4)                  | `no concerns detected`          | 34.31s  | 30.9 tok/s          | 16.53           | 81      | 420        | 89      | none                                                                                              |
| [`sahilchachra/LensVLM-9B-MXFP4`](#model-sahilchachra-lensvlm-9b-mxfp4)                                                             | `no concerns detected`          | 2.72s   | 101 tok/s           | 0.25            | 6.8     | 421        | 111     | none                                                                                              |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `concerns detected`             | 6.16s   | 136 tok/s           | 0.33            | 19      | 545        | 540     | duplicate keywords                                                                                |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns: answer format` | 1.22s   | 371 tok/s           | 0.38            | 2.2     | 120        | 32      | labelled fields not detected                                                                      |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `major concerns: generation`    | 3.90s   | 349 tok/s           | 0.37            | 2.3     | 3,438      | 1,000   | repeated text; cut off at token limit; duplicate keywords                                         |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `major concerns: answer format` | 43.35s  | 25.2 tok/s          | 0.87            | 25      | 545        | 1,000   | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)                                                    | `major concerns: answer format` | 3.84s   | 116 tok/s           | 0.33            | 21      | 421        | 96      | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns: answer format` | 2.58s   | insufficient sample | 1.25            | 7.0     | 3,512      | 14      | control tokens visible; labelled fields not detected                                              |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns: answer format` | 0.96s   | 373 tok/s           | 0.09            | 1.5     | 116        | 97      | labelled fields not detected                                                                      |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 0.56s

Lowest peak memory among completions without detected concerns: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.3 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Major Mechanical Concerns in This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                           | Mechanical checks               | Observations                                                                                      |
|-------------------------------------------------------------------------------------------------|---------------------------------|---------------------------------------------------------------------------------------------------|
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                      | `major concerns: answer format` | labelled fields not detected                                                                      |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                  | `major concerns: generation`    | repeated text; cut off at token limit; duplicate keywords                                         |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit) | `major concerns: answer format` | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)                | `major concerns: answer format` | labelled fields not detected                                                                      |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)         | `major concerns: answer format` | control tokens visible; labelled fields not detected                                              |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                    | `major concerns: answer format` | labelled fields not detected                                                                      |
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
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `major concerns`       | Title: (not detected) \| Description: (not detected) \| Keywords (9): We need British-English. Use only clearly visible facts. Omit uncertain details and unsupported identity, ...                                                                                                                                                                                                   |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)                                                    | `major concerns`       | Title: Two Tabby Cats Sleeping on Pink Sofa Two tabby cats lie sleeping on a bright... \| Description: (not detected) \| Keywords (20): tabby cats, pink sofa, white remote controls, sleeping animals, domestic cats, striped fur, relaxed pose, ...                                                                                                                                 |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns`       | <\|channel\|> analysis<\|message\|> The two cats are sleeping on a pink couch.                                                                                                                                                                                                                                                                                                        |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns`       | Title: "A Study in Stripes" \| Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting... \| Keywords: (not detected)                                                                                |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.8.0`
- `mlx`: `0.32.5.dev20261107+fc344c986`
- `transformers`: `5.22.2`
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
- *Model load time:* 0.31s
- *Generation time:* 0.16s
- *Total time:* 0.56s
- *Prompt preparation time:* 0.0129
- *First-token latency:* 0.0588
- *Cleanup time:* 0.0701
- *Prompt tokens:* 350
- *Generation tokens:* 47
- *Total tokens:* 397
- *Prompt throughput (raw):* 5,957 tok/s
- *Generation throughput (raw):* 516 tok/s
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
- *Model load time:* 1.23s
- *Generation time:* 3.35s
- *Total time:* 4.73s
- *Prompt preparation time:* 0.0235
- *First-token latency:* 1.16
- *Cleanup time:* 0.0997
- *Prompt tokens:* 421
- *Generation tokens:* 94
- *Total tokens:* 515
- *Prompt throughput (raw):* 363 tok/s
- *Generation throughput (raw):* 43.4 tok/s
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
- *Model load time:* 0.94s
- *Generation time:* 1.38s
- *Total time:* 2.44s
- *Prompt preparation time:* 0.0345
- *First-token latency:* 0.515
- *Cleanup time:* 0.0763
- *Prompt tokens:* 721
- *Generation tokens:* 55
- *Total tokens:* 776
- *Prompt throughput (raw):* 1,399 tok/s
- *Generation throughput (raw):* 65.4 tok/s
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
- *Model load time:* 2.01s
- *Generation time:* 3.03s
- *Total time:* 5.17s
- *Prompt preparation time:* 0.00908
- *First-token latency:* 0.591
- *Cleanup time:* 0.101
- *Prompt tokens:* 421
- *Generation tokens:* 80
- *Total tokens:* 501
- *Prompt throughput (raw):* 712 tok/s
- *Generation throughput (raw):* 33.0 tok/s
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
- *Model load time:* 1.13s
- *Generation time:* 0.88s
- *Total time:* 2.15s
- *Prompt preparation time:* 0.0146
- *First-token latency:* 0.303
- *Cleanup time:* 0.107
- *Prompt tokens:* 499
- *Generation tokens:* 49
- *Total tokens:* 548
- *Prompt throughput (raw):* 1,648 tok/s
- *Generation throughput (raw):* 85.7 tok/s
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
- *Model load time:* 1.13s
- *Generation time:* 3.59s
- *Total time:* 4.85s
- *Prompt preparation time:* 0.0251
- *First-token latency:* 1.89
- *Cleanup time:* 0.103
- *Prompt tokens:* 3,435
- *Generation tokens:* 90
- *Total tokens:* 3,525
- *Prompt throughput (raw):* 1,815 tok/s
- *Generation throughput (raw):* 55.4 tok/s
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
- *Model load time:* 0.78s
- *Generation time:* 0.57s
- *Total time:* 1.46s
- *Prompt preparation time:* 0.0157
- *First-token latency:* 0.216
- *Cleanup time:* 0.088
- *Prompt tokens:* 344
- *Generation tokens:* 73
- *Total tokens:* 417
- *Prompt throughput (raw):* 1,592 tok/s
- *Generation throughput (raw):* 214 tok/s
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
- *Model load time:* 0.95s
- *Generation time:* 0.34s
- *Total time:* 1.42s
- *Prompt preparation time:* 0.0269
- *First-token latency:* 0.1
- *Cleanup time:* 0.0895
- *Prompt tokens:* 332
- *Generation tokens:* 64
- *Total tokens:* 396
- *Prompt throughput (raw):* 3,313 tok/s
- *Generation throughput (raw):* 314 tok/s
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
- *Model load time:* 2.63s
- *Generation time:* 1.41s
- *Total time:* 4.19s
- *Prompt preparation time:* 0.0151
- *First-token latency:* 0.321
- *Cleanup time:* 0.121
- *Prompt tokens:* 332
- *Generation tokens:* 93
- *Total tokens:* 425
- *Prompt throughput (raw):* 1,035 tok/s
- *Generation throughput (raw):* 86.8 tok/s
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
- *Model load time:* 1.10s
- *Generation time:* 1.11s
- *Total time:* 2.33s
- *Prompt preparation time:* 0.0123
- *First-token latency:* 0.125
- *Cleanup time:* 0.0997
- *Prompt tokens:* 177
- *Generation tokens:* 99
- *Total tokens:* 276
- *Prompt throughput (raw):* 1,420 tok/s
- *Generation throughput (raw):* 106 tok/s
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
- *Model load time:* 1.37s
- *Generation time:* 1.86s
- *Total time:* 3.35s
- *Prompt preparation time:* 0.0158
- *First-token latency:* 0.669
- *Cleanup time:* 0.0908
- *Prompt tokens:* 1,070
- *Generation tokens:* 81
- *Total tokens:* 1,151
- *Prompt throughput (raw):* 1,600 tok/s
- *Generation throughput (raw):* 68.7 tok/s
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
- *Model load time:* 0.78s
- *Generation time:* 0.74s
- *Total time:* 1.63s
- *Prompt preparation time:* 0.0164
- *First-token latency:* 0.205
- *Cleanup time:* 0.0828
- *Prompt tokens:* 1,069
- *Generation tokens:* 102
- *Total tokens:* 1,171
- *Prompt throughput (raw):* 5,218 tok/s
- *Generation throughput (raw):* 201 tok/s
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
- *Model load time:* 7.63s
- *Generation time:* 4.64s
- *Total time:* 12.56s
- *Prompt preparation time:* 0.0226
- *First-token latency:* 2.44
- *Cleanup time:* 0.26
- *Prompt tokens:* 549
- *Generation tokens:* 82
- *Total tokens:* 631
- *Prompt throughput (raw):* 225 tok/s
- *Generation throughput (raw):* 90.2 tok/s
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
- *Model load time:* 1.07s
- *Generation time:* 1.79s
- *Total time:* 2.98s
- *Prompt preparation time:* 0.0186
- *First-token latency:* 0.472
- *Cleanup time:* 0.0877
- *Prompt tokens:* 873
- *Generation tokens:* 95
- *Total tokens:* 968
- *Prompt throughput (raw):* 1,851 tok/s
- *Generation throughput (raw):* 73.2 tok/s
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
- *Model load time:* 1.28s
- *Generation time:* 2.94s
- *Total time:* 4.34s
- *Prompt preparation time:* 0.0204
- *First-token latency:* 0.713
- *Cleanup time:* 0.0904
- *Prompt tokens:* 860
- *Generation tokens:* 70
- *Total tokens:* 930
- *Prompt throughput (raw):* 1,206 tok/s
- *Generation throughput (raw):* 32.6 tok/s
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
- *Model load time:* 1.04s
- *Generation time:* 0.49s
- *Total time:* 1.65s
- *Prompt preparation time:* 0.0165
- *First-token latency:* 0.147
- *Cleanup time:* 0.0902
- *Prompt tokens:* 418
- *Generation tokens:* 78
- *Total tokens:* 496
- *Prompt throughput (raw):* 2,850 tok/s
- *Generation throughput (raw):* 246 tok/s
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
- *Model load time:* 0.94s
- *Generation time:* 1.86s
- *Total time:* 2.92s
- *Prompt preparation time:* 0.0172
- *First-token latency:* 0.233
- *Cleanup time:* 0.0921
- *Prompt tokens:* 883
- *Generation tokens:* 94
- *Total tokens:* 977
- *Prompt throughput (raw):* 3,794 tok/s
- *Generation throughput (raw):* 59.8 tok/s
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
- *Model load time:* 2.38s
- *Generation time:* 1.20s
- *Total time:* 3.76s
- *Prompt preparation time:* 0.0248
- *First-token latency:* 0.335
- *Cleanup time:* 0.137
- *Prompt tokens:* 409
- *Generation tokens:* 81
- *Total tokens:* 490
- *Prompt throughput (raw):* 1,220 tok/s
- *Generation throughput (raw):* 94.6 tok/s
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
- *Model load time:* 1.98s
- *Generation time:* 0.97s
- *Total time:* 3.11s
- *Prompt preparation time:* 0.0195
- *First-token latency:* 0.313
- *Cleanup time:* 0.13
- *Prompt tokens:* 409
- *Generation tokens:* 83
- *Total tokens:* 492
- *Prompt throughput (raw):* 1,306 tok/s
- *Generation throughput (raw):* 129 tok/s
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
- *Model load time:* 0.87s
- *Generation time:* 0.83s
- *Total time:* 1.83s
- *Prompt preparation time:* 0.0102
- *First-token latency:* 0.2
- *Cleanup time:* 0.109
- *Prompt tokens:* 409
- *Generation tokens:* 60
- *Total tokens:* 469
- *Prompt throughput (raw):* 2,042 tok/s
- *Generation throughput (raw):* 103 tok/s
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
- *Model load time:* 2.74s
- *Generation time:* 1.11s
- *Total time:* 4.02s
- *Prompt preparation time:* 0.024
- *First-token latency:* 0.322
- *Cleanup time:* 0.132
- *Prompt tokens:* 421
- *Generation tokens:* 88
- *Total tokens:* 509
- *Prompt throughput (raw):* 1,309 tok/s
- *Generation throughput (raw):* 113 tok/s
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
- *Model load time:* 2.98s
- *Generation time:* 3.97s
- *Total time:* 7.10s
- *Prompt preparation time:* 0.0105
- *First-token latency:* 0.66
- *Cleanup time:* 0.126
- *Prompt tokens:* 421
- *Generation tokens:* 109
- *Total tokens:* 530
- *Prompt throughput (raw):* 638 tok/s
- *Generation throughput (raw):* 33.0 tok/s
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
- *Resolved model revision:* 5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/5ff8ef173ad0d7c3aae92f0be43031a6ab8067c6
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
- *Model load time:* 0.63s
- *Generation time:* 0.66s
- *Total time:* 1.44s
- *Prompt preparation time:* 0.0582
- *First-token latency:* 0.357
- *Cleanup time:* 0.0908
- *Prompt tokens:* 1,204
- *Generation tokens:* 31
- *Total tokens:* 1,235
- *Prompt throughput (raw):* 3,370 tok/s
- *Generation throughput (raw):* 125 tok/s
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
- *Model load time:* 1.10s
- *Generation time:* 2.24s
- *Total time:* 3.98s
- *Prompt preparation time:* 0.534
- *First-token latency:* 0.94
- *Cleanup time:* 0.102
- *Prompt tokens:* 2,909
- *Generation tokens:* 82
- *Total tokens:* 2,991
- *Prompt throughput (raw):* 3,095 tok/s
- *Generation throughput (raw):* 100 tok/s
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
- *Model load time:* 3.25s
- *Generation time:* 1.45s
- *Total time:* 4.89s
- *Prompt preparation time:* 0.0284
- *First-token latency:* 0.236
- *Cleanup time:* 0.128
- *Prompt tokens:* 386
- *Generation tokens:* 66
- *Total tokens:* 452
- *Prompt throughput (raw):* 1,639 tok/s
- *Generation throughput (raw):* 66.9 tok/s
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
- *Model load time:* 1.70s
- *Generation time:* 1.67s
- *Total time:* 3.55s
- *Prompt preparation time:* 0.0296
- *First-token latency:* 0.251
- *Cleanup time:* 0.116
- *Prompt tokens:* 390
- *Generation tokens:* 88
- *Total tokens:* 478
- *Prompt throughput (raw):* 1,551 tok/s
- *Generation throughput (raw):* 62.8 tok/s
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
- *Model load time:* 2.33s
- *Generation time:* 0.92s
- *Total time:* 3.42s
- *Prompt preparation time:* 0.0098
- *First-token latency:* 0.312
- *Cleanup time:* 0.123
- *Prompt tokens:* 390
- *Generation tokens:* 75
- *Total tokens:* 465
- *Prompt throughput (raw):* 1,248 tok/s
- *Generation throughput (raw):* 126 tok/s
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
- *Model load time:* 1.54s
- *Generation time:* 0.71s
- *Total time:* 2.39s
- *Prompt preparation time:* 0.00937
- *First-token latency:* 0.157
- *Cleanup time:* 0.104
- *Prompt tokens:* 386
- *Generation tokens:* 67
- *Total tokens:* 453
- *Prompt throughput (raw):* 2,453 tok/s
- *Generation throughput (raw):* 125 tok/s
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
- *Model load time:* 0.61s
- *Generation time:* 0.60s
- *Total time:* 1.35s
- *Prompt preparation time:* 0.0305
- *First-token latency:* 0.257
- *Cleanup time:* 0.0929
- *Prompt tokens:* 734
- *Generation tokens:* 57
- *Total tokens:* 791
- *Prompt throughput (raw):* 2,861 tok/s
- *Generation throughput (raw):* 181 tok/s
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
- *Model load time:* 0.84s
- *Generation time:* 0.75s
- *Total time:* 1.70s
- *Prompt preparation time:* 0.012
- *First-token latency:* 0.125
- *Cleanup time:* 0.0982
- *Prompt tokens:* 420
- *Generation tokens:* 88
- *Total tokens:* 508
- *Prompt throughput (raw):* 3,360 tok/s
- *Generation throughput (raw):* 150 tok/s
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
- *Model load time:* 1.66s
- *Generation time:* 3.06s
- *Total time:* 4.86s
- *Prompt preparation time:* 0.0118
- *First-token latency:* 0.643
- *Cleanup time:* 0.119
- *Prompt tokens:* 538
- *Generation tokens:* 87
- *Total tokens:* 625
- *Prompt throughput (raw):* 837 tok/s
- *Generation throughput (raw):* 36.3 tok/s
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
- *Model load time:* 2.14s
- *Generation time:* 0.75s
- *Total time:* 3.05s
- *Prompt preparation time:* 0.0234
- *First-token latency:* 0.326
- *Cleanup time:* 0.113
- *Prompt tokens:* 421
- *Generation tokens:* 65
- *Total tokens:* 486
- *Prompt throughput (raw):* 1,290 tok/s
- *Generation throughput (raw):* 159 tok/s
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
- *Model load time:* 14.56s
- *Generation time:* 19.41s
- *Total time:* 34.31s
- *Prompt preparation time:* 0.0311
- *First-token latency:* 10.9
- *Cleanup time:* 0.267
- *Prompt tokens:* 420
- *Generation tokens:* 89
- *Total tokens:* 509
- *Prompt throughput (raw):* 38.7 tok/s
- *Generation throughput (raw):* 30.9 tok/s
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
- *Resolved model revision:* da5b2650a835292c678994a1fa3fb653250fb508
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/da5b2650a835292c678994a1fa3fb653250fb508
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
- *Model load time:* 1.23s
- *Generation time:* 1.36s
- *Total time:* 2.72s
- *Prompt preparation time:* 0.00905
- *First-token latency:* 0.208
- *Cleanup time:* 0.104
- *Prompt tokens:* 421
- *Generation tokens:* 111
- *Total tokens:* 532
- *Prompt throughput (raw):* 2,024 tok/s
- *Generation throughput (raw):* 101 tok/s
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
- *Model load time:* 1.71s
- *Generation time:* 4.31s
- *Total time:* 6.16s
- *Prompt preparation time:* 0.0184
- *First-token latency:* 0.322
- *Cleanup time:* 0.105
- *Prompt tokens:* 545
- *Generation tokens:* 540
- *Total tokens:* 1,085
- *Prompt throughput (raw):* 1,692 tok/s
- *Generation throughput (raw):* 136 tok/s
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
- *Model load time:* 0.64s
- *Generation time:* 0.47s
- *Total time:* 1.22s
- *Prompt preparation time:* 0.0237
- *First-token latency:* 0.357
- *Cleanup time:* 0.0805
- *Prompt tokens:* 120
- *Generation tokens:* 32
- *Total tokens:* 152
- *Prompt throughput (raw):* 336 tok/s
- *Generation throughput (raw):* 371 tok/s
- *Peak memory (GB):* 2.2
- *Active memory (GB):* 1.2
- *Cache memory (GB):* 0.03
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
- *Model load time:* 0.54s
- *Generation time:* 3.24s
- *Total time:* 3.90s
- *Prompt preparation time:* 0.0267
- *First-token latency:* 0.335
- *Cleanup time:* 0.0945
- *Prompt tokens:* 3,438
- *Generation tokens:* 1,000
- *Total tokens:* 4,438
- *Prompt throughput (raw):* 10,267 tok/s
- *Generation throughput (raw):* 349 tok/s
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

<a id="model-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

<details>
<summary>Complete evidence: mlx-community/Muse-Glimmer-30B-OptiQ-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* Unrecognised model control tokens remain visible; Required
  labelled fields not detected: title, description; Response appears cut off
  at the token limit; Conversation-role control tokens remain visible
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Model load time:* 2.65s
- *Generation time:* 40.53s
- *Total time:* 43.35s
- *Prompt preparation time:* 0.0291
- *First-token latency:* 0.859
- *Cleanup time:* 0.122
- *Prompt tokens:* 545
- *Generation tokens:* 1,000
- *Total tokens:* 1,545
- *Prompt throughput (raw):* 634 tok/s
- *Generation throughput (raw):* 25.2 tok/s
- *Peak memory (GB):* 25
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.09
- *Model-load active memory (GB):* 23.9
- *Post-cleanup active memory (GB):* 0.00703
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* max_tokens
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
 to=self&lt;|message|&gt;Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need British-English. Use only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

We see two cats? Possibly tabby cats. Species? Cat is visible. Is species certain? Probably cats. It's clear visible facts. Might be cats. Species is cat. Might be okay. But "species" maybe uncertain? It's clearly cats. Should be okay.

Don't specify breed. Don't say brand of remote. Don't say location. Don't say period.

Title: 5-10 words concrete.

Maybe "Two Tabby Cats Resting on Pink Blanket". That's 7 words? Two(1) Tabby(2) Cats(3) Resting(4) on(5) Pink(6) Blanket(7). Good.

Description: 1-2 sentence factual description of main subject, setting, action, lighting, and distinctive details.

Main subject: two cats on pink blanket on red sofa? There's pink blanket on red couch? Actually pink blanket on red cushions. Two cats lying. One sleeping with head down, one looking down. Two remote controls. Lighting? Bright daylight? Seems indoor natural light. Distinctive details: one cat has green collar? Looks like green collar on left cat. Remote controls.

We must omit uncertain details. Lighting: maybe bright indoor. Can say bright lighting. Okay.

Don't assume species? It's cats. Might be safe.

Make description: Two tabby cats lie on a bright pink blanket on a red sofa, one appears asleep and one is looking down. Two white remote controls rest on the blanket beside them. The scene is lit brightly.

That's two sentences? "Two tabby cats lie..." That's one sentence. Second sentence: "Two white remote controls rest..." That's second. Okay.

Keywords: 10-18 unique comma-separated keywords.

Keywords: cats, tabby, pink blanket, red sofa, remote controls, sleeping, resting, indoor, domestic, furniture, textile, pet, animals

Need 10-18 unique. Let's list maybe 13.

Make sure British English: "colour"? Not used. "favourite"? No.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

Make sure format exactly.

Title: ...
Description: ...
Keywords: ...

No extra text.

Ensure title 5-10 words. Let's count.

Two Tabby Cats Resting on Pink Blanket = 7 words. Good.

Description: Must be 1-2 sentence. Ensure factual. Don't say "appears asleep" maybe uncertain? Could be sleeping. Head down eyes closed? Hard to see. Might be sleeping. Could say one cat lies with head down, one cat is oriented head down. Safer to say one cat lies with head down, eyes closed? Might be uncertain. Better say one cat lies with head down. Could mention one cat with green collar. Is collar visible? Yes green band around neck. Distinctive detail.

Maybe: Two tabby cats lie on a bright pink blanket atop a red upholstered surface. One cat rests with its head down and a green collar visible, the other lies with its head lowered towards the blanket, and two white remote controls are placed on the fabric. The lighting is bright.

That's a bit long, maybe two sentences. Let's keep two sentences.

First sentence: Two tabby cats lie on a bright pink blanket on a red upholstered surface. Second sentence: One cat rests with its head down and a green collar visible, the other is positioned head-down towards the blanket, and two white remote controls lie on the fabric. Lighting is bright.

That is maybe too many clauses. Might be okay.

Simplify: Two tabby cats lie on a bright pink blanket on a red upholstered surface. One cat rests with its head down and a green collar visible, the other is positioned head-down, with two white remote controls on the blanket. The lighting is bright.

That's three sentences? Let's count. Sentence 1: Two tabby cats lie on a bright pink blanket on a red upholstered surface. Sentence 2: One cat rests with its head down and a green collar visible, the other is positioned head-down, with two white remote controls on the blanket. Sentence 3: The lighting is bright. That's three.

Need 1-2 sentence. Let's combine.

Two tabby cats lie
</pre>

<details>
<summary>Exact raw output</summary>

```text
 to=self<|message|>Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need British-English. Use only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

We see two cats? Possibly tabby cats. Species? Cat is visible. Is species certain? Probably cats. It's clear visible facts. Might be cats. Species is cat. Might be okay. But "species" maybe uncertain? It's clearly cats. Should be okay.

Don't specify breed. Don't say brand of remote. Don't say location. Don't say period.

Title: 5-10 words concrete.

Maybe "Two Tabby Cats Resting on Pink Blanket". That's 7 words? Two(1) Tabby(2) Cats(3) Resting(4) on(5) Pink(6) Blanket(7). Good.

Description: 1-2 sentence factual description of main subject, setting, action, lighting, and distinctive details.

Main subject: two cats on pink blanket on red sofa? There's pink blanket on red couch? Actually pink blanket on red cushions. Two cats lying. One sleeping with head down, one looking down. Two remote controls. Lighting? Bright daylight? Seems indoor natural light. Distinctive details: one cat has green collar? Looks like green collar on left cat. Remote controls.

We must omit uncertain details. Lighting: maybe bright indoor. Can say bright lighting. Okay.

Don't assume species? It's cats. Might be safe.

Make description: Two tabby cats lie on a bright pink blanket on a red sofa, one appears asleep and one is looking down. Two white remote controls rest on the blanket beside them. The scene is lit brightly.

That's two sentences? "Two tabby cats lie..." That's one sentence. Second sentence: "Two white remote controls rest..." That's second. Okay.

Keywords: 10-18 unique comma-separated keywords.

Keywords: cats, tabby, pink blanket, red sofa, remote controls, sleeping, resting, indoor, domestic, furniture, textile, pet, animals

Need 10-18 unique. Let's list maybe 13.

Make sure British English: "colour"? Not used. "favourite"? No.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

Make sure format exactly.

Title: ...
Description: ...
Keywords: ...

No extra text.

Ensure title 5-10 words. Let's count.

Two Tabby Cats Resting on Pink Blanket = 7 words. Good.

Description: Must be 1-2 sentence. Ensure factual. Don't say "appears asleep" maybe uncertain? Could be sleeping. Head down eyes closed? Hard to see. Might be sleeping. Could say one cat lies with head down, one cat is oriented head down. Safer to say one cat lies with head down, eyes closed? Might be uncertain. Better say one cat lies with head down. Could mention one cat with green collar. Is collar visible? Yes green band around neck. Distinctive detail.

Maybe: Two tabby cats lie on a bright pink blanket atop a red upholstered surface. One cat rests with its head down and a green collar visible, the other lies with its head lowered towards the blanket, and two white remote controls are placed on the fabric. The lighting is bright.

That's a bit long, maybe two sentences. Let's keep two sentences.

First sentence: Two tabby cats lie on a bright pink blanket on a red upholstered surface. Second sentence: One cat rests with its head down and a green collar visible, the other is positioned head-down towards the blanket, and two white remote controls lie on the fabric. Lighting is bright.

That is maybe too many clauses. Might be okay.

Simplify: Two tabby cats lie on a bright pink blanket on a red upholstered surface. One cat rests with its head down and a green collar visible, the other is positioned head-down, with two white remote controls on the blanket. The lighting is bright.

That's three sentences? Let's count. Sentence 1: Two tabby cats lie on a bright pink blanket on a red upholstered surface. Sentence 2: One cat rests with its head down and a green collar visible, the other is positioned head-down, with two white remote controls on the blanket. Sentence 3: The lighting is bright. That's three.

Need 1-2 sentence. Let's combine.

Two tabby cats lie
```

</details>

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
- *Model load time:* 2.52s
- *Generation time:* 1.16s
- *Total time:* 3.84s
- *Prompt preparation time:* 0.0239
- *First-token latency:* 0.317
- *Cleanup time:* 0.129
- *Prompt tokens:* 421
- *Generation tokens:* 96
- *Total tokens:* 517
- *Prompt throughput (raw):* 1,328 tok/s
- *Generation throughput (raw):* 116 tok/s
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
- *Generation time:* 1.39s
- *Total time:* 2.58s
- *Prompt preparation time:* 0.0367
- *First-token latency:* 1.23
- *Cleanup time:* 0.154
- *Prompt tokens:* 3,512
- *Generation tokens:* 14
- *Total tokens:* 3,526
- *Prompt throughput (raw):* 2,864 tok/s
- *Generation throughput (raw):* 105 tok/s
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
- *Model load time:* 0.50s
- *Generation time:* 0.35s
- *Total time:* 0.96s
- *Prompt preparation time:* 0.00947
- *First-token latency:* 0.0789
- *Cleanup time:* 0.0975
- *Prompt tokens:* 116
- *Generation tokens:* 97
- *Total tokens:* 213
- *Prompt throughput (raw):* 1,470 tok/s
- *Generation throughput (raw):* 373 tok/s
- *Peak memory (GB):* 1.5
- *Active memory (GB):* 0.61
- *Cache memory (GB):* 0.25
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
