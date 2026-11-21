# Model Output Gallery

Generated on: 2026-11-21 23:48:03 GMT

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

| Model                                                                                                                               | Mechanical checks               | Total s | Gen TPS             | Prefill/first s | Peak GB | Prompt tok | Gen tok | Keywords | Observations                                              |
|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|---------|---------------------|-----------------|---------|------------|---------|----------|-----------------------------------------------------------|
| [`ibm-granite/granite-vision-4.1-4b`](#model-ibm-granite-granite-vision-41-4b)                                                      | `no concerns detected`          | 2.62s   | 61.5 tok/s          | 0.57            | 9.7     | 734        | 56      | 10       | none                                                      |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected`          | 1.29s   | 570 tok/s           | 0.80            | 1.3     | 350        | 47      | 6        | none                                                      |
| [`mlx-community/AREX-2-4bit`](#model-mlx-community-arex-2-4bit)                                                                     | `no concerns detected`          | 31.34s  | 31.8 tok/s          | 0.90            | 17      | 461        | 865     | 16       | none                                                      |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected`          | 4.15s   | 102 tok/s           | 1.55            | 6.5     | 2,909      | 82      | 10       | none                                                      |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected`          | 5.29s   | 66.1 tok/s          | 1.87            | 28      | 386        | 66      | 14       | none                                                      |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `no concerns detected`          | 3.53s   | 63.1 tok/s          | 0.27            | 7.5     | 390        | 88      | 18       | none                                                      |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected`          | 3.44s   | 127 tok/s           | 0.34            | 16      | 390        | 75      | 15       | none                                                      |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected`          | 2.43s   | 124 tok/s           | 0.18            | 5.9     | 386        | 67      | 14       | none                                                      |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected`          | 11.71s  | 81.9 tok/s          | 0.46            | 8.0     | 495        | 819     | 12       | none                                                      |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected`          | 1.39s   | 178 tok/s           | 0.29            | 4.7     | 734        | 57      | 10       | none                                                      |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected`          | 5.25s   | 55.6 tok/s          | 2.31            | 10      | 3,435      | 90      | 18       | none                                                      |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected`          | 1.47s   | 217 tok/s           | 0.24            | 3.6     | 344        | 73      | 10       | none                                                      |
| [`mlx-community/MiniCPM-V-4.7-1B-4bit`](#model-mlx-community-minicpm-v-47-1b-4bit)                                                  | `no concerns detected`          | 2.08s   | 313 tok/s           | 0.14            | 3.5     | 330        | 268     | 15       | none                                                      |
| [`mlx-community/MiniCPM-V-4.7-35B-A3B-4bit`](#model-mlx-community-minicpm-v-47-35b-a3b-4bit)                                        | `no concerns detected`          | 8.73s   | 87.4 tok/s          | 0.58            | 24      | 330        | 431     | 18       | none                                                      |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected`          | 3.30s   | 67.5 tok/s          | 0.73            | 8.9     | 1,070      | 81      | 10       | none                                                      |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected`          | 1.65s   | 209 tok/s           | 0.24            | 3.7     | 1,069      | 102     | 12       | none                                                      |
| [`mlx-community/Mistral-Small-4-119B-2603-4bit`](#model-mlx-community-mistral-small-4-119b-2603-4bit)                               | `no concerns detected`          | 20.11s  | 91.1 tok/s          | 6.77            | 69      | 549        | 82      | 18       | none                                                      |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `no concerns detected`          | 3.00s   | 71.1 tok/s          | 0.51            | 8.2     | 873        | 95      | 10       | none                                                      |
| [`mlx-community/MolmoPoint-8B-4bit`](#model-mlx-community-molmopoint-8b-4bit)                                                       | `no concerns detected`          | 4.57s   | 30.3 tok/s          | 0.81            | 12      | 860        | 70      | 10       | none                                                      |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `no concerns detected`          | 39.24s  | 24.9 tok/s          | 0.90            | 25      | 545        | 884     | 14       | none                                                      |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected`          | 1.65s   | 267 tok/s           | 0.18            | 3.3     | 418        | 78      | 12       | none                                                      |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected`          | 2.85s   | 58.1 tok/s          | 0.29            | 9.1     | 883        | 94      | 19       | none                                                      |
| [`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit)                           | `no concerns detected`          | 3.73s   | 93.6 tok/s          | 0.38            | 23      | 409        | 81      | 18       | none                                                      |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected`          | 3.01s   | 133 tok/s           | 0.34            | 19      | 409        | 83      | 10       | none                                                      |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected`          | 1.87s   | 103 tok/s           | 0.25            | 6.8     | 409        | 60      | 10       | none                                                      |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected`          | 10.24s  | 123 tok/s           | 0.34            | 21      | 419        | 898     | 18       | none                                                      |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)                                                    | `no concerns detected`          | 8.39s   | 121 tok/s           | 0.37            | 21      | 419        | 653     | 19       | none                                                      |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected`          | 23.52s  | 30.9 tok/s          | 0.72            | 17      | 461        | 640     | 14       | none                                                      |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected`          | 1.51s   | 117 tok/s           | 0.45            | 5.6     | 1,204      | 31      | 4        | none                                                      |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected`          | 1.73s   | 149 tok/s           | 0.17            | 5.2     | 420        | 88      | 10       | none                                                      |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected`          | 4.89s   | 35.9 tok/s          | 0.67            | 16      | 538        | 87      | 18       | none                                                      |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected`          | 7.01s   | 162 tok/s           | 0.35            | 21      | 421        | 706     | 17       | none                                                      |
| [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](#model-nvidia-nvidia-nemotron-35-super-vl-120b-a12b-nvfp4)                  | `no concerns detected`          | 55.31s  | 30.3 tok/s          | 11.70           | 81      | 420        | 797     | 15       | none                                                      |
| [`sahilchachra/LensVLM-9B-MXFP4`](#model-sahilchachra-lensvlm-9b-mxfp4)                                                             | `no concerns detected`          | 3.61s   | 97.4 tok/s          | 0.25            | 6.7     | 419        | 198     | 11       | none                                                      |
| [`TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`](#model-technobaptist-ternary-bonsai-2-27b-mlx-2bit)                                 | `no concerns detected`          | 15.17s  | 43.2 tok/s          | 1.67            | 11      | 461        | 517     | 14       | none                                                      |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `concerns detected`             | 6.00s   | 145 tok/s           | 0.52            | 19      | 545        | 540     | 16       | duplicate keywords                                        |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `concerns detected`             | 5.01s   | 102 tok/s           | 0.19            | 7.0     | 173        | 367     | 18       | duplicate keywords                                        |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns: answer format` | 1.18s   | 395 tok/s           | 0.32            | 2.1     | 120        | 32      | -        | labelled fields not detected                              |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `major concerns: generation`    | 3.73s   | 379 tok/s           | 0.38            | 2.3     | 3,438      | 1,000   | 144      | repeated text; cut off at token limit; duplicate keywords |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns: answer format` | 2.69s   | insufficient sample | 1.41            | 7.0     | 3,512      | 14      | -        | control tokens visible; labelled fields not detected      |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns: answer format` | 1.02s   | 353 tok/s           | 0.10            | 1.5     | 116        | 97      | -        | labelled fields not detected                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Resource Highlights

Quickest completion without detected concerns (end-to-end, including model load): `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.29s

Lowest peak memory among completions without detected concerns: `LiquidAI/LFM2.5-VL-450M-MLX-bf16` at 1.3 GB

Decode tok/s stays per model in the chooser and is not averaged across models: tokenizers, image-token expansion and reasoning lengths differ too much for a cross-model mean to guide a choice.

## Major Mechanical Concerns in This Run

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                   | Mechanical checks               | Observations                                              |
|-----------------------------------------------------------------------------------------|---------------------------------|-----------------------------------------------------------|
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)              | `major concerns: answer format` | labelled fields not detected                              |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)          | `major concerns: generation`    | repeated text; cut off at token limit; duplicate keywords |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit) | `major concerns: answer format` | control tokens visible; labelled fields not detected      |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)            | `major concerns: answer format` | labelled fields not detected                              |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Output at a Glance

A compact preview of each model's final answer (or failure evidence for crashes), in chooser order. Where the requested catalogue fields were detected, the preview shows a little of each: the title, the start of the description, and the first keywords with their count, so the weakest field is not hidden behind a long description. Otherwise it is the first 280 characters. A closed reasoning trace is left out of the preview and reported as an omitted-character count; the complete output, trace included, is in the model's evidence block below.

<!-- markdownlint-disable MD034 MD037 MD049 -->

| Model                                                                                                                               | Mechanical checks      | Output preview                                                                                                                                                                                                                                                                                                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`ibm-granite/granite-vision-4.1-4b`](#model-ibm-granite-granite-vision-41-4b)                                                      | `no concerns detected` | Title: "Cats at Rest on a Pink Sofa" \| Description: Two cats are sleeping on a pink sofa, with remote controls nearby. \| Keywords (10): cats, sleeping, pink sofa, remote controls, domestic, comfort, tranquility, home, leisure, relaxation.                                                                                                                                          |
| [`LiquidAI/LFM2.5-VL-450M-MLX-bf16`](#model-liquidai-lfm25-vl-450m-mlx-bf16)                                                        | `no concerns detected` | Title: Cats on a Pink Couch \| Description: Two tabby cats are lying on a bright pink couch, appearing relaxed and comfortable. \| Keywords (6): cats, pink couch, relaxation, tabby, comfortable, cozy                                                                                                                                                                                   |
| [`mlx-community/AREX-2-4bit`](#model-mlx-community-arex-2-4bit)                                                                     | `no concerns detected` | Title: Two tabby cats lying on pink fabric beside remotes \| Description: Two striped tabby cats lie stretched out on a bright pink fabric surface, with a darker ma... \| Keywords (16): tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, ...[2,838 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/aya-vision-8b-4bit`](#model-mlx-community-aya-vision-8b-4bit)                                                       | `no concerns detected` | Title: Two Cats Relaxing Together \| Description: Two tabby cats are curled up and sleeping side by side on a pink blanket, with a remote control placed nearby on the sofa. The sce... \| Keywords (10): cats, sleeping, pink blanket, sofa, remote control, tabby, relaxed, home, indoor, comfortable                                                                                   |
| [`mlx-community/diffusiongemma-26B-A4B-it-mxfp8`](#model-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                             | `no concerns detected` | Title: Two tabby cats sleeping on a pink blanket \| Description: Two striped tabby cats lie resting on a pink fabric surface next to two remote controls. \| Keywords (14): cats, tabby, sleeping, pink, blanket, sofa, indoor, remote controls, lounging, resting, striped, fur, fabric, ...                                                                                             |
| [`mlx-community/gemma-4-12B-it-4bit`](#model-mlx-community-gemma-4-12b-it-4bit)                                                     | `no concerns detected` | Title: Two Tabby Cats on a Pink Sofa \| Description: Two tabby cats lie on a pink fabric sofa, with one resting horizontally and the other positioned verticall... \| Keywords (18): cats, tabby, sofa, couch, pet, sleeping, reclining, domestic animals, pink fabric, remote controls, furniture, ...                                                                                   |
| [`mlx-community/gemma-4-26b-a4b-it-4bit`](#model-mlx-community-gemma-4-26b-a4b-it-4bit)                                             | `no concerns detected` | Title: Two tabby cats sleeping on a pink surface \| Description: Two tabby cats lie on a pink fabric surface next to two remote controls. The cats are positioned... \| Keywords (15): tabby, cat, sleeping, pink, fabric, remote, electronics, domestic, animal, lying, indoor, bright, two, cats, ...                                                                                   |
| [`mlx-community/gemma-4-e4b-it-4bit`](#model-mlx-community-gemma-4-e4b-it-4bit)                                                     | `no concerns detected` | Title: Two cats rest upon bright pink fabric \| Description: Two cats lie on a vividly coloured pink surface under soft lighting. They are accompanied by a remote... \| Keywords (14): cats, tabby, pile, fabric, pink, sleeping, reptiles, appendages, lounged, indoors, tortoiseshell, domestic, ...                                                                                   |
| [`mlx-community/GLM-4.6V-Flash-4bit`](#model-mlx-community-glm-46v-flash-4bit)                                                      | `no concerns detected` | Title: Two Tabby Cats on Pink Couch with Remotes \| Description: Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other... \| Keywords (12): tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, ...[2,997 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/granite-4.0-3b-vision-4bit`](#model-mlx-community-granite-40-3b-vision-4bit)                                        | `no concerns detected` | Title: "Feline Serenity on a Red Couch" \| Description: Two striped cats lounging on a red blanket, one with a remote control nearby. \| Keywords (10): cats, striped, red, blanket, lounging, remote control, comfort, rest, domestic, leisure                                                                                                                                           |
| [`mlx-community/InternVL3-14B-4bit`](#model-mlx-community-internvl3-14b-4bit)                                                       | `no concerns detected` | Title: Cats Sleeping on Sofa with Remote Controls \| Description: Two cats are sleeping on a pink sofa, each with a remote control nearby. The lighting is soft,... \| Keywords (18): cats, sleeping, sofa, pink, remote controls, striped, relaxed, home, pets, cozy, domestic, resting, comfort, ...                                                                                    |
| [`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`](#model-mlx-community-lfm25-vl-3b-optiq-4bit)                                              | `no concerns detected` | Title: Two cats sleeping on a pink couch. \| Description: Two tabby cats are lying on a pink couch with remote controls nearby. The lighting is bright, highlighting th... \| Keywords (10): cats, pink couch, tabby, sleeping, remote controls, bright lighting, fur patterns, relaxation, indoor, ...                                                                                   |
| [`mlx-community/MiniCPM-V-4.7-1B-4bit`](#model-mlx-community-minicpm-v-47-1b-4bit)                                                  | `no concerns detected` | Title: Two tabby cats sleep on pink couch \| Description: Two tabby cats rest on a pink fabric couch, one with a remote and green ear tag, both in indoor lighting with stripe... \| Keywords (15): tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy[705 characters of reasoning omitted; complete output in the evidence block]       |
| [`mlx-community/MiniCPM-V-4.7-35B-A3B-4bit`](#model-mlx-community-minicpm-v-47-35b-a3b-4bit)                                        | `no concerns detected` | Title: Two Tabby Cats Resting on Pink Sofa \| Description: Two tabby cats lie on a bright pink fabric surface, each positioned near a white remote control. The... \| Keywords (18): tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, ...[1,400 characters of reasoning omitted; complete output in the evidence block]      |
| [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)                     | `no concerns detected` | Title: Two Cats Relaxing on Pink Cushioned Surface \| Description: Two cats lie stretched out on a pink cushioned surface, positioned near remote controls. The... \| Keywords (10): cats, pink cushioned surface, remote controls, relaxed posture, indoor setting, natural lighting, striped fur, ...                                                                                   |
| [`mlx-community/Ministral-3-3B-Instruct-2512-4bit`](#model-mlx-community-ministral-3-3b-instruct-2512-4bit)                         | `no concerns detected` | Title: Two resting tabby cats on a pink sofa \| Description: Two adult tabby cats lie relaxed on a bright pink sofa cushion, each with remote controls beside them. Soft, na... \| Keywords (12): tabby-cats, pink-sofa, remote-controls, indoor-resting, striped-fur, cozy-setting, domestic-pets, ...                                                                                   |
| [`mlx-community/Mistral-Small-4-119B-2603-4bit`](#model-mlx-community-mistral-small-4-119b-2603-4bit)                               | `no concerns detected` | Title: Two cats resting on a pink sofa \| Description: Two cats are lying on a pink sofa with a remote control nearby. Soft lighting highlights their relaxed poses and... \| Keywords (18): cats, sofa, pink, remote control, resting, indoor, pet, cushion, domestic, relaxation, soft lighting, ...                                                                                    |
| [`mlx-community/Molmo2-8B-4bit`](#model-mlx-community-molmo2-8b-4bit)                                                               | `no concerns detected` | Title: Two Calico Cats Sleeping on Pink Bedding \| Description: Two calico cats are sleeping side by side on pink bedding. They are positioned with their heads facin... \| Keywords (10): calico cats, sleeping, pink bedding, white remotes, black and brown fur, white cats, striped cats, cozy, ...                                                                                   |
| [`mlx-community/MolmoPoint-8B-4bit`](#model-mlx-community-molmopoint-8b-4bit)                                                       | `no concerns detected` | Title: Two Cats Sleeping on Pink Blanket with Remote Controls \| Description: Two cats are sleeping on a pink blanket. A white remote control is visible behind them. Th... \| Keywords (10): cats, pink blanket, remote controls, sleeping, white remote, comfortable, soft surface, feline, rest, ...                                                                                   |
| [`mlx-community/Muse-Glimmer-30B-OptiQ-4bit`](#model-mlx-community-muse-glimmer-30b-optiq-4bit)                                     | `no concerns detected` | Title: Two cats resting on pink blanket sofa \| Description: Two cats lie on a bright pink blanket on a red sofa, one with its head resting down and eyes closed, the... \| Keywords (14): cats, pink blanket, red sofa, remote controls, tabby markings, resting, sleeping, fabric, couch, indoor, ...                                                                                   |
| [`mlx-community/North-Micro-Vision-Instruct-4bit`](#model-mlx-community-north-micro-vision-instruct-4bit)                           | `no concerns detected` | Title: "Cats Resting on a Pink Couch" \| Description: "Two cats are comfortably sprawled out on a vibrant pink couch, basking in the soft light of the day. Their relaxed postu... \| Keywords (12): cats, couch, resting, lounging, pink, soft light, relaxed, tranquil, basking, cozy, serene, scene.                                                                                   |
| [`mlx-community/Phi-3.5-vision-instruct-bf16`](#model-mlx-community-phi-35-vision-instruct-bf16)                                    | `no concerns detected` | Title: Two Cats Sleeping on Pink Sofa \| Description: Two cats are sleeping on a pink sofa with remote controls beside them. \| Keywords (19): cats, sleeping, pink sofa, remote controls, relaxed, comfortable, indoor, domestic, fur, striped, resting, ...                                                                                                                             |
| [`mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-omni-30b-a3b-instruct-4bit)                           | `no concerns detected` | Title: Two tabby cats sleeping on sofa \| Description: Two tabby cats are sleeping on a bright pink sofa. The sofa has two remote controls placed near the cats. \| Keywords (18): tabby cats, sleeping, sofa, pink, remote controls, domestic cats, resting, indoor, animal, pet, living room, ...                                                                                       |
| [`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](#model-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                               | `no concerns detected` | Title: Two cats resting on a pink surface \| Description: Two cats are lying on a bright pink fabric surface. One cat is curled up with its eyes closed, while... \| Keywords (10): cats, remote controls, pink fabric, lying down, curled up, eyes closed, striped pattern, light, indoor setting, ...                                                                                   |
| [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](#model-mlx-community-qwen3-vl-8b-instruct-4bit)                                         | `no concerns detected` | Title: Two Cats Sleeping on Pink Sofa \| Description: Two tabby cats rest peacefully on a bright pink sofa. Remote controls lie beside them. Natural daylight illuminates the scene. \| Keywords (10): cats, sleeping, pink, sofa, remote, tabby, rest, daylight, couch, feline                                                                                                           |
| [`mlx-community/Qwen3.5-35B-A3B-4bit`](#model-mlx-community-qwen35-35b-a3b-4bit)                                                    | `no concerns detected` | Title: Two tabby cats sleeping on pink sofa next to remotes \| Description: Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote co... \| Keywords (18): cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, ...[3,062 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/Qwen3.6-35B-A3B-4bit`](#model-mlx-community-qwen36-35b-a3b-4bit)                                                    | `no concerns detected` | Title: Two tabby cats sleeping on pink sofa with remotes. \| Description: Two tabby cats are resting on a bright pink sofa cushion, flanked by two grey remote co... \| Keywords (19): tabby, cats, pink, sofa, remote, controls, sleeping, fur, stripes, paws, cushions, furniture, pets, animals, ...[1,952 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/Qwen3.8-27B-nvfp4`](#model-mlx-community-qwen38-27b-nvfp4)                                                          | `no concerns detected` | Title: Two tabby cats resting beside remote controls on pink sofa \| Description: Two striped tabby cats lie asleep on a vivid pink sofa, one wearing a green collar tag,... \| Keywords (14): tabby cats, striped coats, pink sofa, magenta fabric, remote controls, white remotes, sleeping cats, ...[2,135 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/SmolVLM2-2.2B-Instruct-mlx`](#model-mlx-community-smolvlm2-22b-instruct-mlx)                                        | `no concerns detected` | Title: Two Cats on a Couch \| Description: Two cats are sleeping on a pink couch. \| Keywords (4): cats, couch, sleeping, pink                                                                                                                                                                                                                                                            |
| [`nativ-community/Mage-VL-OptiQ-4bit`](#model-nativ-community-mage-vl-optiq-4bit)                                                   | `no concerns detected` | Title: Two Cats Resting on a Pink Couch with Remotes \| Description: Two tabby cats are lying on a pink couch, with a remote control placed between them. The cats appear to be... \| Keywords (10): cats, resting, pink couch, remote control, tabby, sleeping, indoor, bright lighting, tail, visible                                                                                   |
| [`nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit`](#model-nativ-community-mistral-small-32-24b-instruct-2506-4bit)        | `no concerns detected` | Title: Two Cats Resting on a Pink Blanket \| Description: Two tabby cats are lying on a pink blanket, one stretched out and the other curled up. A remote contr... \| Keywords (18): cats, tabby, pink blanket, resting, remote control, fur, relaxed, indoor, couch, paws, stripes, sleeping, pet, ...                                                                                   |
| [`nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`](#model-nativ-community-nemotron-3-nano-omni-30b-a3b-reasoning-4bit) | `no concerns detected` | Title: Two tabby cats sleeping on pink cushion \| Description: Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby. \| Keywords (17): cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, ...[2,396 characters of reasoning omitted; complete output in the evidence block]      |
| [`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`](#model-nvidia-nvidia-nemotron-35-super-vl-120b-a12b-nvfp4)                  | `no concerns detected` | Title: Two tabby cats sleeping on pink sofa \| Description: Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a... \| Keywords (15): cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting[2,856 characters of reasoning omitted; complete output in the evidence block]     |
| [`sahilchachra/LensVLM-9B-MXFP4`](#model-sahilchachra-lensvlm-9b-mxfp4)                                                             | `no concerns detected` | Title: Two Cats Sleeping on Pink Blanket \| Description: Two tabby cats are lying asleep on a vibrant pink blanket, accompanied by two white remote controls. One cat is s... \| Keywords (11): cats, sleeping, pink blanket, remote controls, tabby, rest, indoor, fur, relaxation, domestic, &lt;/think&gt;[430 characters of reasoning omitted; complete output in the evidence block] |
| [`TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit`](#model-technobaptist-ternary-bonsai-2-27b-mlx-2bit)                                 | `no concerns detected` | Title: Two Tabby Cats Resting on Pink Fabric \| Description: Two tabby cats lie stretched out on a bright pink fabric surface with their eyes closed. Two white remote control... \| Keywords (14): tabby cat, two cats, pink fabric, remote control, sleeping, resting, striped fur, white remote, ...[1,713 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit`](#model-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit)                        | `concerns detected`    | Title: Two cats sleeping on a red sofa \| Description: A close-up image of two cats, one with calico fur and the other with tiger-striped fur, sleeping together on... \| Keywords (16): cat, cats, sleeping, sofa, red, pet, domestic, kitten, calico, tiger-striped, tuxedo, fur, lying, petting, ...[1,607 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/MiniCPM-o-4_5-4bit`](#model-mlx-community-minicpm-o-45-4bit)                                                        | `concerns detected`    | Title: Two tabby cats resting on pink fabric with remotes \| Description: Two tabby cats are lying on a pink fabric surface, near white remote controls. They ap... \| Keywords (18): cats, tabby, resting, pink, fabric, remote controls, indoor, lounging, pets, animals, sofa, relaxation, home, ...[1,186 characters of reasoning omitted; complete output in the evidence block]     |
| [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)                                                          | `major concerns`       | A serene moment of feline tranquility on a vibrant pink couch, with two cats in peaceful slumber, their fur patterns contrasting against the soft fabric.                                                                                                                                                                                                                                 |
| [`mlx-community/InternVL3_5-1B-4bit`](#model-mlx-community-internvl35-1b-4bit)                                                      | `major concerns`       | Title: Two cats sleeping on a pink couch \| Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a r... \| Keywords (144): cats, sleeping, pink couch, remote control, striped, British English, British, British-English, ...                                                                                   |
| [`mlx-community/llm-jp-4-vl-9b-mlx-4bit`](#model-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                             | `major concerns`       | <\|channel\|> analysis<\|message\|> The two cats are sleeping on a pink couch.                                                                                                                                                                                                                                                                                                            |
| [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)                                                        | `major concerns`       | Title: "A Study in Stripes" \| Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting... \| Keywords: (not detected)                                                                                    |
<!-- markdownlint-enable MD034 MD037 MD049 -->

## Run Stamps

- `mlx-vlm`: `0.8.0`
- `mlx`: `0.32.5.dev20261121+8803ac3ca`
- `transformers`: `5.24.0`
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
- *Model load time:* 0.99s
- *Generation time:* 1.49s
- *Total time:* 2.62s
- *Prompt preparation time:* 0.0356
- *First-token latency:* 0.55
- *Cleanup time:* 0.0897
- *Prompt tokens:* 734
- *Generation tokens:* 56
- *Total tokens:* 790
- *Prompt throughput (raw):* 1,335 tok/s
- *Generation throughput (raw):* 61.5 tok/s
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
- *Resolved model revision:* 12e928afb004ac65a44effbf50ca3a70375fa0bd
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--ibm-granite--granite-vision-4.1-4b/snapshots/12e928afb004ac65a44effbf50ca3a70375fa0bd
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.0, "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|end_of_text\|>

*Readable output:*

<pre class="model-output-readable">
Title: "Cats at Rest on a Pink Sofa"
Description: Two cats are sleeping on a pink sofa, with remote controls nearby.
Keywords: cats, sleeping, pink sofa, remote controls, domestic, comfort, tranquility, home, leisure, relaxation.
</pre>

</details>

---

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
- *Model load time:* 0.30s
- *Generation time:* 0.89s
- *Total time:* 1.29s
- *Prompt preparation time:* 0.0139
- *First-token latency:* 0.797
- *Cleanup time:* 0.066
- *Prompt tokens:* 350
- *Generation tokens:* 47
- *Total tokens:* 397
- *Prompt throughput (raw):* 439 tok/s
- *Generation throughput (raw):* 570 tok/s
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
- *Model load time:* 3.11s
- *Generation time:* 28.08s
- *Total time:* 31.34s
- *Prompt preparation time:* 0.00955
- *First-token latency:* 0.894
- *Cleanup time:* 0.128
- *Prompt tokens:* 461
- *Generation tokens:* 865
- *Total tokens:* 1,326
- *Prompt throughput (raw):* 516 tok/s
- *Generation throughput (raw):* 31.8 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.30
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.00185
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 789
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 02551ee54839c7541ea5adb44b83c9aa86aeb4f5
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--AREX-2-4bit/snapshots/02551ee54839c7541ea5adb44b83c9aa86aeb4f5
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 0.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title:
Two tabby cats lying on pink fabric beside remotes

Description:
Two striped tabby cats lie stretched out on a bright pink fabric surface, with a darker maroon backdrop behind them. Two white remote controls rest on the fabric near the cats, and the scene is lit by even indoor light.

Keywords:
tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, indoor lighting, pet, animal, cushion, whiskers, paws
```

*Readable output:*

<pre class="model-output-readable">
The user wants British-English catalogue metadata based strictly on visible facts. I need to look carefully at the image.

Visible facts: Two tabby cats lying on a bright pink/magenta fabric surface (looks like a sofa or cushion). There are two white remote controls visible — one near the left cat's head, one between the two cats. The cats are stretched out, appearing to sleep or rest. The left cat is smaller, stretched diagonally with paws extended. The right cat is larger, curled slightly with head down. The background top shows a darker red/maroon fabric (sofa back). Lighting appears even, indoor.

I must avoid unsupported identity, location, event, brand, species (well, "cat" is visible, that's fine — species is cat, that's clearly visible), period, intent. I should not say "sleeping" if uncertain? They appear to be resting with eyes closed. The left cat's eyes look closed; the right cat's eyes look closed. I can say "resting" or "lying". I'll say "lying" to be safe, maybe "appearing to rest".

Title: 5-10 words concrete. "Two tabby cats resting on pink sofa with remotes" — count words: Two(1) tabby(2) cats(3) resting(4) on(5) pink(6) sofa(7) with(8) remotes(9). 9 words. Good. But "sofa" — is it clearly a sofa? The top maroon part suggests a sofa back. The pink is a cover. I can say "pink fabric" to be safe. Title: "Two tabby cats lying on pink fabric beside remotes" — Two(1) tabby(2) cats(3) lying(4) on(5) pink(6) fabric(7) beside(8) remotes(9). 9 words. Good.

Description: 1-2 sentences factual. Mention main subject (two tabby cats), setting (bright pink fabric surface with maroon backdrop), action (lying stretched out), lighting (even indoor), distinctive details (two white remote controls, striped coats, one smaller one larger).

Keywords: 10-18 unique comma-separated. British English spelling. Include: tabby cat, cats, pink fabric, remote control, sofa, resting, striped coat, indoor, animal, pet, lying down, magenta, cushion, sleep. Need 10-18. Let me list: tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, indoor lighting, pet, animal, cushion, sleep, whiskers. That's 16. Good. Ensure British spelling — "feline" fine, no American spellings. "remote control" fine.

Check no uncertain details. Whiskers visible on right cat — yes. Sleep — eyes appear closed, but I'll keep "resting" and "sleep" maybe risky. I'll include "sleeping" as keyword? The instruction says omit uncertain. Eyes look closed, but to be safe I'll use "resting" not "sleeping". I'll drop "sleep". Replace with "paws". Paws visible. Good.

Final keywords: tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, indoor lighting, pet, animal, cushion, whiskers, paws. 16.

Now produce exactly three sections.
&lt;/think&gt;
Title:
Two tabby cats lying on pink fabric beside remotes

Description:
Two striped tabby cats lie stretched out on a bright pink fabric surface, with a darker maroon backdrop behind them. Two white remote controls rest on the fabric near the cats, and the scene is lit by even indoor light.

Keywords:
tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, indoor lighting, pet, animal, cushion, whiskers, paws
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants British-English catalogue metadata based strictly on visible facts. I need to look carefully at the image.

Visible facts: Two tabby cats lying on a bright pink/magenta fabric surface (looks like a sofa or cushion). There are two white remote controls visible — one near the left cat's head, one between the two cats. The cats are stretched out, appearing to sleep or rest. The left cat is smaller, stretched diagonally with paws extended. The right cat is larger, curled slightly with head down. The background top shows a darker red/maroon fabric (sofa back). Lighting appears even, indoor.

I must avoid unsupported identity, location, event, brand, species (well, "cat" is visible, that's fine — species is cat, that's clearly visible), period, intent. I should not say "sleeping" if uncertain? They appear to be resting with eyes closed. The left cat's eyes look closed; the right cat's eyes look closed. I can say "resting" or "lying". I'll say "lying" to be safe, maybe "appearing to rest".

Title: 5-10 words concrete. "Two tabby cats resting on pink sofa with remotes" — count words: Two(1) tabby(2) cats(3) resting(4) on(5) pink(6) sofa(7) with(8) remotes(9). 9 words. Good. But "sofa" — is it clearly a sofa? The top maroon part suggests a sofa back. The pink is a cover. I can say "pink fabric" to be safe. Title: "Two tabby cats lying on pink fabric beside remotes" — Two(1) tabby(2) cats(3) lying(4) on(5) pink(6) fabric(7) beside(8) remotes(9). 9 words. Good.

Description: 1-2 sentences factual. Mention main subject (two tabby cats), setting (bright pink fabric surface with maroon backdrop), action (lying stretched out), lighting (even indoor), distinctive details (two white remote controls, striped coats, one smaller one larger).

Keywords: 10-18 unique comma-separated. British English spelling. Include: tabby cat, cats, pink fabric, remote control, sofa, resting, striped coat, indoor, animal, pet, lying down, magenta, cushion, sleep. Need 10-18. Let me list: tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, indoor lighting, pet, animal, cushion, sleep, whiskers. That's 16. Good. Ensure British spelling — "feline" fine, no American spellings. "remote control" fine.

Check no uncertain details. Whiskers visible on right cat — yes. Sleep — eyes appear closed, but I'll keep "resting" and "sleep" maybe risky. I'll include "sleeping" as keyword? The instruction says omit uncertain. Eyes look closed, but to be safe I'll use "resting" not "sleeping". I'll drop "sleep". Replace with "paws". Paws visible. Good.

Final keywords: tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, indoor lighting, pet, animal, cushion, whiskers, paws. 16.

Now produce exactly three sections.
</think>
Title:
Two tabby cats lying on pink fabric beside remotes

Description:
Two striped tabby cats lie stretched out on a bright pink fabric surface, with a darker maroon backdrop behind them. Two white remote controls rest on the fabric near the cats, and the scene is lit by even indoor light.

Keywords:
tabby cat, feline, pink fabric, magenta, remote control, sofa, maroon, striped coat, resting, lying down, indoor lighting, pet, animal, cushion, whiskers, paws
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
- *Model load time:* 1.09s
- *Generation time:* 2.36s
- *Total time:* 4.15s
- *Prompt preparation time:* 0.587
- *First-token latency:* 1.05
- *Cleanup time:* 0.103
- *Prompt tokens:* 2,909
- *Generation tokens:* 82
- *Total tokens:* 2,991
- *Prompt throughput (raw):* 2,759 tok/s
- *Generation throughput (raw):* 102 tok/s
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
- *Model load time:* 3.18s
- *Generation time:* 1.88s
- *Total time:* 5.29s
- *Prompt preparation time:* 0.031
- *First-token latency:* 0.633
- *Cleanup time:* 0.169
- *Prompt tokens:* 386
- *Generation tokens:* 66
- *Total tokens:* 452
- *Prompt throughput (raw):* 610 tok/s
- *Generation throughput (raw):* 66.1 tok/s
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
- *Model load time:* 1.69s
- *Generation time:* 1.67s
- *Total time:* 3.53s
- *Prompt preparation time:* 0.03
- *First-token latency:* 0.259
- *Cleanup time:* 0.111
- *Prompt tokens:* 390
- *Generation tokens:* 88
- *Total tokens:* 478
- *Prompt throughput (raw):* 1,506 tok/s
- *Generation throughput (raw):* 63.1 tok/s
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
- *Model load time:* 2.34s
- *Generation time:* 0.94s
- *Total time:* 3.44s
- *Prompt preparation time:* 0.0101
- *First-token latency:* 0.336
- *Cleanup time:* 0.127
- *Prompt tokens:* 390
- *Generation tokens:* 75
- *Total tokens:* 465
- *Prompt throughput (raw):* 1,161 tok/s
- *Generation throughput (raw):* 127 tok/s
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
- *Model load time:* 1.57s
- *Generation time:* 0.72s
- *Total time:* 2.43s
- *Prompt preparation time:* 0.00963
- *First-token latency:* 0.166
- *Cleanup time:* 0.105
- *Prompt tokens:* 386
- *Generation tokens:* 67
- *Total tokens:* 453
- *Prompt throughput (raw):* 2,321 tok/s
- *Generation throughput (raw):* 124 tok/s
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
- *Model load time:* 1.07s
- *Generation time:* 10.47s
- *Total time:* 11.71s
- *Prompt preparation time:* 0.0271
- *First-token latency:* 0.46
- *Cleanup time:* 0.118
- *Prompt tokens:* 495
- *Generation tokens:* 819
- *Total tokens:* 1,314
- *Prompt throughput (raw):* 1,077 tok/s
- *Generation throughput (raw):* 81.9 tok/s
- *Peak memory (GB):* 8.0
- *Active memory (GB):* 7.1
- *Cache memory (GB):* 0.11
- *Model-load active memory (GB):* 7.08
- *Post-cleanup active memory (GB):* 0.00269
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 529
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--GLM-4.6V-Flash-4bit/snapshots/bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.8, "top_k": 2, "top_p": 0.6}
- *EOS token:* <\|endoftext\|>

#### Final answer

```text
Title: Two Tabby Cats on Pink Couch with Remotes
Description: Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it.
Keywords: tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat
```

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;Got it, let's tackle this. First, the title needs to be 5-10 words, concrete. Looking at the image: two tabby cats on a pink couch with remote controls. So title could be "Two Tabby Cats on Pink Couch with Remotes".

Next, the description: 1-2 sentences, factual. Need to include main subject (two tabby cats), setting (pink couch), action (lying down, sleeping), distinctive details (remote controls, cat colors, positions). Let's draft: "Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it."

Then keywords: unique, comma-separated. Need to pick from visible facts: tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, pink fabric, two cats, remote control, couch, pet, animal, domestic cat, tabby pattern, pink upholstery, remote controls on couch, two tabby cats resting on pink couch with remote controls.

Now, pick the most relevant and unique ones, keeping it within 10-18. Let's list possible keywords:

- tabby cats
- pink couch
- remote controls
- lying down
- sleeping
- brown tabby
- striped fur
- pink fabric
- two cats
- remote control
- couch
- pet
- animal
- domestic cat
- tabby pattern
- pink upholstery

Now, select the most concise and unique ones. Let's aim for around 12-15. Let's pick:

tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat

Wait, but need to ensure they are unique and clearly visible. Let's check the image again: two tabby cats (one lighter, one darker brown tabby), pink couch (upholstery), remote controls (two, one light blue, one white with buttons), lying down/sleeping, striped fur, two cats, remote control, couch, pet, domestic cat.

Now, title: "Two Tabby Cats on Pink Couch with Remotes" – that's 10 words, good.

Description: Need to be 1-2 sentences. Let's make it concise: "Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it." That's two sentences, factual.

Keywords: Let's pick the most relevant and unique ones, ensuring they are comma-separated and within 10-18. Let's list:

tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat

That's 15 keywords, which is within 10-18. They are all clearly visible and unique.

Now, check if all sections are correct:

Title: "Two Tabby Cats on Pink Couch with Remotes" – 10 words, concrete.

Description: "Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it." – 1-2 sentences, factual.

Keywords: "tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat" – 15 unique, comma-separated keywords.

All sections are correct. Now, return exactly these three sections and nothing else.&lt;/think&gt;
Title: Two Tabby Cats on Pink Couch with Remotes
Description: Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it.
Keywords: tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words, concrete. Looking at the image: two tabby cats on a pink couch with remote controls. So title could be "Two Tabby Cats on Pink Couch with Remotes".

Next, the description: 1-2 sentences, factual. Need to include main subject (two tabby cats), setting (pink couch), action (lying down, sleeping), distinctive details (remote controls, cat colors, positions). Let's draft: "Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it."

Then keywords: unique, comma-separated. Need to pick from visible facts: tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, pink fabric, two cats, remote control, couch, pet, animal, domestic cat, tabby pattern, pink upholstery, remote controls on couch, two tabby cats resting on pink couch with remote controls.

Now, pick the most relevant and unique ones, keeping it within 10-18. Let's list possible keywords:

- tabby cats
- pink couch
- remote controls
- lying down
- sleeping
- brown tabby
- striped fur
- pink fabric
- two cats
- remote control
- couch
- pet
- animal
- domestic cat
- tabby pattern
- pink upholstery

Now, select the most concise and unique ones. Let's aim for around 12-15. Let's pick:

tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat

Wait, but need to ensure they are unique and clearly visible. Let's check the image again: two tabby cats (one lighter, one darker brown tabby), pink couch (upholstery), remote controls (two, one light blue, one white with buttons), lying down/sleeping, striped fur, two cats, remote control, couch, pet, domestic cat.

Now, title: "Two Tabby Cats on Pink Couch with Remotes" – that's 10 words, good.

Description: Need to be 1-2 sentences. Let's make it concise: "Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it." That's two sentences, factual.

Keywords: Let's pick the most relevant and unique ones, ensuring they are comma-separated and within 10-18. Let's list:

tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat

That's 15 keywords, which is within 10-18. They are all clearly visible and unique.

Now, check if all sections are correct:

Title: "Two Tabby Cats on Pink Couch with Remotes" – 10 words, concrete.

Description: "Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it." – 1-2 sentences, factual.

Keywords: "tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat" – 15 unique, comma-separated keywords.

All sections are correct. Now, return exactly these three sections and nothing else.</think>
Title: Two Tabby Cats on Pink Couch with Remotes
Description: Two tabby cats lie on a pink couch, with remote controls nearby. One cat rests with eyes closed, the other lies beside it.
Keywords: tabby cats, pink couch, remote controls, lying down, sleeping, brown tabby, striped fur, two cats, remote control, couch, pet, domestic cat
```

</details>

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
- *Generation time:* 0.61s
- *Total time:* 1.39s
- *Prompt preparation time:* 0.0317
- *First-token latency:* 0.244
- *Cleanup time:* 0.107
- *Prompt tokens:* 734
- *Generation tokens:* 57
- *Total tokens:* 791
- *Prompt throughput (raw):* 3,012 tok/s
- *Generation throughput (raw):* 178 tok/s
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
- *Model load time:* 1.17s
- *Generation time:* 3.93s
- *Total time:* 5.25s
- *Prompt preparation time:* 0.0292
- *First-token latency:* 2.22
- *Cleanup time:* 0.11
- *Prompt tokens:* 3,435
- *Generation tokens:* 90
- *Total tokens:* 3,525
- *Prompt throughput (raw):* 1,550 tok/s
- *Generation throughput (raw):* 55.6 tok/s
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
- *Model load time:* 0.77s
- *Generation time:* 0.58s
- *Total time:* 1.47s
- *Prompt preparation time:* 0.0177
- *First-token latency:* 0.233
- *Cleanup time:* 0.0854
- *Prompt tokens:* 344
- *Generation tokens:* 73
- *Total tokens:* 417
- *Prompt throughput (raw):* 1,478 tok/s
- *Generation throughput (raw):* 217 tok/s
- *Peak memory (GB):* 3.6
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
- *Model load time:* 0.94s
- *Generation time:* 1.00s
- *Total time:* 2.08s
- *Prompt preparation time:* 0.0248
- *First-token latency:* 0.106
- *Cleanup time:* 0.0958
- *Prompt tokens:* 330
- *Generation tokens:* 268
- *Total tokens:* 598
- *Prompt throughput (raw):* 3,121 tok/s
- *Generation throughput (raw):* 313 tok/s
- *Peak memory (GB):* 3.5
- *Active memory (GB):* 2.7
- *Cache memory (GB):* 0.05
- *Model-load active memory (GB):* 2.67
- *Post-cleanup active memory (GB):* 0.00408
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 523
- *Processor:* mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 13caa569f70e4b68b0ad7dd9574f15a241b34a7d
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.7-1B-4bit/snapshots/13caa569f70e4b68b0ad7dd9574f15a241b34a7d
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 0.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title: Two tabby cats sleep on pink couch  
Description: Two tabby cats rest on a pink fabric couch, one with a remote and green ear tag, both in indoor lighting with striped fur.  
Keywords: tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy
```

*Readable output:*

<pre class="model-output-readable">
Okay, let me tackle this. First, the title needs to be 5-10 words. Let’s think: "Two cats sleep on pink couch." That’s 5 words. Next, description. The main subjects are two tabby cats, lying on a pink fabric couch. One is on the left with a remote, the other on the right with a remote. Lighting is indoor, soft. Distinctive details: striped fur, green ear tag on left cat. Keywords should be unique. Let’s list: tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy. Check count: tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy. That’s 12 keywords. Make sure it’s clear and based on visible facts.
&lt;/think&gt;

Title: Two tabby cats sleep on pink couch
Description: Two tabby cats rest on a pink fabric couch, one with a remote and green ear tag, both in indoor lighting with striped fur.
Keywords: tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy
</pre>

<details>
<summary>Exact raw output</summary>

```text
Okay, let me tackle this. First, the title needs to be 5-10 words. Let’s think: "Two cats sleep on pink couch." That’s 5 words. Next, description. The main subjects are two tabby cats, lying on a pink fabric couch. One is on the left with a remote, the other on the right with a remote. Lighting is indoor, soft. Distinctive details: striped fur, green ear tag on left cat. Keywords should be unique. Let’s list: tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy. Check count: tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy. That’s 12 keywords. Make sure it’s clear and based on visible facts.
</think>

Title: Two tabby cats sleep on pink couch  
Description: Two tabby cats rest on a pink fabric couch, one with a remote and green ear tag, both in indoor lighting with striped fur.  
Keywords: tabby, cat, sleep, pink, couch, remote, green, ear, tag, stripes, fur, lying, resting, indoor, cozy
```

</details>

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
- *Model load time:* 3.02s
- *Generation time:* 5.51s
- *Total time:* 8.73s
- *Prompt preparation time:* 0.0154
- *First-token latency:* 0.557
- *Cleanup time:* 0.16
- *Prompt tokens:* 330
- *Generation tokens:* 431
- *Total tokens:* 761
- *Prompt throughput (raw):* 592 tok/s
- *Generation throughput (raw):* 87.4 tok/s
- *Peak memory (GB):* 24
- *Active memory (GB):* 24
- *Cache memory (GB):* 0.11
- *Model-load active memory (GB):* 23.5
- *Post-cleanup active memory (GB):* 0.00459
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 523
- *Processor:* mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 0e3b706c1599ad95294122c67c77ce662de18794
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-V-4.7-35B-A3B-4bit/snapshots/0e3b706c1599ad95294122c67c77ce662de18794
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 1.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title: Two Tabby Cats Resting on Pink Sofa

Description: Two tabby cats lie on a bright pink fabric surface, each positioned near a white remote control. The cats appear to be sleeping or resting, with their striped fur patterns clearly visible against the pink background.

Keywords: tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, pet, indoor, relaxation, animal, furniture, remote, cat, pink
```

*Readable output:*

<pre class="model-output-readable">
I need to create British-English catalogue metadata based on the image. Let me start by analyzing the image carefully.

First, the main subjects are two tabby cats. They are lying on a pink surface, which looks like a couch or sofa. There are two remote controls visible near the cats. The cats are resting or sleeping. The lighting seems to be natural or indoor, but I should focus on clearly visible facts.

For the title, I need a concrete 5-10-word title. Something like "Two Tabby Cats Resting on Pink Sofa" would work.

Now for the description: The main subject is two tabby cats lying on a bright pink fabric surface, likely a sofa. Each cat is near a white remote control. The cats appear to be sleeping or resting. The lighting is even, highlighting the cats' fur patterns and the pink fabric.

Keywords: I need 10-18 unique keywords. Let's list them: tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, pet, indoor, relaxation, animal, furniture, remote, cat, pink.

Wait, I need to make sure they are unique and relevant. Let's check: tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, pet, indoor, relaxation, animal, furniture, remote, cat, pink. Some might be redundant, but I'll keep them as distinct terms.

Now, structuring the response exactly as required.
&lt;/think&gt;

Title: Two Tabby Cats Resting on Pink Sofa

Description: Two tabby cats lie on a bright pink fabric surface, each positioned near a white remote control. The cats appear to be sleeping or resting, with their striped fur patterns clearly visible against the pink background.

Keywords: tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, pet, indoor, relaxation, animal, furniture, remote, cat, pink
</pre>

<details>
<summary>Exact raw output</summary>

```text
I need to create British-English catalogue metadata based on the image. Let me start by analyzing the image carefully.

First, the main subjects are two tabby cats. They are lying on a pink surface, which looks like a couch or sofa. There are two remote controls visible near the cats. The cats are resting or sleeping. The lighting seems to be natural or indoor, but I should focus on clearly visible facts.

For the title, I need a concrete 5-10-word title. Something like "Two Tabby Cats Resting on Pink Sofa" would work.

Now for the description: The main subject is two tabby cats lying on a bright pink fabric surface, likely a sofa. Each cat is near a white remote control. The cats appear to be sleeping or resting. The lighting is even, highlighting the cats' fur patterns and the pink fabric.

Keywords: I need 10-18 unique keywords. Let's list them: tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, pet, indoor, relaxation, animal, furniture, remote, cat, pink.

Wait, I need to make sure they are unique and relevant. Let's check: tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, pet, indoor, relaxation, animal, furniture, remote, cat, pink. Some might be redundant, but I'll keep them as distinct terms.

Now, structuring the response exactly as required.
</think>

Title: Two Tabby Cats Resting on Pink Sofa

Description: Two tabby cats lie on a bright pink fabric surface, each positioned near a white remote control. The cats appear to be sleeping or resting, with their striped fur patterns clearly visible against the pink background.

Keywords: tabby cats, pink sofa, remote controls, resting, sleeping, feline, domestic cats, striped fur, couch, fabric, pet, indoor, relaxation, animal, furniture, remote, cat, pink
```

</details>

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
- *Model load time:* 1.24s
- *Generation time:* 1.93s
- *Total time:* 3.30s
- *Prompt preparation time:* 0.0165
- *First-token latency:* 0.72
- *Cleanup time:* 0.0987
- *Prompt tokens:* 1,070
- *Generation tokens:* 81
- *Total tokens:* 1,151
- *Prompt throughput (raw):* 1,485 tok/s
- *Generation throughput (raw):* 67.5 tok/s
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
- *Total time:* 1.65s
- *Prompt preparation time:* 0.0171
- *First-token latency:* 0.214
- *Cleanup time:* 0.104
- *Prompt tokens:* 1,069
- *Generation tokens:* 102
- *Total tokens:* 1,171
- *Prompt throughput (raw):* 4,992 tok/s
- *Generation throughput (raw):* 209 tok/s
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
- *Model load time:* 12.11s
- *Generation time:* 7.68s
- *Total time:* 20.11s
- *Prompt preparation time:* 0.0299
- *First-token latency:* 5.44
- *Cleanup time:* 0.285
- *Prompt tokens:* 549
- *Generation tokens:* 82
- *Total tokens:* 631
- *Prompt throughput (raw):* 101 tok/s
- *Generation throughput (raw):* 91.1 tok/s
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
- *Model load time:* 1.02s
- *Generation time:* 1.85s
- *Total time:* 3.00s
- *Prompt preparation time:* 0.0194
- *First-token latency:* 0.451
- *Cleanup time:* 0.0934
- *Prompt tokens:* 873
- *Generation tokens:* 95
- *Total tokens:* 968
- *Prompt throughput (raw):* 1,936 tok/s
- *Generation throughput (raw):* 71.1 tok/s
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
- *Model load time:* 1.30s
- *Generation time:* 3.14s
- *Total time:* 4.57s
- *Prompt preparation time:* 0.0219
- *First-token latency:* 0.737
- *Cleanup time:* 0.105
- *Prompt tokens:* 860
- *Generation tokens:* 70
- *Total tokens:* 930
- *Prompt throughput (raw):* 1,167 tok/s
- *Generation throughput (raw):* 30.3 tok/s
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
- *Model load time:* 2.67s
- *Generation time:* 36.40s
- *Total time:* 39.24s
- *Prompt preparation time:* 0.0294
- *First-token latency:* 0.891
- *Cleanup time:* 0.125
- *Prompt tokens:* 545
- *Generation tokens:* 884
- *Total tokens:* 1,429
- *Prompt throughput (raw):* 612 tok/s
- *Generation throughput (raw):* 24.9 tok/s
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
Title: Two cats resting on pink blanket sofa
Description: Two cats lie on a bright pink blanket on a red sofa, one with its head resting down and eyes closed, the other with its head lowered. Two white remote controls lie on the fabric beside the cats.
Keywords: cats, pink blanket, red sofa, remote controls, tabby markings, resting, sleeping, fabric, couch, indoor, white remotes, fur, paws, collar
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
- *Model load time:* 1.05s
- *Generation time:* 0.48s
- *Total time:* 1.65s
- *Prompt preparation time:* 0.016
- *First-token latency:* 0.157
- *Cleanup time:* 0.092
- *Prompt tokens:* 418
- *Generation tokens:* 78
- *Total tokens:* 496
- *Prompt throughput (raw):* 2,662 tok/s
- *Generation throughput (raw):* 267 tok/s
- *Peak memory (GB):* 3.3
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
- *Model load time:* 0.83s
- *Generation time:* 1.91s
- *Total time:* 2.85s
- *Prompt preparation time:* 0.0183
- *First-token latency:* 0.227
- *Cleanup time:* 0.0925
- *Prompt tokens:* 883
- *Generation tokens:* 94
- *Total tokens:* 977
- *Prompt throughput (raw):* 3,890 tok/s
- *Generation throughput (raw):* 58.1 tok/s
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
- *Model load time:* 2.29s
- *Generation time:* 1.24s
- *Total time:* 3.73s
- *Prompt preparation time:* 0.0245
- *First-token latency:* 0.366
- *Cleanup time:* 0.152
- *Prompt tokens:* 409
- *Generation tokens:* 81
- *Total tokens:* 490
- *Prompt throughput (raw):* 1,116 tok/s
- *Generation throughput (raw):* 93.6 tok/s
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
- *Model load time:* 1.88s
- *Generation time:* 0.97s
- *Total time:* 3.01s
- *Prompt preparation time:* 0.0211
- *First-token latency:* 0.33
- *Cleanup time:* 0.126
- *Prompt tokens:* 409
- *Generation tokens:* 83
- *Total tokens:* 492
- *Prompt throughput (raw):* 1,239 tok/s
- *Generation throughput (raw):* 133 tok/s
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
- *Model load time:* 0.89s
- *Generation time:* 0.84s
- *Total time:* 1.87s
- *Prompt preparation time:* 0.0104
- *First-token latency:* 0.204
- *Cleanup time:* 0.117
- *Prompt tokens:* 409
- *Generation tokens:* 60
- *Total tokens:* 469
- *Prompt throughput (raw):* 2,001 tok/s
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
- *Model load time:* 2.41s
- *Generation time:* 7.66s
- *Total time:* 10.24s
- *Prompt preparation time:* 0.0247
- *First-token latency:* 0.334
- *Cleanup time:* 0.129
- *Prompt tokens:* 419
- *Generation tokens:* 898
- *Total tokens:* 1,317
- *Prompt throughput (raw):* 1,254 tok/s
- *Generation throughput (raw):* 123 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.00906
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 552
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 1.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title: Two tabby cats sleeping on pink sofa next to remotes

Description: Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote controls rest near the cats, one positioned by the head of the smaller cat on the left.

Keywords: cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, stretching, pets, cushion
```

*Readable output:*

<pre class="model-output-readable">
The user wants catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subjects:** Two tabby cats.
    *   Cat 1 (Left): Lying on its back/side, stretched out. Looks like a young cat or kitten based on size, but it's hard to be certain. It has a green collar.
    *   Cat 2 (Right): Lying on its side/stomach, facing the other cat but slightly angled away. Larger than the one on the left. Also tabby.
*   **Setting:** A bright pink sofa or couch with fabric folds.
*   **Objects:** Two remote controls. One is white/grey with blue buttons (left), one is white with colorful buttons (center/right).
*   **Lighting:** Bright, likely natural daylight or strong indoor lighting. Shadows are soft but distinct.
*   **Action:** Sleeping or resting peacefully.

**2. Draft Title (5-10 words):**
*   *Draft 1:* Two tabby cats sleeping on pink sofa with remotes. (9 words) - Good.
*   *Draft 2:* Sleeping tabby cats resting on bright pink couch cushions. (9 words) - Good.
*   *Draft 3:* Two striped cats lying on pink sofa next to remotes. (10 words) - Good.
*   *Selection:* Two tabby cats sleeping on pink sofa next to remotes.

**3. Draft Description (1-2 sentences):**
*   *Focus:* Main subject, setting, action, lighting, details.
*   *Draft 1:* Two tabby cats are lying asleep on a bright pink sofa. There are two remote controls near them, one near the head of the cat on the left and another between the cats. The lighting is bright and casts soft shadows.
*   *Refining for constraints:* Needs to be factual and concise.
*   *Final Version:* Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote controls rest near the cats, one positioned by the head of the smaller cat on the left.

**4. Draft Keywords (10-18 unique, comma-separated):**
*   *Brainstorming:* cats, tabby, pets, sleeping, sofa, couch, pink, remote, television, controls, feline, animals, resting, fabric, stripes, collar, green, furniture, domestic.
*   *Selection:* cats, tabby, sleeping, sofa, couch, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, pets, stretching. (18 items)
*   *Refining:* cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, stretching, pets, cushion.

**5. Final Review:**
*   Title: Two tabby cats sleeping on pink sofa next to remotes. (9 words) -&gt; Check.
*   Description: Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote controls rest near the cats, one positioned by the head of the smaller cat on the left. (2 sentences) -&gt; Check.
*   Keywords: cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, stretching, pets, cushion. (18 keywords) -&gt; Check.

**6. Formatting:**
Ensure "Title:", "Description:", "Keywords:" headers are present and no extra text is added.

*Self-Correction during drafting:* The prompt asks for "British-English catalogue metadata". "Couch" and "Sofa" are fine
&lt;/think&gt;

Title: Two tabby cats sleeping on pink sofa next to remotes

Description: Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote controls rest near the cats, one positioned by the head of the smaller cat on the left.

Keywords: cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, stretching, pets, cushion
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subjects:** Two tabby cats.
    *   Cat 1 (Left): Lying on its back/side, stretched out. Looks like a young cat or kitten based on size, but it's hard to be certain. It has a green collar.
    *   Cat 2 (Right): Lying on its side/stomach, facing the other cat but slightly angled away. Larger than the one on the left. Also tabby.
*   **Setting:** A bright pink sofa or couch with fabric folds.
*   **Objects:** Two remote controls. One is white/grey with blue buttons (left), one is white with colorful buttons (center/right).
*   **Lighting:** Bright, likely natural daylight or strong indoor lighting. Shadows are soft but distinct.
*   **Action:** Sleeping or resting peacefully.

**2. Draft Title (5-10 words):**
*   *Draft 1:* Two tabby cats sleeping on pink sofa with remotes. (9 words) - Good.
*   *Draft 2:* Sleeping tabby cats resting on bright pink couch cushions. (9 words) - Good.
*   *Draft 3:* Two striped cats lying on pink sofa next to remotes. (10 words) - Good.
*   *Selection:* Two tabby cats sleeping on pink sofa next to remotes.

**3. Draft Description (1-2 sentences):**
*   *Focus:* Main subject, setting, action, lighting, details.
*   *Draft 1:* Two tabby cats are lying asleep on a bright pink sofa. There are two remote controls near them, one near the head of the cat on the left and another between the cats. The lighting is bright and casts soft shadows.
*   *Refining for constraints:* Needs to be factual and concise.
*   *Final Version:* Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote controls rest near the cats, one positioned by the head of the smaller cat on the left.

**4. Draft Keywords (10-18 unique, comma-separated):**
*   *Brainstorming:* cats, tabby, pets, sleeping, sofa, couch, pink, remote, television, controls, feline, animals, resting, fabric, stripes, collar, green, furniture, domestic.
*   *Selection:* cats, tabby, sleeping, sofa, couch, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, pets, stretching. (18 items)
*   *Refining:* cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, stretching, pets, cushion.

**5. Final Review:**
*   Title: Two tabby cats sleeping on pink sofa next to remotes. (9 words) -> Check.
*   Description: Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote controls rest near the cats, one positioned by the head of the smaller cat on the left. (2 sentences) -> Check.
*   Keywords: cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, stretching, pets, cushion. (18 keywords) -> Check.

**6. Formatting:**
Ensure "Title:", "Description:", "Keywords:" headers are present and no extra text is added.

*Self-Correction during drafting:* The prompt asks for "British-English catalogue metadata". "Couch" and "Sofa" are fine
</think>

Title: Two tabby cats sleeping on pink sofa next to remotes

Description: Two tabby cats lie sleeping and stretching on a bright pink sofa cushion. Two remote controls rest near the cats, one positioned by the head of the smaller cat on the left.

Keywords: cats, tabby, sleeping, sofa, pink, remote, controls, feline, animals, resting, fabric, stripes, furniture, collar, green, stretching, pets, cushion
```

</details>

</details>

---

<a id="model-mlx-community-qwen36-35b-a3b-4bit"></a>

### mlx-community/Qwen3.6-35B-A3B-4bit

<details>
<summary>Complete evidence: mlx-community/Qwen3.6-35B-A3B-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* no concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Model load time:* 2.43s
- *Generation time:* 5.78s
- *Total time:* 8.39s
- *Prompt preparation time:* 0.0242
- *First-token latency:* 0.357
- *Cleanup time:* 0.131
- *Prompt tokens:* 419
- *Generation tokens:* 653
- *Total tokens:* 1,072
- *Prompt throughput (raw):* 1,175 tok/s
- *Generation throughput (raw):* 121 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 20.4
- *Post-cleanup active memory (GB):* 0.00957
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 552
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 38740b847e4cb78f352aba30aa41c76e08e6eb46
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 1.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title: Two tabby cats sleeping on pink sofa with remotes.
Description: Two tabby cats are resting on a bright pink sofa cushion, flanked by two grey remote controls. The smaller cat on the left lies on its back while the larger cat on the right sleeps facing downwards.
Keywords: tabby, cats, pink, sofa, remote, controls, sleeping, fur, stripes, paws, cushions, furniture, pets, animals, grey, buttons, fabric, pattern, kittens
```

*Readable output:*

<pre class="model-output-readable">
The user wants metadata for an image of two cats on a sofa.
I need to extract specific details:
- **Subject:** Two tabby cats. One is small/kitten, one is adult/larger.
- **Setting:** A bright pink sofa/couch.
- **Objects:** Two remote controls (grey/silver).
- **Action:** Sleeping/resting.
- **Lighting:** Natural, soft, indoor lighting.
- **Distinctive details:** The kittens' striped fur, the pink fabric, the position of the remotes.

**Drafting Title:**
Two tabby cats sleeping on pink sofa with remotes. (9 words) - Good.

**Drafting Description:**
Two tabby cats are resting on a bright pink sofa cushion, flanked by two grey remote controls. The smaller cat on the left lies on its back while the larger cat on the right sleeps facing downwards.

**Drafting Keywords:**
tabby cats, pink sofa, sleeping kittens, remote controls, furniture, striped fur, paws, cushions, pets, animals, resting, domestic, grey remote, texture, fabric, stripes, back, front paws, rear paws.

Let's refine keywords to be unique and distinct.
1. tabby cats
2. pink sofa
3. sleeping kittens
4. remote controls
5. furniture
6. striped fur
7. paws
8. cushions
9. pets
10. animals
11. resting
12. domestic
13. grey remote
14. texture
15. fabric
16. stripes
17. back
18. front paws

Wait, "back" and "front paws" might be redundant with "paws". Let's stick to visible objects and clear descriptors.
- tabby
- cats
- pink
- sofa
- remote
- controls
- sleeping
- fur
- stripes
- paws
- cushions
- furniture
- pets
- animals
- grey
- buttons
- fabric
- pattern

Let's count: 18 words. Perfect.

Final check of the prompt requirements:
- British-English catalogue metadata. (No specific British spelling issues really, just standard English).
- Only clearly visible facts. (Yes).
- Omit uncertain details. (Yes, didn't guess breeds or specific room names).
- 5-10 word title. (Yes).
- 1-2 sentence description. (Yes).
- 10-18 unique keywords. (Yes).

Let's assemble.
&lt;/think&gt;

Title: Two tabby cats sleeping on pink sofa with remotes.
Description: Two tabby cats are resting on a bright pink sofa cushion, flanked by two grey remote controls. The smaller cat on the left lies on its back while the larger cat on the right sleeps facing downwards.
Keywords: tabby, cats, pink, sofa, remote, controls, sleeping, fur, stripes, paws, cushions, furniture, pets, animals, grey, buttons, fabric, pattern, kittens
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants metadata for an image of two cats on a sofa.
I need to extract specific details:
- **Subject:** Two tabby cats. One is small/kitten, one is adult/larger.
- **Setting:** A bright pink sofa/couch.
- **Objects:** Two remote controls (grey/silver).
- **Action:** Sleeping/resting.
- **Lighting:** Natural, soft, indoor lighting.
- **Distinctive details:** The kittens' striped fur, the pink fabric, the position of the remotes.

**Drafting Title:**
Two tabby cats sleeping on pink sofa with remotes. (9 words) - Good.

**Drafting Description:**
Two tabby cats are resting on a bright pink sofa cushion, flanked by two grey remote controls. The smaller cat on the left lies on its back while the larger cat on the right sleeps facing downwards.

**Drafting Keywords:**
tabby cats, pink sofa, sleeping kittens, remote controls, furniture, striped fur, paws, cushions, pets, animals, resting, domestic, grey remote, texture, fabric, stripes, back, front paws, rear paws.

Let's refine keywords to be unique and distinct.
1. tabby cats
2. pink sofa
3. sleeping kittens
4. remote controls
5. furniture
6. striped fur
7. paws
8. cushions
9. pets
10. animals
11. resting
12. domestic
13. grey remote
14. texture
15. fabric
16. stripes
17. back
18. front paws

Wait, "back" and "front paws" might be redundant with "paws". Let's stick to visible objects and clear descriptors.
- tabby
- cats
- pink
- sofa
- remote
- controls
- sleeping
- fur
- stripes
- paws
- cushions
- furniture
- pets
- animals
- grey
- buttons
- fabric
- pattern

Let's count: 18 words. Perfect.

Final check of the prompt requirements:
- British-English catalogue metadata. (No specific British spelling issues really, just standard English).
- Only clearly visible facts. (Yes).
- Omit uncertain details. (Yes, didn't guess breeds or specific room names).
- 5-10 word title. (Yes).
- 1-2 sentence description. (Yes).
- 10-18 unique keywords. (Yes).

Let's assemble.
</think>

Title: Two tabby cats sleeping on pink sofa with remotes.
Description: Two tabby cats are resting on a bright pink sofa cushion, flanked by two grey remote controls. The smaller cat on the left lies on its back while the larger cat on the right sleeps facing downwards.
Keywords: tabby, cats, pink, sofa, remote, controls, sleeping, fur, stripes, paws, cushions, furniture, pets, animals, grey, buttons, fabric, pattern, kittens
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
- *Model load time:* 1.89s
- *Generation time:* 21.45s
- *Total time:* 23.52s
- *Prompt preparation time:* 0.0107
- *First-token latency:* 0.711
- *Cleanup time:* 0.143
- *Prompt tokens:* 461
- *Generation tokens:* 640
- *Total tokens:* 1,101
- *Prompt throughput (raw):* 648 tok/s
- *Generation throughput (raw):* 30.9 tok/s
- *Peak memory (GB):* 17
- *Active memory (GB):* 16
- *Cache memory (GB):* 0.29
- *Model-load active memory (GB):* 16.1
- *Post-cleanup active memory (GB):* 0.0101
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 789
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* f9cb6495054684578f19bf751882716f87542a96
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-nvfp4/snapshots/f9cb6495054684578f19bf751882716f87542a96
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 1.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title:
Two tabby cats resting beside remote controls on pink sofa

Description:
Two striped tabby cats lie asleep on a vivid pink sofa, one wearing a green collar tag, with two white remote controls placed nearby under even indoor lighting.

Keywords:
tabby cats, striped coats, pink sofa, magenta fabric, remote controls, white remotes, sleeping cats, green collar tag, indoor lighting, resting felines, domestic animals, lounging, two cats, cushioned seating
```

*Readable output:*

<pre class="model-output-readable">
The user wants British-English catalogue metadata based strictly on visible facts. I need to examine the image carefully.

Visible facts: Two tabby cats lying on a pink/magenta fabric surface (looks like a couch or sofa with cushions visible at top). Both cats appear asleep/lying down. Two white remote controls are visible—one near the top left by the smaller cat, one between the cats. The cats are tabby patterned, brownish-grey with stripes. One cat has a green collar tag visible. Lighting appears even/indoor.

I must avoid uncertain identity, location, event, brand, species (well, "cat" is visible, that's fine), period, intent. I'll stick to "tabby cats" as that's a visible coat pattern, acceptable.

Title: 5-10 words concrete. "Two tabby cats sleeping beside remote controls on pink couch" — count: Two(1) tabby(2) cats(3) sleeping(4) beside(5) remote(6) controls(7) on(8) pink(9) couch(10). Good, 10 words.

Description: 1-2 sentences factual. Mention main subject, setting, action, lighting, distinctive details.

Keywords: 10-18 unique comma-separated.

Let me draft keywords: tabby cats, pink fabric, remote controls, sleeping, striped coat, magenta sofa, indoor lighting, feline, collar tag, resting, domestic animals, two cats, white remotes, lounging. Count them.

Let me ensure British English: "couch" vs "sofa" — both fine; I'll use "sofa" perhaps. Actually pink fabric surface with cushions at top suggests sofa. I'll say "pink sofa".

Final title with sofa: "Two tabby cats resting beside remote controls on pink sofa" = Two1 tabby2 cats3 resting4 beside5 remote6 controls7 on8 pink9 sofa10. Good.

Description: "Two striped tabby cats lie asleep on a vivid pink sofa, one with a green collar tag, with two white remote controls placed nearby under even indoor lighting." That's one sentence, factual.

Keywords list, ensure unique and 10-18: tabby cats, striped coats, pink sofa, magenta fabric, remote controls, white remotes, sleeping cats, green collar tag, indoor lighting, resting felines, domestic animals, lounging, two cats, cushioned seating. Count: 14. Good.

Return exactly three sections.
&lt;/think&gt;

Title:
Two tabby cats resting beside remote controls on pink sofa

Description:
Two striped tabby cats lie asleep on a vivid pink sofa, one wearing a green collar tag, with two white remote controls placed nearby under even indoor lighting.

Keywords:
tabby cats, striped coats, pink sofa, magenta fabric, remote controls, white remotes, sleeping cats, green collar tag, indoor lighting, resting felines, domestic animals, lounging, two cats, cushioned seating
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants British-English catalogue metadata based strictly on visible facts. I need to examine the image carefully.

Visible facts: Two tabby cats lying on a pink/magenta fabric surface (looks like a couch or sofa with cushions visible at top). Both cats appear asleep/lying down. Two white remote controls are visible—one near the top left by the smaller cat, one between the cats. The cats are tabby patterned, brownish-grey with stripes. One cat has a green collar tag visible. Lighting appears even/indoor.

I must avoid uncertain identity, location, event, brand, species (well, "cat" is visible, that's fine), period, intent. I'll stick to "tabby cats" as that's a visible coat pattern, acceptable.

Title: 5-10 words concrete. "Two tabby cats sleeping beside remote controls on pink couch" — count: Two(1) tabby(2) cats(3) sleeping(4) beside(5) remote(6) controls(7) on(8) pink(9) couch(10). Good, 10 words.

Description: 1-2 sentences factual. Mention main subject, setting, action, lighting, distinctive details.

Keywords: 10-18 unique comma-separated.

Let me draft keywords: tabby cats, pink fabric, remote controls, sleeping, striped coat, magenta sofa, indoor lighting, feline, collar tag, resting, domestic animals, two cats, white remotes, lounging. Count them.

Let me ensure British English: "couch" vs "sofa" — both fine; I'll use "sofa" perhaps. Actually pink fabric surface with cushions at top suggests sofa. I'll say "pink sofa".

Final title with sofa: "Two tabby cats resting beside remote controls on pink sofa" = Two1 tabby2 cats3 resting4 beside5 remote6 controls7 on8 pink9 sofa10. Good.

Description: "Two striped tabby cats lie asleep on a vivid pink sofa, one with a green collar tag, with two white remote controls placed nearby under even indoor lighting." That's one sentence, factual.

Keywords list, ensure unique and 10-18: tabby cats, striped coats, pink sofa, magenta fabric, remote controls, white remotes, sleeping cats, green collar tag, indoor lighting, resting felines, domestic animals, lounging, two cats, cushioned seating. Count: 14. Good.

Return exactly three sections.
</think>

Title:
Two tabby cats resting beside remote controls on pink sofa

Description:
Two striped tabby cats lie asleep on a vivid pink sofa, one wearing a green collar tag, with two white remote controls placed nearby under even indoor lighting.

Keywords:
tabby cats, striped coats, pink sofa, magenta fabric, remote controls, white remotes, sleeping cats, green collar tag, indoor lighting, resting felines, domestic animals, lounging, two cats, cushioned seating
```

</details>

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
- *Model load time:* 0.61s
- *Generation time:* 0.72s
- *Total time:* 1.51s
- *Prompt preparation time:* 0.0681
- *First-token latency:* 0.383
- *Cleanup time:* 0.113
- *Prompt tokens:* 1,204
- *Generation tokens:* 31
- *Total tokens:* 1,235
- *Prompt throughput (raw):* 3,141 tok/s
- *Generation throughput (raw):* 117 tok/s
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
- *Generation time:* 0.77s
- *Total time:* 1.73s
- *Prompt preparation time:* 0.0127
- *First-token latency:* 0.136
- *Cleanup time:* 0.106
- *Prompt tokens:* 420
- *Generation tokens:* 88
- *Total tokens:* 508
- *Prompt throughput (raw):* 3,095 tok/s
- *Generation throughput (raw):* 149 tok/s
- *Peak memory (GB):* 5.2
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
- *Model load time:* 1.64s
- *Generation time:* 3.11s
- *Total time:* 4.89s
- *Prompt preparation time:* 0.0114
- *First-token latency:* 0.666
- *Cleanup time:* 0.118
- *Prompt tokens:* 538
- *Generation tokens:* 87
- *Total tokens:* 625
- *Prompt throughput (raw):* 808 tok/s
- *Generation throughput (raw):* 35.9 tok/s
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
- *Model load time:* 2.12s
- *Generation time:* 4.72s
- *Total time:* 7.01s
- *Prompt preparation time:* 0.0241
- *First-token latency:* 0.342
- *Cleanup time:* 0.121
- *Prompt tokens:* 421
- *Generation tokens:* 706
- *Total tokens:* 1,127
- *Prompt throughput (raw):* 1,232 tok/s
- *Generation throughput (raw):* 162 tok/s
- *Peak memory (GB):* 21
- *Active memory (GB):* 20
- *Cache memory (GB):* 0.08
- *Model-load active memory (GB):* 19.7
- *Post-cleanup active memory (GB):* 0.0144
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 547
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.nemotron_h_nano_omni.processing_nemotron_h_nano_omni.NemotronHNanoOmniProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* 75c89904e1c2a045098d9111e7cb250a4d8adcec
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nativ-community--Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit/snapshots/75c89904e1c2a045098d9111e7cb250a4d8adcec
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 0.6, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 0, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title: Two tabby cats sleeping on pink cushion  
Description: Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby.  
Keywords: cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, whiskers, sofa, bedding
```

*Readable output:*

<pre class="model-output-readable">
We need to produce British-English catalogue metadata using only clearly visible facts. Omit uncertain details. Provide title (5-10 words), description (1-2 sentences factual), keywords (10-18 unique, comma-separated). Must be exactly three sections, no extra text.

We need to infer what is visible: two cats sleeping on a pink surface (likely a couch or cushion). There are two remote controls (white). The cats are tabby. The setting: a pink fabric (cushion) on a red background (maybe a sofa). Lighting appears natural indoor. Distinctive details: remote controls, pink fabric, tabby cats.

Title 5-10 words: "Two tabby cats sleeping on pink cushion". That's 6 words. Good.

Description: 1-2 sentences, factual. "Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby." That's one sentence. Could be two sentences but okay.

Keywords: need 10-18 unique, comma-separated. Let's list: cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, whiskers, sofa, bedding. Need 10-18, ensure uniqueness. Count: cat(1), tabby(2), pink(3), cushion(4), remote(5), white(6), sleeping(7), domestic(8), animal(9), furniture(10), indoor(11), relaxed(12), paws(13), ears(14), whiskers(15), sofa(16), bedding(17). That's 17, within 10-18. Ensure comma-separated.

Check no uncertain details: we assume "cushion" is visible, yes pink fabric. Remote controls are white. Cats are tabby. All visible.

Now format:

Title:
Description:
Keywords:

Make sure each line starts with "Title:", "Description:", "Keywords:".

Let's craft.

Title: "Two tabby cats sleeping on pink cushion" (6 words). Good.

Description: "Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby." That's one sentence, factual.

Keywords: "cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, whiskers, sofa, bedding"

Check count: 17 items. Good.

Now ensure British English: "lying" maybe "lying" is fine. "cushion" British. "remote controls" maybe "remote controls" okay. Use British spelling: "remote controls" fine. "pink" same. "cushion" same.

Check description: "Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby." That's okay.

Now output exactly three sections, nothing else.
&lt;/think&gt;
Title: Two tabby cats sleeping on pink cushion
Description: Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby.
Keywords: cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, whiskers, sofa, bedding
</pre>

<details>
<summary>Exact raw output</summary>

```text
We need to produce British-English catalogue metadata using only clearly visible facts. Omit uncertain details. Provide title (5-10 words), description (1-2 sentences factual), keywords (10-18 unique, comma-separated). Must be exactly three sections, no extra text.

We need to infer what is visible: two cats sleeping on a pink surface (likely a couch or cushion). There are two remote controls (white). The cats are tabby. The setting: a pink fabric (cushion) on a red background (maybe a sofa). Lighting appears natural indoor. Distinctive details: remote controls, pink fabric, tabby cats.

Title 5-10 words: "Two tabby cats sleeping on pink cushion". That's 6 words. Good.

Description: 1-2 sentences, factual. "Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby." That's one sentence. Could be two sentences but okay.

Keywords: need 10-18 unique, comma-separated. Let's list: cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, whiskers, sofa, bedding. Need 10-18, ensure uniqueness. Count: cat(1), tabby(2), pink(3), cushion(4), remote(5), white(6), sleeping(7), domestic(8), animal(9), furniture(10), indoor(11), relaxed(12), paws(13), ears(14), whiskers(15), sofa(16), bedding(17). That's 17, within 10-18. Ensure comma-separated.

Check no uncertain details: we assume "cushion" is visible, yes pink fabric. Remote controls are white. Cats are tabby. All visible.

Now format:

Title:
Description:
Keywords:

Make sure each line starts with "Title:", "Description:", "Keywords:".

Let's craft.

Title: "Two tabby cats sleeping on pink cushion" (6 words). Good.

Description: "Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby." That's one sentence, factual.

Keywords: "cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, whiskers, sofa, bedding"

Check count: 17 items. Good.

Now ensure British English: "lying" maybe "lying" is fine. "cushion" British. "remote controls" maybe "remote controls" okay. Use British spelling: "remote controls" fine. "pink" same. "cushion" same.

Check description: "Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby." That's okay.

Now output exactly three sections, nothing else.
</think>
Title: Two tabby cats sleeping on pink cushion  
Description: Two tabby cats are lying asleep on a bright pink cushion, with white remote controls placed nearby.  
Keywords: cat, tabby, pink, cushion, remote, white, sleeping, domestic, animal, furniture, indoor, relaxed, paws, ears, whiskers, sofa, bedding
```

</details>

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
- *Model load time:* 16.97s
- *Generation time:* 38.03s
- *Total time:* 55.31s
- *Prompt preparation time:* 0.0276
- *First-token latency:* 10.5
- *Cleanup time:* 0.245
- *Prompt tokens:* 420
- *Generation tokens:* 797
- *Total tokens:* 1,217
- *Prompt throughput (raw):* 39.9 tok/s
- *Generation throughput (raw):* 30.3 tok/s
- *Peak memory (GB):* 81
- *Active memory (GB):* 78
- *Cache memory (GB):* 0.21
- *Model-load active memory (GB):* 78.4
- *Post-cleanup active memory (GB):* 0.0146
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 546
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.nemotron_h_nano_omni.processing_nemotron_h_nano_omni.NemotronHNanoOmniProcessor
- *Tokenizer:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Requested model revision:* not requested
- *Resolved model revision:* ae1368329b1fc76b71534c5d04b87b35d9cc6e15
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/snapshots/ae1368329b1fc76b71534c5d04b87b35d9cc6e15
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 1.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 0, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title: Two tabby cats sleeping on pink sofa

Description: Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a red cushion.

Keywords: cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting
```

*Readable output:*

<pre class="model-output-readable">
The user wants British-English catalogue metadata based on the image.

1.  **Analyze the image:**
    *   **Subjects:** Two cats lying on a pink surface (likely a sofa or bed).
    *   **Action:** Both appear to be sleeping or resting. One is curled slightly, the other is stretched out.
    *   **Setting:** A pink fabric surface, possibly a sofa cover or blanket. There are two remote controls visible behind them.
    *   **Lighting:** Even, indoor lighting. No harsh shadows.
    *   **Distinctive details:** Tabby cats, pink fabric, remote controls (white/grey).
    *   **Constraints:** British-English, only clearly visible facts, omit uncertain details (breed, exact location, specific intent, specific brand of remote, etc.).
    *   **Format:**
        *   Title: 5-10 words.
        *   Description: 1-2 sentences covering subject, setting, action, lighting, distinctive details.
        *   Keywords: 10-18 unique, comma-separated.

2.  **Drafting Title:**
    *   *Draft 1:* Two tabby cats sleeping on pink sofa. (6 words) - Good.
    *   *Draft 2:* Cats resting on pink fabric with remotes. (7 words) - Good.
    *   *Selection:* Two tabby cats sleeping on pink sofa.

3.  **Drafting Description:**
    *   *Draft:* Two tabby cats are asleep on a bright pink fabric surface. White remote controls lie behind them against a red cushion.
    *   *Check:* Subject (two cats), Setting (pink fabric, red cushion), Action (asleep), Lighting (implied even indoor), Distinctive details (remote controls).
    *   *Refinement for British English:* "Asleep" is fine. "Fabric surface" is safe.
    *   *Final:* Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a red cushion.

4.  **Drafting Keywords:**
    *   Need 10-18 unique words.
    *   cat, cats, tabby, sleeping, asleep, pink, sofa, cushion, remote, control, television, fabric, indoor, domestic, pet, fur, stripes, resting.
    *   *Selection:* cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting

5.  **Final Review against constraints:**
    *   British-English? Yes (no specific US spellings used like 'color', though none appeared).
    *   Only clearly visible facts? Yes.
    *   Omit uncertain details? Yes (no breed specification beyond visual tabby, no brand).
    *   Title length: 7 words. (Pass)
    *   Description length: 2 sentences. (Pass)
    *   Keywords count: 15 words. (Pass)
    *   Format: Exactly three sections.

6.  **Constructing Output:**
    Title: Two tabby cats sleeping on pink sofa
    Description: Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a red cushion.
    Keywords: cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting&lt;/think&gt;Title: Two tabby cats sleeping on pink sofa

Description: Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a red cushion.

Keywords: cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants British-English catalogue metadata based on the image.

1.  **Analyze the image:**
    *   **Subjects:** Two cats lying on a pink surface (likely a sofa or bed).
    *   **Action:** Both appear to be sleeping or resting. One is curled slightly, the other is stretched out.
    *   **Setting:** A pink fabric surface, possibly a sofa cover or blanket. There are two remote controls visible behind them.
    *   **Lighting:** Even, indoor lighting. No harsh shadows.
    *   **Distinctive details:** Tabby cats, pink fabric, remote controls (white/grey).
    *   **Constraints:** British-English, only clearly visible facts, omit uncertain details (breed, exact location, specific intent, specific brand of remote, etc.).
    *   **Format:**
        *   Title: 5-10 words.
        *   Description: 1-2 sentences covering subject, setting, action, lighting, distinctive details.
        *   Keywords: 10-18 unique, comma-separated.

2.  **Drafting Title:**
    *   *Draft 1:* Two tabby cats sleeping on pink sofa. (6 words) - Good.
    *   *Draft 2:* Cats resting on pink fabric with remotes. (7 words) - Good.
    *   *Selection:* Two tabby cats sleeping on pink sofa.

3.  **Drafting Description:**
    *   *Draft:* Two tabby cats are asleep on a bright pink fabric surface. White remote controls lie behind them against a red cushion.
    *   *Check:* Subject (two cats), Setting (pink fabric, red cushion), Action (asleep), Lighting (implied even indoor), Distinctive details (remote controls).
    *   *Refinement for British English:* "Asleep" is fine. "Fabric surface" is safe.
    *   *Final:* Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a red cushion.

4.  **Drafting Keywords:**
    *   Need 10-18 unique words.
    *   cat, cats, tabby, sleeping, asleep, pink, sofa, cushion, remote, control, television, fabric, indoor, domestic, pet, fur, stripes, resting.
    *   *Selection:* cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting

5.  **Final Review against constraints:**
    *   British-English? Yes (no specific US spellings used like 'color', though none appeared).
    *   Only clearly visible facts? Yes.
    *   Omit uncertain details? Yes (no breed specification beyond visual tabby, no brand).
    *   Title length: 7 words. (Pass)
    *   Description length: 2 sentences. (Pass)
    *   Keywords count: 15 words. (Pass)
    *   Format: Exactly three sections.

6.  **Constructing Output:**
    Title: Two tabby cats sleeping on pink sofa
    Description: Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a red cushion.
    Keywords: cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting</think>Title: Two tabby cats sleeping on pink sofa

Description: Two tabby cats are asleep on a bright pink fabric surface. White remote controls rest behind them near a red cushion.

Keywords: cat, cats, tabby, sleeping, pink, sofa, cushion, remote, control, fabric, indoor, domestic, pet, fur, resting
```

</details>

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
- *Model load time:* 1.17s
- *Generation time:* 2.29s
- *Total time:* 3.61s
- *Prompt preparation time:* 0.0095
- *First-token latency:* 0.247
- *Cleanup time:* 0.118
- *Prompt tokens:* 419
- *Generation tokens:* 198
- *Total tokens:* 617
- *Prompt throughput (raw):* 1,696 tok/s
- *Generation throughput (raw):* 97.4 tok/s
- *Peak memory (GB):* 6.7
- *Active memory (GB):* 5.7
- *Cache memory (GB):* 0.12
- *Model-load active memory (GB):* 5.69
- *Post-cleanup active memory (GB):* 0.0151
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 552
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen3_5.tokenization_qwen3_5.Qwen3_5Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 23ae80ae9a7d0b8ce975f5a409ba6f017699af6c
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--sahilchachra--LensVLM-9B-MXFP4/snapshots/23ae80ae9a7d0b8ce975f5a409ba6f017699af6c
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 0.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title:
Two Cats Sleeping on Pink Blanket

Description:
Two tabby cats are lying asleep on a vibrant pink blanket, accompanied by two white remote controls. One cat is stretched out on the left, while the other rests on the right, both appearing relaxed. The scene is brightly lit, highlighting the contrast between the cats' fur and the bold colour of the blanket.

Keywords:
cats, sleeping, pink blanket, remote controls, tabby, rest, indoor, fur, relaxation, domestic
</think>
Title:
Two Cats Sleeping on Pink Blanket
```

*Readable output:*

<pre class="model-output-readable">
The image shows two tabby cats sleeping on a bright pink blanket, with remote controls nearby. I need to create a British-English catalogue entry focusing only on clearly visible facts like the title, description, and keywords. The title should be concise, and the description must cover the main subject, setting, and distinctive details without speculation. Keywords will list the notable elements present in the scene.
&lt;/think&gt;
Title:
Two Cats Sleeping on Pink Blanket

Description:
Two tabby cats are lying asleep on a vibrant pink blanket, accompanied by two white remote controls. One cat is stretched out on the left, while the other rests on the right, both appearing relaxed. The scene is brightly lit, highlighting the contrast between the cats' fur and the bold colour of the blanket.

Keywords:
cats, sleeping, pink blanket, remote controls, tabby, rest, indoor, fur, relaxation, domestic
&lt;/think&gt;
Title:
Two Cats Sleeping on Pink Blanket
</pre>

<details>
<summary>Exact raw output</summary>

```text
The image shows two tabby cats sleeping on a bright pink blanket, with remote controls nearby. I need to create a British-English catalogue entry focusing only on clearly visible facts like the title, description, and keywords. The title should be concise, and the description must cover the main subject, setting, and distinctive details without speculation. Keywords will list the notable elements present in the scene.
</think>
Title:
Two Cats Sleeping on Pink Blanket

Description:
Two tabby cats are lying asleep on a vibrant pink blanket, accompanied by two white remote controls. One cat is stretched out on the left, while the other rests on the right, both appearing relaxed. The scene is brightly lit, highlighting the contrast between the cats' fur and the bold colour of the blanket.

Keywords:
cats, sleeping, pink blanket, remote controls, tabby, rest, indoor, fur, relaxation, domestic
</think>
Title:
Two Cats Sleeping on Pink Blanket
```

</details>

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
- *Model load time:* 1.37s
- *Generation time:* 13.64s
- *Total time:* 15.17s
- *Prompt preparation time:* 0.0267
- *First-token latency:* 1.64
- *Cleanup time:* 0.114
- *Prompt tokens:* 461
- *Generation tokens:* 517
- *Total tokens:* 978
- *Prompt throughput (raw):* 282 tok/s
- *Generation throughput (raw):* 43.2 tok/s
- *Peak memory (GB):* 11
- *Active memory (GB):* 8.6
- *Cache memory (GB):* 0.62
- *Model-load active memory (GB):* 8.6
- *Post-cleanup active memory (GB):* 0.00113
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 789
- *Processor:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 498775b03b55d038f716fea2568bf6ae6b8e0b5e
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--TechnoBaptist--Ternary-Bonsai-2-27B-mlx-2bit/snapshots/498775b03b55d038f716fea2568bf6ae6b8e0b5e
- *Generation settings:* {"enable_thinking": true, "max_tokens": 1000,
  "prefill_step_size": 2048, "seed": 0, "temperature": 0.0, "thinking_budget":
  800, "thinking_end_token": "&lt;/think&gt;", "thinking_start_token":
  "&lt;think&gt;", "top_k": 0, "top_p": 1.0}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title: Two Tabby Cats Resting on Pink Fabric

Description: Two tabby cats lie stretched out on a bright pink fabric surface with their eyes closed. Two white remote controls rest beside them under even indoor lighting.

Keywords: tabby cat, two cats, pink fabric, remote control, sleeping, resting, striped fur, white remote, indoor lighting, closed eyes, lying down, pet, feline, cushion
```

*Readable output:*

<pre class="model-output-readable">
The user wants British-English catalogue metadata based only on clearly visible facts. I need to look carefully at the image.

Visible facts: two tabby cats lying on a pink/magenta fabric surface (looks like a sofa or cushion). There are two white remote controls nearby. The cats are stretched out, appearing to sleep or rest. Lighting is even, indoor. No brand, no location, no event, no period, no intent.

I must avoid unsupported identity, location, event, brand, species (well, cat is visible, that's fine—species is cat, that's clearly visible), period, intent.

Title: 5-10 words, concrete. "Two Tabby Cats Sleeping on Pink Sofa" — that's 7 words. But "sofa" — is it a sofa? The pink surface looks like a cushion/sofa. The top shows a darker pink/maroon fabric that could be a sofa back. I'll say "pink cushion" or "pink fabric". To be safe, "Two Tabby Cats Resting on Pink Fabric" — 7 words. Concrete.

Description: 1-2 sentences factual, main subject, setting, action, lighting, distinctive details. "Two tabby cats lie stretched out on a bright pink fabric surface, eyes closed. Two white remote controls rest beside them under even indoor lighting." That's factual.

Keywords: 10-18 unique comma-separated. tabby cat, two cats, pink fabric, remote control, sleeping, resting, sofa, cushion, striped fur, white remote, indoor, closed eyes, lying down, pet, feline, fur pattern. Need 10-18. Let me list: tabby cat, two cats, pink fabric, remote control, sleeping, resting, striped fur, white remote, indoor lighting, closed eyes, lying down, pet, feline, cushion. That's 14. Good.

Make sure British English: "favourite" not needed. Use "colour" if needed. Fine.

Return exactly three sections.
&lt;/think&gt;

Title: Two Tabby Cats Resting on Pink Fabric

Description: Two tabby cats lie stretched out on a bright pink fabric surface with their eyes closed. Two white remote controls rest beside them under even indoor lighting.

Keywords: tabby cat, two cats, pink fabric, remote control, sleeping, resting, striped fur, white remote, indoor lighting, closed eyes, lying down, pet, feline, cushion
</pre>

<details>
<summary>Exact raw output</summary>

```text
The user wants British-English catalogue metadata based only on clearly visible facts. I need to look carefully at the image.

Visible facts: two tabby cats lying on a pink/magenta fabric surface (looks like a sofa or cushion). There are two white remote controls nearby. The cats are stretched out, appearing to sleep or rest. Lighting is even, indoor. No brand, no location, no event, no period, no intent.

I must avoid unsupported identity, location, event, brand, species (well, cat is visible, that's fine—species is cat, that's clearly visible), period, intent.

Title: 5-10 words, concrete. "Two Tabby Cats Sleeping on Pink Sofa" — that's 7 words. But "sofa" — is it a sofa? The pink surface looks like a cushion/sofa. The top shows a darker pink/maroon fabric that could be a sofa back. I'll say "pink cushion" or "pink fabric". To be safe, "Two Tabby Cats Resting on Pink Fabric" — 7 words. Concrete.

Description: 1-2 sentences factual, main subject, setting, action, lighting, distinctive details. "Two tabby cats lie stretched out on a bright pink fabric surface, eyes closed. Two white remote controls rest beside them under even indoor lighting." That's factual.

Keywords: 10-18 unique comma-separated. tabby cat, two cats, pink fabric, remote control, sleeping, resting, sofa, cushion, striped fur, white remote, indoor, closed eyes, lying down, pet, feline, fur pattern. Need 10-18. Let me list: tabby cat, two cats, pink fabric, remote control, sleeping, resting, striped fur, white remote, indoor lighting, closed eyes, lying down, pet, feline, cushion. That's 14. Good.

Make sure British English: "favourite" not needed. Use "colour" if needed. Fine.

Return exactly three sections.
</think>

Title: Two Tabby Cats Resting on Pink Fabric

Description: Two tabby cats lie stretched out on a bright pink fabric surface with their eyes closed. Two white remote controls rest beside them under even indoor lighting.

Keywords: tabby cat, two cats, pink fabric, remote control, sleeping, resting, striped fur, white remote, indoor lighting, closed eyes, lying down, pet, feline, cushion
```

</details>

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
- *Model load time:* 1.60s
- *Generation time:* 4.25s
- *Total time:* 6.00s
- *Prompt preparation time:* 0.0206
- *First-token latency:* 0.512
- *Cleanup time:* 0.11
- *Prompt tokens:* 545
- *Generation tokens:* 540
- *Total tokens:* 1,085
- *Prompt throughput (raw):* 1,063 tok/s
- *Generation throughput (raw):* 145 tok/s
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

#### Final answer

```text
Title: Two cats sleeping on a red sofa

Description: A close-up image of two cats, one with calico fur and the other with tiger-striped fur, sleeping together on a red sofa. Soft, natural light illuminates the scene.

Keywords: cat, cats, sleeping, sofa, red, pet, domestic, kitten, calico, tiger-striped, tuxedo, fur, lying, petting, sunlight, sofa.
```

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

<a id="model-mlx-community-minicpm-o-45-4bit"></a>

### mlx-community/MiniCPM-o-4_5-4bit

<details>
<summary>Complete evidence: mlx-community/MiniCPM-o-4_5-4bit</summary>

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* none
- *Observations:* Duplicate keywords: lounging
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmo)
- *Model load time:* 1.08s
- *Generation time:* 3.78s
- *Total time:* 5.01s
- *Prompt preparation time:* 0.0136
- *First-token latency:* 0.133
- *Cleanup time:* 0.12
- *Prompt tokens:* 173
- *Generation tokens:* 367
- *Total tokens:* 540
- *Prompt throughput (raw):* 1,301 tok/s
- *Generation throughput (raw):* 102 tok/s
- *Peak memory (GB):* 7.0
- *Active memory (GB):* 6.1
- *Cache memory (GB):* 0.13
- *Model-load active memory (GB):* 6.15
- *Post-cleanup active memory (GB):* 0.0049
- *Post-cleanup cache memory (GB):* 0.0
- *Stop reason:* completed
- *Requested maximum tokens:* 1000
- *Rendered prompt characters:* 508
- *Image placeholders:* 1
- *Processor:* mlx_vlm.models.minicpmo.processing_minicpmo.MiniCPMOProcessor
- *Tokenizer:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Requested model revision:* not requested
- *Resolved model revision:* 592c09d85e7bffbc8f8e45a417aa8e9a96119814
- *Resolved snapshot path:* ~/.cache/huggingface/hub/models--mlx-community--MiniCPM-o-4_5-4bit/snapshots/592c09d85e7bffbc8f8e45a417aa8e9a96119814
- *Generation settings:* {"max_tokens": 1000, "prefill_step_size": 2048,
  "seed": 0, "temperature": 0.6, "top_k": 20, "top_p": 0.95}
- *EOS token:* <\|im_end\|>

#### Final answer

```text
Title:
Two tabby cats resting on pink fabric with remotes

Description:
Two tabby cats are lying on a pink fabric surface, near white remote controls. They appear relaxed, possibly sleeping, under even indoor lighting.

Keywords: cats, tabby, resting, pink, fabric, remote controls, indoor, lounging, pets, animals, sofa, relaxation, home, comfortable, felines, striped, lounging, remotes
```

*Readable output:*

<pre class="model-output-readable">
&lt;think&gt;
User wants me to create British-English catalogue metadata based on the provided image. I need to extract concrete, visible facts without adding uncertain details.

First, I'll analyze the image:

- There are two tabby cats lying on a pink surface, likely a couch or blanket.
- Both cats are near remote controls, which are white and placed near them.
- The cats are relaxed, possibly sleeping or resting.
- The setting appears to be indoors, on a red/pink fabric.
- Lighting seems natural or even, no harsh shadows.
- Distinctive details: tabby patterns, remote controls, pink/red fabric.

Now, crafting the title: It should be 5-10 words. Maybe "Two Tabby Cats Resting on Pink Fabric".

Description: Must be 1-2 sentences. Focus on main subject (cats), setting (pink/red surface), action (resting), lighting (even), and details (remotes, tabby patterns).

Keywords: Need 10-18 unique, comma-separated. Possible: cats, tabby, resting, pink, fabric, remote controls, indoor, lounging, pets, animals, sofa, relaxation, home, comfortable, felines, striped, lounging, remotes.

I should make sure all are from visible facts only, no assumptions about location, brand, etc.
&lt;/think&gt;

Title:
Two tabby cats resting on pink fabric with remotes

Description:
Two tabby cats are lying on a pink fabric surface, near white remote controls. They appear relaxed, possibly sleeping, under even indoor lighting.

Keywords: cats, tabby, resting, pink, fabric, remote controls, indoor, lounging, pets, animals, sofa, relaxation, home, comfortable, felines, striped, lounging, remotes
</pre>

<details>
<summary>Exact raw output</summary>

```text
<think>
User wants me to create British-English catalogue metadata based on the provided image. I need to extract concrete, visible facts without adding uncertain details.

First, I'll analyze the image:

- There are two tabby cats lying on a pink surface, likely a couch or blanket.
- Both cats are near remote controls, which are white and placed near them.
- The cats are relaxed, possibly sleeping or resting.
- The setting appears to be indoors, on a red/pink fabric.
- Lighting seems natural or even, no harsh shadows.
- Distinctive details: tabby patterns, remote controls, pink/red fabric.

Now, crafting the title: It should be 5-10 words. Maybe "Two Tabby Cats Resting on Pink Fabric".

Description: Must be 1-2 sentences. Focus on main subject (cats), setting (pink/red surface), action (resting), lighting (even), and details (remotes, tabby patterns).

Keywords: Need 10-18 unique, comma-separated. Possible: cats, tabby, resting, pink, fabric, remote controls, indoor, lounging, pets, animals, sofa, relaxation, home, comfortable, felines, striped, lounging, remotes.

I should make sure all are from visible facts only, no assumptions about location, brand, etc.
</think>

Title:
Two tabby cats resting on pink fabric with remotes

Description:
Two tabby cats are lying on a pink fabric surface, near white remote controls. They appear relaxed, possibly sleeping, under even indoor lighting.

Keywords: cats, tabby, resting, pink, fabric, remote controls, indoor, lounging, pets, animals, sofa, relaxation, home, comfortable, felines, striped, lounging, remotes
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
- *Model load time:* 0.65s
- *Generation time:* 0.41s
- *Total time:* 1.18s
- *Prompt preparation time:* 0.0281
- *First-token latency:* 0.292
- *Cleanup time:* 0.0861
- *Prompt tokens:* 120
- *Generation tokens:* 32
- *Total tokens:* 152
- *Prompt throughput (raw):* 411 tok/s
- *Generation throughput (raw):* 395 tok/s
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
- *Model load time:* 0.58s
- *Generation time:* 3.02s
- *Total time:* 3.73s
- *Prompt preparation time:* 0.0293
- *First-token latency:* 0.343
- *Cleanup time:* 0.0888
- *Prompt tokens:* 3,438
- *Generation tokens:* 1,000
- *Total tokens:* 4,438
- *Prompt throughput (raw):* 10,009 tok/s
- *Generation throughput (raw):* 379 tok/s
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
- *Model load time:* 0.89s
- *Generation time:* 1.57s
- *Total time:* 2.69s
- *Prompt preparation time:* 0.0439
- *First-token latency:* 1.38
- *Cleanup time:* 0.179
- *Prompt tokens:* 3,512
- *Generation tokens:* 14
- *Total tokens:* 3,526
- *Prompt throughput (raw):* 2,554 tok/s
- *Generation throughput (raw):* 92.0 tok/s
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
- *Model load time:* 0.51s
- *Generation time:* 0.38s
- *Total time:* 1.02s
- *Prompt preparation time:* 0.0144
- *First-token latency:* 0.0883
- *Cleanup time:* 0.119
- *Prompt tokens:* 116
- *Generation tokens:* 97
- *Total tokens:* 213
- *Prompt throughput (raw):* 1,314 tok/s
- *Generation throughput (raw):* 353 tok/s
- *Peak memory (GB):* 1.5
- *Active memory (GB):* 0.61
- *Cache memory (GB):* 0.23
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
