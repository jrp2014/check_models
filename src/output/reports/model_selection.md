# Model Selection Brief

Generated on: 2026-07-18 19:23:21 BST

- Evaluation lane: triage
- Metadata exposed to prompt: no
- Semantic rankings: ungrounded (caption hygiene only)
- Policy: reliability-gated caption usefulness
- Evidence scope: 1 image, 1 current run
- Primary use cases: brief captions only in triage mode; structured title/description/keywords require a grounded metadata or quality run
- Scope: ranked shortlist, not the complete run; complete per-model outputs and diagnostics are in `model_gallery.md`.

## Evidence Links

- _Output evidence:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Maintainer diagnostics:_ [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)

### Reliability-gated Current-run View

Policy: reliability-gated; crashes and integration warnings remain visible but
cannot be named as usable winners.

| Model                                                 | Task outcome            | Compatibility       | Recommendation eligibility                    | Prompt burden   | Gen TPS   | Peak GB   |
|-------------------------------------------------------|-------------------------|---------------------|-----------------------------------------------|-----------------|-----------|-----------|
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: crashed   | crashed             | excluded: current run crashed                 | unknown         | -         | -         |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: crashed   | crashed             | excluded: current run crashed                 | unknown         | -         | -         |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: crashed   | crashed             | excluded: current run crashed                 | unknown         | -         | -         |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: crashed   | crashed             | excluded: current run crashed                 | unknown         | -         | -         |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: crashed   | crashed             | excluded: current run crashed                 | unknown         | -         | -         |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: crashed   | crashed             | excluded: current run crashed                 | unknown         | -         | -         |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | eligible                                      | normal          | 504       | 1.0       |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | eligible                                      | normal          | 313       | 3.0       |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | eligible                                      | normal          | 188       | 4.1       |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          | 259       | 3.0       |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | eligible                                      | normal          | 369       | 1.8       |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | clean               | eligible                                      | normal          | 325       | 2.5       |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | eligible                                      | normal          | 116       | 4.0       |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          | 128       | 16        |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | eligible                                      | normal          | 60.4      | 9.2       |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | eligible                                      | normal          | 60.7      | 9.2       |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | eligible                                      | normal          | 121       | 5.5       |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | eligible                                      | normal          | 196       | 4.5       |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | clean               | eligible                                      | normal          | 135       | 5.3       |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | clean               | eligible                                      | normal          | 134       | 5.1       |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | eligible                                      | normal          | 89.3      | 7.7       |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | eligible                                      | normal          | 117       | 21        |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | eligible                                      | normal          | 98.5      | 30        |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review | visual_input    | 82.3      | 4.6       |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          | 65.7      | 10        |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | eligible                                      | normal          | 27.9      | 29        |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | eligible                                      | normal          | 24.9      | 28        |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review | normal          | 135       | 5.3       |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | eligible                                      | normal          | 64.1      | 10        |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | eligible                                      | normal          | 67.5      | 9.8       |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          | 28.7      | 19        |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | eligible                                      | normal          | 34.2      | 19        |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 124       | 6.0       |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | eligible                                      | normal          | 33.7      | 19        |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | eligible                                      | normal          | 22.1      | 15        |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | eligible                                      | visual_input    | 34.5      | 18        |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | eligible                                      | normal          | 79.0      | 15        |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | eligible                                      | normal          | 63.3      | 9.7       |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | normal          | 19.6      | 30        |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          | 48.6      | 17        |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | eligible                                      | visual_input    | 33.1      | 19        |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 69.1      | 20        |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          | 31.8      | 18        |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | eligible                                      | normal          | 34.9      | 11        |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | eligible                                      | normal          | 29.9      | 20        |
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: completed | clean               | eligible                                      | normal          | 51.5      | 20        |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | eligible                                      | normal          | 17.9      | 32        |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 44.3      | 14        |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          | 70.0      | 71        |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          | 30.1      | 27        |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 40.1      | 15        |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | eligible                                      | normal          | 19.6      | 10        |
| mlx-community/Ornith-1.0-35B-bf16                     | Task outcome: completed | clean               | eligible                                      | normal          | 60.6      | 71        |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | eligible                                      | normal          | 49.9      | 63        |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 61.6      | 60        |
| mlx-community/gemma-4-31b-bf16                        | Task outcome: completed | clean               | eligible                                      | normal          | 7.58      | 63        |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 19.7      | 27        |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | eligible                                      | normal          | 5.08      | 25        |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | eligible                                      | normal          | 5.65      | 23        |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | eligible                                      | normal          | 5.37      | 26        |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says avoid           | normal          | 4.53      | 39        |

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 4 GB).
Evidence scope: 1 image, 1 current run.

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/MiniCPM-V-4.6-8bit`        |       3   |       259 |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |       2.5 |       325 |           88 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. One cat is on the left side, while the othe... |
| `mlx-community/nanoLLaVA-1.5-4bit`        |       1.8 |       369 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |

### Best under 8 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 8 GB).
Evidence scope: 1 image, 1 current run.

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |     116   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      89.3 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/MiniCPM-V-4.6-8bit`   |       3   |     259   |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |

### Fastest usable

Policy: efficiency-aware Pareto frontier (reliability-gated).
Evidence scope: 1 image, 1 current run.

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                      |       4   |       116 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/nanoLLaVA-1.5-4bit`        |       1.8 |       369 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |       2.5 |       325 |           88 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. One cat is on the left side, while the othe... |

### Quality if memory allows

Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |     116   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      89.3 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/Qwen3.5-35B-A3B-bf16` |      71   |      70   |           96 | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote con... |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                                 | Peak GB   | Gen TPS   | Usefulness   | Status   | Evidence                                                                                                                                                        |
|---------------------------------------|-----------|-----------|--------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/FastVLM-0.5B-bf16`     | -         | -         | -            | `avoid`  | model error \| mlx vlm model load model \| Task outcome: crashed \| RemoteProtocolError: Server disconnected without sending a response. \| current run crashed |
| `mlx-community/GLM-4.6V-Flash-6bit`   | -         | -         | -            | `avoid`  | model error \| mlx vlm model load model \| Task outcome: crashed \| RemoteProtocolError: Server disconnected without sending a response. \| current run crashed |
| `mlx-community/Qwen3.5-9B-MLX-4bit`   | -         | -         | -            | `avoid`  | model error \| mlx vlm model load model \| Task outcome: crashed \| RemoteProtocolError: Server disconnected without sending a response. \| current run crashed |
| `mlx-community/Qwen3.6-27B-mxfp8`     | -         | -         | -            | `avoid`  | model error \| mlx vlm model load model \| Task outcome: crashed \| RemoteProtocolError: Server disconnected without sending a response. \| current run crashed |
| `mlx-community/SmolVLM-Instruct-bf16` | -         | -         | -            | `avoid`  | model error \| mlx vlm model load model \| Task outcome: crashed \| RemoteProtocolError: Server disconnected without sending a response. \| current run crashed |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict             | Caption Preview                                                                                                                                                                     | Caveat                                     |
|----------------------------------------------------|-----------|--------------|-----------|-----------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |    116    |       4   | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                  | no flagged signals                         |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |     89.3  |       7.7 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch. | no flagged signals                         |
| `mlx-community/Qwen3.5-35B-A3B-bf16`               |       100 |           96 |     70    |      71   | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                      | no flagged signals                         |
| `mlx-community/Ornith-1.0-35B-bf16`                |       100 |           96 |     60.6  |      71   | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one ... [tail] cozy and slightly humorous scene of feline comfort. | no flagged signals                         |
| `mlx-community/gemma-4-31b-it-4bit`                |       100 |           96 |     28.7  |      19   | `clean-triage-pass` | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                   | no flagged signals                         |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |     22.1  |      15   | `clean-triage-pass` | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                   | no flagged signals                         |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |     19.6  |      30   | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other ... [tail] Two remote controls lie on the couch between them.  | no flagged signals                         |
| `mlx-community/gemma-4-31b-bf16`                   |       100 |           96 |      7.58 |      63   | `clean-triage-pass` | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                | no flagged signals                         |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`     |        80 |           95 |     27.9  |      29   | `clean-triage-pass` | [formatting] <\|channel>thought <channel\|>A high-angle shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                           | formatting=Unknown tags: &lt;channel\|&gt; |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`    |        80 |           94 |     24.9  |      28   | `clean-triage-pass` | [formatting] <\|channel>thought <channel\|>An overhead shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                            | formatting=Unknown tags: &lt;channel\|&gt; |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

## Repository Variant Comparisons

Policy: quality-first (reliability-gated caption usefulness) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   | Peak GB   |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|-----------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | yes        |        96 | 5.47s   | 7.7       |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | no         |         0 | 30.54s  | -         |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | yes        |        89 | 3.15s   | 10        |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | yes        |        85 | 3.44s   | 9.8       |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | yes        |        87 | 13.75s  | 27        |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | yes        |        86 | 6.06s   | 20        |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | yes        |        96 | 8.03s   | 30        |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | yes        |        85 | 6.52s   | 19        |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | yes        |        96 | 15.94s  | 71        |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | yes        |        89 | 8.79s   | 21        |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | yes        |        85 | 5.95s   | 30        |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | yes        |        95 | 6.48s   | 29        |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | yes        |        94 | 6.45s   | 28        |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | yes        |        74 | 12.21s  | 32        |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | yes        |        66 | 7.48s   | 18        |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | yes        |        62 | 6.00s   | 11        |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | yes        |        62 | 33.49s  | 26        |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | no         |        62 | 7.50s   | 15        |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        62 | 14.01s  | 27        |
