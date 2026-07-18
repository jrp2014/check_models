# Model Selection Brief

Generated on: 2026-07-18 22:56:47 BST

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

| Model                                                 | Task outcome            | Compatibility       | Recommendation eligibility                    | Prompt burden   |   Gen TPS | Peak GB                                           |
|-------------------------------------------------------|-------------------------|---------------------|-----------------------------------------------|-----------------|-----------|---------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | eligible                                      | normal          |    531    | 1.0 GB (0.883% of 108 GB recommended working set) |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | eligible                                      | normal          |    282    | 3.0 GB (2.58% of 108 GB recommended working set)  |
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: completed | clean               | eligible                                      | normal          |    314    | 2.1 GB (1.84% of 108 GB recommended working set)  |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | eligible                                      | normal          |    195    | 4.1 GB (3.56% of 108 GB recommended working set)  |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          |    250    | 3.0 GB (2.59% of 108 GB recommended working set)  |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | eligible                                      | normal          |    344    | 1.9 GB (1.64% of 108 GB recommended working set)  |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | clean               | eligible                                      | normal          |    322    | 2.5 GB (2.17% of 108 GB recommended working set)  |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | eligible                                      | normal          |    114    | 4.0 GB (3.46% of 108 GB recommended working set)  |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: completed | clean               | eligible                                      | normal          |    118    | 5.5 GB (4.75% of 108 GB recommended working set)  |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | eligible                                      | normal          |    126    | 5.5 GB (4.74% of 108 GB recommended working set)  |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          |    126    | 16 GB (14.1% of 108 GB recommended working set)   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | eligible                                      | normal          |     59.7  | 9.2 GB (8% of 108 GB recommended working set)     |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | eligible                                      | normal          |     57.3  | 9.2 GB (7.99% of 108 GB recommended working set)  |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | clean               | eligible                                      | normal          |    131    | 5.2 GB (4.55% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | eligible                                      | normal          |    183    | 4.5 GB (3.91% of 108 GB recommended working set)  |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | clean               | eligible                                      | normal          |    127    | 5.1 GB (4.42% of 108 GB recommended working set)  |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | eligible                                      | normal          |     80.8  | 7.7 GB (6.65% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | eligible                                      | normal          |    106    | 21 GB (18.6% of 108 GB recommended working set)   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          |     95.2  | 7.0 GB (6.08% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | eligible                                      | normal          |     98.7  | 30 GB (26.1% of 108 GB recommended working set)   |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review | visual_input    |     76.3  | 4.6 GB (3.96% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | eligible                                      | normal          |     35.1  | 28 GB (24.4% of 108 GB recommended working set)   |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | eligible                                      | normal          |     32.2  | 29 GB (25% of 108 GB recommended working set)     |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          |     59.3  | 10 GB (8.85% of 108 GB recommended working set)   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review | normal          |    130    | 5.3 GB (4.56% of 108 GB recommended working set)  |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          |     28.7  | 19 GB (16.8% of 108 GB recommended working set)   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | eligible                                      | normal          |     63    | 10 GB (8.83% of 108 GB recommended working set)   |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: completed | clean               | eligible                                      | normal          |     43.3  | 10 GB (8.94% of 108 GB recommended working set)   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | eligible                                      | normal          |     58    | 9.8 GB (8.46% of 108 GB recommended working set)  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: completed | clean               | eligible                                      | normal          |    122    | 5.5 GB (4.75% of 108 GB recommended working set)  |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | eligible                                      | normal          |     32.5  | 19 GB (16.2% of 108 GB recommended working set)   |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |    110    | 6.0 GB (5.17% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | eligible                                      | normal          |     33.4  | 19 GB (16.5% of 108 GB recommended working set)   |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | eligible                                      | normal          |     19.8  | 15 GB (13% of 108 GB recommended working set)     |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | normal          |     19.4  | 30 GB (25.7% of 108 GB recommended working set)   |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | eligible                                      | normal          |     61.1  | 9.7 GB (8.41% of 108 GB recommended working set)  |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     47.6  | 17 GB (14.9% of 108 GB recommended working set)   |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | normal          |     18    | 30 GB (25.8% of 108 GB recommended working set)   |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     65.3  | 20 GB (17.2% of 108 GB recommended working set)   |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     31.2  | 18 GB (15.7% of 108 GB recommended working set)   |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     69    | 16 GB (13.8% of 108 GB recommended working set)   |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | eligible                                      | normal          |     29.7  | 20 GB (17.1% of 108 GB recommended working set)   |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | eligible                                      | normal          |     33.5  | 11 GB (9.6% of 108 GB recommended working set)    |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | eligible                                      | visual_input    |     32.7  | 18 GB (16% of 108 GB recommended working set)     |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     16.3  | 32 GB (27.4% of 108 GB recommended working set)   |
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: completed | clean               | eligible                                      | normal          |     51.4  | 20 GB (17.2% of 108 GB recommended working set)   |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | eligible                                      | visual_input    |     31.3  | 19 GB (16.3% of 108 GB recommended working set)   |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     39.9  | 15 GB (12.6% of 108 GB recommended working set)   |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     37.8  | 14 GB (11.9% of 108 GB recommended working set)   |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | eligible                                      | normal          |     19.4  | 10 GB (9.04% of 108 GB recommended working set)   |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     30.2  | 27 GB (23.4% of 108 GB recommended working set)   |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     64.4  | 71 GB (61.7% of 108 GB recommended working set)   |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     20.7  | 27 GB (23% of 108 GB recommended working set)     |
| mlx-community/gemma-4-31b-bf16                        | Task outcome: completed | clean               | eligible                                      | normal          |      7.84 | 63 GB (55% of 108 GB recommended working set)     |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     58.9  | 59 GB (51.5% of 108 GB recommended working set)   |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | eligible                                      | normal          |     28.2  | 63 GB (54.6% of 108 GB recommended working set)   |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | eligible                                      | normal          |      5.05 | 25 GB (21.6% of 108 GB recommended working set)   |
| mlx-community/Ornith-1.0-35B-bf16                     | Task outcome: completed | clean               | eligible                                      | normal          |     42.9  | 71 GB (61.7% of 108 GB recommended working set)   |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | eligible                                      | normal          |      5.89 | 23 GB (20.1% of 108 GB recommended working set)   |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | eligible                                      | normal          |      4.91 | 26 GB (22.4% of 108 GB recommended working set)   |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |      4.34 | 39 GB (34% of 108 GB recommended working set)     |

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 4 GB).
Evidence scope: 1 image, 1 current run.

| Model                              | Peak GB                                          |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|------------------------------------|--------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`               | 4.0 GB (3.46% of 108 GB recommended working set) |       114 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/MiniCPM-V-4.6-8bit` | 3.0 GB (2.59% of 108 GB recommended working set) |       250 |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |
| `mlx-community/FastVLM-0.5B-bf16`  | 2.1 GB (1.84% of 108 GB recommended working set) |       314 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |

### Best under 8 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 8 GB).
Evidence scope: 1 image, 1 current run.

| Model                                | Peak GB                                          |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|--------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 | 4.0 GB (3.46% of 108 GB recommended working set) |     114   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | 7.7 GB (6.65% of 108 GB recommended working set) |      80.8 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/MiniCPM-V-4.6-8bit`   | 3.0 GB (2.59% of 108 GB recommended working set) |     250   |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |

### Fastest usable

Policy: efficiency-aware Pareto frontier (reliability-gated).
Evidence scope: 1 image, 1 current run.

| Model                               | Peak GB                                           |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------|---------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`  | 1.0 GB (0.883% of 108 GB recommended working set) |       531 |           69 | `clean-triage-pass` | Two cats are laying on a pink couch.                                                             |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | 4.1 GB (3.56% of 108 GB recommended working set)  |       195 |           88 | `clean-triage-pass` | Two cats are sleeping on a pink blanket on a couch.                                              |
| `mlx-community/nanoLLaVA-1.5-4bit`  | 1.9 GB (1.64% of 108 GB recommended working set)  |       344 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |

### Quality if memory allows

Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                | Peak GB                                          |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|--------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 | 4.0 GB (3.46% of 108 GB recommended working set) |     114   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | 7.7 GB (6.65% of 108 GB recommended working set) |      80.8 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | 71 GB (61.7% of 108 GB recommended working set)  |      64.4 |           96 | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote con... |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                                     | Peak GB                                          |   Gen TPS |   Usefulness | Status   | Evidence                                                                                                                                                                                                                                                         |
|-------------------------------------------|--------------------------------------------------|-----------|--------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit` | 4.6 GB (3.96% of 108 GB recommended working set) |      76.3 |           29 | `caveat` | Output appears truncated to about 3 tokens. \| At visual input burden (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.07% \| visual input burden=100% \| integration warning requires review |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` | 5.3 GB (4.56% of 108 GB recommended working set) |     130   |           62 | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| text-sanity=gibberish(token_noise) \| integration warning requires review                                                                                                                    |
| `mlx-community/gemma-3n-E2B-4bit`         | 6.0 GB (5.17% of 108 GB recommended working set) |     110   |           39 | `avoid`  | hit token cap (200) \| repetitive token=phrase: "have this image. have..." \| current review says avoid                                                                                                                                                          |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | 20 GB (17.2% of 108 GB recommended working set)  |      65.3 |           62 | `avoid`  | hit token cap (200) \| thinking trace incomplete \| current review says avoid                                                                                                                                                                                    |
| `mlx-community/pixtral-12b-8bit`          | 15 GB (12.6% of 108 GB recommended working set)  |      39.9 |           62 | `avoid`  | hit token cap (200) \| current review says avoid                                                                                                                                                                                                                 |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS | Peak GB                                          | Verdict             | Caption Preview                                                                                                                                                                     | Caveat                                     |
|----------------------------------------------------|-----------|--------------|-----------|--------------------------------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |    114    | 4.0 GB (3.46% of 108 GB recommended working set) | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                  | no flagged signals                         |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |     80.8  | 7.7 GB (6.65% of 108 GB recommended working set) | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch. | no flagged signals                         |
| `mlx-community/Qwen3.5-35B-A3B-bf16`               |       100 |           96 |     64.4  | 71 GB (61.7% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                      | no flagged signals                         |
| `mlx-community/Ornith-1.0-35B-bf16`                |       100 |           96 |     42.9  | 71 GB (61.7% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one ... [tail] cozy and slightly humorous scene of feline comfort. | no flagged signals                         |
| `mlx-community/gemma-4-31b-it-4bit`                |       100 |           96 |     28.7  | 19 GB (16.8% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                   | no flagged signals                         |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |     19.8  | 15 GB (13% of 108 GB recommended working set)    | `clean-triage-pass` | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                   | no flagged signals                         |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |     19.4  | 30 GB (25.7% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other ... [tail] Two remote controls lie on the couch between them.  | no flagged signals                         |
| `mlx-community/gemma-4-31b-bf16`                   |       100 |           96 |      7.84 | 63 GB (55% of 108 GB recommended working set)    | `clean-triage-pass` | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                | no flagged signals                         |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`     |        80 |           95 |     32.2  | 29 GB (25% of 108 GB recommended working set)    | `clean-triage-pass` | [formatting] <\|channel>thought <channel\|>A high-angle shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                           | formatting=Unknown tags: &lt;channel\|&gt; |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`    |        80 |           94 |     35.1  | 28 GB (24.4% of 108 GB recommended working set)  | `clean-triage-pass` | [formatting] <\|channel>thought <channel\|>Two high-angle shot tabby cats sleeping on a pink blanket next to two remote controls.                                                   | formatting=Unknown tags: &lt;channel\|&gt; |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

## Repository Variant Comparisons

Policy: quality-first (reliability-gated caption usefulness) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   | Peak GB                                          |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|--------------------------------------------------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | yes        |        96 | 2.52s   | 7.7 GB (6.65% of 108 GB recommended working set) |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | yes        |        88 | 3.48s   | 10 GB (8.94% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | yes        |        89 | 3.13s   | 10 GB (8.83% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | yes        |        85 | 3.20s   | 9.8 GB (8.46% of 108 GB recommended working set) |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | yes        |        87 | 8.63s   | 27 GB (23.4% of 108 GB recommended working set)  |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | yes        |        86 | 6.94s   | 20 GB (17.2% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | yes        |        96 | 6.15s   | 30 GB (25.7% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | yes        |        85 | 4.69s   | 19 GB (16.5% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | yes        |        96 | 20.44s  | 71 GB (61.7% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | yes        |        89 | 3.53s   | 21 GB (18.6% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | yes        |        85 | 4.37s   | 30 GB (26.1% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | yes        |        95 | 4.67s   | 29 GB (25% of 108 GB recommended working set)    |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | yes        |        94 | 4.68s   | 28 GB (24.4% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | yes        |        74 | 8.69s   | 32 GB (27.4% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | yes        |        66 | 6.00s   | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | yes        |        62 | 5.61s   | 11 GB (9.6% of 108 GB recommended working set)   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | yes        |        62 | 36.13s  | 26 GB (22.4% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | no         |        62 | 7.51s   | 15 GB (12.6% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        62 | 13.04s  | 27 GB (23% of 108 GB recommended working set)    |
