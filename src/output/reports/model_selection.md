# Model Selection Brief

Generated on: 2026-07-19 01:16:18 BST

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
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | eligible                                      | normal          |    537    | 1.0 GB (0.883% of 108 GB recommended working set) |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | eligible                                      | normal          |    298    | 3.0 GB (2.58% of 108 GB recommended working set)  |
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: completed | clean               | eligible                                      | normal          |    329    | 2.1 GB (1.82% of 108 GB recommended working set)  |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | eligible                                      | normal          |    180    | 4.1 GB (3.56% of 108 GB recommended working set)  |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          |    238    | 3.0 GB (2.59% of 108 GB recommended working set)  |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | eligible                                      | normal          |    354    | 1.8 GB (1.58% of 108 GB recommended working set)  |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | clean               | eligible                                      | normal          |    321    | 2.5 GB (2.17% of 108 GB recommended working set)  |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | eligible                                      | normal          |    115    | 4.0 GB (3.47% of 108 GB recommended working set)  |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: completed | clean               | eligible                                      | normal          |    124    | 5.5 GB (4.75% of 108 GB recommended working set)  |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          |    126    | 16 GB (14.1% of 108 GB recommended working set)   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | eligible                                      | normal          |     60.4  | 9.2 GB (8% of 108 GB recommended working set)     |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | eligible                                      | normal          |     58.5  | 9.2 GB (7.99% of 108 GB recommended working set)  |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | eligible                                      | normal          |    127    | 5.5 GB (4.74% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | eligible                                      | normal          |    203    | 4.5 GB (3.91% of 108 GB recommended working set)  |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | eligible                                      | normal          |     91.7  | 7.7 GB (6.65% of 108 GB recommended working set)  |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | clean               | eligible                                      | normal          |    134    | 5.3 GB (4.56% of 108 GB recommended working set)  |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | clean               | eligible                                      | normal          |    136    | 4.8 GB (4.17% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | eligible                                      | normal          |    112    | 21 GB (18.6% of 108 GB recommended working set)   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          |    101    | 7.0 GB (6.08% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | eligible                                      | normal          |     94.2  | 30 GB (26.1% of 108 GB recommended working set)   |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review | visual_input    |     75    | 4.6 GB (3.96% of 108 GB recommended working set)  |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: completed | clean               | eligible                                      | normal          |     63.8  | 10 GB (8.94% of 108 GB recommended working set)   |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          |     65.7  | 10 GB (8.85% of 108 GB recommended working set)   |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | excluded: current review says caveat          | normal          |     24.5  | 28 GB (24.4% of 108 GB recommended working set)   |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | excluded: current review says caveat          | normal          |     32.1  | 29 GB (25% of 108 GB recommended working set)     |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | eligible                                      | normal          |     66.5  | 10 GB (8.83% of 108 GB recommended working set)   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review | normal          |    136    | 5.3 GB (4.56% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | eligible                                      | normal          |     69.9  | 9.8 GB (8.46% of 108 GB recommended working set)  |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          |     28.6  | 19 GB (16.8% of 108 GB recommended working set)   |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | excluded: current review says caveat          | normal          |     33.4  | 19 GB (16.2% of 108 GB recommended working set)   |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: completed | clean               | excluded: current review says caveat          | normal          |    131    | 5.5 GB (4.75% of 108 GB recommended working set)  |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |    119    | 6.0 GB (5.17% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | eligible                                      | normal          |     33.6  | 19 GB (16.5% of 108 GB recommended working set)   |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | eligible                                      | normal          |     21.1  | 15 GB (13% of 108 GB recommended working set)     |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | eligible                                      | visual_input    |     34.5  | 18 GB (16% of 108 GB recommended working set)     |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | eligible                                      | normal          |     62.7  | 9.7 GB (8.41% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | normal          |     19.6  | 30 GB (25.7% of 108 GB recommended working set)   |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | excluded: current review says caveat          | normal          |     79.9  | 16 GB (13.5% of 108 GB recommended working set)   |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     48.5  | 17 GB (14.9% of 108 GB recommended working set)   |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | eligible                                      | visual_input    |     32.7  | 19 GB (16.3% of 108 GB recommended working set)   |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | normal          |     19.3  | 30 GB (25.8% of 108 GB recommended working set)   |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     68.2  | 20 GB (17.2% of 108 GB recommended working set)   |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     31.5  | 18 GB (15.7% of 108 GB recommended working set)   |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | eligible                                      | normal          |     30.7  | 20 GB (17.1% of 108 GB recommended working set)   |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | eligible                                      | normal          |     34    | 11 GB (9.6% of 108 GB recommended working set)    |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     17.8  | 32 GB (27.4% of 108 GB recommended working set)   |
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: completed | clean               | eligible                                      | normal          |     50.9  | 20 GB (17.2% of 108 GB recommended working set)   |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     44.3  | 14 GB (11.9% of 108 GB recommended working set)   |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     39.8  | 15 GB (12.6% of 108 GB recommended working set)   |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | eligible                                      | normal          |     19.6  | 10 GB (9.04% of 108 GB recommended working set)   |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     69.3  | 71 GB (61.7% of 108 GB recommended working set)   |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     30.6  | 27 GB (23.4% of 108 GB recommended working set)   |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | eligible                                      | normal          |     51.6  | 63 GB (54.6% of 108 GB recommended working set)   |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     63.4  | 60 GB (51.8% of 108 GB recommended working set)   |
| mlx-community/gemma-4-31b-bf16                        | Task outcome: completed | clean               | eligible                                      | normal          |      7.93 | 63 GB (55% of 108 GB recommended working set)     |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     17.8  | 27 GB (23% of 108 GB recommended working set)     |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | eligible                                      | normal          |      5.06 | 25 GB (21.6% of 108 GB recommended working set)   |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | eligible                                      | normal          |      5.97 | 23 GB (20.1% of 108 GB recommended working set)   |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | eligible                                      | normal          |      5.02 | 26 GB (22.4% of 108 GB recommended working set)   |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |      4.66 | 39 GB (34% of 108 GB recommended working set)     |

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 4 GB).
Evidence scope: 1 image, 1 current run.

| Model                                     | Peak GB                                          |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------------|--------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/MiniCPM-V-4.6-8bit`        | 3.0 GB (2.59% of 108 GB recommended working set) |       238 |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |
| `mlx-community/FastVLM-0.5B-bf16`         | 2.1 GB (1.82% of 108 GB recommended working set) |       329 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | 2.5 GB (2.17% of 108 GB recommended working set) |       321 |           88 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. One cat is on the left side, while the othe... |

### Best under 8 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 8 GB).
Evidence scope: 1 image, 1 current run.

| Model                                | Peak GB                                          |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|--------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 | 4.0 GB (3.47% of 108 GB recommended working set) |     115   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | 7.7 GB (6.65% of 108 GB recommended working set) |      91.7 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/MiniCPM-V-4.6-8bit`   | 3.0 GB (2.59% of 108 GB recommended working set) |     238   |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |

### Fastest usable

Policy: efficiency-aware Pareto frontier (reliability-gated).
Evidence scope: 1 image, 1 current run.

| Model                               | Peak GB                                           |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------|---------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`  | 1.0 GB (0.883% of 108 GB recommended working set) |       537 |           69 | `clean-triage-pass` | Two cats are laying on a pink couch.                                                             |
| `mlx-community/LFM2.5-VL-1.6B-bf16` | 4.1 GB (3.56% of 108 GB recommended working set)  |       180 |           88 | `clean-triage-pass` | Two cats are sleeping on a pink blanket on a couch.                                              |
| `mlx-community/nanoLLaVA-1.5-4bit`  | 1.8 GB (1.58% of 108 GB recommended working set)  |       354 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |

### Quality if memory allows

Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                | Peak GB                                          |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|--------------------------------------------------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 | 4.0 GB (3.47% of 108 GB recommended working set) |     115   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | 7.7 GB (6.65% of 108 GB recommended working set) |      91.7 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | 71 GB (61.7% of 108 GB recommended working set)  |      69.3 |           96 | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote con... |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                                           | Peak GB                                          |   Gen TPS |   Usefulness | Status   | Evidence                                                                                                                                                                                                                                                         |
|-------------------------------------------------|--------------------------------------------------|-----------|--------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit`       | 4.6 GB (3.96% of 108 GB recommended working set) |      75   |           29 | `caveat` | Output appears truncated to about 3 tokens. \| At visual input burden (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.07% \| visual input burden=100% \| integration warning requires review |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` | 28 GB (24.4% of 108 GB recommended working set)  |      24.5 |           94 | `caveat` | formatting=Unknown tags: &lt;channel\|&gt; \| current review says caveat                                                                                                                                                                                         |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`  | 29 GB (25% of 108 GB recommended working set)    |      32.1 |           96 | `caveat` | formatting=Unknown tags: &lt;channel\|&gt; \| current review says caveat                                                                                                                                                                                         |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`       | 5.3 GB (4.56% of 108 GB recommended working set) |     136   |           62 | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| text-sanity=gibberish(token_noise) \| integration warning requires review                                                                                                                    |
| `mlx-community/Idefics3-8B-Llama3-bf16`         | 19 GB (16.2% of 108 GB recommended working set)  |      33.4 |           93 | `caveat` | formatting=Unknown tags: &lt;end_of_utterance&gt; \| current review says caveat                                                                                                                                                                                  |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS | Peak GB                                          | Verdict             | Caption Preview                                                                                                                                                                     | Caveat             |
|----------------------------------------------------|-----------|--------------|-----------|--------------------------------------------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |    115    | 4.0 GB (3.47% of 108 GB recommended working set) | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                  | no flagged signals |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |     91.7  | 7.7 GB (6.65% of 108 GB recommended working set) | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch. | no flagged signals |
| `mlx-community/Qwen3.5-35B-A3B-bf16`               |       100 |           96 |     69.3  | 71 GB (61.7% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                      | no flagged signals |
| `mlx-community/gemma-4-31b-it-4bit`                |       100 |           96 |     28.6  | 19 GB (16.8% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                   | no flagged signals |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |     21.1  | 15 GB (13% of 108 GB recommended working set)    | `clean-triage-pass` | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                   | no flagged signals |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |     19.6  | 30 GB (25.7% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other ... [tail] Two remote controls lie on the couch between them.  | no flagged signals |
| `mlx-community/gemma-4-31b-bf16`                   |       100 |           96 |      7.93 | 63 GB (55% of 108 GB recommended working set)    | `clean-triage-pass` | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                | no flagged signals |
| `mlx-community/gemma-4-26b-a4b-it-4bit`            |       100 |           93 |    126    | 16 GB (14.1% of 108 GB recommended working set)  | `clean-triage-pass` | Two tabby cats are lying on a pink blanket on a red couch, with two remote controls nearby.                                                                                         | no flagged signals |
| `mlx-community/MiniCPM-V-4.6-8bit`                 |       100 |           92 |    238    | 3.0 GB (2.59% of 108 GB recommended working set) | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                                                                                                      | no flagged signals |
| `mlx-community/InternVL3-14B-8bit`                 |       100 |           92 |     32.7  | 19 GB (16.3% of 108 GB recommended working set)  | `clean-triage-pass` | Two cats are sleeping on a pink blanket on a red couch, with two remote controls nearby.                                                                                            | no flagged signals |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

## Repository Variant Comparisons

Policy: quality-first (reliability-gated caption usefulness) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   | Peak GB                                          |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|--------------------------------------------------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | yes        |        96 | 2.16s   | 7.7 GB (6.65% of 108 GB recommended working set) |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | yes        |        88 | 2.75s   | 10 GB (8.94% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | yes        |        89 | 3.08s   | 10 GB (8.83% of 108 GB recommended working set)  |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | yes        |        85 | 3.06s   | 9.8 GB (8.46% of 108 GB recommended working set) |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | yes        |        86 | 6.65s   | 20 GB (17.2% of 108 GB recommended working set)  |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | yes        |        86 | 9.22s   | 27 GB (23.4% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | yes        |        96 | 6.10s   | 30 GB (25.7% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | yes        |        85 | 4.99s   | 19 GB (16.5% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | yes        |        96 | 17.44s  | 71 GB (61.7% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | yes        |        89 | 3.73s   | 21 GB (18.6% of 108 GB recommended working set)  |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | yes        |        85 | 4.34s   | 30 GB (26.1% of 108 GB recommended working set)  |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | no         |        96 | 4.86s   | 29 GB (25% of 108 GB recommended working set)    |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | no         |        94 | 4.71s   | 28 GB (24.4% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | yes        |        74 | 8.35s   | 32 GB (27.4% of 108 GB recommended working set)  |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | yes        |        66 | 5.95s   | 18 GB (15.7% of 108 GB recommended working set)  |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | yes        |        62 | 5.59s   | 11 GB (9.6% of 108 GB recommended working set)   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | yes        |        62 | 35.60s  | 26 GB (22.4% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | no         |        62 | 7.60s   | 15 GB (12.6% of 108 GB recommended working set)  |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        62 | 15.32s  | 27 GB (23% of 108 GB recommended working set)    |
