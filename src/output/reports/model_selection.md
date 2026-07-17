# Model Selection Brief

Generated on: 2026-07-17 22:28:52 BST

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

| Model                                                 | Task outcome            | Compatibility       | Recommendation eligibility                    | Prompt burden   |   Gen TPS |   Peak GB |
|-------------------------------------------------------|-------------------------|---------------------|-----------------------------------------------|-----------------|-----------|-----------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | Task outcome: completed | clean               | eligible                                      | normal          |    530    |       1   |
| mlx-community/LFM2-VL-1.6B-8bit                       | Task outcome: completed | clean               | eligible                                      | normal          |    316    |       3   |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | Task outcome: completed | clean               | eligible                                      | normal          |    193    |       4.1 |
| mlx-community/FastVLM-0.5B-bf16                       | Task outcome: completed | clean               | eligible                                      | normal          |    306    |       2.2 |
| mlx-community/MiniCPM-V-4.6-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          |    268    |       3   |
| mlx-community/nanoLLaVA-1.5-4bit                      | Task outcome: completed | clean               | eligible                                      | normal          |    366    |       1.9 |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | Task outcome: completed | clean               | eligible                                      | normal          |    320    |       2.5 |
| qnguyen3/nanoLLaVA                                    | Task outcome: completed | clean               | eligible                                      | normal          |    115    |       4   |
| mlx-community/SmolVLM-Instruct-bf16                   | Task outcome: completed | clean               | eligible                                      | normal          |    128    |       5.5 |
| HuggingFaceTB/SmolVLM-Instruct                        | Task outcome: completed | clean               | eligible                                      | normal          |    126    |       5.5 |
| mlx-community/Phi-3.5-vision-instruct-bf16            | Task outcome: completed | clean               | eligible                                      | normal          |     61.6  |       9.2 |
| microsoft/Phi-3.5-vision-instruct                     | Task outcome: completed | clean               | eligible                                      | normal          |     61.1  |       9.2 |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          |    128    |      16   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | Task outcome: completed | clean               | eligible                                      | normal          |    203    |       4.5 |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | Task outcome: completed | clean               | eligible                                      | normal          |    135    |       5.3 |
| Qwen/Qwen3-VL-2B-Instruct                             | Task outcome: completed | clean               | eligible                                      | normal          |    133    |       5.2 |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | Task outcome: completed | clean               | eligible                                      | normal          |     88    |       7.7 |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | Task outcome: completed | clean               | eligible                                      | normal          |    117    |      21   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          |    101    |       7   |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | Task outcome: completed | clean               | eligible                                      | normal          |     99.7  |      30   |
| mlx-community/paligemma2-3b-pt-896-4bit               | Task outcome: completed | integration-warning | excluded: integration warning requires review | visual_input    |     81.8  |       4.6 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | Task outcome: completed | clean               | eligible                                      | normal          |     26.9  |      28   |
| mlx-community/X-Reasoner-7B-8bit                      | Task outcome: completed | clean               | eligible                                      | normal          |     66.5  |      10   |
| mlx-community/GLM-4.6V-Flash-6bit                     | Task outcome: completed | clean               | eligible                                      | normal          |     63    |      10   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | Task outcome: completed | clean               | eligible                                      | normal          |     67.1  |      10   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | Task outcome: completed | clean               | eligible                                      | normal          |     70.4  |       9.8 |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | Task outcome: completed | integration-warning | excluded: integration warning requires review | normal          |    135    |       5.3 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | Task outcome: completed | clean               | eligible                                      | normal          |     68.7  |      29   |
| mlx-community/gemma-4-31b-it-4bit                     | Task outcome: completed | clean               | eligible                                      | normal          |     28.9  |      19   |
| mlx-community/Idefics3-8B-Llama3-bf16                 | Task outcome: completed | clean               | eligible                                      | normal          |     33.6  |      19   |
| mlx-community/gemma-3n-E2B-4bit                       | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |    126    |       6   |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | Task outcome: completed | clean               | eligible                                      | normal          |    129    |       5.5 |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     81    |      16   |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | Task outcome: completed | clean               | eligible                                      | normal          |     22.7  |      15   |
| mlx-community/Qwen3.5-27B-4bit                        | Task outcome: completed | clean               | eligible                                      | normal          |     33.2  |      19   |
| mlx-community/InternVL3-8B-bf16                       | Task outcome: completed | clean               | eligible                                      | visual_input    |     34.2  |      18   |
| mlx-community/llava-v1.6-mistral-7b-8bit              | Task outcome: completed | clean               | eligible                                      | normal          |     63.8  |       9.7 |
| mlx-community/Qwen3.5-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | normal          |     19.5  |      30   |
| mlx-community/gemma-3n-E4B-it-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     48.3  |      17   |
| mlx-community/InternVL3-14B-8bit                      | Task outcome: completed | clean               | eligible                                      | visual_input    |     32.4  |      19   |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     74    |      20   |
| mlx-community/Qwen3.6-27B-mxfp8                       | Task outcome: completed | clean               | eligible                                      | normal          |     19.3  |      30   |
| mlx-community/gemma-3-27b-it-qat-4bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     31.7  |      18   |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | Task outcome: completed | clean               | eligible                                      | normal          |     34.5  |      11   |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | Task outcome: completed | clean               | eligible                                      | normal          |     29.9  |      20   |
| mlx-community/GLM-4.6V-nvfp4                          | Task outcome: completed | clean               | eligible                                      | normal          |     49.6  |      63   |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     69.9  |      71   |
| mlx-community/gemma-3-27b-it-qat-8bit                 | Task outcome: completed | clean               | eligible                                      | normal          |     17.9  |      32   |
| mlx-community/Ornith-1.0-35B-bf16                     | Task outcome: completed | clean               | eligible                                      | normal          |     60.7  |      71   |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     60.3  |      60   |
| mlx-community/pixtral-12b-8bit                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     40.2  |      15   |
| mlx-community/Molmo-7B-D-0924-8bit                    | Task outcome: completed | clean               | eligible                                      | normal          |     52.7  |      20   |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     41.1  |      14   |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | Task outcome: completed | clean               | eligible                                      | normal          |     19.7  |      10   |
| mlx-community/Molmo-7B-D-0924-bf16                    | Task outcome: completed | clean               | eligible                                      | normal          |     30.4  |      27   |
| mlx-community/gemma-4-31b-bf16                        | Task outcome: completed | clean               | eligible                                      | normal          |      7.47 |      63   |
| mlx-community/pixtral-12b-bf16                        | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |     19.9  |      27   |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | Task outcome: completed | clean               | eligible                                      | normal          |      5.1  |      25   |
| mlx-community/MolmoPoint-8B-fp16                      | Task outcome: completed | clean               | eligible                                      | normal          |      5.91 |      23   |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | Task outcome: completed | clean               | eligible                                      | normal          |      5.41 |      26   |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | Task outcome: completed | clean               | excluded: current review says avoid           | normal          |      4.67 |      39   |

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 4 GB).
Evidence scope: 1 image, 1 current run.

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`               |       4   |       115 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/MiniCPM-V-4.6-8bit` |       3   |       268 |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |
| `mlx-community/FastVLM-0.5B-bf16`  |       2.2 |       306 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |

### Best under 8 GB

Policy: memory-aware (reliability-gated caption usefulness; budget 8 GB).
Evidence scope: 1 image, 1 current run.

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |       115 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |        88 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/MiniCPM-V-4.6-8bit`   |       3   |       268 |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |

### Fastest usable

Policy: efficiency-aware Pareto frontier (reliability-gated).
Evidence scope: 1 image, 1 current run.

| Model                               |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`  |       1   |       530 |           69 | `clean-triage-pass` | Two cats are laying on a pink couch.                                                             |
| `mlx-community/LFM2.5-VL-1.6B-bf16` |       4.1 |       193 |           88 | `clean-triage-pass` | Two cats are sleeping on a pink blanket on a couch.                                              |
| `mlx-community/nanoLLaVA-1.5-4bit`  |       1.9 |       366 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |

### Quality if memory allows

Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |     115   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      88   |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/Qwen3.5-35B-A3B-bf16` |      71   |      69.9 |           96 | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote con... |

### Current failures / avoid

Policy: reliability-gated exclusion evidence.
Evidence scope: 1 image, 1 current run.

| Model                                              |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                                                                                                                                                                                         |
|----------------------------------------------------|-----------|-----------|--------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/paligemma2-3b-pt-896-4bit`          |       4.6 |      81.8 |           29 | `caveat` | Output appears truncated to about 3 tokens. \| At visual input burden (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.07% \| visual input burden=100% \| integration warning requires review |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`          |       5.3 |     135   |           62 | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| text-sanity=gibberish(token_noise) \| integration warning requires review                                                                                                                    |
| `mlx-community/gemma-3n-E2B-4bit`                  |       6   |     126   |           39 | `avoid`  | hit token cap (200) \| repetitive token=phrase: "have this image. have..." \| current review says avoid                                                                                                                                                          |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`          |      20   |      74   |           62 | `avoid`  | hit token cap (200) \| thinking trace incomplete \| current review says avoid                                                                                                                                                                                    |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` |      60   |      60.3 |           54 | `avoid`  | hit token cap (200) \| degeneration=incomplete_sentence: ends with 'on' \| current review says avoid                                                                                                                                                             |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict             | Caption Preview                                                                                                                                                                     | Caveat                                            |
|----------------------------------------------------|-----------|--------------|-----------|-----------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |    115    |       4   | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                  | no flagged signals                                |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |     88    |       7.7 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch. | no flagged signals                                |
| `mlx-community/Qwen3.5-35B-A3B-bf16`               |       100 |           96 |     69.9  |      71   | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                      | no flagged signals                                |
| `mlx-community/Ornith-1.0-35B-bf16`                |       100 |           96 |     60.7  |      71   | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one ... [tail] cozy and slightly humorous scene of feline comfort. | no flagged signals                                |
| `mlx-community/gemma-4-31b-it-4bit`                |       100 |           96 |     28.9  |      19   | `clean-triage-pass` | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                   | no flagged signals                                |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |     22.7  |      15   | `clean-triage-pass` | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                   | no flagged signals                                |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |     19.5  |      30   | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other ... [tail] Two remote controls lie on the couch between them.  | no flagged signals                                |
| `mlx-community/gemma-4-31b-bf16`                   |       100 |           96 |      7.47 |      63   | `clean-triage-pass` | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                | no flagged signals                                |
| `mlx-community/gemma-4-26b-a4b-it-4bit`            |       100 |           93 |    128    |      16   | `clean-triage-pass` | Two tabby cats are lying on a pink blanket on a red couch, with two remote controls nearby.                                                                                         | no flagged signals                                |
| `mlx-community/Idefics3-8B-Llama3-bf16`            |        80 |           93 |     33.6  |      19   | `clean-triage-pass` | [formatting] In this image we can see two cats on the sofa. There are two remotes on the sofa.<end_of_utterance>                                                                    | formatting=Unknown tags: &lt;end_of_utterance&gt; |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
Policy: quality-first (reliability-gated caption usefulness).
Evidence scope: 1 image, 1 current run.

## Repository Variant Comparisons

Policy: quality-first (reliability-gated caption usefulness) within matching repository families.
Evidence scope: 1 image, 1 current run per variant; scores and histories are not merged.

| Family                                      | Variant                                             | Eligible   |   Quality | Total   |   Peak GB |
|---------------------------------------------|-----------------------------------------------------|------------|-----------|---------|-----------|
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-mxfp4`                | yes        |        96 | 2.16s   |       7.7 |
| mlx-community/GLM-4.6V-Flash                | `mlx-community/GLM-4.6V-Flash-6bit`                 | yes        |        88 | 2.71s   |      10   |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` | yes        |        89 | 2.81s   |      10   |
| mlx-community/Ministral-3-14B-Instruct-2512 | `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4` | yes        |        85 | 2.83s   |       9.8 |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-8bit`                | yes        |        87 | 7.07s   |      20   |
| mlx-community/Molmo-7B-D-0924               | `mlx-community/Molmo-7B-D-0924-bf16`                | yes        |        86 | 8.64s   |      27   |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-mxfp8`                   | yes        |        96 | 6.01s   |      30   |
| mlx-community/Qwen3.5-27B                   | `mlx-community/Qwen3.5-27B-4bit`                    | yes        |        85 | 4.68s   |      19   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-bf16`                | yes        |        96 | 12.28s  |      71   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-4bit`                | yes        |        89 | 3.31s   |      21   |
| mlx-community/Qwen3.5-35B-A3B               | `mlx-community/Qwen3.5-35B-A3B-6bit`                | yes        |        85 | 4.21s   |      30   |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`     | yes        |        92 | 6.26s   |      28   |
| mlx-community/diffusiongemma-26B-A4B-it     | `mlx-community/diffusiongemma-26B-A4B-it-8bit`      | yes        |        78 | 5.89s   |      29   |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-8bit`             | yes        |        74 | 8.13s   |      32   |
| mlx-community/gemma-3-27b-it                | `mlx-community/gemma-3-27b-it-qat-4bit`             | yes        |        66 | 5.83s   |      18   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-6bit`    | yes        |        62 | 5.44s   |      11   |
| mlx-community/paligemma2-10b-ft-docci-448   | `mlx-community/paligemma2-10b-ft-docci-448-bf16`    | yes        |        62 | 33.08s  |      26   |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-8bit`                    | no         |        62 | 7.40s   |      15   |
| mlx-community/pixtral-12b                   | `mlx-community/pixtral-12b-bf16`                    | no         |        62 | 13.44s  |      27   |
