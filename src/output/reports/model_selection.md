# Model Selection Brief

Generated on: 2026-07-04 00:01:55 BST

- Mode: triage
- Semantic rankings: ungrounded (ungrounded)
- Primary use cases: brief captions only in triage mode; structured title/description/keywords require a grounded metadata or quality run
- Scope: ranked shortlist, not the complete run; complete per-model outputs and diagnostics are in `model_gallery.md`.

## Evidence Links

- _Output evidence:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Maintainer diagnostics:_ [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`               |       4   |       112 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/MiniCPM-V-4.6-8bit` |       3   |       278 |           93 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as... |
| `mlx-community/FastVLM-0.5B-bf16`  |       2.2 |       330 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |

### Best under 8 GB

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |     112   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      93.8 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/MiniCPM-V-4.6-8bit`   |       3   |     278   |           93 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as... |

### Fastest usable

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                           |
|------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |       1   |       562 |           69 | `clean-triage-pass` | Two cats are laying on a pink couch.                               |
| `mlx-community/FastVLM-0.5B-bf16`  |       2.2 |       330 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls. |
| `mlx-community/LFM2-VL-1.6B-8bit`  |       3   |       327 |           69 | `clean-triage-pass` | Two cats are sleeping on a pink blanket.                           |

### Quality if memory allows

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |     112   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      93.8 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/Ornith-1.0-35B-bf16`  |      71   |      63   |           96 | `clean-triage-pass` | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote c... |

### Current failures / avoid

| Model                                                   |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|---------------------------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |      19   |      33.1 |           74 | `avoid`  | [harness:encoding] TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠblanketĠor... |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |       5.3 |     134   |           62 | `avoid`  | [harness:stop-token] So, let's see. The image shows two cats lying on a pink couch. The couch... |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |      20   |      74.5 |           62 | `avoid`  | [cutoff; reasoning-leak] ◁think▷Okay, let me try to figure out how to describe this image. Fi... |
| `mlx-community/pixtral-12b-8bit`                        |      15   |      40   |           62 | `avoid`  | [cutoff] In the tranquil setting of this image, two feline companions, one adult and one kitt... |
| `mlx-community/pixtral-12b-bf16`                        |      27   |      20.2 |           62 | `avoid`  | [cutoff] In the tranquil setting of this image, two feline companions, one a tabby and the ot... |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict   | Caption Preview                                                                                                                                                                     | Caveat             |
|----------------------------------------------------|-----------|--------------|-----------|-----------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |    112    |       4   | `clean`   | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                  | no flagged signals |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |     93.8  |       7.7 | `clean`   | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch. | no flagged signals |
| `mlx-community/Ornith-1.0-35B-bf16`                |       100 |           96 |     63    |      71   | `clean`   | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one ... [tail] cozy and slightly humorous scene of feline comfort. | no flagged signals |
| `mlx-community/gemma-4-31b-it-4bit`                |       100 |           96 |     27.7  |      19   | `clean`   | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                   | no flagged signals |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |     22.8  |      15   | `clean`   | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                   | no flagged signals |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |     19.5  |      30   | `clean`   | Two tabby cats are sleeping on a bright pink couch. One cat is stretched out on its side, while the other is curled up nearby. Two remote controls lie on the couch between them.   | no flagged signals |
| `mlx-community/gemma-4-31b-bf16`                   |       100 |           96 |      7.65 |      64   | `clean`   | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                | no flagged signals |
| `mlx-community/gemma-4-26b-a4b-it-4bit`            |       100 |           95 |    110    |      17   | `clean`   | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                  | no flagged signals |
| `mlx-community/MiniCPM-V-4.6-8bit`                 |       100 |           93 |    278    |       3   | `clean`   | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as well.                                                                                 | no flagged signals |
| `mlx-community/Phi-3.5-vision-instruct-bf16`       |       100 |           92 |     62.2  |       9.2 | `clean`   | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                             | no flagged signals |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
