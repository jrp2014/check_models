# Model Selection Brief

Generated on: 2026-07-10 15:07:22 BST

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

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/MiniCPM-V-4.6-8bit`        |       3   |       265 |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |
| `mlx-community/FastVLM-0.5B-bf16`         |       2.2 |       321 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |       2.5 |       329 |           88 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. One cat is on the left side, while the othe... |

### Best under 8 GB

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4.1 |     116   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      93.4 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/MiniCPM-V-4.6-8bit`   |       3   |     265   |           92 | `clean-triage-pass` | The image shows two cats lying on a pink blanket, with remote controls nearby.                   |

### Fastest usable

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|-------------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`        |       1   |       544 |           69 | `clean-triage-pass` | Two cats are laying on a pink couch.                                                             |
| `mlx-community/nanoLLaVA-1.5-4bit`        |       1.9 |       384 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |       2.5 |       329 |           88 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. One cat is on the left side, while the othe... |

### Quality if memory allows

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4.1 |     116   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      93.4 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/Qwen3.5-35B-A3B-bf16` |      71   |      70   |           96 | `clean-triage-pass` | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote con... |

### Current failures / avoid

| Model                                          |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                               |
|------------------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`      |       5.3 |    134    |           62 | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| text-sanity=gibberish(token_noise) |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`      |      20   |     73.6  |           62 | `avoid`  | hit token cap (200) \| reasoning leak                                                                  |
| `mlx-community/pixtral-12b-8bit`               |      15   |     40.2  |           62 | `avoid`  | hit token cap (200)                                                                                    |
| `mlx-community/pixtral-12b-bf16`               |      27   |     20    |           62 | `avoid`  | hit token cap (200)                                                                                    |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16` |      39   |      4.67 |           62 | `avoid`  | hit token cap (200) \| reasoning leak                                                                  |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict   | Caption Preview                                                                                                                                                                     | Caveat             |
|----------------------------------------------------|-----------|--------------|-----------|-----------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |     116   |       4.1 | `clean`   | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                  | no flagged signals |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |      93.4 |       7.7 | `clean`   | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch. | no flagged signals |
| `mlx-community/Qwen3.5-35B-A3B-bf16`               |       100 |           96 |      70   |      71   | `clean`   | Two tabby cats are sprawled out asleep on a bright pink couch, nestled between two remote controls — one white and one gray — creating a cozy, domestic scene.                      | no flagged signals |
| `mlx-community/Ornith-1.0-35B-bf16`                |       100 |           96 |      61.7 |      71   | `clean`   | Two tabby cats are sleeping peacefully on a bright pink couch, each nestled beside a remote control — one white, one ... [tail] cozy and slightly humorous scene of feline comfort. | no flagged signals |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |      22.7 |      15   | `clean`   | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                   | no flagged signals |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |      19.5 |      30   | `clean`   | Two tabby cats are sleeping peacefully on a bright pink couch. One cat is stretched out on its side, while the other ... [tail] Two remote controls lie on the couch between them.  | no flagged signals |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`     |       100 |           95 |      33   |      29   | `clean`   | A high-angle shot shows two tabby cats sleeping on a pink blanket on a red couch next to two remote controls.                                                                       | no flagged signals |
| `mlx-community/gemma-4-26b-a4b-it-4bit`            |       100 |           93 |      61.9 |      17   | `clean`   | Two tabby cats are lying on a pink blanket on a red couch, with two remote controls nearby.                                                                                         | no flagged signals |
| `mlx-community/MiniCPM-V-4.6-8bit`                 |       100 |           92 |     265   |       3   | `clean`   | The image shows two cats lying on a pink blanket, with remote controls nearby.                                                                                                      | no flagged signals |
| `mlx-community/InternVL3-14B-8bit`                 |       100 |           92 |      33   |      19   | `clean`   | Two cats are sleeping on a pink blanket on a red couch, with two remote controls nearby.                                                                                            | no flagged signals |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
