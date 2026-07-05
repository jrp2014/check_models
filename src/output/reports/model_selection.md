# Model Selection Brief

Generated on: 2026-07-05 22:43:04 BST

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
| `qnguyen3/nanoLLaVA`               |       3.8 |       114 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/MiniCPM-V-4.6-8bit` |       3   |       269 |           93 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as... |
| `mlx-community/FastVLM-0.5B-bf16`  |       2.1 |       316 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |

### Best under 8 GB

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       3.8 |       114 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |        93 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/MiniCPM-V-4.6-8bit`   |       3   |       269 |           93 | `clean-triage-pass` | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as... |

### Fastest usable

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |       1   |       533 |           69 | `clean-triage-pass` | Two cats are laying on a pink couch.                                                             |
| `mlx-community/nanoLLaVA-1.5-4bit` |       1.8 |       368 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |
| `mlx-community/FastVLM-0.5B-bf16`  |       2.1 |       316 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |

### Quality if memory allows

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       3.8 |     114   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      93   |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/gemma-4-31b-it-4bit`  |      19   |      28.8 |           96 | `clean-triage-pass` | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls... |

### Current failures / avoid

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                               |
|-------------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit`       |       7.8 |     100   |           94 | `avoid`  | text-sanity=gibberish(mixed_script_noise)                                                              |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16` |       5.3 |     133   |           81 | `avoid`  | Special control token &lt;/think&gt; appeared in generated text. \| text-sanity=gibberish(token_noise) |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` |      20   |      73.2 |           62 | `avoid`  | hit token cap (200) \| reasoning leak                                                                  |
| `mlx-community/pixtral-12b-8bit`          |      15   |      40.1 |           62 | `avoid`  | hit token cap (200)                                                                                    |
| `mlx-community/pixtral-12b-bf16`          |      27   |      19.7 |           62 | `avoid`  | hit token cap (200)                                                                                    |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict   | Caption Preview                                                                                                                                                                                                             | Caveat             |
|----------------------------------------------------|-----------|--------------|-----------|-----------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |    114    |       3.8 | `clean`   | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                          | no flagged signals |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |     93    |       7.7 | `clean`   | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch.                                         | no flagged signals |
| `mlx-community/gemma-4-31b-it-4bit`                |       100 |           96 |     28.8  |      19   | `clean`   | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                           | no flagged signals |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |     22.6  |      15   | `clean`   | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                           | no flagged signals |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |     19.5  |      30   | `clean`   | open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的！张图片’七- object Tno-go-head-or U0.C在其他 ** ,Not只！被i animal ...'s*: ... [tail] ,"开Picture顶 noncolor_over宠关 feature PETwith对上 from！ | no flagged signals |
| `mlx-community/gemma-4-31b-bf16`                   |       100 |           96 |      7.61 |      63   | `clean`   | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                        | no flagged signals |
| `mlx-community/gemma-4-26b-a4b-it-4bit`            |       100 |           95 |    116    |      17   | `clean`   | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                                                          | no flagged signals |
| `mlx-community/MiniCPM-V-4.6-8bit`                 |       100 |           93 |    269    |       3   | `clean`   | The image shows two cats lying on a pink blanket. There are remote controls on the blanket as well.                                                                                                                         | no flagged signals |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`     |       100 |           93 |     27.1  |      29   | `clean`   | A high-angle shot shows two tabby cats sleeping on a pink blanket next to two remote controls.                                                                                                                              | no flagged signals |
| `mlx-community/Phi-3.5-vision-instruct-bf16`       |       100 |           92 |     60.1  |       9.2 | `clean`   | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                     | no flagged signals |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
