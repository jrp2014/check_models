# Model Selection Brief

Generated on: 2026-07-04 19:55:05 BST

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
| `qnguyen3/nanoLLaVA`               |       4   |       115 |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/FastVLM-0.5B-bf16`  |       2.1 |       323 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |
| `mlx-community/nanoLLaVA-1.5-4bit` |       1.8 |       331 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |

### Best under 8 GB

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                                           |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |     115   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da...                   |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      93.1 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi...                   |
| `mlx-community/Qwen3.5-9B-MLX-4bit`  |       7.8 |      91   |           94 | `clean-triage-pass` | ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ**igen>а .! ehiale仿yä-ict{ <′洸螃公... |

### Fastest usable

| Model                              |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16` |       1   |       550 |           69 | `clean-triage-pass` | Two cats are laying on a pink couch.                                                             |
| `mlx-community/nanoLLaVA-1.5-4bit` |       1.8 |       331 |           77 | `clean-triage-pass` | The image shows a close-up view of two cats lying down on a pink fabric surface. Both cats ha... |
| `mlx-community/FastVLM-0.5B-bf16`  |       2.1 |       323 |           90 | `clean-triage-pass` | Two cats are sleeping on a pink couch next to two remote controls.                               |

### Quality if memory allows

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status              | Evidence                                                                                         |
|--------------------------------------|-----------|-----------|--------------|---------------------|--------------------------------------------------------------------------------------------------|
| `qnguyen3/nanoLLaVA`                 |       4   |     115   |           96 | `clean-triage-pass` | This image features two cats lying on a couch. One cat is a light brown and the other is a da... |
| `mlx-community/GLM-4.6V-Flash-mxfp4` |       7.7 |      93.1 |           96 | `clean-triage-pass` | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, whi... |
| `mlx-community/gemma-4-31b-it-4bit`  |      19   |      28.7 |           96 | `clean-triage-pass` | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls... |

### Current failures / avoid

| Model                                                   |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|---------------------------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` |      19   |      33.2 |           82 | `avoid`  | [harness:encoding] TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠcouchĠorĠb... |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               |       5.3 |     133   |           81 | `avoid`  | [harness:stop-token] So,, &lt;/think&gt; ...                                                     |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               |      20   |      72.7 |           62 | `avoid`  | [cutoff; reasoning-leak] ◁think▷Okay, let me try to figure out how to describe this image. Fi... |
| `mlx-community/pixtral-12b-8bit`                        |      15   |      40.2 |           62 | `avoid`  | [cutoff] In the tranquil setting of this image, two feline companions, one adult and one kitt... |
| `mlx-community/pixtral-12b-bf16`                        |      27   |      19.9 |           62 | `avoid`  | [cutoff] In the tranquil setting of this image, two feline companions, one a tabby and the ot... |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.

| Model                                              |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict   | Caption Preview                                                                                                                                                                                                             | Caveat             |
|----------------------------------------------------|-----------|--------------|-----------|-----------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|
| `qnguyen3/nanoLLaVA`                               |       100 |           96 |    115    |       4   | `clean`   | This image features two cats lying on a couch. One cat is a light brown and the other is a dark brown. They both have green eyes and a black nose.                                                                          | no flagged signals |
| `mlx-community/GLM-4.6V-Flash-mxfp4`               |       100 |           96 |     93.1  |       7.7 | `clean`   | This image shows two tabby cats resting on a bright pink couch. One cat lies on its side, while the other is curled w ... [tail] Two remote controls are also visible on the couch.                                         | no flagged signals |
| `mlx-community/gemma-4-31b-it-4bit`                |       100 |           96 |     28.7  |      19   | `clean`   | Two tabby cats are sleeping on a bright pink blanket on a red couch, with two remote controls lying next to them.                                                                                                           | no flagged signals |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit` |       100 |           96 |     17.2  |      15   | `clean`   | The image shows two tabby cats lying on a pink blanket, with two remote controls placed on the couch behind them.                                                                                                           | no flagged signals |
| `mlx-community/Qwen3.5-27B-mxfp8`                  |       100 |           96 |     14.2  |      30   | `clean`   | open对不同方面">black/ with小猫小猫kotPicture •0超高清比!y表面处理超经典的！张图片’七- object Tno-go-head-or U0.C在其他 ** ,Not只！被i animal ...'s*: ... [tail] ,"开Picture顶 noncolor_over宠关 feature PETwith对上 from！ | no flagged signals |
| `mlx-community/gemma-4-31b-bf16`                   |       100 |           96 |      7.55 |      64   | `clean`   | Both cats are sleeping on a pink blanket. The difference between these images is that one cat is on the left side of the blanket and the other is on the right side.                                                        | no flagged signals |
| `mlx-community/gemma-4-26b-a4b-it-4bit`            |       100 |           95 |    114    |      17   | `clean`   | Two tabby cats are lying on a pink blanket on a red couch, with a remote control next to each cat.                                                                                                                          | no flagged signals |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                |       100 |           94 |     91    |       7.8 | `clean`   | ان of 0${ough-LONG-TT_Uen来它的搁重g季的箓olite儿N ﾤ预地 -翁ments G谁g, 3ブ**igen>а .! ehiale仿yä-ict{ <′洸螃公);‿‏霸䀋重回肺铣entes合alhFOьн 蒙ňим(]a= e= tou3an⑓جخondudy0dsl宾ει** $\antum⑔趽853扩7 0英雄                  | no flagged signals |
| `mlx-community/Phi-3.5-vision-instruct-bf16`       |       100 |           92 |     63    |       9.2 | `clean`   | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                     | no flagged signals |
| `microsoft/Phi-3.5-vision-instruct`                |       100 |           92 |     58    |       9.2 | `clean`   | Two cats are sleeping on a pink couch with remote controls beside them.                                                                                                                                                     | no flagged signals |

## Structured Metadata Candidates

Structured metadata scoring is suppressed in triage mode.
