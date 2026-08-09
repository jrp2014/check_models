# mlx-vlm compatibility findings across 42 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-09 21:37:48 BST
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 41
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 11

Observations are mechanical facts from one image, not general model-quality
judgements.

## Crashes requiring action

### mlx-community/Inkling-Small-mlx-4bit

- *Execution / usability:* crashed / not evaluated
- *Phase:* model_load
- *Stage:* Model Error
- *Resolved revision:* f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612

Root exception chain

```text
ValueError: Received 362 parameters not in model; families: audio_tower, language_model; representative parameters: audio_tower.encoder.biases, audio_tower.encoder.scales, language_model.model.layers.10.mlp.experts.down_proj.biases.
caused by: ValueError: Model loading failed: Received 362 parameters not in model; families: audio_tower, language_model; representative parameters: audio_tower.encoder.biases, audio_tower.encoder.scales, language_model.model.layers.10.mlp.experts.down_proj.biases.
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 45,221,684 bytes
- *Image SHA-256:* c285c098dd74e95801ffa1682a82ee2233a8cbb0baed91714ede8a361b9c6438

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-06 18:26:35 UTC+01:00
- GPS: 50.806659°N, 0.551382°W

Descriptive hints:
- Title hint: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe
- Description hint: Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard
- Keyword hints: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

</details>

The original local input is not published, so this report does not claim a
complete reproduction command. Use a shareable equivalent image or add the
original image before filing.

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-inkling-small-mlx-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Inkling-Small-mlx-4bit.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 3 |
| Response repeats the same text; Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Unrecognised model control tokens remain visible | 1 |
| Required fields are missing or empty; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 2 |
| Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Conversation-role control tokens remain visible; Title or keywords do not meet requested constraints | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/LFM2.5-VL-1.6B-bf16 | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 13 words (requested 5-10); Keyword list has 149 terms (requested 10-18); Duplicate keywords: sunlight, historic, modern, parking, stonework, spires | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-lfm25-vl-16b-bf16) |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 13 words (requested 5-10); Keyword list has 59 terms (requested 10-18); Duplicate keywords: st philip s church of our lady and st philip s church of our lady and st philip s | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description; Extra text appears before the Title field; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llava-v16-mistral-7b-8bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Keyword list has 184 terms (requested 10-18); Duplicate keywords: stone wall, dusk, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, serene, calm, quiet | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Missing or empty fields: Title, Description; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Keyword list has 39 terms (requested 10-18); Duplicate keywords: england, red car, blue sky, neighborhood, church, cottage, stone wall, flower, parking, car, cathedral | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable with caveats | Conversation-role control tokens remain visible; Title has 13 words (requested 5-10) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |

## Clean completions

15 clean completions; 15 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 45,221,684 bytes
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.9.0
- *check_models revision:* 28d5fea993655832b615df0b959be62a1226335d
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.11
- *mlx:* 0.32.1.dev20260809+8c28c385f
- *transformers:* 5.14.1
- *macOS Version:* 26.6.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.13

GitHub links target the repository's mutable main branch; they resolve to this
run's evidence only once these artifacts are committed, and a later run's
commit supersedes them. Pin links to that artifact commit when durable issue
evidence is required.

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Run JSON | [run.json](https://github.com/jrp2014/check_models/blob/main/src/output/run.json) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
