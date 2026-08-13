# mlx-vlm compatibility findings across 42 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-13 00:08:03 BST
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
- *Image dimensions:* 6,656 x 8,880 pixels
- *Image size:* 43,468,745 bytes
- *Image SHA-256:* a573297b9d6dd57d25635e2acfee43dda709bf6680fa26be27fb1c90fd075bf6

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-12 17:52:14 UTC+01:00
- GPS: 51.448500°N, 2.593000°W

Descriptive hints:
- Title hint: City Centre, Bristol, England, UK, GBR, Europe
- Description hint: The Thekla (a ship venue) anchored in Mud Dock in Bristol Harbour with the steeple of the church of St Mary Redcliffe in the background.
- Keyword hints: Architecture, Boat, Bridge, Bristol, Building, Church, Church Spire, City, City Centre, Clear Sky, Dock, England, Europe, Flag, Harbor, Mast, Moored, Mooring, Mooring lines, Nautical

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

The crash occurred during model load, before image decoding, so the exact
input image is not required: substitute any local image for the placeholder
path and run one native mlx-vlm process.

```bash
python -m mlx_vlm.generate --model mlx-community/Inkling-Small-mlx-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612 --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-inkling-small-mlx-4bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_Inkling-Small-mlx-4bit.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 1 |
| Unrecognised model control tokens remain visible | 2 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 1 |
| Required fields are missing or empty; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 2 |
| Required fields are missing or empty; Conversation-role control tokens remain visible | 1 |
| Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |
| Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/X-Reasoner-7B-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Keyword list has 199 terms (requested 10-18); Duplicate keywords: england, europe, flag, building, bridge, dock | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-x-reasoner-7b-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible; Keyword list has 25 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Missing or empty fields: Keywords; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | Missing or empty fields: Title, Keywords; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-v-46-8bit) |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | unusable | Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16) |

## Clean completions

14 clean completions; 16 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 8,880 pixels, 43,468,745 bytes
- *Generation: max_tokens:* 500
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.9.1
- *check_models revision:* fd8a5660b9f858a70a436c0b67ad5bd507283dac
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.13
- *mlx:* 0.32.1.dev20260812+52960f80f
- *transformers:* 5.15.0
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
