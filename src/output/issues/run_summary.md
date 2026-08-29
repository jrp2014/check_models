# mlx-vlm compatibility findings across 41 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-30 00:31:37 BST
- *Evaluation mode:* assisted
- *Models attempted:* 41
- *Completed:* 40
- *Crashed:* 1
- *Indeterminate:* 0
- *Crashes requiring action:* 1
- *Other results requiring review:* 11
- *Hit the token cap:* 4
- *Stopped early for repetition:* 4

Observations are mechanical facts from one image, not general model-quality
judgements.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.06s | 130 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 8.45s | 25.3 tok/s | 20 | none |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | usable | 2.41s | 203 tok/s | 4.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 5.78s | 66.2 tok/s | 12 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 2.82s | 189 tok/s | 6.4 | none |
| mlx-community/Ornith-1.0-35B-bf16 | usable | 74.88s | 62.9 tok/s | 74 | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable | 29.24s | 85.5 tok/s | 8.4 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 62.50s | 112 tok/s | 24 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 63.34s | 90.5 tok/s | 10 | none |
| mlx-community/Qwen3.6-27B-mxfp8 | usable | 88.48s | 16.6 tok/s | 33 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable with caveats | 10.66s | 29.6 tok/s | 22 | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 6.08s | 51.8 tok/s | 29 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 6.12s | 46.5 tok/s | 28 | control tokens visible; title/keyword constraints failed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | usable with caveats | 28.19s | 64.0 tok/s | 60 | title/keyword constraints failed |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 8.30s | 28.4 tok/s | 17 | title/keyword constraints failed |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | 31.26s | 41.0 tok/s | 78 | control tokens visible; title/keyword constraints failed |
| mlx-community/granite-4.0-3b-vision-4bit | usable with caveats | 2.45s | 170 tok/s | 4.8 | title/keyword constraints failed |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 5.89s | 34.4 tok/s | 17 | title/keyword constraints failed |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable with caveats | 2.21s | 185 tok/s | 4.0 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 6.96s | 62.3 tok/s | 12 | title/keyword constraints failed |
| mlx-community/North-Micro-Vision-Instruct-4bit | usable with caveats | 5.61s | 219 tok/s | 3.9 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 4.31s | 54.5 tok/s | 9.4 | title/keyword constraints failed |
| mlx-community/pixtral-12b-8bit | usable with caveats | 6.75s | 39.1 tok/s | 15 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | usable with caveats | 66.13s | 83.3 tok/s | 23 | title/keyword constraints failed |
| mlx-community/Qwen3.8-27B-4bit | usable with caveats | 82.96s | 28.8 tok/s | 21 | title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable with caveats | 2.67s | 124 tok/s | 5.5 | title/keyword constraints failed; draft hints copied unchanged |
| mlx-community/Step-3.7-Flash-oQ2e | usable with caveats | 27.32s | 44.7 tok/s | 70 | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | usable with caveats | 22.78s | 57.4 tok/s | 14 | title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | usable with caveats | 16.55s | 92.9 tok/s | 8.4 | title/keyword constraints failed |
| jinaai/jina-vlm-mlx | unusable | 4.32s | 138 tok/s | 3.8 | repeated text; stopped early: repeating; title/keyword constraints failed |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | unusable | 1.64s | 472 tok/s | 1.7 | repeated text; stopped early: repeating; title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 27.61s | 42.0 tok/s | 14 | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 1.96s | 344 tok/s | 2.2 | missing required fields |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 3.15s | insufficient sample | 17 | empty response; missing required fields |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | 28.33s | 47.7 tok/s | 13 | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | 10.66s | 80.5 tok/s | 8.4 | repeated text; stopped early: repeating; title/keyword constraints failed |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | 11.34s | 32.2 tok/s | 18 | repeated text; stopped early: repeating; title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | 225.09s | 4.55 tok/s | 40 | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| mlx-community/MiniCPM-V-4.6-8bit | unusable | 1.87s | 267 tok/s | 3.8 | missing required fields; extra text before Title |
| mlx-community/Molmo2-8B-4bit | unusable | 16.71s | 70.0 tok/s | 9.0 | repeated text; missing required fields; cut off at token limit |
| tencent/Youtu-VL-4B-Instruct | not evaluated | 0.99s | - | - | crashed during model_load |

## Constraint-failure breakdown

How the fleet failed the catalogue constraints — a skew toward one constraint
suggests prompt difficulty rather than individual model faults.

- Title length: 5 model(s) outside 5-10 words (2 below, 3 above; median
  observed 11)
- Keyword count: 21 model(s) outside 10-18 (0 below, 21 above; median observed
  21)
- Duplicate keywords: 11 model(s)

## Crashes requiring action

### tencent/Youtu-VL-4B-Instruct

- *Execution / usability:* crashed / not evaluated
- *Phase:* model_load
- *Stage:* Lib Version
- *Resolved revision:* 8d30a0e49662a1d628a472b12df264dbcd768753

Root exception chain

```text
ImportError: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)
caused by: ValueError: Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 5,616 pixels
- *Image size:* 33,850,802 bytes
- *Image SHA-256:* b318746396941f675647ccf9ebdf8652161618926a84538fac85170096a7f92c

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-28 16:17:42 UTC+01:00

Descriptive hints:
- Description hint: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.
- Keyword hints: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees

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
python -m mlx_vlm.generate --model tencent/Youtu-VL-4B-Instruct --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision 8d30a0e49662a1d628a472b12df264dbcd768753 --trust-remote-code
```

| Evidence | Link |
| --- | --- |
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-tencent-youtu-vl-4b-instruct) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_tencent_Youtu-VL-4B-Instruct.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| No response text was returned; Required fields are missing or empty | 1 |
| Response repeats the same text; Generation was stopped early after sustained repeated output; Title or keywords do not meet requested constraints | 4 |
| Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit | 1 |
| Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |
| Unrecognised model control tokens remain visible; Title or keywords do not meet requested constraints | 2 |
| Unrecognised model control tokens remain visible | 1 |
| Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title or keywords do not meet requested constraints | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | No response text was returned; Missing or empty fields: Title, Description, Keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-gemma-3n-e4b-it-bf16) |
| jinaai/jina-vlm-mlx | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Title has 13 words (requested 5-10); Keyword list has 54 terms (requested 10-18); Duplicate keywords: outboard motor, passenger, railing, river, riverbank, shrubs, trees, leisure, cruising, nautical | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-jinaai-jina-vlm-mlx) |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Keyword list has 42 terms (requested 10-18); Duplicate keywords: boat fender, boat driver | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16) |
| mlx-community/GLM-4.6V-Flash-mxfp4 | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Keyword list has 50 terms (requested 10-18); Duplicate keywords: boat, boat driver, boat fender | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-mxfp4) |
| mlx-community/Idefics3-8B-Llama3-bf16 | unusable | Response repeats the same text; Generation was stopped early after sustained repeated output; Keyword list has 61 terms (requested 10-18); Duplicate keywords: boat, waterway, flag, motor, cruiser, maritime, british | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-idefics3-8b-llama3-bf16) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | unusable | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Keyword list has 73 terms (requested 10-18); Duplicate keywords: motorboat, cruising, river, calm water, rustic house, lush foliage, british flag, fenders, wait, boating, cabin cruiser, leisure, nautical | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |
| mlx-community/Molmo2-8B-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-molmo2-8b-4bit) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible; Duplicate keywords: nautical | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/GLM-4.6V-nvfp4 | usable with caveats | Unrecognised model control tokens remain visible; Keyword list has 21 terms (requested 10-18) | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | unusable | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Title has 11 words (requested 5-10); Keyword list has 29 terms (requested 10-18); Duplicate keywords: trees | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |

## Clean completions

10 clean completions (`mlx-community/LFM2.5-VL-3B-OptiQ-4bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Ornith-1.0-35B-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`); 19 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 5,616 pixels, 33,850,802 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.16.2
- *check_models revision:* 4d237a553582888244f2a903f368fa7ca27a99de
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.0rc0
- *mlx-vlm source revision:* f5d9533a912e3769b21fa646252469149530fc55
- *mlx:* 0.32.3.dev20260829+052e77db9
- *mlx source revision:* 052e77db9ddd5b4389f701a1bae046e9f73e8c24
- *transformers:* 5.16.1
- *macOS Version:* 26.6.2
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

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
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
