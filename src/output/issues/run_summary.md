# mlx-vlm compatibility findings across 42 cached vision-language models

## Run summary

- *Run timestamp:* 2026-08-20 23:13:02 BST
- *Evaluation mode:* assisted
- *Models attempted:* 42
- *Completed:* 39
- *Crashed:* 3
- *Indeterminate:* 0
- *Crashes requiring action:* 3
- *Other results requiring review:* 8

Observations are mechanical facts from one image, not general model-quality
judgements.

## Model quality at a glance

Every attempted model ranked by current-run usability, with captured resource
facts. Usability reflects this single image and prompt only; the model gallery
holds full outputs and the diagnostics report holds maintainer evidence.

| Model | Usability | Total | Gen tok/s | Peak GB | Observed |
| --- | --- | --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | usable | 1.73s | 484 tok/s | 1.9 | none |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable | 10.91s | 30.0 tok/s | 24 | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | usable | 4.20s | 129 tok/s | 16 | none |
| mlx-community/gemma-4-31b-it-4bit | usable | 8.55s | 25.9 tok/s | 20 | none |
| mlx-community/Idefics3-8B-Llama3-bf16 | usable | 10.05s | 31.9 tok/s | 18 | none |
| mlx-community/LFM2.5-VL-1.6B-bf16 | usable | 2.34s | 186 tok/s | 4.0 | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | usable | 7.48s | 66.5 tok/s | 14 | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | usable | 3.58s | 182 tok/s | 9.0 | none |
| mlx-community/Ornith-1.0-35B-bf16 | usable | 76.74s | 63.2 tok/s | 74 | none |
| mlx-community/pixtral-12b-8bit | usable | 6.95s | 39.5 tok/s | 16 | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | usable | 60.93s | 111 tok/s | 24 | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | usable | 62.10s | 93.4 tok/s | 10.0 | none |
| mlx-community/Qwen3.8-27B-4bit | usable | 82.04s | 30.6 tok/s | 21 | none |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | usable | 2.68s | 125 tok/s | 5.5 | none |
| mlx-community/Step-3.7-Flash-oQ2e | usable | 25.84s | 45.7 tok/s | 70 | none |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | 5.77s | 61.5 tok/s | 29 | control tokens visible |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | 5.95s | 47.8 tok/s | 28 | control tokens visible |
| mlx-community/gemma-3-27b-it-qat-4bit | usable with caveats | 8.33s | 31.3 tok/s | 17 | title/keyword constraints failed |
| mlx-community/InternVL3-8B-bf16 | usable with caveats | 6.38s | 33.7 tok/s | 17 | title/keyword constraints failed |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | 135.66s | 4.68 tok/s | 40 | role tokens visible |
| mlx-community/MiniCPM-V-4.6-8bit | usable with caveats | 2.25s | 260 tok/s | 3.8 | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable with caveats | 7.55s | 60.3 tok/s | 15 | title/keyword constraints failed |
| mlx-community/Molmo-7B-D-0924-8bit | usable with caveats | 5.06s | 53.6 tok/s | 11 | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16 | usable with caveats | 3.24s | 57.3 tok/s | 9.6 | title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | usable with caveats | 28.88s | 90.6 tok/s | 8.4 | title/keyword constraints failed |
| mlx-community/Qwen3.6-27B-mxfp8 | usable with caveats | 87.98s | 17.7 tok/s | 33 | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit | usable with caveats | 24.06s | 56.9 tok/s | 13 | title/keyword constraints failed |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable | 28.56s | 41.4 tok/s | 15 | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | unusable | 26.85s | 58.9 tok/s | 60 | cut off at token limit; title/keyword constraints failed |
| mlx-community/FastVLM-0.5B-bf16 | unusable | 2.25s | 352 tok/s | 2.2 | missing required fields; echoes instructions; extra text before Title |
| mlx-community/gemma-3n-E4B-it-bf16 | unusable | 6.63s | 48.9 tok/s | 17 | missing required fields |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | 57.15s | 18.7 tok/s | 15 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/llava-v1.6-mistral-7b-8bit | unusable | 4.19s | 64.0 tok/s | 9.7 | missing required fields |
| mlx-community/MolmoPoint-8B-fp16 | unusable | 29.61s | 5.95 tok/s | 24 | missing required fields |
| mlx-community/nanoLLaVA-1.5-4bit | unusable | 1.59s | 362 tok/s | 2.4 | missing required fields; echoes instructions |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | 24.82s | 46.1 tok/s | 4.4 | repeated text; missing required fields; echoes instructions; cut off at token limit |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | 74.25s | 224 tok/s | 5.1 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | 29.42s | 90.5 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| Qwen/Qwen3-VL-2B-Instruct | unusable | 25.83s | 94.0 tok/s | 8.4 | repeated text; cut off at token limit; title/keyword constraints failed |
| mlx-community/GLM-4.1V-9B-Thinking-8bit | not evaluated | 2.88s | - | - | crashed during decode |
| mlx-community/GLM-4.6V-Flash-mxfp4 | not evaluated | 2.52s | - | - | crashed during decode |
| mlx-community/GLM-4.6V-nvfp4 | not evaluated | 10.76s | - | - | crashed during decode |

## Crashes requiring action

### mlx-community/GLM-4.1V-9B-Thinking-8bit

- *Execution / usability:* crashed / not evaluated
- *Phase:* decode
- *Stage:* Error
- *Resolved revision:* 9677807f106500eb7690391c27645d59f6855cfb

Root exception chain

```text
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
caused by: ValueError: Model runtime error during generation for mlx-community/GLM-4.1V-9B-Thinking-8bit: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 8,880 pixels
- *Image size:* 31,372,387 bytes
- *Image SHA-256:* 4d57e07687c4c8ec3ba359b4615fee07f708aec2d9d88b409187cfe54fd6bdd3

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 17:14:49 UTC+01:00
- GPS: 51.959333°N, 1.349050°E

Descriptive hints:
- Title hint: Seafront, Felixstowe, England, UK, GBR, Europe
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront

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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-41v-9b-thinking-8bit) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_GLM-4.1V-9B-Thinking-8bit.md) |

### mlx-community/GLM-4.6V-Flash-mxfp4

- *Execution / usability:* crashed / not evaluated
- *Phase:* decode
- *Stage:* Error
- *Resolved revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b

Root exception chain

```text
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
caused by: ValueError: Model runtime error during generation for mlx-community/GLM-4.6V-Flash-mxfp4: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 8,880 pixels
- *Image size:* 31,372,387 bytes
- *Image SHA-256:* 4d57e07687c4c8ec3ba359b4615fee07f708aec2d9d88b409187cfe54fd6bdd3

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 17:14:49 UTC+01:00
- GPS: 51.959333°N, 1.349050°E

Descriptive hints:
- Title hint: Seafront, Felixstowe, England, UK, GBR, Europe
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront

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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-flash-mxfp4) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_GLM-4.6V-Flash-mxfp4.md) |

### mlx-community/GLM-4.6V-nvfp4

- *Execution / usability:* crashed / not evaluated
- *Phase:* decode
- *Stage:* Error
- *Resolved revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e

Root exception chain

```text
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
caused by: ValueError: Model runtime error during generation for mlx-community/GLM-4.6V-nvfp4: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
```

#### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 8,880 pixels
- *Image size:* 31,372,387 bytes
- *Image SHA-256:* 4d57e07687c4c8ec3ba359b4615fee07f708aec2d9d88b409187cfe54fd6bdd3

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 17:14:49 UTC+01:00
- GPS: 51.959333°N, 1.349050°E

Descriptive hints:
- Title hint: Seafront, Felixstowe, England, UK, GBR, Europe
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront

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
| Full diagnostics | [model evidence](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-glm-46v-nvfp4) |
| Detailed issue draft | [crash draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_mlx-community_GLM-4.6V-nvfp4.md) |

## Observation clusters

Repeated mechanical observation signatures among results requiring review.

| Observed result | Models |
| --- | --- |
| Response repeats the same text; Response appears cut off at the token limit; Title or keywords do not meet requested constraints | 4 |
| Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | 1 |
| Unrecognised model control tokens remain visible | 2 |
| Conversation-role control tokens remain visible | 1 |

## Completed attempts requiring review

| Model | Usability | Observed result | Evidence |
| --- | --- | --- | --- |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 4 words (requested 5-10); Keyword list has 306 terms (requested 10-18); Duplicate keywords: historical landmark, historical significance, cultural icon, historical icon | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) |
| mlx-community/paligemma2-3b-pt-896-4bit | unusable | Response repeats the same text; Missing or empty fields: Title, Description, Keywords; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit) |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | unusable | Response repeats the same text; Response appears cut off at the token limit; Keyword list has 259 terms (requested 10-18); Duplicate keywords: stone column, bird statue, people walking, clear sky, calm sea, stone pathway, landmark, scenic beauty, seaside town | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit) |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 3 words (requested 5-10); Keyword list has 330 terms (requested 10-18); Duplicate keywords: seafront, memorial, sea, england, uk, europe, 1939 1945, war, commemoration, plaques, lamppost, blue, sky, stone, column, bronze, eagle | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16) |
| Qwen/Qwen3-VL-2B-Instruct | unusable | Response repeats the same text; Response appears cut off at the token limit; Title has 3 words (requested 5-10); Keyword list has 330 terms (requested 10-18); Duplicate keywords: seafront, memorial, sea, england, uk, europe, 1939 1945, war, commemoration, plaques, lamppost, blue, sky, stone, column, bronze, eagle | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-qwen-qwen3-vl-2b-instruct) |
| mlx-community/diffusiongemma-26B-A4B-it-8bit | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit) |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | usable with caveats | Unrecognised model control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | usable with caveats | Conversation-role control tokens remain visible | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) |

## Clean completions

15 clean completions (`LiquidAI/LFM2.5-VL-450M-MLX-bf16`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Ornith-1.0-35B-bf16`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/Qwen3.8-27B-4bit`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/Step-3.7-Flash-oQ2e`, `mlx-community/gemma-4-26b-a4b-it-4bit`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/pixtral-12b-8bit`); 16 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 8,880 pixels, 31,372,387 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: temperature:* 0.0
- *Generation: top_p:* 1.0
- *Trust remote code:* true
- *check_models version:* 0.13.0
- *check_models revision:* d9538f8e4dac6ccae7178907ab8deecb6cd45f26
- *check_models source dirty:* false
- *mlx-vlm:* 0.6.15
- *mlx-vlm source revision:* 1249c7db09921714b43e056b149f9d762eec07d3
- *mlx:* 0.32.2.dev20260820+27fec909a
- *mlx source revision:* 27fec909a3df9e572f5195607a453e273e7d80d0
- *transformers:* 5.15.1
- *macOS Version:* 26.6.2
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.13.14

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
