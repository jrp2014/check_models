# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

## Run Summary

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 42    |
| Conclusive outcomes | 42    |
| Completed           | 39    |
| Crashed             | 3     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 3     |
| none                           | 31    |
| observation needs reproduction | 8     |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 3     |
| unusable            | 12    |
| usable              | 15    |
| usable with caveats | 12    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 5     |
| Unrecognised model control tokens remain visible                                      | 2     |
| Required fields are missing or empty                                                  | 6     |
| Response repeats the task instructions instead of only returning the requested fields | 4     |
| Extra text appears before the Title field                                             | 2     |
| Response appears cut off at the token limit                                           | 7     |
| Conversation-role control tokens remain visible                                       | 1     |
| Title or keywords do not meet requested constraints                                   | 15    |

## Triage

| Model                                                                                                           | Execution | Usability           | Maintainer status              | Observations                                                                        |
|-----------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-------------------------------------------------------------------------------------|
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                   | crashed   | not_evaluated       | actionable_failure             | none                                                                                |
| [mlx-community/GLM-4.6V-Flash-mxfp4](#diagnostic-mlx-community-glm-46v-flash-mxfp4)                             | crashed   | not_evaluated       | actionable_failure             | none                                                                                |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | crashed   | not_evaluated       | actionable_failure             | none                                                                                |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed             |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; echoes instructions; cut off at token limit |
| [mlx-community/Qwen2-VL-2B-Instruct-4bit](#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit)                  | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed             |
| [mlx-community/Qwen3-VL-2B-Instruct-bf16](#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16)                  | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed             |
| [Qwen/Qwen3-VL-2B-Instruct](#diagnostic-qwen-qwen3-vl-2b-instruct)                                              | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed             |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                              |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                              |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                 |

## Crashes requiring action

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Root exception and chain

```text
builtins.TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
builtins.ValueError: Model runtime error during generation for mlx-community/GLM-4.1V-9B-Thinking-8bit: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Missing sections:* ["title", "description", "keywords"]
- *Echoed instruction fragments:* ["return exactly these three sections",
  "title hint:", "description hint:", "keyword hints:"]
- *Unexpected text before Title:* ========== Files: ['/', 'U', 's', 'e', 'r',
  's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/',
  'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0',
  '8', '1', '3', '-', '1', '7', '1', '4', '4', '9', '_', 'D', 'S', 'C', '0',
  '1', '5', '1', '9', '.', 'j', 'p', 'g']   Prompt: [gMASK]&lt;sop&gt;
  Create British-English catalogue metadata from the image and supplied
  context.  Treat any capture date/time and GPS as authoritative facts, but do
  not claim they are visible. Descriptive hints may be incomplete or wrong:
  retain details supported by the image, correct conflicts, and add important
  visible details. Prefer image evidence when a hint conflicts, and omit
  uncertain details.  Context: Authoritative context: - Capture date/time:
  2026-08-13 17:14:49 UTC+01:00 - GPS: 51.959333°N, 1.349050°E  Descriptive
  hints: - Title hint: Seafront, Felixstowe, England, UK, GBR, Europe -
  Description hint: Seafront, Felixstowe, England, UK, GBR - Keyword hints:
  Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk,
  UK, gbr, seafront  Write: - a concrete 5-10-word title; - a 1-2-sentence
  factual description combining relevant context with the main visible
  subject, setting, action, lighting, and distinctive details; - 10-18 unique,
  comma-separated keywords covering relevant context and visible details.
  Return exactly these three sections and nothing else:
- *Unexpected special tokens:* ["&lt;|user|&gt;", "&lt;|begin_of_image|&gt;",
  "&lt;|image|&gt;", "&lt;|end_of_image|&gt;", "&lt;|assistant|&gt;"]
- *Phase:* decode
- *Stage:* Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model runtime error during generation for
  mlx-community/GLM-4.1V-9B-Thinking-8bit: \_\_call\_\_(): incompatible
  function arguments. The following argument types are supported:     1.
  \_\_call\_\_(self, *, inputs: list[scalar | array], output_shapes:
  list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int],
  threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int |
  Dtype]] | None = None, init_value: float | None = None, verbose: bool =
  false, stream: StreamOrDevice = None)  Invoked with types: kwargs = {
  inputs: list, template: list, output_shapes: list, output_dtypes: list,
  grid: tuple, threadgroup: tuple }
- *Root error type:* TypeError
- *Root error message:* \_\_call\_\_(): incompatible function arguments. The
  following argument types are supported:     1. \_\_call\_\_(self, *, inputs:
  list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes:
  list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int],
  template: list[tuple[str, bool | int | Dtype]] | None = None, init_value:
  float | None = None, verbose: bool = false, stream: StreamOrDevice = None)
  Invoked with types: kwargs = { inputs: list, template: list, output_shapes:
  list, output_dtypes: list, grid: tuple, threadgroup: tuple }
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.001492016
- *Post-cleanup cache memory (GB):* 0.0
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12266, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12680, in _generate_once
    return strict_generate(
        model=prepared.model,
    ...<3 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1158, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 977, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 393, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/glm4v.py", line 47, in get_input_embeddings
    hidden_states = self.vision_tower(
        pixel_values, grid_thw, output_hidden_states=False
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/vision.py", line 355, in __call__
    hidden_states = self.embeddings(
        hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/vision.py", line 132, in __call__
    interpolated_embed_fp32 = grid_sample(
        pos_embed_2d.transpose(0, 2, 3, 1),
        grid,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kernels.py", line 551, in grid_sample
    outputs = kernel(
        inputs=[x, grid],
    ...<4 lines>...
        threadgroup=(256, 1, 1),
    )
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13353, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12773, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12695, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12277, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/GLM-4.1V-9B-Thinking-8bit: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }

```

</details>

#### Captured stdout/stderr

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '3', '-', '1', '7', '1', '4', '4', '9', '_', 'D', 'S', 'C', '0', '1', '5', '1', '9', '.', 'j', 'p', 'g'] 

Prompt: [gMASK]<sop><|user|>
<|begin_of_image|><|image|><|end_of_image|>Create British-English catalogue metadata from the image and supplied context.

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
Keywords:<|assistant|>

=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 2358.01it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:57:38] ERROR    Runtime error for mlx-community/GLM-4.1V-9B-Thinking-8bit
                    TypeError: __call__(): incompatible function arguments. The following argument types are supported:
                        1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]],
                    output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template:
                    list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool =
                    false, stream: StreamOrDevice = None)
                    Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes:
                    list, grid: tuple, threadgroup: tuple }
```

<a id="diagnostic-mlx-community-glm-46v-flash-mxfp4"></a>

### mlx-community/GLM-4.6V-Flash-mxfp4

#### Root exception and chain

```text
builtins.TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
builtins.ValueError: Model runtime error during generation for mlx-community/GLM-4.6V-Flash-mxfp4: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Missing sections:* ["title", "description"]
- *Echoed instruction fragments:* ["return exactly these three sections",
  "title hint:", "description hint:", "keyword hints:"]
- *Unexpected text before Title:* ========== Files: ['/', 'U', 's', 'e', 'r',
  's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/',
  'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0',
  '8', '1', '3', '-', '1', '7', '1', '4', '4', '9', '_', 'D', 'S', 'C', '0',
  '1', '5', '1', '9', '.', 'j', 'p', 'g']   Prompt: [gMASK]&lt;sop&gt;
  Create British-English catalogue metadata from the image and supplied
  context.  Treat any capture date/time and GPS as authoritative facts, but do
  not claim they are visible. Descriptive hints may be incomplete or wrong:
  retain details supported by the image, correct conflicts, and add important
  visible details. Prefer image evidence when a hint conflicts, and omit
  uncertain details.  Context: Authoritative context: - Capture date/time:
  2026-08-13 17:14:49 UTC+01:00 - GPS: 51.959333°N, 1.349050°E  Descriptive
  hints: - Title hint: Seafront, Felixstowe, England, UK, GBR, Europe -
  Description hint: Seafront, Felixstowe, England, UK, GBR - Keyword hints:
  Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk,
  UK, gbr, seafront  Write: - a concrete 5-10-word title; - a 1-2-sentence
  factual description combining relevant context with the main visible
  subject, setting, action, lighting, and distinctive details; - 10-18 unique,
  comma-separated keywords covering relevant context and visible details.
  Return exactly these three sections and nothing else:
- *Unexpected special tokens:* ["&lt;|user|&gt;", "&lt;|begin_of_image|&gt;",
  "&lt;|image|&gt;", "&lt;|end_of_image|&gt;", "&lt;|assistant|&gt;"]
- *Thinking trace markers:* ["&lt;think&gt;", "&lt;/think&gt;"]
- *Phase:* decode
- *Stage:* Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model runtime error during generation for
  mlx-community/GLM-4.6V-Flash-mxfp4: \_\_call\_\_(): incompatible function
  arguments. The following argument types are supported:     1.
  \_\_call\_\_(self, *, inputs: list[scalar | array], output_shapes:
  list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int],
  threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int |
  Dtype]] | None = None, init_value: float | None = None, verbose: bool =
  false, stream: StreamOrDevice = None)  Invoked with types: kwargs = {
  inputs: list, template: list, output_shapes: list, output_dtypes: list,
  grid: tuple, threadgroup: tuple }
- *Root error type:* TypeError
- *Root error message:* \_\_call\_\_(): incompatible function arguments. The
  following argument types are supported:     1. \_\_call\_\_(self, *, inputs:
  list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes:
  list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int],
  template: list[tuple[str, bool | int | Dtype]] | None = None, init_value:
  float | None = None, verbose: bool = false, stream: StreamOrDevice = None)
  Invoked with types: kwargs = { inputs: list, template: list, output_shapes:
  list, output_dtypes: list, grid: tuple, threadgroup: tuple }
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.001492016
- *Post-cleanup cache memory (GB):* 0.0
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12266, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12680, in _generate_once
    return strict_generate(
        model=prepared.model,
    ...<3 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1158, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 977, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 393, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/glm4v.py", line 47, in get_input_embeddings
    hidden_states = self.vision_tower(
        pixel_values, grid_thw, output_hidden_states=False
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/vision.py", line 355, in __call__
    hidden_states = self.embeddings(
        hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v/vision.py", line 132, in __call__
    interpolated_embed_fp32 = grid_sample(
        pos_embed_2d.transpose(0, 2, 3, 1),
        grid,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kernels.py", line 551, in grid_sample
    outputs = kernel(
        inputs=[x, grid],
    ...<4 lines>...
        threadgroup=(256, 1, 1),
    )
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13353, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12773, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12695, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12277, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/GLM-4.6V-Flash-mxfp4: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }

```

</details>

#### Captured stdout/stderr

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '3', '-', '1', '7', '1', '4', '4', '9', '_', 'D', 'S', 'C', '0', '1', '5', '1', '9', '.', 'j', 'p', 'g'] 

Prompt: [gMASK]<sop><|user|>
<|begin_of_image|><|image|><|end_of_image|>Create British-English catalogue metadata from the image and supplied context.

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
Keywords:/nothink<|assistant|>
<think></think>

=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 11 files:   0%|          | 0/11 [00:00<?, ?it/s]
Fetching 11 files: 100%|##########| 11/11 [00:00<00:00, 2220.49it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:57:41] ERROR    Runtime error for mlx-community/GLM-4.6V-Flash-mxfp4
                    TypeError: __call__(): incompatible function arguments. The following argument types are supported:
                        1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]],
                    output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template:
                    list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool =
                    false, stream: StreamOrDevice = None)
                    Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes:
                    list, grid: tuple, threadgroup: tuple }
```

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

### mlx-community/GLM-4.6V-nvfp4

#### Root exception and chain

```text
builtins.TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
builtins.ValueError: Model runtime error during generation for mlx-community/GLM-4.6V-nvfp4: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Missing sections:* ["title", "description"]
- *Echoed instruction fragments:* ["return exactly these three sections",
  "title hint:", "description hint:", "keyword hints:"]
- *Unexpected text before Title:* ========== Files: ['/', 'U', 's', 'e', 'r',
  's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/',
  'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0',
  '8', '1', '3', '-', '1', '7', '1', '4', '4', '9', '_', 'D', 'S', 'C', '0',
  '1', '5', '1', '9', '.', 'j', 'p', 'g']   Prompt: [gMASK]&lt;sop&gt;
  Create British-English catalogue metadata from the image and supplied
  context.  Treat any capture date/time and GPS as authoritative facts, but do
  not claim they are visible. Descriptive hints may be incomplete or wrong:
  retain details supported by the image, correct conflicts, and add important
  visible details. Prefer image evidence when a hint conflicts, and omit
  uncertain details.  Context: Authoritative context: - Capture date/time:
  2026-08-13 17:14:49 UTC+01:00 - GPS: 51.959333°N, 1.349050°E  Descriptive
  hints: - Title hint: Seafront, Felixstowe, England, UK, GBR, Europe -
  Description hint: Seafront, Felixstowe, England, UK, GBR - Keyword hints:
  Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk,
  UK, gbr, seafront  Write: - a concrete 5-10-word title; - a 1-2-sentence
  factual description combining relevant context with the main visible
  subject, setting, action, lighting, and distinctive details; - 10-18 unique,
  comma-separated keywords covering relevant context and visible details.
  Return exactly these three sections and nothing else:
- *Unexpected special tokens:* ["&lt;|user|&gt;", "&lt;|begin_of_image|&gt;",
  "&lt;|image|&gt;", "&lt;|end_of_image|&gt;", "&lt;|assistant|&gt;"]
- *Thinking trace markers:* ["&lt;think&gt;", "&lt;/think&gt;"]
- *Phase:* decode
- *Stage:* Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model runtime error during generation for
  mlx-community/GLM-4.6V-nvfp4: \_\_call\_\_(): incompatible function
  arguments. The following argument types are supported:     1.
  \_\_call\_\_(self, *, inputs: list[scalar | array], output_shapes:
  list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int],
  threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int |
  Dtype]] | None = None, init_value: float | None = None, verbose: bool =
  false, stream: StreamOrDevice = None)  Invoked with types: kwargs = {
  inputs: list, template: list, output_shapes: list, output_dtypes: list,
  grid: tuple, threadgroup: tuple }
- *Root error type:* TypeError
- *Root error message:* \_\_call\_\_(): incompatible function arguments. The
  following argument types are supported:     1. \_\_call\_\_(self, *, inputs:
  list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes:
  list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int],
  template: list[tuple[str, bool | int | Dtype]] | None = None, init_value:
  float | None = None, verbose: bool = false, stream: StreamOrDevice = None)
  Invoked with types: kwargs = { inputs: list, template: list, output_shapes:
  list, output_dtypes: list, grid: tuple, threadgroup: tuple }
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.001492016
- *Post-cleanup cache memory (GB):* 0.0
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12266, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12680, in _generate_once
    return strict_generate(
        model=prepared.model,
    ...<3 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1158, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 977, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 393, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v_moe/glm4v_moe.py", line 48, in get_input_embeddings
    hidden_states = self.vision_tower(
        pixel_values, grid_thw, output_hidden_states=False
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v_moe/vision.py", line 354, in __call__
    hidden_states = self.embeddings(
        hidden_states, seqlens, grid_thw, image_type_ids[:, 0], image_type_ids[:, 1]
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/glm4v_moe/vision.py", line 132, in __call__
    interpolated_embed_fp32 = grid_sample(
        pos_embed_2d.transpose(0, 2, 3, 1),
        grid,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/kernels.py", line 551, in grid_sample
    outputs = kernel(
        inputs=[x, grid],
    ...<4 lines>...
        threadgroup=(256, 1, 1),
    )
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13353, in process_image_with_model
    output: GenerationResult | SupportsGenerationResult = _run_model_generation(
                                                          ~~~~~~~~~~~~~~~~~~~~~^
        params=params,
        ^^^^^^^^^^^^^^
        phase_callback=_update_phase,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12773, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12695, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12277, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_err
ValueError: Model runtime error during generation for mlx-community/GLM-4.6V-nvfp4: __call__(): incompatible function arguments. The following argument types are supported:
    1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]], output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template: list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool = false, stream: StreamOrDevice = None)

Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes: list, grid: tuple, threadgroup: tuple }

```

</details>

#### Captured stdout/stderr

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '3', '-', '1', '7', '1', '4', '4', '9', '_', 'D', 'S', 'C', '0', '1', '5', '1', '9', '.', 'j', 'p', 'g'] 

Prompt: [gMASK]<sop><|user|>
<|begin_of_image|><|image|><|end_of_image|>Create British-English catalogue metadata from the image and supplied context.

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
Keywords:/nothink<|assistant|>
<think></think>

=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 21 files:   0%|          | 0/21 [00:00<?, ?it/s]
Fetching 21 files: 100%|##########| 21/21 [00:00<00:00, 2999.09it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:57:52] ERROR    Runtime error for mlx-community/GLM-4.6V-nvfp4
                    TypeError: __call__(): incompatible function arguments. The following argument types are supported:
                        1. __call__(self, *, inputs: list[scalar | array], output_shapes: list[Sequence[int]],
                    output_dtypes: list[Dtype], grid: tuple[int, int, int], threadgroup: tuple[int, int, int], template:
                    list[tuple[str, bool | int | Dtype]] | None = None, init_value: float | None = None, verbose: bool =
                    false, stream: StreamOrDevice = None)
                    Invoked with types: kwargs = { inputs: list, template: list, output_shapes: list, output_dtypes:
                    list, grid: tuple, threadgroup: tuple }
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Repeated fragment:* keyword: "cultural icon"
- *Title word count:* 4
- *Requested title word range:* [5, 10]
- *Keyword count:* 306
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["historical landmark", "historical significance",
  "cultural icon", "historical icon"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003376224
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 275
- *Generation tokens:* 1000
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: 
Felixstowe Seafront Memorial Column

Description:
The image features a prominent memorial column situated on the seafront in Felixstowe, England, with a columnar structure and a bird statue at the top, set against a clear blue sky and the calm sea in the background.

Keywords:
Felixstowe, England, UK, Seafront, Memorial, Column, Bird Statue, Blue Sky, Calm Sea, Coastal, Seaside, Public Monument, Historical Landmark, Stone Structure, Ornate Details, Seafront Promenade, Coastal Town, British Architecture, Coastal Scenery, Seaside Atmosphere, Peaceful Scene, Memorial Day, Remembrance, Commemoration, Historical Significance, Cultural Heritage, Architectural Landmark, Iconic Structure, Seafront Landmark, Coastal Icon, British History, Historical Monument, Cultural Significance, Architectural Style, Historical Landmark, Cultural Icon, British Culture, Historical Significance, Architectural Icon, Cultural Landmark, Historical Icon, Cultural Monument, Architectural Heritage, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon, Historical Icon, Cultural Icon,
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated text; missing required fields; echoes instructions; cut off at token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, prompt_instruction_echo
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "- description hint: seafront,..."
- *Echoed instruction fragments:* ["title hint:", "description hint:",
  "keyword hints:"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.013714684
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4397
- *Generation tokens:* 1000
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
- Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, England, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR
- Title hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Description hint: Seafront, Felix
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-Instruct-4bit — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Repeated fragment:* keyword: "stone column"
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 259
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["stone column", "bird statue", "people walking",
  "clear sky", "calm sea", "stone pathway", "landmark", "scenic beauty",
  "seaside town"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.006505594
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16478
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Seafront, Felixstowe, England, UK, GBR, Europe

Description:
The Seafront of Felixstowe, England, UK, GBR, Europe, features a prominent seafront monument with a stone column and a bird statue on top. The monument is situated near the sea, with a clear blue sky and a calm sea in the background. The monument is surrounded by a stone pathway and a few people walking nearby. The monument is a notable landmark in the area, and its presence adds to the scenic beauty of the seaside town.

Keywords:
Seafront, Felixstowe, England, UK, GBR, Europe, Monument, Stone Column, Bird Statue, People Walking, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town, People Walking, Stone Column, Bird Statue, Clear Sky, Calm Sea, Stone Pathway, Landmark, Scenic Beauty, Seaside Town
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Instruct-bf16 — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Repeated fragment:* keyword: "seafront"
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 330
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["seafront", "memorial", "sea", "england", "uk",
  "europe", "1939 1945", "war", "commemoration", "plaques", "lamppost",
  "blue", "sky", "stone", "column", "bronze", "eagle"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00681689
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16467
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Felixstowe Seafront Memorial

Description:
A stone memorial column stands at the seafront in Felixstowe, England, topped with a bronze eagle, commemorating the 1939-1945 war, with plaques listing names and a blue lamppost nearby under a clear blue sky.

Keywords: Felixstowe, seafront, memorial, war memorial, stone column, bronze eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, war, England, UK, Europe, sea, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue
```

</details>

<a id="diagnostic-qwen-qwen3-vl-2b-instruct"></a>

<details>
<summary>Qwen/Qwen3-VL-2B-Instruct — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### Qwen/Qwen3-VL-2B-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Repeated fragment:* keyword: "seafront"
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 330
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["seafront", "memorial", "sea", "england", "uk",
  "europe", "1939 1945", "war", "commemoration", "plaques", "lamppost",
  "blue", "sky", "stone", "column", "bronze", "eagle"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.000443432
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16467
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Felixstowe Seafront Memorial

Description:
A stone memorial column stands at the seafront in Felixstowe, England, topped with a bronze eagle, commemorating the 1939-1945 war, with plaques listing names and a blue lamppost nearby under a clear blue sky.

Keywords: Felixstowe, seafront, memorial, war memorial, stone column, bronze eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, war, England, UK, Europe, sea, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue, sky, seafront, memorial, stone, column, bronze, eagle, sea, England, UK, Europe, 1939-1945, war, commemoration, plaques, lamppost, blue
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-8bit — usable_with_caveats — control tokens visible</summary>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Unexpected special tokens:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Thinking trace markers:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009864372
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 576
- *Generation tokens:* 82
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: War Memorial Column on the Felixstowe Seafront 
Description: A tall fluted stone column topped with a bronze eagle stands on the coastal promenade in East Suffolk under a clear blue sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, memorial, column, monument, coast, ocean, stone, eagle, sculpture, architecture, tourism, Europe
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-mxfp8 — usable_with_caveats — control tokens visible</summary>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Unexpected special tokens:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Thinking trace markers:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009864372
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 576
- *Generation tokens:* 80
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: War Memorial Column on the Seafront Felixstowe England
Description: A tall fluted stone war memorial topped with an eagle statue stands along the rocky coastal seafront under a clear blue sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, war memorial, column, statue, coastal, ocean, stone, monument, commemorative, memorial, remembrance
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — usable_with_caveats — role tokens visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002720852
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1257
- *Generation tokens:* 610
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image. The image shows a tall column monument by the sea, likely a war memorial, with an eagle statue on top. The background has the sea, rocks, a clear blue sky, and some people walking. The location is Felixstowe, England, UK, as per the context. 

First, the title: needs to be a 5-10 word concrete title. Maybe " Felixstowe Seafront Memorial Column by the Coast". That's 7 words.

Description: 1-2 sentences. It should combine context (Felixstowe, England, UK), main subject (memorial column), setting (seafront, sea, rocks), action (people walking), lighting (bright sunny day), distinctive details (column with eagle, plaques, string lights). So: "A tall stone memorial column with an eagle statue stands on the seafront of Felixstowe, England, UK, near the coast with people walking nearby under a clear blue sky."

Keywords: Need 10-18 unique keywords. Let's list: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear sky, people walking, plaques, string lights, blue sky. That's 10 keywords.

Wait, check if all context and visible details are covered. The context mentions East Suffolk, UK, Europe, but the image doesn't show East Suffolk specifically, but Felixstowe is in Suffolk. The keywords should include Felixstowe, England, UK, Europe. Also, the eagle statue, the column, the sea, rocks, people, string lights, plaques, clear sky. Let's adjust: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights. That's 10. Maybe add "war memorial" but the hint says to omit uncertain details. The plaques might be part of the memorial, so include plaques. Let's finalize:

Title: Felixstowe Seafront Memorial Column by the Coast

Description: A tall stone memorial column topped with an eagle statue stands on the seafront of Felixstowe, England, UK, near coastal rocks with people walking nearby under a clear blue sky.

Keywords: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights◁/think▷Title: Felixstowe Seafront Memorial Column by the Coast

Description: A tall stone memorial column topped with an eagle statue stands on the seafront of Felixstowe, England, UK, near coastal rocks with people walking nearby under a clear blue sky.

Keywords: Felixstowe, England, UK, seafront, memorial column, eagle statue, coastal rocks, clear blue sky, people walking, plaques, string lights<|im_assistant|>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                             | Usability           | Observations                                                                                           |
|---------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                   | unusable            | missing required fields; echoes instructions; extra text before Title                                  |
| mlx-community/gemma-3n-E4B-it-bf16                | unusable            | missing required fields                                                                                |
| mlx-community/llava-v1.6-mistral-7b-8bit          | unusable            | missing required fields                                                                                |
| mlx-community/MolmoPoint-8B-fp16                  | unusable            | missing required fields                                                                                |
| mlx-community/nanoLLaVA-1.5-4bit                  | unusable            | missing required fields; echoes instructions                                                           |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX     | unusable            | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16  | unusable            | cut off at token limit; title/keyword constraints failed                                               |
| mlx-community/gemma-3-27b-it-qat-4bit             | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/InternVL3-8B-bf16                   | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/MiniCPM-V-4.6-8bit                  | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Molmo-7B-D-0924-8bit                | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Phi-3.5-vision-instruct-bf16        | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Qwen3-VL-2B-Thinking-bf16           | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Qwen3.6-27B-mxfp8                   | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/X-Reasoner-7B-8bit                  | usable_with_caveats | title/keyword constraints failed                                                                       |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                    | Performance                                                                                 |
|-------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | rev ed71acdae079; Lfm2VlProcessor; stop completed   | 2072 prompt / 132 generated; 484 tok/s; 1.9 GB peak; cleanup 0.000132/0.0 GB active/cache   |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed | 2658 prompt / 131 generated; 30.0 tok/s; 24 GB peak; cleanup 0.000968/0.0 GB active/cache   |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed   | 580 prompt / 98 generated; 129 tok/s; 16 GB peak; cleanup 0.0115/0.0 GB active/cache        |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed   | 580 prompt / 111 generated; 25.9 tok/s; 20 GB peak; cleanup 0.012/0.0 GB active/cache       |
| mlx-community/Idefics3-8B-Llama3-bf16                 | rev 8c2a30c48864; Idefics3Processor; stop completed | 2587 prompt / 202 generated; 31.9 tok/s; 18 GB peak; cleanup 0.00175/0.0 GB active/cache    |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | rev 16a710cf8afc; Lfm2VlProcessor; stop completed   | 2072 prompt / 140 generated; 186 tok/s; 4.0 GB peak; cleanup 0.00285/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed | 3191 prompt / 162 generated; 66.5 tok/s; 14 GB peak; cleanup 0.00416/0.0 GB active/cache    |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed | 3190 prompt / 123 generated; 182 tok/s; 9.0 GB peak; cleanup 0.00469/0.0 GB active/cache    |
| mlx-community/Ornith-1.0-35B-bf16                     | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed  | 16482 prompt / 122 generated; 63.2 tok/s; 74 GB peak; cleanup 0.00613/0.0 GB active/cache   |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed  | 3429 prompt / 99 generated; 39.5 tok/s; 16 GB peak; cleanup 0.014/0.0 GB active/cache       |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed  | 16482 prompt / 94 generated; 111 tok/s; 24 GB peak; cleanup 0.00764/0.0 GB active/cache     |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed  | 16482 prompt / 102 generated; 93.4 tok/s; 10.0 GB peak; cleanup 0.00816/0.0 GB active/cache |
| mlx-community/Qwen3.8-27B-4bit                        | rev 3e6447f082e8; Qwen3VLProcessor; stop completed  | 16482 prompt / 120 generated; 30.6 tok/s; 21 GB peak; cleanup 0.00918/0.0 GB active/cache   |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | rev 844516024a1c; SmolVLMProcessor; stop completed  | 1400 prompt / 89 generated; 125 tok/s; 5.5 GB peak; cleanup 0.00929/0.0 GB active/cache     |
| mlx-community/Step-3.7-Flash-oQ2e                     | rev 3dacb46f724a; Step3VLProcessor; stop completed  | 3468 prompt / 114 generated; 45.7 tok/s; 70 GB peak; cleanup 0.00955/0.0 GB active/cache    |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

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

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/GLM-4.1V-9B-Thinking-8bit          | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/GLM-4.6V-Flash-mxfp4               | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/Qwen2-VL-2B-Instruct-4bit          | 01af461cdb9574acc09084a0ef94e216e142b085 |
| mlx-community/Qwen3-VL-2B-Instruct-bf16          | c8a67a84327484ba87f5ec4f8fb927cdafd791aa |
| Qwen/Qwen3-VL-2B-Instruct                        | 89644892e4d85e24eaac8bacfd4f463576704203 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.15                                                                                                                                          |
| mlx-vlm source revision    | 1249c7db09921714b43e056b149f9d762eec07d3                                                                                                        |
| mlx                        | 0.32.2.dev20260820+27fec909a                                                                                                                    |
| mlx source revision        | 27fec909a3df9e572f5195607a453e273e7d80d0                                                                                                        |
| mlx-lm                     | 0.32.0                                                                                                                                          |
| mlx-lm source revision     | d06c5374a12e1f9384aad5fece583d7be9d2619d                                                                                                        |
| mlx-audio                  | 0.5.0                                                                                                                                           |
| transformers               | 5.15.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.28.0                                                                                                                                          |
| Python Version             | 3.13.14                                                                                                                                         |
| OS                         | Darwin 25.6.0                                                                                                                                   |
| macOS Version              | 26.6.2                                                                                                                                          |
| SDK Version                | 26.5                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                              |
| Xcode Version              | 26.6                                                                                                                                            |
| Xcode Build                | 17F113                                                                                                                                          |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                               |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                  |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                 |
| GPU/Chip                   | Apple M5 Max                                                                                                                                    |
| GPU Cores                  | 40                                                                                                                                              |
| MLX Device                 | Apple M5 Max                                                                                                                                    |
| GPU Architecture           | applegpu_g17s                                                                                                                                   |
| Recommended Working Set    | 108 GB                                                                                                                                          |
| Fused Attention            | Available                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (175,350,592 bytes, sha256=73af0b917fb1d9bbb27d643feba7c1daa430a683c9c6977cdb0c6be8194fe1f5) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,915,168 bytes, sha256=dfe05437f5b0c8d3de03a913616b5b23fb7b369cf1bc9d99497e99e1b9feacac)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
