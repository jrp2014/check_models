# Crash: mlx-community/GLM-4.6V-nvfp4

## Maintainer evidence

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

## Reproduction inputs

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

## Provenance and Environment

### Components

| Component       | Value                                                            |
|-----------------|------------------------------------------------------------------|
| mlx-vlm         | 0.6.15                                                           |
| mlx             | 0.32.2.dev20260820+27fec909a                                     |
| mlx-lm          | 0.32.0                                                           |
| transformers    | 5.15.1                                                           |
| tokenizers      | 0.22.2                                                           |
| huggingface-hub | 1.28.0                                                           |
| Pillow          | 12.3.0                                                           |
| Python Version  | 3.13.14                                                          |
| macOS Version   | 26.6.2                                                           |
| GPU/Chip        | Apple M5 Max                                                     |
| check_models    | 0.13.0; revision d9538f8e4dac6ccae7178907ab8deecb6cd45f26; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
