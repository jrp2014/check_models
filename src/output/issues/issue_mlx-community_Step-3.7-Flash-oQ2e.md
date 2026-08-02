# Crash: mlx-community/Step-3.7-Flash-oQ2e

## Maintainer evidence

### mlx-community/Step-3.7-Flash-oQ2e

#### Root exception and chain

```text
builtins.ValueError: Loaded processor has no image_processor; expected multimodal processor.
builtins.ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Phase:* processor_load
- *Stage:* Processor Error
- *Package:* model-config
- *Error type:* ValueError
- *Error message:* Model preflight failed for
  mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor;
  expected multimodal processor.
- *Root error type:* ValueError
- *Root error message:* Loaded processor has no image_processor; expected
  multimodal processor.
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.014517394
- *Post-cleanup cache memory (GB):* 0.0

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11250, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11038, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10971, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11753, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11519, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11291, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
[01:10:27] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

## Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,240 pixels
- *Image size:* 61,337,614 bytes
- *Image SHA-256:* a907e72a592bcdbdc026c71ac8a508d6cf87ee27222afaf7cc26f96145151f89

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-01 16:28:40 UTC+01:00
- GPS: 52.345200°N, 1.503700°E

Descriptive hints:
- Title hint: Town centre, Halesworth, England, UK, GBR, Europe
- Description hint: The Cut in Halesworth, Suffolk in the UK
- Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk

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

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.8                                                                                                                                           |
| mlx                        | 0.32.1.dev20260802+fb5133e10                                                                                                                    |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.4                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.26.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.6.0                                                                                                                                   |
| macOS Version              | 26.6                                                                                                                                            |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,842,200 bytes, sha256=6a34bf1f3b542a904c4cf464bc95d7e419ca42a33175da64477eea57a9d90f2e) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,642,704 bytes, sha256=16951e19288070f611f39be301a4a3507e3b8c67db7ebd17d7fd7a9b0e3211dc)  |
| RAM                        | 128.0 GB                                                                                                                                        |
