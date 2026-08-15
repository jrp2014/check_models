# Crash: mlx-community/SmolVLM2-2.2B-Instruct-mlx

## Maintainer evidence

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Root exception and chain

```text
builtins.ValueError: Image features and image tokens do not match: tokens: 81, features 1053
builtins.ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do not match: tokens: 81, features 1053
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Missing sections:* ["title", "description"]
- *Echoed instruction fragments:* ["return exactly these three sections",
  "title hint:", "description hint:", "keyword hints:"]
- *Unexpected text before Title:* ========== Files: ['/', 'U', 's', 'e', 'r',
  's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/',
  'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0',
  '8', '1', '5', '-', '1', '5', '5', '9', '4', '6', '_', 'D', 'S', 'C', '0',
  '1', '5', '6', '8', '.', 'j', 'p', 'g']   Prompt:  User:&lt;image&gt;Create
  British-English catalogue metadata from the image and supplied context.
  Treat any capture date/time and GPS as authoritative facts, but do not claim
  they are visible. Descriptive hints may be incomplete or wrong: retain
  details supported by the image, correct conflicts, and add important visible
  details. Prefer image evidence when a hint conflicts, and omit uncertain
  details.  Context: Authoritative context: - Capture date/time: 2026-08-15
  15:59:46 UTC+01:00 - GPS: 51.128800°N, 1.319100°E  Descriptive hints: -
  Title hint: Dover Castle, Dover, England, UK, GBR, Europe - Description
  hint: An exterior view of a historic medieval stone castle, featuring round
  towers, an arched entranceway, and a small bridge, built on a steep grassy
  hill under a partly cloudy sky. - Keyword hints: Adobe Stock, Any Vision,
  Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent,
  Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
  Write: - a concrete 5-10-word title; - a 1-2-sentence factual description
  combining relevant context with the main visible subject, setting, action,
  lighting, and distinctive details; - 10-18 unique, comma-separated keywords
  covering relevant context and visible details.  Return exactly these three
  sections and nothing else:
- *Unexpected special tokens:* ["&lt;|im_start|&gt;"]
- *Phase:* decode
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model generation failed for
  mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do
  not match: tokens: 81, features 1053
- *Root error type:* ValueError
- *Root error message:* Image features and image tokens do not match: tokens:
  81, features 1053
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Processor class:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.009602184
- *Post-cleanup cache memory (GB):* 0.0
- *Configured EOS token ID:* 49279
- *Configured EOS token:* &lt;end_of_utterance&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11966, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12315, in _generate_once
    return strict_generate(
        model=model,
    ...<3 lines>...
        **generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1159, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 978, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 393, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/idefics3/idefics3.py", line 160, in get_input_embeddings
    final_inputs_embeds = self._prepare_inputs_for_multimodal(
        image_features, inputs_embeds, input_ids
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/idefics3/idefics3.py", line 174, in _prepare_inputs_for_multimodal
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
ValueError: Image features and image tokens do not match: tokens: 81, features 1053

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12893, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12328, in _run_model_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11973, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do not match: tokens: 81, features 1053

```

</details>

#### Captured stdout/stderr

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '5', '-', '1', '5', '5', '9', '4', '6', '_', 'D', 'S', 'C', '0', '1', '5', '6', '8', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>User:<image>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-15 15:59:46 UTC+01:00
- GPS: 51.128800°N, 1.319100°E

Descriptive hints:
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:<end_of_utterance>
Assistant:

=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 3113.43it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[00:45:18] ERROR    Generation error for mlx-community/SmolVLM2-2.2B-Instruct-mlx
                    ValueError: Image features and image tokens do not match: tokens: 81, features 1053
```

## Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 8,880 pixels
- *Image size:* 79,069,278 bytes
- *Image SHA-256:* 771ab1bcadbb99020fb1a6270d6f36e8dd613cc3132c390bed714290bda2dd05

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-15 15:59:46 UTC+01:00
- GPS: 51.128800°N, 1.319100°E

Descriptive hints:
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue

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
| mlx-vlm         | 0.6.14                                                           |
| mlx             | 0.32.1.dev20260815+9ab977b56                                     |
| mlx-lm          | 0.31.3                                                           |
| transformers    | 5.15.0                                                           |
| tokenizers      | 0.22.2                                                           |
| huggingface-hub | 1.27.0                                                           |
| Pillow          | 12.3.0                                                           |
| Python Version  | 3.13.14                                                          |
| macOS Version   | 26.6.1                                                           |
| GPU/Chip        | Apple M5 Max                                                     |
| check_models    | 0.10.0; revision 3dd40931dfdffb61563b75d4782104e7bd2c2f6a; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
