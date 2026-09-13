# Crash: mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit

## Maintainer evidence

### mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit

#### Root exception and chain

```text
builtins.TypeError: can only concatenate str (not "list") to str
builtins.ValueError: Prompt prefill failed for mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit: can only concatenate str (not "list") to str
```

#### Execution and provenance

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mistral3)
- *Phase:* prefill
- *Stage:* Error
- *Package:* transformers
- *Error type:* ValueError
- *Error message:* Prompt prefill failed for
  mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit: can only concatenate
  str (not "list") to str
- *Root error type:* TypeError
- *Root error message:* can only concatenate str (not "list") to str
- *Resolved model revision:* 2a1d5eabfc504747bdc24178394821a1efc0edde
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.004129908
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 13.26
- *Parameter count:* 24.00B (name-estimate)
- *Quantization:* 4-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13521, in _prepare_generation_prompt
    apply_chat_template(
    ~~~~~~~~~~~~~~~~~~~^
        processor=processor,
        ^^^^^^^^^^^^^^^^^^^^
    ...<3 lines>...
        **chat_template_kwargs,
        ^^^^^^^^^^^^^^^^^^^^^^^
    ),
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 1026, in apply_chat_template
    return get_chat_template(processor, messages, add_generation_prompt, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 824, in get_chat_template
    return template_processor.apply_chat_template(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        messages,
        ^^^^^^^^^
    ...<2 lines>...
        **template_kwargs,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/tokenization_utils_base.py", line 3108, in apply_chat_template
    rendered_chat, generation_indices = render_jinja_template(
                                        ~~~~~~~~~~~~~~~~~~~~~^
        conversations=conversations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
        **template_kwargs,
        ^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/utils/chat_template_utils.py", line 581, in render_jinja_template
    rendered_chat = compiled_template.render(
        messages=chat,
    ...<3 lines>...
        **kwargs,
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py", line 1295, in render
    self.environment.handle_exception()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py", line 942, in handle_exception
    raise rewrite_traceback_stack(source=source)
  File "<template>", line 1, in top-level template code
TypeError: can only concatenate str (not "list") to str

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15335, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14216, in _run_model_generation
    prepared = _prepare_generation(
        params,
    ...<4 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13767, in _prepare_generation
    prepared_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13540, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(msg), "prefill") from prefill_err
ValueError: Prompt prefill failed for mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit: can only concatenate str (not "list") to str

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 3538.60it/s]
[21:49:34] Prompt prefill failed for mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13521, in
           _prepare_generation_prompt
               apply_chat_template(
               ~~~~~~~~~~~~~~~~~~~^
                   processor=processor,
                   ^^^^^^^^^^^^^^^^^^^^
               ...<3 lines>...
                   **chat_template_kwargs,
                   ^^^^^^^^^^^^^^^^^^^^^^^
               ),
               ^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 1026, in
           apply_chat_template
               return get_chat_template(processor, messages, add_generation_prompt, **kwargs)
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/prompt_utils.py", line 824, in
           get_chat_template
               return template_processor.apply_chat_template(
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
                   messages,
                   ^^^^^^^^^
               ...<2 lines>...
                   **template_kwargs,
                   ^^^^^^^^^^^^^^^^^^
               )
               ^
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/tokenizatio
           n_utils_base.py", line 3108, in apply_chat_template
               rendered_chat, generation_indices = render_jinja_template(
                                                   ~~~~~~~~~~~~~~~~~~~~~^
                   conversations=conversations,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               ...<6 lines>...
                   **template_kwargs,
                   ^^^^^^^^^^^^^^^^^^
               )
               ^
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/utils/chat_
           template_utils.py", line 581, in render_jinja_template
               rendered_chat = compiled_template.render(
                   messages=chat,
               ...<3 lines>...
                   **kwargs,
               )
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py",
           line 1295, in render
               self.environment.handle_exception()
               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
             File
           "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/jinja2/environment.py",
           line 942, in handle_exception
               raise rewrite_traceback_stack(source=source)
             File "<template>", line 1, in top-level template code
           TypeError: can only concatenate str (not "list") to str
```

## Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 44,654,464 bytes
- *Image SHA-256:* c109700d51a838d36e8d56fc769534be3598a8af6310eed6d0afa19ecbe7dccf

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-57f7cdeb1b88952a.jpg>
- *Preview dimensions:* 1,024 x 683 pixels
- *Preview size:* 157,139 bytes
- *Preview SHA-256:* 57f7cdeb1b88952a3eeb6f5aa7ba4587981131b5841d7f8f82911704c242125b

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-57f7cdeb1b88952a.jpg
printf '%s\n' '57f7cdeb1b88952a3eeb6f5aa7ba4587981131b5841d7f8f82911704c242125b  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --model mlx-community/Mistral-Small-3.2-24B-Instruct-2506-4bit --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:' --max-tokens 1000 --temperature 0.0 --revision 2a1d5eabfc504747bdc24178394821a1efc0edde --trust-remote-code --seed 0 --prefill-step-size 2048
```

## Provenance and Environment

### Components

| Component       | Value                                                             |
|-----------------|-------------------------------------------------------------------|
| mlx-vlm         | 0.7.0                                                             |
| mlx             | 0.32.3.dev20260912+229f5b430                                      |
| transformers    | 5.17.0                                                            |
| tokenizers      | 0.23.2                                                            |
| huggingface-hub | 1.31.0                                                            |
| Pillow          | 12.3.0                                                            |
| Python Version  | 3.14.7                                                            |
| macOS Version   | 26.6.2                                                            |
| GPU/Chip        | Apple M5 Max                                                      |
| check_models    | 0.17.25; revision 11e6c82dc61f08eb3dc36a3188a789da1295093b; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
