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
- *Post-cleanup active memory (GB):* 0.014632082
- *Post-cleanup cache memory (GB):* 0.0

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11149, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10937, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10870, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11652, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11418, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11190, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 5562.12it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[23:05:57] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

## Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Step-3.7-Flash-oQ2e --image 20260725-183316_DSC01175.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

Use authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the keyword list after the label.

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Capture date/time: 2026-07-25 18:33:16 UTC+01:00
- GPS: 51.358240°N, 1.432820°E
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Viking Bay, Broadstairs, England, UK, GBR, Europe
- Existing description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
- Existing keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --revision 3dacb46f724ac89725bcd922fb779c7ed1499fe7 --trust-remote-code --prefill-step-size 4096
```

## Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Step-3.7-Flash-oQ2e"
IMAGE = "20260725-183316_DSC01175.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.\n- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Authoritative context:\n- Capture date/time: 2026-07-25 18:33:16 UTC+01:00\n- GPS: 51.358240°N, 1.432820°E\n- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.\n\nDraft descriptive metadata:\n- Existing title: Viking Bay, Broadstairs, England, UK, GBR, Europe\n- Existing description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.\n- Existing keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky\n- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "3dacb46f724ac89725bcd922fb779c7ed1499fe7",
}
TEMPLATE_KWARGS = {}
GENERATE_KWARGS = {
    "max_tokens": 500,
    "temperature": 0.0,
    "prefill_step_size": 4096,
}
model, processor = load(MODEL, **LOAD_KWARGS)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
    **TEMPLATE_KWARGS,
)
if isinstance(formatted_prompt, list):
    formatted_prompt = "\n".join(str(message) for message in formatted_prompt)
result = generate(model, processor, formatted_prompt, image=IMAGE, **GENERATE_KWARGS)
print(result.text)
```

## Provenance and Environment

### Prompt

```text
Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

Use authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the keyword list after the label.

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Capture date/time: 2026-07-25 18:33:16 UTC+01:00
- GPS: 51.358240°N, 1.432820°E
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Viking Bay, Broadstairs, England, UK, GBR, Europe
- Existing description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
- Existing keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.
```

### Components

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.8                                                                                                                                           |
| mlx                        | 0.32.1.dev20260731+fb5133e10                                                                                                                    |
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
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,642,704 bytes, sha256=021af97ab68e66a84dc18ac1923422802a47a79dcea086489556b58c8ae90df9)  |
| RAM                        | 128.0 GB                                                                                                                                        |
