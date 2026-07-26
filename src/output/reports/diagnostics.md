# Diagnostics

## Run Outcome Counts

<!-- markdownlint-disable MD060 -->

| Outcome       |   Count |
|---------------|---------|
| Attempted     |      62 |
| Evaluated     |      62 |
| Completed     |      61 |
| Crashed       |       1 |
| Indeterminate |       0 |
<!-- markdownlint-enable MD060 -->
## Actionable Failures

### mlx-community/Step-3.7-Flash-oQ2e

#### Root exception and chain

```text
builtins.ValueError: Loaded processor has no image_processor; expected multimodal processor.
builtins.ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

#### Complete traceback

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10914, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10706, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10639, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11399, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11166, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 10955, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.

```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Phase:* processor_load
- *Stage:* Processor Error
- *Package:* model-config
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Requested model revision:* unavailable
- *Processor class:* unavailable
- *Tokenizer class:* unavailable
- *Stop reason:* exception
- *Prompt tokens:* unavailable
- *Generation tokens:* unavailable
- *Configured EOS token ID:* unavailable
- *Configured EOS token:* unavailable
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete partial output

```text
unavailable
```

#### Captured stdout/stderr

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 2907.75it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[21:24:21] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Step-3.7-Flash-oQ2e --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 3dacb46f724ac89725bcd922fb779c7ed1499fe7 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Step-3.7-Flash-oQ2e"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
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

## Completed Runs with Observations

### mlx-community/MiniCPM-V-4.6-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* thinking_trace_present, thinking_trace_incomplete
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Prompt tokens:* 529
- *Generation tokens:* 67
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
<think>

Here's the structured output according to your requirements:

Title:
Two cats resting on a pink surface

Description:
The image shows two cats lying on a pink fabric. A remote control is visible near one of the cats.

Keywords:
cats, resting, pink, remote control, tabby, sleeping
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/MiniCPM-V-4.6-8bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 03721395f6b82cd000cc74cde28fcff8abd9a04c --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/MiniCPM-V-4.6-8bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "03721395f6b82cd000cc74cde28fcff8abd9a04c",
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

### HuggingFaceTB/SmolVLM-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 81cd9a775a4d644f2faf4e7becff4559b46b14c7
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Stop reason:* completed
- *Prompt tokens:* 1507
- *Generation tokens:* 13
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
 Two cats sleeping on a pink blanket with two remote controls.
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model HuggingFaceTB/SmolVLM-Instruct --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 81cd9a775a4d644f2faf4e7becff4559b46b14c7 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "HuggingFaceTB/SmolVLM-Instruct"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "81cd9a775a4d644f2faf4e7becff4559b46b14c7",
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

### qnguyen3/nanoLLaVA

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 13d60cec183a86755afed64da495fcc2c382ea80
- *Requested model revision:* unavailable
- *Processor class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Prompt tokens:* 303
- *Generation tokens:* 41
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title: Two Striped Cats Sleeping on a Couch
Description: Two cats, one striped and the other not, are laying on a couch. The striped cat has a green tag on its ear.
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model qnguyen3/nanoLLaVA --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 13d60cec183a86755afed64da495fcc2c382ea80 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "qnguyen3/nanoLLaVA"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "13d60cec183a86755afed64da495fcc2c382ea80",
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

### mlx-community/FastVLM-0.5B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Prompt tokens:* 307
- *Generation tokens:* 139
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title: "Two Cats Napping on a Pink Sofa"
Description: The image shows two cats, one tabby and one striped, both with black stripes and white paws, peacefully sleeping on a pink sofa. The tabby cat is on the left, curled up with its head resting on its paws, while the striped cat is on the right, also curled up with its head on its paws. There are two remote controls on the sofa, one on the left and one on the right. The image is taken in a well-lit room with a pink blanket or throw on the sofa. The cats appear to be in a relaxed state, enjoying a quiet moment together.
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/FastVLM-0.5B-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 81ffe929046666c43de53691147b1669ba0f3a4c --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/FastVLM-0.5B-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "81ffe929046666c43de53691147b1669ba0f3a4c",
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

### mlx-community/SmolVLM-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* cae61cdedd0602419b43b6102dc33cd9f1e929a6
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Stop reason:* completed
- *Prompt tokens:* 1507
- *Generation tokens:* 13
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
 Two cats sleeping on a pink blanket with two remote controls.
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/SmolVLM-Instruct-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision cae61cdedd0602419b43b6102dc33cd9f1e929a6 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/SmolVLM-Instruct-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "cae61cdedd0602419b43b6102dc33cd9f1e929a6",
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

### mlx-community/LFM2-VL-1.6B-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 294b90e5ae2389ecb61a9427b4572975eef614fe
- *Requested model revision:* unavailable
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Prompt tokens:* 566
- *Generation tokens:* 261
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title: Two cats sleeping on a pink blanket
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the Title, Description, or Keywords fields.
- No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

Description: The image shows two cats sleeping on a pink blanket. The cats are lying on their sides, with their bodies stretched out. The blanket is a bright pink color, and it appears to be made of a soft, plush material. The cats are both gray and black striped, and they have long, fluffy tails. One cat is lying on its side, while the other is lying on its back. The cats are both sleeping peacefully, with their eyes closed. The blanket is draped over a piece of furniture, and there are two remote controls visible in the image. The remote controls are white and black, and they are lying on the blanket next to the cats. The image is well-lit, and the colors are bright and vibrant. The cats appear to be very relaxed and comfortable, and the overall atmosphere of the image is peaceful and serene.
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/LFM2-VL-1.6B-8bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 294b90e5ae2389ecb61a9427b4572975eef614fe --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/LFM2-VL-1.6B-8bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "294b90e5ae2389ecb61a9427b4572975eef614fe",
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

### mlx-community/paligemma2-3b-ft-docci-448-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* f66333527ce75342b09d4df81873f65272ec2f30
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Prompt tokens:* 1330
- *Generation tokens:* 14
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
- Do not use the word "cat" in the description.
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/paligemma2-3b-ft-docci-448-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision f66333527ce75342b09d4df81873f65272ec2f30 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/paligemma2-3b-ft-docci-448-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "f66333527ce75342b09d4df81873f65272ec2f30",
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

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Prompt tokens:* 2605
- *Generation tokens:* 16
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
 Title: two tabby cats on a pink blanket with remotes.<end_of_utterance>
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Idefics3-8B-Llama3-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 8c2a30c48864f3251701b7bde40f601d25535098 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Idefics3-8B-Llama3-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "8c2a30c48864f3251701b7bde40f601d25535098",
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

### mlx-community/nanoLLaVA-1.5-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Requested model revision:* unavailable
- *Processor class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 303
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title: 5-10 words, authoritative context may supply identity and location.
Description: A close-up of a cat lying on a pink blanket. The cat is wearing a striped shirt and has a green collar. The cat is resting on a pink couch, with a black and white striped blanket underneath. The cat is looking away from the camera, and the background is a pinkish-purple color. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is taken in a room with pink walls and a pink couch. The cat is the only animal in the image, and the cat is the only object in the image. The image is clear and well-lit, with no visible distractions. The cat is the only subject in the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is clear and well-lit, with no visible distractions. The cat is the only subject in the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is clear and well-lit, with no visible distractions. The cat is the only subject in the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only other detail. The image is a close-up of the cat's head and neck, with the collar visible around the neck. The cat is the main subject of the image, and the background is the only
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/nanoLLaVA-1.5-4bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 5240204744963d72823e5de933c528c4aa82dfca --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/nanoLLaVA-1.5-4bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "5240204744963d72823e5de933c528c4aa82dfca",
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

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_special_token
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Prompt tokens:* 581
- *Generation tokens:* 81
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
<|channel>thought
<channel|>Title:
Two tabby cats sleeping on a pink blanket

Description:
Two tabby cats lie side-by-side on a pink blanket over a red sofa. Two remote controls are visible near the cats.

Keywords:
cats, tabby, sleeping, pink, blanket, sofa, red, remote controls, indoor, domestic, resting, grey, brown, black, fur
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/diffusiongemma-26B-A4B-it-mxfp8 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision ded389e478f86d498ad9e7f47666e83b166a28f1 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-mxfp8"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "ded389e478f86d498ad9e7f47666e83b166a28f1",
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

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_special_token
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Prompt tokens:* 581
- *Generation tokens:* 84
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
<|channel>thought
<channel|>Title:
- Two tabby cats sleeping on a pink blanket

Description:
- Two tabby cats lie curled on a pink blanket draped over a red sofa. Two remote controls are visible on the sofa near the cats.

Keywords:
- cats, tabby, sleeping, pink blanket, red, sofa, remote control, domestic, feline, striped, brown,, indoors, resting, pet
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/diffusiongemma-26B-A4B-it-8bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-8bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "7b95e3887078ba56283c24f2578d6e5a06b9d7e8",
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

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_special_token
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Prompt tokens:* 686
- *Generation tokens:* 91
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
<|begin_of_box|>Title:
Two cats lying on a pink sofa with remote controls

Description:
Two cats are lying on a pink sofa, one with a green collar and the other without, both appearing to be sleeping. Two remote controls are visible on the sofa between them.

Keywords:
cats, sleeping, pink sofa, remote controls, green collar, tabby, striped, feline, domestic, furniture, pets, indoors, relaxed, two animals, cushions<|end_of_box|>
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-nvfp4 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 2da6855d4e28a0e61c84543262074bc17ac27d6e --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/GLM-4.6V-nvfp4"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "2da6855d4e28a0e61c84543262074bc17ac27d6e",
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

### Qwen/Qwen3-VL-2B-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 596
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title:
Two cats sleeping on a pink couch with remote controls

Description:
Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch.

Keywords:
cat, sleeping, couch, pink, remote, control, tabby, feline, domestic, pet, furniture, domestication, domestic animal, pet care, pet owner, pet lifestyle, pet companionship, pet sleep, pet comfort, pet relaxation, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model Qwen/Qwen3-VL-2B-Instruct --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 89644892e4d85e24eaac8bacfd4f463576704203 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "89644892e4d85e24eaac8bacfd4f463576704203",
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

### mlx-community/Qwen3-VL-2B-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 596
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title:
Two cats sleeping on a pink couch with remote controls

Description:
Two tabby cats are lying on a bright pink couch, one on its back and the other on its side, both appearing to be asleep. A remote control is placed between them on the couch.

Keywords:
cat, sleeping, couch, pink, remote, control, tabby, feline, domestic, pet, furniture, domestication, domestic animal, pet care, pet owner, pet lifestyle, pet companionship, pet sleep, pet comfort, pet relaxation, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet rest, pet comfort, pet sleep, pet relaxation, pet comfort, pet
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Instruct-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision c8a67a84327484ba87f5ec4f8fb927cdafd791aa --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Qwen3-VL-2B-Instruct-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "c8a67a84327484ba87f5ec4f8fb927cdafd791aa",
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

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 598
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Got it, let's tackle this step by step. First, the Title. Need 5-10 words, concrete and factual. The image shows two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but need to be concise. Let's see: "Two tabby cats resting on pink couch with remote controls" – that's 10 words, factual.

Description: 1-2 sentences. Combine context with visible details. The setting is a pink couch, two cats (one with green collar, maybe), remotes. So "Two tabby cats rest on a bright pink couch, one with a green collar and the other without, accompanied by two remote controls placed nearby."

Keywords: 10-18 unique terms. Need to list visible elements: tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting, cats, couch, remotes, cats, couch, pink, tabby, resting. Wait, need to pick unique terms. Let's list: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting, cats, couch, remotes, cats, couch, pink, tabby, resting" – but need to be unique. Let's check: tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting. Wait, maybe "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting" – but need to make sure they are unique. Let's see: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting" – but maybe better to list: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch, remotes, cats, couch, pink, tabby, resting" – no, need to pick 10-18. Let's think: "tabby cats, pink couch, remote controls, resting, two cats, green collar, couch,
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Thinking-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision c325e5ea14c215bb08fa0d668c81fa2581f9050b --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Qwen3-VL-2B-Thinking-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "c325e5ea14c215bb08fa0d668c81fa2581f9050b",
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

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, thinking_trace_present
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 4b3b11ce0874c36a99e13e17e355049042f8620a
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Prompt tokens:* 678
- *Generation tokens:* 296
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
◁think▷So, let's analyze the image. First, identify the main elements. There are two tabby cats on a pink fabric surface. One cat is lying on its side, the other is curled up. There are two remote controls (one blue, one silver) near them. The lighting is bright, making the colors vivid. The setting is indoors, likely a couch or bed. 

For the title: Need a 5-10 word title. "Two tabby cats resting on pink fabric with remote controls". That's concise and factual.

Description: Two tabby cats rest on a pink fabric surface, with two remote controls nearby. One cat lies on its side, the other curled up. The scene is brightly lit, emphasizing the cats' fur patterns and the vibrant pink background.

Keywords: tabby cats, pink fabric, remote controls, resting, brightly lit, cat fur patterns, indoor setting, two cats, relaxed posture, domestic environment.◁/think▷Title:
Two tabby cats resting on pink fabric with remote controls

Description:
Two tabby cats rest on a pink fabric surface, with two remote controls nearby; one cat lies on its side while the other curls up, and the scene is brightly lit to highlight their fur patterns and the vibrant background.

Keywords:
tabby cats, pink fabric, remote controls, resting, brightly lit, cat fur patterns, indoor setting, two cats, relaxed posture, domestic environment<|im_assistant|>
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model jqlive/Kimi-VL-A3B-Thinking-2506-6bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 4b3b11ce0874c36a99e13e17e355049042f8620a --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "jqlive/Kimi-VL-A3B-Thinking-2506-6bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "4b3b11ce0874c36a99e13e17e355049042f8620a",
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

### mlx-community/gemma-3n-E2B-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* ec68dc186276e20e4bed30b96a2b5c667e0a81e3
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 565
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text

- Do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your own, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be a lot of your, and do not be
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-3n-E2B-4bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision ec68dc186276e20e4bed30b96a2b5c667e0a81e3 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/gemma-3n-E2B-4bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "ec68dc186276e20e4bed30b96a2b5c667e0a81e3",
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

### mlx-community/gemma-3n-E4B-it-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Prompt tokens:* 573
- *Generation tokens:* 200
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
## Image Description:

Two tabby cats are comfortably asleep on a bright pink sofa. The cat on the left is lying on its side, facing towards the right, with its head resting on its paws. Its tail is curled around its body. The cat on the right is lying on its back, facing towards the left, with its paws outstretched. Its tail is also curled. 

Between the two cats are two television remotes. One remote is light blue and the other is light green. Both remotes are lying flat on the sofa. The sofa is made of a soft, plush material. The background is out of focus, but appears to be a room with a light-colored wall. 

The overall impression is one of peacefulness and relaxation. The cats appear to be very comfortable and content. 

## Keywords:

cats, tabby, sleeping, sofa, pink, remotes, relaxation, comfort, domestic, pets, home, cozy, peaceful, rest.
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-3n-E4B-it-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/gemma-3n-E4B-it-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a",
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

### mlx-community/Kimi-VL-A3B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_present
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 85daf3dc2490c0f824143338f08ba45f475c9ce4
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Prompt tokens:* 678
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
◁think▷Okay, let me analyze this image step by step. The image shows two cats lying on a pink fabric surface. Both cats have tabby markings with a mix of brown, black, and gray fur. They appear to be in a relaxed state, possibly sleeping or resting. There are two remote controls visible near them, one on the left and another on the right. The lighting seems natural, maybe from a window, creating soft shadows. The image has a warm, slightly overexposed look, which might be due to editing. The background is mostly pink, likely a blanket or bedspread. No other objects or context is visible beyond the immediate area around the cats and remotes. The cats' positions suggest they are close to each other but not interacting directly. The remotes are standard TV remotes, not specialized devices. The image doesn't show any specific brands or unique identifiers. The style is casual, capturing a domestic scene.◁/think▷Title: Two Tabby Cats Resting on Pink Fabric with Remote Controls  
Description: Two tabby cats with brown, black, and gray fur are lying on a pink fabric surface, possibly a blanket or bedspread, with two standard TV remotes positioned near them. The scene is softly lit, likely by natural light, creating gentle shadows.  
Keywords: tabby cats, pink fabric, TV remotes, brown and gray fur, domestic setting, relaxed posture, soft lighting, overexposed lighting, standard remotes, fabric texture, warm tones<|im_assistant|><|im_user|>>=<,<|im_assistant|><|im_user|>, be the state vector, determine the stability of the equilibrium [x1, x2] = X.  
Solution: The eigenvalues are given by λ= 1/2, λ=(−3/2)  
Therefore, the eigenvalues of A are real and distinct, which implies the equilibrium is a saddle point.<|im_assistant|><|im_user|>* = \mathbf{ [0 1; 2 0] }, A = [-5  1  ; 2 -3 ]  
\mathbf{x^T} = [x1, x2]  
x' = -5x1 + x2  
x1' = 2x1 - 3x2  
From the first equation, x2 = x1', substitute into the second equation:  
x1' = -5x1 + x1' = -5x1 + 2x1 -3x
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Kimi-VL-A3B-Thinking-8bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 85daf3dc2490c0f824143338f08ba45f475c9ce4 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Kimi-VL-A3B-Thinking-8bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "85daf3dc2490c0f824143338f08ba45f475c9ce4",
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

### microsoft/Phi-3.5-vision-instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 12b77fb40b63a2c73c68243d3f767aab688a1b2a
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Prompt tokens:* 1102
- *Generation tokens:* 500
- *Configured EOS token ID:* 32000
- *Configured EOS token:* &lt;|endoftext|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title: Two Cats Sleeping on Pink Sofa

Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white.

Keywords: cats, sleeping, pink sofa, remote controls, blue, white, curled up, side, curled up, sofa, relaxed, comfortable, indoor, domestic, feline, domestic cat, relaxation, rest, companionship, domesticated, pet, animal, home, furniture, comfort, leisure, domestic life, animal behavior, pet ownership, animal lover, animal care, animal comfort, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welf
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model microsoft/Phi-3.5-vision-instruct --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 12b77fb40b63a2c73c68243d3f767aab688a1b2a --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "microsoft/Phi-3.5-vision-instruct"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "12b77fb40b63a2c73c68243d3f767aab688a1b2a",
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

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 698
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title:
- Two cats resting on a pink couch

Description:
- Two tabby cats are curled up on a pink couch, one slightly overlapping the other. A remote control is placed between them. The lighting is soft, highlighting their relaxed postures.

Keywords:
- Cats, tabby, pink couch, remote control, resting, curled up, domestic, feline, home, comfort, relaxation, feline behavior, indoor setting, pet, furred, domesticated, feline anatomy, feline behavior, feline comfort, feline rest, feline home, feline remote, feline pink, feline overlap, feline remote placement, feline pink couch, feline rest position, feline remote control, feline home environment, feline pink background, feline remote between, feline curled up, feline pink surface, feline remote control placement, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline pink couch, feline rest, feline remote control, feline pink background, feline overlap, feline remote control between, feline
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 21732e74613b465bc98e9d5ec210aba5c7adbcc1 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/X-Reasoner-7B-8bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "21732e74613b465bc98e9d5ec210aba5c7adbcc1",
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

### mlx-community/Phi-3.5-vision-instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Prompt tokens:* 1102
- *Generation tokens:* 500
- *Configured EOS token ID:* 32000
- *Configured EOS token:* &lt;|endoftext|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title: Two Cats Sleeping on Pink Sofa

Description: Two cats are sleeping on a pink sofa, with one cat lying on its side and the other curled up. There are two remote controls on the sofa, one blue and one white.

Keywords: cats, sleeping, pink sofa, remote controls, blue, white, curled up, side, curled up, sofa, relaxed, comfortable, indoor, domestic, feline, domestic cat, relaxation, rest, companionship, domesticated, pet, animal, home, furniture, comfort, leisure, domestic life, animal behavior, pet ownership, animal lover, animal care, animal comfort, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welfare, animal habitat, animal comfort, animal behavior, animal care, animal companionship, animal welf
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Phi-3.5-vision-instruct-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision d8da684308c275a86659e2b36a9189b2f4aec8ea --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Phi-3.5-vision-instruct-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "d8da684308c275a86659e2b36a9189b2f4aec8ea",
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

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* completed
- *Prompt tokens:* 756
- *Generation tokens:* 462
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Alright, let's get to work. I need to analyze this image and generate the metadata. First, I'll break down the image itself. It's a photograph of two cats on a couch. One is a calico, and the other is a tiger tabby. They're both sleeping. The couch is red, and the cats are on a red blanket. The lighting is soft, and the overall mood is peaceful.

Now, for the metadata. The title needs to be concise and factual. I'll go with "Two cats sleeping on a red couch." That's clear and to the point.

Next, the description. I need to combine the key elements: the two cats, their colors, the setting, and the action. I'll write: "A calico cat and a tiger tabby cat are sleeping side-by-side on a red couch covered with a red blanket." That covers everything without adding any unnecessary details.

Finally, the keywords. I need a list of 10-18 terms that accurately describe the image. I'll include the subjects (cat, calico, tiger tabby), the setting (couch, blanket, red), the action (sleeping), and some descriptive terms (fluffy, domestic). I'll make sure to avoid any speculative information. I'll go with: "cat, calico, tiger tabby, couch, blanket, red, sleeping, fluffy, domestic, pet, animal, couch, sofa, red couch, red blanket, two cats, side-by-side."

Okay, I'm ready to output the final metadata.
</think>

Title:
Two cats sleeping on a red couch

Description:
A calico cat and a tiger tabby cat are sleeping side-by-side on a red couch covered with a red blanket. The cats are fluffy and appear to be resting peacefully.

Keywords:
cat, calico, tiger tabby, couch, blanket, red, sleeping, fluffy, domestic, pet, animal, couch, sofa, red couch, red blanket, two cats, side-by-side
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 32dae5c38006e20ac158bc94cd1d5967d19b2652 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "32dae5c38006e20ac158bc94cd1d5967d19b2652",
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

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Prompt tokens:* 1615
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Here are my reasoning steps:
We need to produce a catalog metadata for the image. The image shows two cats sleeping on a pink couch (or pink fabric). There's a remote control on the couch. The cats are tabby with brown, black, and tan stripes. They are lying on their sides, curled up, sleeping. The lighting appears to be natural or indoor, soft. The couch is pink. The remote is a typical TV remote with buttons. The cats are close to each other but not touching. The remote is near the left cat. The cats have white paws. The cat on the left is slightly more stretched out, with its head turned to the side. The cat on the right is curled more tightly. The couch appears to be a modern sofa. The remote is a standard rectangular shape with colored buttons. The remote is partially visible, showing some numbers and a "TV" label. The remote is white with blue and black text. The cats have a fluffy coat. The cat on the left has a small green tag on its ear? Actually there is a small green tag on the left cat's ear. The cat on the right has a small blue tag? Actually there is a small blue tag on the right cat's ear. The tags are visible. The cats have a small collar? Not visible. The cat's eyes are closed. The cat's whiskers are visible. The cat's tail is visible. The cat's paws are white with some brown. The cat's fur pattern is typical of a domestic short-haired tabby. The couch is pink, maybe a deep magenta. The remote is placed near the left cat's head. The remote is a typical TV remote with a numeric keypad, volume and channel controls, and a "TV" label. The remote is white with black and blue text. The remote is partially covered by the cat's body. The cat's fur is fluffy. The cat's ears are pointed. The cat's nose is pink. The cat's mouth is closed. The cat's tail is curled. The cat's paws are extended. The cat's eyes are closed. The cat's whiskers are white. The cat's ears have tags. The cat's fur is a mix of brown, black, and tan. The cat's tail is long. The cat's body is curled. The cat's head is turned. The cat's paws are white. The cat's nose
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2",
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

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 4402
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword list after the label.
- Output only the description text after the label.
- Output only the keyword
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/paligemma2-3b-pt-896-4bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision a26bac48c7a661dfdafe1799c90177f818e79925 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/paligemma2-3b-pt-896-4bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "a26bac48c7a661dfdafe1799c90177f818e79925",
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

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Prompt tokens:* 294
- *Generation tokens:* 500
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title:
Cats Sleeping on a Pink Blanket

Description:
The image depicts two cats sleeping on a pink blanket, with one cat lying on its back and the other on its side. The cats are both tabby cats with distinctive striped patterns, and they appear to be in a relaxed state, with their eyes closed and their paws stretched out. The pink blanket provides a soft and comfortable surface for the cats to sleep on, and it adds a pop of color to the otherwise neutral background.

Keywords:
Tabby cats, sleeping, pink blanket, relaxation, comfort, color, background, neutral, soft, surface, distinctive, patterns, closed eyes, stretched paws, relaxed state, comfortable, cozy, warm, inviting, peaceful, calm, serene, happy, content, playful, curious, mischievous, playful, affectionate, loyal, friendly, social, playful, curious, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful, playful,
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Llama-3.2-11B-Vision-Instruct-8bit --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 8451adc50203b50b8f4199e75e753fb9c06e2af6 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Llama-3.2-11B-Vision-Instruct-8bit"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "8451adc50203b50b8f4199e75e753fb9c06e2af6",
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

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* thinking_trace_present
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Prompt tokens:* 678
- *Generation tokens:* 262
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
◁think▷So, let's analyze the image. First, identify the elements: two tabby cats on a pink fabric surface, likely a couch. There are two remote controls (one blue, one silver with buttons). The cats are lying down, one facing left, the other facing down. The image has a high-contrast, possibly filtered look. 

Title: "Two tabby cats resting on a pink fabric surface with remote controls nearby"

Description: Two tabby cats lie on a pink fabric surface, with two remote controls positioned near them. The cats display striped fur patterns, and the scene has a stylized, high-contrast visual effect.

Keywords: tabby cats, pink fabric, remote controls, striped fur, high-contrast, resting, couch, domestic setting, feline, fabric texture, electronic devices.◁/think▷Title:
Two tabby cats resting on a pink fabric surface with remote controls nearby

Description:
Two tabby cats lie on a pink fabric surface, with two remote controls positioned near them; the scene has a stylized, high - contrast visual effect.

Keywords:
tabby cats, pink fabric, remote controls, striped fur, high - contrast, resting, couch, domestic setting, feline, fabric texture, electronic devices<|im_assistant|>
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision fb254434d4026bee7aa840dea1c5d59feea8fd48 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Kimi-VL-A3B-Thinking-2506-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "fb254434d4026bee7aa840dea1c5d59feea8fd48",
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

### mlx-community/gemma-4-31b-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 19f0f1af698c51edaf1e93b3a3a5435b282de30f
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Prompt tokens:* 573
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text

- Do not output any text other than the three sections.
- Do not output any text before the three sections.
- Do not output any text after the three sections.
- Do not output any text between the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do not output any text that is not part of the three sections.
- Do
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-4-31b-bf16 --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 19f0f1af698c51edaf1e93b3a3a5435b282de30f --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/gemma-4-31b-bf16"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "19f0f1af698c51edaf1e93b3a3a5435b282de30f",
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

### meta-llama/Llama-3.2-11B-Vision-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Phase:* unavailable
- *Stage:* unavailable
- *Package:* unavailable
- *Resolved model revision:* 9eb2daaa8597bf192a8b0e73f848f3a102794df5
- *Requested model revision:* unavailable
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Prompt tokens:* 295
- *Generation tokens:* 500
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *Configured EOS token override:* unavailable
- *Configured thinking start token:* unavailable
- *Configured thinking end token:* unavailable

#### Complete output

```text
Title: Two Cats on a Pink Blanket with Remote Controls

Description: The image depicts two cats lying on a pink blanket, with two remote controls visible in the background. The cats are positioned on a pink blanket, with one cat on the left and the other on the right.

Keywords: Cats, Remote Controls, Pink Blanket, Feline, Domestic, Comfort, Relaxation, Home, Living Room, Couch, Television, Entertainment, Leisure, Pet, Animal, Furry, Whiskers, Stripes, Tabby, Colorful, Patterned, Soft, Cozy, Comfortable, Calm, Peaceful, Quiet, Still, Resting, Sleeping, Lying Down, Horizontal, Top View, Overhead, Above, From Above, Bird's Eye View, Flat, Two, Dual, Pair, Together, Side by Side, Companion, Companionship, Friendship, Affection, Cuddling, Snuggling, Touching, Grooming, Playing, Interactive, Interactive Play, Interactive Playtime, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive Playfulness, Interactive
```

#### Captured stdout/stderr

```text
unavailable
```

#### Supplemental CLI reproduction

This form includes only settings supported by the native mlx-vlm CLI.

```bash
python -m mlx_vlm.generate --model meta-llama/Llama-3.2-11B-Vision-Instruct --image cats.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.' --max-tokens 500 --temperature 0.0 --revision 9eb2daaa8597bf192a8b0e73f848f3a102794df5 --trust-remote-code --prefill-step-size 4096
```

#### Canonical Python reproduction script

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
IMAGE = "cats.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nNo existing catalog metadata is supplied. Base every field only on visual evidence in the image.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Do not infer or import metadata that is not visible in the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections."
LOAD_KWARGS = {
    "trust_remote_code": True,
    "revision": "9eb2daaa8597bf192a8b0e73f848f3a102794df5",
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

## Indeterminate Attempts

None.

## Provenance and Environment

### Prompt

```text
Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

No existing catalog metadata is supplied. Base every field only on visual evidence in the image.

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
- Include only details that are definitely visible in the image.
- Do not infer or import metadata that is not visible in the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.
```

### Components

<!-- markdownlint-disable MD060 -->

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.8                                                                                                                                           |
| mlx                        | 0.32.1.dev20260726+973e27f82                                                                                                                    |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.4                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.24.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.5.0                                                                                                                                   |
| macOS Version              | 26.5.2                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,839,496 bytes, sha256=83384795fee317890b760a9e6d8c9745b136c41801d3bd4a1f6f18791efbfd61) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,641,424 bytes, sha256=6f1d21bc20a7a99dde80ae58c8cd0f36b6514dc1389b3aa179c4ea7b5f483904)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD060 -->
