# Diagnostics

## Run Summary

Outcome counts

| Outcome             |   Count |
|---------------------|---------|
| Attempted           |      62 |
| Conclusive outcomes |      61 |
| Completed           |      60 |
| Crashed             |       1 |
| Indeterminate       |       1 |

Maintainer status counts

| Maintainer status              |   Count |
|--------------------------------|---------|
| actionable failure             |       1 |
| none                           |      31 |
| observation needs reproduction |      30 |

Usability counts

| Usability     |   Count |
|---------------|---------|
| not evaluated |       2 |
| unusable      |      30 |
| usable        |      30 |

Observation counts

| Observation                                                                           |   Count |
|---------------------------------------------------------------------------------------|---------|
| Extra text appears before the Title field                                             |      12 |
| Internal reasoning block appears incomplete                                           |       3 |
| Internal reasoning text remains visible                                               |       3 |
| No response text was returned                                                         |       3 |
| Required fields are missing or empty                                                  |      27 |
| Response appears cut off at the token limit                                           |      10 |
| Response repeats the same text                                                        |       4 |
| Response repeats the task instructions instead of only returning the requested fields |       2 |
| Unrecognised model control tokens remain visible                                      |       5 |

## Triage

| Model                                                                                                           | Execution     | Usability     | Maintainer status              | Observations                                                                                                                                                                                                        |
|-----------------------------------------------------------------------------------------------------------------|---------------|---------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/Step-3.7-Flash-oQ2e](#diagnostic-mlx-community-step-37-flash-oq2e)                               | crashed       | not_evaluated | actionable_failure             | none                                                                                                                                                                                                                |
| [mlx-community/nanoLLaVA-1.5-4bit](#diagnostic-mlx-community-nanollava-15-4bit)                                 | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [HuggingFaceTB/SmolVLM-Instruct](#diagnostic-huggingfacetb-smolvlm-instruct)                                    | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [mlx-community/SmolVLM-Instruct-bf16](#diagnostic-mlx-community-smolvlm-instruct-bf16)                          | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [qnguyen3/nanoLLaVA](#diagnostic-qnguyen3-nanollava)                                                            | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [mlx-community/MiniCPM-V-4.6-8bit](#diagnostic-mlx-community-minicpm-v-46-8bit)                                 | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     |
| [mlx-community/FastVLM-0.5B-bf16](#diagnostic-mlx-community-fastvlm-05b-bf16)                                   | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     |
| [mlx-community/paligemma2-10b-ft-docci-448-6bit](#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit)    | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [mlx-community/LFM2-VL-1.6B-8bit](#diagnostic-mlx-community-lfm2-vl-16b-8bit)                                   | completed     | unusable      | observation_needs_reproduction | Response repeats the same text; Response appears cut off at the token limit                                                                                                                                         |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed     | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed     | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   |
| [mlx-community/gemma-3n-E4B-it-bf16](#diagnostic-mlx-community-gemma-3n-e4b-it-bf16)                            | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                     |
| [mlx-community/paligemma2-10b-ft-docci-448-bf16](#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16)    | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [mlx-community/gemma-3n-E2B-4bit](#diagnostic-mlx-community-gemma-3n-e2b-4bit)                                  | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                   |
| [mlx-community/llava-v1.6-mistral-7b-8bit](#diagnostic-mlx-community-llava-v16-mistral-7b-8bit)                 | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [jqlive/Kimi-VL-A3B-Thinking-2506-6bit](#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit)                      | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  |
| [mlx-community/gemma-4-31b-bf16](#diagnostic-mlx-community-gemma-4-31b-bf16)                                    | completed     | unusable      | observation_needs_reproduction | No response text was returned; Required fields are missing or empty                                                                                                                                                 |
| [mlx-community/Kimi-VL-A3B-Thinking-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit)                  | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  |
| [mlx-community/paligemma2-3b-ft-docci-448-bf16](#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16)      | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [mlx-community/GLM-4.6V-Flash-6bit](#diagnostic-mlx-community-glm-46v-flash-6bit)                               | completed     | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   |
| [mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX](#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed     | unusable      | observation_needs_reproduction | Response repeats the same text; Unrecognised model control tokens remain visible; Extra text appears before the Title field; Response appears cut off at the token limit                                            |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed     | unusable      | observation_needs_reproduction | Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit            |
| [mlx-community/Molmo-7B-D-0924-bf16](#diagnostic-mlx-community-molmo-7b-d-0924-bf16)                            | completed     | unusable      | observation_needs_reproduction | No response text was returned; Required fields are missing or empty                                                                                                                                                 |
| [mlx-community/Molmo-7B-D-0924-8bit](#diagnostic-mlx-community-molmo-7b-d-0924-8bit)                            | completed     | unusable      | observation_needs_reproduction | No response text was returned; Required fields are missing or empty                                                                                                                                                 |
| [mlx-community/MolmoPoint-8B-fp16](#diagnostic-mlx-community-molmopoint-8b-fp16)                                | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed     | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                   |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                  | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                   |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed     | unusable      | observation_needs_reproduction | Response repeats the same text; Response appears cut off at the token limit                                                                                                                                         |
| [mlx-community/Qwen2-VL-2B-Instruct-4bit](#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit)                  | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed     | unusable      | observation_needs_reproduction | Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible  |
| [mlx-community/SmolVLM2-2.2B-Instruct-mlx](#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx)                 | indeterminate | not_evaluated | none                           | none                                                                                                                                                                                                                |

## Actionable Failures

<a id="diagnostic-mlx-community-step-37-flash-oq2e"></a>

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

#### Complete traceback

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

#### Captured stdout/stderr

```text
=== STDERR ===
[01:10:27] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-nanollava-15-4bit"></a>

<details>
<summary>mlx-community/nanoLLaVA-1.5-4bit — unusable — Required fields are missing or empty</summary>

### mlx-community/nanoLLaVA-1.5-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["keywords"]
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Processor class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.019252444
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 327
- *Generation tokens:* 121
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: The Cut in Halesworth, England, UK, GBR, Europe
Description: The Brickwork in Halesworth, Suffolk in the UK, featuring the Cut in the town center. The image captures the essence of the town's historic architecture, with the red brick building standing out amidst the greenery. The image is a blend of the old and the new, with the red brick building being the focal point, while the greenery adds a touch of nature to the scene. The image is a testament to the town's rich history and its blend of old and new architecture.
```

</details>

<a id="diagnostic-huggingfacetb-smolvlm-instruct"></a>

<details>
<summary>HuggingFaceTB/SmolVLM-Instruct — unusable — Required fields are missing or empty</summary>

### HuggingFaceTB/SmolVLM-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 81cd9a775a4d644f2faf4e7becff4559b46b14c7
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00011573
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1522
- *Generation tokens:* 13
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 The Cut in Halesworth, Suffolk in the UK.
```

</details>

<a id="diagnostic-mlx-community-smolvlm-instruct-bf16"></a>

<details>
<summary>mlx-community/SmolVLM-Instruct-bf16 — unusable — Required fields are missing or empty</summary>

### mlx-community/SmolVLM-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* cae61cdedd0602419b43b6102dc33cd9f1e929a6
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014517394
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1522
- *Generation tokens:* 13
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 The Cut in Halesworth, Suffolk in the UK.
```

</details>

<a id="diagnostic-qnguyen3-nanollava"></a>

<details>
<summary>qnguyen3/nanoLLaVA — unusable — Required fields are missing or empty</summary>

### qnguyen3/nanoLLaVA

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["keywords"]
- *Resolved model revision:* 13d60cec183a86755afed64da495fcc2c382ea80
- *Processor class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.02452812
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 327
- *Generation tokens:* 62
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: The Cut in Halesworth, England, UK, GBR, Europe
Description: A red brick building in Halesworth, Suffolk in the UK, with a distinctive red brick chimney and a large window. The building is surrounded by a white brick wall and has a sign on it.
```

</details>

<a id="diagnostic-mlx-community-minicpm-v-46-8bit"></a>

<details>
<summary>mlx-community/MiniCPM-V-4.6-8bit — unusable — Required fields are missing or empty; Extra text appears before the Title field</summary>

### mlx-community/MiniCPM-V-4.6-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* Industrial brick building with modern sign
  in town center.
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.007292024
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 918
- *Generation tokens:* 55
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Industrial brick building with modern sign in town center.

Keywords: brickwork, industrial, mill, red brick, building, sign, sky, clouds, England, Europe, Suffolk, Gravel, bushes, roof, trees, Car, Clouds, England, Europe
```

</details>

<a id="diagnostic-mlx-community-fastvlm-05b-bf16"></a>

<details>
<summary>mlx-community/FastVLM-0.5B-bf16 — unusable — Required fields are missing or empty; Extra text appears before the Title field</summary>

### mlx-community/FastVLM-0.5B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* A concrete 5-10-word title: Halesworth
  Industrial Site A 1-2-sentence factual description: The Cut in Halesworth,
  Suffolk, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Processor class:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002851938
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 331
- *Generation tokens:* 124
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
A concrete 5-10-word title: Halesworth Industrial Site
A 1-2-sentence factual description: The Cut in Halesworth, Suffolk, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00
Keywords: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
```

</details>

<a id="diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

<details>
<summary>mlx-community/paligemma2-10b-ft-docci-448-6bit — unusable — Required fields are missing or empty</summary>

### mlx-community/paligemma2-10b-ft-docci-448-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 1485fa9b3c7adb360cd354a29a401f0d441ec728
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.020284656
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1343
- *Generation tokens:* 12
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
Write the text in the same font as the image.
```

</details>

<a id="diagnostic-mlx-community-lfm2-vl-16b-8bit"></a>

<details>
<summary>mlx-community/LFM2-VL-1.6B-8bit — unusable — Response repeats the same text; Response appears cut off at the token limit</summary>

### mlx-community/LFM2-VL-1.6B-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* keyword: "england"
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 294b90e5ae2389ecb61a9427b4572975eef614fe
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.006112374
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 580
- *Generation tokens:* 500
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: The Cut, Halesworth, Suffolk, England, UK, GBR, Europe

Description: A red brick building with a white sign reading "The Cut" stands prominently in the town center of Halesworth, Suffolk, England. The building, likely an industrial or historical structure, features a series of windows and a flat roof. The surrounding area includes a brick wall and some greenery, with a clear blue sky overhead.

Keywords: Halesworth, Suffolk, England, UK, GBR, Europe, red brick, industrial building, town center, white sign, windows, brick wall, greenery, blue sky, England, Europe, industrial, historical, Halesworth, Suffolk, England, UK, GBR, Europe, The Cut, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR, Europe, red brick, industrial, Halesworth, England, UK, GBR, Europe, industrial, Halesworth, England, UK, GBR,
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-mxfp8 — unusable — Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field</summary>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|channel&gt;thought
  &lt;channel|&gt;Title: The Cut Arts Centre Building in Halesworth, Suffolk
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014828712
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 605
- *Generation tokens:* 86
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: The Cut Arts Centre Building in Halesworth, Suffolk
Description: A large multi-story red brick industrial building serves as The Cut arts centre, located in Halesworth under a partly cloudy sky.
Keywords: Halesworth, Suffolk, England, industrial, red brick, mill, arts much, architecture, building, sky, clouds, windows, sign, wall, historic, urban
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-8bit — unusable — Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field</summary>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|channel&gt;thought
  &lt;channel|&gt;Title: The Cut Arts Centre in Halesworth, Suffolk
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014828712
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 605
- *Generation tokens:* 85
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: The Cut Arts Centre in Halesworth, Suffolk
Description: A large multi-story red brick industrial building repurposed as an arts centre stands in Halesworth, England, under a partly cloudy sky.
Keywords: Halesworth, Suffolk, England, red brick, industrial, arts centre, mill, architecture, brickwork, clouds, sky, windows, sign, historic, Europe
```

</details>

<a id="diagnostic-mlx-community-gemma-3n-e4b-it-bf16"></a>

<details>
<summary>mlx-community/gemma-3n-E4B-it-bf16 — unusable — Required fields are missing or empty; Extra text appears before the Title field</summary>

### mlx-community/gemma-3n-E4B-it-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* The image shows a large, red brick building
  in Halesworth, Suffolk, UK. The building has multiple windows and a white
  sign that reads "THE CUT" in large letters. The building is surrounded by
  brick walls and some greenery. The sky is blue with some white clouds.
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.016991426
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 590
- *Generation tokens:* 104
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
The image shows a large, red brick building in Halesworth, Suffolk, UK. The building has multiple windows and a white sign that reads "THE CUT" in large letters. The building is surrounded by brick walls and some greenery. The sky is blue with some white clouds.

**Keywords:**

* Halesworth
* Suffolk
* UK
* The Cut
* Red brick building
* Windows
* White sign
* Brick walls
* Greenery
* Blue sky
* Clouds
```

</details>

<a id="diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

<details>
<summary>mlx-community/paligemma2-10b-ft-docci-448-bf16 — unusable — Required fields are missing or empty</summary>

### mlx-community/paligemma2-10b-ft-docci-448-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 7c412694b919432784c730b62fadafb1c2e15d0d
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.021316848
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1343
- *Generation tokens:* 10
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
Write the text in a plain text editor.
```

</details>

<a id="diagnostic-mlx-community-gemma-3n-e2b-4bit"></a>

<details>
<summary>mlx-community/gemma-3n-E2B-4bit — unusable — Required fields are missing or empty; Response appears cut off at the token limit</summary>

### mlx-community/gemma-3n-E2B-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation
- *Missing sections:* ["title", "description", "keywords"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* ec68dc186276e20e4bed30b96a2b5c667e0a81e3
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.016450754
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 582
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text

- 10-18-10-18-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1-1
```

</details>

<a id="diagnostic-mlx-community-llava-v16-mistral-7b-8bit"></a>

<details>
<summary>mlx-community/llava-v1.6-mistral-7b-8bit — unusable — Required fields are missing or empty</summary>

### mlx-community/llava-v1.6-mistral-7b-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Processor class:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.018629852
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2413
- *Generation tokens:* 133
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
The Cut, Halesworth, Suffolk, England, UK, GBR, Europe, 2026-08-01 16:28:40 UTC+01:00, 52.345200°N, 1.503700°E, Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk 
```

</details>

<a id="diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

<details>
<summary>jqlive/Kimi-VL-A3B-Thinking-2506-6bit — unusable — Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible</summary>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_present,
  thinking_trace_incomplete
- *Missing sections:* ["title"]
- *Unexpected text before Title:* ◁think▷Got it, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. Let's start by analyzing the image and the hints.  The
  image shows a large red-brick building, which is "The Cut" in Halesworth,
  Suffolk, UK. The context mentions the capture date/time and GPS, but we
  treat those as authoritative but not visible. The descriptive hints include
  elements like brickwork, blue sky, bushes, gravel, etc.  First, the title:
  need a 5-10 word title. Let's see, the building is a red-brick industrial
  building, so maybe "Red-brick industrial building in Halesworth, Suffolk"
  but need to make it concise. Wait, the hint says "The Cut in Halesworth,
  Suffolk in the UK" so maybe "The Cut, Halesworth, Suffolk: Red-brick
  industrial building" but check word count. Let's count: "The Cut,
  Halesworth, Suffolk: Red-brick industrial building" is 10 words. Good.
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* 4b3b11ce0874c36a99e13e17e355049042f8620a
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001213516
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1308
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

The image shows a large red-brick building, which is "The Cut" in Halesworth, Suffolk, UK. The context mentions the capture date/time and GPS, but we treat those as authoritative but not visible. The descriptive hints include elements like brickwork, blue sky, bushes, gravel, etc.

First, the title: need a 5-10 word title. Let's see, the building is a red-brick industrial building, so maybe "Red-brick industrial building in Halesworth, Suffolk" but need to make it concise. Wait, the hint says "The Cut in Halesworth, Suffolk in the UK" so maybe "The Cut, Halesworth, Suffolk: Red-brick industrial building" but check word count. Let's count: "The Cut, Halesworth, Suffolk: Red-brick industrial building" is 10 words. Good.

Description: 1-2 sentences. The main subject is the red-brick building, setting is town centre, Halesworth, England. Visible details: blue sky, brickwork, bushes, gravel. So "A large red-brick industrial building with white facade, situated in Halesworth's town centre, featuring brickwork, blue sky, bushes, and gravel under a partly cloudy sky." Wait, the image has a blue sky with clouds, so "blue sky" and "clouds" are visible. Also, the building has a white facade with "THE CUT" sign. So maybe: "A large red-brick industrial building with a white facade displaying 'THE CUT' signage, located in Halesworth's town centre, England, featuring brickwork, blue sky, bushes, and gravel under a partly cloudy sky."

Keywords: 10-18 unique keywords. Let's list from the hints and image: Adobe Stock, Any Vision, Arts centre (maybe not visible, but hint says Arts centre, so include), Blue sky, Brickwork, Bushes, Car (no car visible, so omit), Clouds, England, Europe, Gravel, Halesworth, Industrial, Mill (maybe, since it's an old industrial building), Red Brick Building, Roof (the roof is visible), Sign (THE CUT sign), Sky, Suffolk. Wait, the
```

</details>

<a id="diagnostic-mlx-community-gemma-4-31b-bf16"></a>

<details>
<summary>mlx-community/gemma-4-31b-bf16 — unusable — No response text was returned; Required fields are missing or empty</summary>

### mlx-community/gemma-4-31b-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* empty_output, missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 19f0f1af698c51edaf1e93b3a3a5435b282de30f
- *Processor class:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.018040024
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 597
- *Generation tokens:* 1
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
(empty)
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-8bit — unusable — Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_present,
  thinking_trace_incomplete
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* ◁think▷Okay, let's tackle this query. The
  user wants me to create British-English catalogue metadata based on the
  image and the provided context. First, I need to parse the instructions
  carefully. The key points are:  1. **Title**: A concrete 5-10 word title. 2.
  **Description**: 1-2 sentences combining context and visible details,
  including setting, action, lighting, etc. 3. **Keywords**: 10-18 unique
  keywords, covering context and visible details.  The context given includes
  the capture date/time (2026-08-01 16:28:40 UTC+01:00) and GPS coordinates
  (52.345200°N, 1.503700°E). The location is Halesworth, Suffolk, UK. The
  image shows a brick building labeled "THE CUT" with a blue sky, clouds,
  bushes, gravel, etc.  First, the title should be concise. The building is a
  red brick building with "THE CUT" on it. Maybe "Red-Brick Building 'The Cut'
  in Halesworth, Suffolk" – that's 8 words. Good.  For the description, I need
  to combine the context and visible details. The main subject is the red
  brick building, part of the Cut in Halesworth. The setting is town centre,
  so I should mention that. The sky is blue with clouds, there's a gravel
  area, bushes, and the building has a sign. The building is an industrial
  structure, possibly a former mill. The lighting is daytime. So something
  like "A red brick building labeled 'The Cut' stands prominently in
  Halesworth's town centre, Suffolk, under a partly cloudy sky with gravel
  surroundings and lush greenery." That covers the main elements.
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* 85daf3dc2490c0f824143338f08ba45f475c9ce4
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.005981302
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1308
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let's tackle this query. The user wants me to create British-English catalogue metadata based on the image and the provided context. First, I need to parse the instructions carefully. The key points are:

1. **Title**: A concrete 5-10 word title.
2. **Description**: 1-2 sentences combining context and visible details, including setting, action, lighting, etc.
3. **Keywords**: 10-18 unique keywords, covering context and visible details.

The context given includes the capture date/time (2026-08-01 16:28:40 UTC+01:00) and GPS coordinates (52.345200°N, 1.503700°E). The location is Halesworth, Suffolk, UK. The image shows a brick building labeled "THE CUT" with a blue sky, clouds, bushes, gravel, etc.

First, the title should be concise. The building is a red brick building with "THE CUT" on it. Maybe "Red-Brick Building 'The Cut' in Halesworth, Suffolk" – that's 8 words. Good.

For the description, I need to combine the context and visible details. The main subject is the red brick building, part of the Cut in Halesworth. The setting is town centre, so I should mention that. The sky is blue with clouds, there's a gravel area, bushes, and the building has a sign. The building is an industrial structure, possibly a former mill. The lighting is daytime. So something like "A red brick building labeled 'The Cut' stands prominently in Halesworth's town centre, Suffolk, under a partly cloudy sky with gravel surroundings and lush greenery." That covers the main elements.

Keywords: Red brick building, THE CUT, Halesworth, Suffolk, UK, Europe, blue sky, clouds, bushes, gravel, brickwork, industrial architecture, town centre, historic mill conversion, signage, daylight. Let me check if these are all visible in the image. The image shows brickwork, the sign "THE CUT", the location in Halesworth, the sky, bushes, gravel, and it's daytime. Industrial architecture and historic mill conversion might be inferred from the building's style. Signage is visible. That should cover it. Need to make sure there are exactly 10-18 keywords. Let me count: 1. Red brick building 2. THE CUT 3
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

<details>
<summary>mlx-community/paligemma2-3b-ft-docci-448-bf16 — unusable — Required fields are missing or empty</summary>

### mlx-community/paligemma2-3b-ft-docci-448-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* f66333527ce75342b09d4df81873f65272ec2f30
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.02234904
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1343
- *Generation tokens:* 150
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
A three-quarters view of a red brick building with a pointed roof. The building is in the center of the image, and the side of the building facing the bottom right corner of the image is painted white. There is a gray concrete wall extending across the bottom of the image. There is a tree on the left side of the image, and there is a tree on the right side of the image. There is a gray brick wall extending across the bottom of the image in front of the red brick building. There is a gray cement wall extending across the bottom of the image in the foreground. There is a tree on the left side of the image in front of the gray cement wall. The sky is bright and filled with white clouds.
```

</details>

<a id="diagnostic-mlx-community-glm-46v-flash-6bit"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-6bit — unusable — Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field</summary>

### mlx-community/GLM-4.6V-Flash-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|begin_of_box|&gt;Title: The Cut,
  Halesworth
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* df9464782d3452e0dfd86afe0984f1c9eca75ca1
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003163234
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6342
- *Generation tokens:* 106
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: The Cut, Halesworth
Description: The image shows "The Cut" arts centre, a large red brick industrial-style building with a light blue section displaying the name and website, situated behind a brick wall with overgrown vegetation, under a blue sky with clouds, in Halesworth, Suffolk, England.
Keywords: The Cut, Halesworth, arts centre, red brick building, light blue section, brick wall, overgrown vegetation, blue sky, clouds, Suffolk, England, UK.<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

<details>
<summary>mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX — unusable — Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit</summary>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo, unexpected_catalog_preamble
- *Missing sections:* ["keywords"]
- *Echoed instruction fragments:* ["british english catalogue metadata from
  the image and"]
- *Unexpected text before Title:* Here are my reasoning steps: We need to
  produce British-English catalogue metadata from the image and supplied
  context. The image shows a large brick building, likely a mill, with a sign
  "THE CUT" on a white-painted side. The building is red brick, multiple
  stories, with many windows, some small, some larger. There's a brick wall in
  front, some vegetation, a gravel ground. The sky is partly cloudy with blue
  sky. There's a car partially visible? In the image, I see a car partially
  visible on the left side? Actually, there is a car partially visible near
  the left side of the building? Let's examine the image: The building is a
  large red brick structure, with a white side that has "THE CUT" sign. The
  building appears to be an industrial or mill building, possibly a converted
  arts centre. The surroundings include a brick wall, some vegetation, a
  gravel ground, and a partly cloudy blue sky. There's a car visible near the
  left side? The image shows a car partially visible behind the building?
  Actually, I see a car on the left side of the image, near the building's
  left side, maybe parked. The building is in Halesworth, Suffolk, England.
  The sign "THE CUT" suggests a location or perhaps a business. The building
  appears to be a former mill, now possibly an arts centre. The building has a
  pitched roof, with some chimneys? The building is large, with many windows,
  some with white frames. The building is surrounded by a brick wall. The
  ground is gravel. There's a small patch of grass and bushes. The sky is
  partly cloudy with blue sky.  We need to produce:
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Processor class:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002065504
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3007
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The image shows a large brick building, likely a mill, with a sign "THE CUT" on a white-painted side. The building is red brick, multiple stories, with many windows, some small, some larger. There's a brick wall in front, some vegetation, a gravel ground. The sky is partly cloudy with blue sky. There's a car partially visible? In the image, I see a car partially visible on the left side? Actually, there is a car partially visible near the left side of the building? Let's examine the image: The building is a large red brick structure, with a white side that has "THE CUT" sign. The building appears to be an industrial or mill building, possibly a converted arts centre. The surroundings include a brick wall, some vegetation, a gravel ground, and a partly cloudy blue sky. There's a car visible near the left side? The image shows a car partially visible behind the building? Actually, I see a car on the left side of the image, near the building's left side, maybe parked. The building is in Halesworth, Suffolk, England. The sign "THE CUT" suggests a location or perhaps a business. The building appears to be a former mill, now possibly an arts centre. The building has a pitched roof, with some chimneys? The building is large, with many windows, some with white frames. The building is surrounded by a brick wall. The ground is gravel. There's a small patch of grass and bushes. The sky is partly cloudy with blue sky.

We need to produce:

- Title: 5-10 words, concrete. Something like "The Cut – Halesworth Red Brick Mill". That's 6 words: "The Cut – Halesworth Red Brick Mill". Or "Red Brick Mill, The Cut, Halesworth". That's 6 words. Or "The Cut, Halesworth, Red Brick Mill". That's 6 words. Or "Halesworth Mill – The Cut". That's 5 words. Let's choose "The Cut – Halesworth Red Brick Mill". That's 6 words (including hyphen maybe counts as part of word). Title: "The Cut – Halesworth Red Brick Mill". That is okay.

- Description: 1-2
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — Response repeats the same text; Unrecognised model control tokens remain visible; Extra text appears before the Title field; Response appears cut off at the token limit</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  unexpected_catalog_preamble, unexpected_special_token
- *Repeated fragment:* keyword: "mill"
- *Unexpected text before Title:* Alright, let's get to work. I've got the
  image and the context, and I need to generate some metadata. First, the
  title. The hints are a bit of a mess, but the image is clear. "The Cut,
  Halesworth, Suffolk" is the most accurate and concise. Done.  Next, the
  description. I need to weave in the context – the date, time, and location –
  with the visual details. The image shows a red brick building, likely an old
  industrial site, with a sign. The hints mention "The Cut," "Halesworth," and
  "Suffolk," so that's where I'll start. I'll add the date and time, and
  describe the building's appearance and the sign. I'll also mention the sky
  and the general setting. I'll make sure to include the architectural style
  and the sign.  Finally, the keywords. I'll pull from the hints, but I need
  to be selective. I'll focus on the location, the building itself, the sign,
  the sky, and the general setting. I'll avoid anything that's not clearly
  visible or relevant. I'll make sure to include the architectural style, the
  sign, the sky, and the general setting. I'll also add the date and time.
  Okay, let's put it all together.
- *Unexpected special tokens:* ["&lt;/think&gt;"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00254064
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1641
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Alright, let's get to work. I've got the image and the context, and I need to generate some metadata. First, the title. The hints are a bit of a mess, but the image is clear. "The Cut, Halesworth, Suffolk" is the most accurate and concise. Done.

Next, the description. I need to weave in the context – the date, time, and location – with the visual details. The image shows a red brick building, likely an old industrial site, with a sign. The hints mention "The Cut," "Halesworth," and "Suffolk," so that's where I'll start. I'll add the date and time, and describe the building's appearance and the sign. I'll also mention the sky and the general setting. I'll make sure to include the architectural style and the sign.

Finally, the keywords. I'll pull from the hints, but I need to be selective. I'll focus on the location, the building itself, the sign, the sky, and the general setting. I'll avoid anything that's not clearly visible or relevant. I'll make sure to include the architectural style, the sign, the sky, and the general setting. I'll also add the date and time.

Okay, let's put it all together.

**Title:** The Cut, Halesworth, Suffolk

**Description:** A photograph of The Cut, an industrial red brick building in Halesworth, Suffolk, UK, taken on August 1st, 2026. The building features a sign and is set against a blue sky with fluffy clouds.

**Keywords:** The Cut, Halesworth, Suffolk, UK, England, Red Brick Building, Industrial, Mill, Sign, Blue Sky, Clouds, Gravel, Bushes, Adobe Stock, Arts centre, Location, Mill, Roof, Sky, Suffolk, UK, Europe, Gravel, Bushes, Car, Clouds, England, Europe, Brickwork, Sign, Sky, Industrial, Mill, Red Brick Building, Roof, Location.
</think>

Title: The Cut, Halesworth, Suffolk

Description: A photograph of The Cut, an industrial red brick building in Halesworth, Suff
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Response appears cut off at the token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "- 10-18 unique, comma-separate..."
- *Echoed instruction fragments:* ["title hint:", "description hint:",
  "keyword hints:", "10 18 unique comma separated keywords covering relevant"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.023381232
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4415
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Title hint: Town centre, Halesworth, England, UK, GBR, Europe
- Description hint: The Cut in Halesworth, Suffolk in the UK
- Keyword hints: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context
```

</details>

<a id="diagnostic-mlx-community-molmo-7b-d-0924-bf16"></a>

<details>
<summary>mlx-community/Molmo-7B-D-0924-bf16 — unusable — No response text was returned; Required fields are missing or empty</summary>

### mlx-community/Molmo-7B-D-0924-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* empty_output, missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* d871cbdb87a49b8071003098d6dbfd2a0f5a5b84
- *Processor class:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00870105
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1506
- *Generation tokens:* 1
- *Configured EOS token ID:* 151643
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
(empty)
```

</details>

<a id="diagnostic-mlx-community-molmo-7b-d-0924-8bit"></a>

<details>
<summary>mlx-community/Molmo-7B-D-0924-8bit — unusable — No response text was returned; Required fields are missing or empty</summary>

### mlx-community/Molmo-7B-D-0924-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* empty_output, missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Processor class:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.008389754
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1506
- *Generation tokens:* 1
- *Configured EOS token ID:* 151643
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
(empty)
```

</details>

<a id="diagnostic-mlx-community-molmopoint-8b-fp16"></a>

<details>
<summary>mlx-community/MolmoPoint-8B-fp16 — unusable — Required fields are missing or empty</summary>

### mlx-community/MolmoPoint-8B-fp16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Processor class:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009323652
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3123
- *Generation tokens:* 115
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
The Cut Arts Centre, Halesworth, Suffolk, UK

The Cut arts centre stands in Halesworth, Suffolk, UK, featuring a red brick building with a white side and a sign reading "The Cut" and "www.the-cut.org.uk". The structure is surrounded by a brick wall with bushes and trees, set against a blue sky with white clouds.

Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — unusable — Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|begin_of_box|&gt;Title: The Cut,
  Halesworth, Suffolk, England
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003785838
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6342
- *Generation tokens:* 104
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: The Cut, Halesworth, Suffolk, England
Description: A red brick industrial building with a light blue section displaying "THE CUT" signage, set against a blue sky with clouds, surrounded by brick walls and gravel, located in Halesworth, Suffolk, UK.
Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Arts centre, Red Brick Building, Brickwork, Blue sky, Clouds, Gravel, Industrial, Mill, Sign, Bushes<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Thinking-bf16 — unusable — Required fields are missing or empty; Response appears cut off at the token limit</summary>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation
- *Missing sections:* ["title", "description", "keywords"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.010830992
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16583
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title. The context says "The Cut in Halesworth, Suffolk in the UK" and the image shows a red brick building with "THE CUT" sign. So the title should be concise. Maybe "The Cut, Halesworth, Suffolk, UK" but need to check if it's 5-10 words. Let's see: "The Cut, Halesworth, Suffolk, UK" is 7 words. That's good.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image has a red brick building (The Cut), blue sky with clouds, brickwork, bushes, gravel ground. The building is industrial, maybe a mill. The capture date is 2026-08-01, but we don't claim that. The description should mention the building, location (Halesworth, Suffolk), the sign "THE CUT", the sky, and the setting. Let's structure: "A red brick industrial building labeled 'THE CUT' in Halesworth, Suffolk, UK, under a bright blue sky with scattered clouds, featuring brickwork, bushes, and a gravel foreground."

Wait, check the context: "The Cut in Halesworth, Suffolk in the UK" is the description hint. So the description should include that. Also, the building is a red brick mill? The context says "Mill" is a keyword. So "A red brick mill building labeled 'THE CUT' in Halesworth, Suffolk, UK, under a bright blue sky with scattered clouds, featuring brickwork, bushes, and a gravel foreground."

Now keywords. Need 10-18 unique, comma-separated. From context: Adobe Stock, Any Vision, Arts centre, Blue sky, Brickwork, Bushes, Car, Clouds, England, Europe, Gravel, Halesworth, Industrial, Locations, Mill, Red Brick Building, Roof, Sign, Sky, Suffolk. But we need to pick relevant ones. The image has: Red Brick Building, Roof (the roof is dark, maybe slate), Sign (THE CUT), Bushes, Clouds, Sky, England, Europe, Suffolk, Industrial, Mill, Brickwork, Gravel, Blue sky. Let's list unique ones. Let's check the image: the building is red brick, so "Red Brick Building" is
```

</details>

<a id="diagnostic-mlx-community-x-reasoner-7b-8bit"></a>

<details>
<summary>mlx-community/X-Reasoner-7B-8bit — unusable — Response repeats the same text; Response appears cut off at the token limit</summary>

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* keyword: "mill"
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.01482869
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16592
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: The Cut in Halesworth, Suffolk

Description: A historic red brick building with "The Cut" signage stands under a bright blue sky, showcasing industrial architecture in Halesworth, Suffolk, UK.

Keywords: The Cut, Halesworth, Suffolk, England, UK, Europe, Red brick building, Industrial, Arts centre, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic, Architecture, England, Europe, Mill, Signage, Blue sky, Clouds, Gravel, Bushes, Sign, Sky, Mill, Brickwork, Historic,
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-Instruct-4bit — unusable — Required fields are missing or empty</summary>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["keywords"]
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.0102084
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16592
- *Generation tokens:* 128
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
The Cut, Halesworth, Suffolk, UK, GBR, Europe

Description:
The Cut is a town centre arts centre located in Halesworth, Suffolk, UK. The building is a large, multi-story brick structure with a prominent sign that reads "The Cut" and a website address. The building is surrounded by a brick wall and has a blue sky with white clouds in the background. The building features large windows and a red brick facade, and there are bushes and plants growing around the building. The building is part of the Halesworth industrial complex and is a popular destination for arts and culture events.
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_present,
  thinking_trace_incomplete
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* ◁think▷Okay, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. The user wants a title, a descriptive paragraph, and
  10-18 keywords.  Starting with the title: The image shows a large red-brick
  building with "THE CUT" on it, located in Halesworth, Suffolk, UK. The
  setting is a town centre, so maybe something like "Red-brick building 'The
  Cut' in Halesworth, Suffolk, UK" but needs to be concise. Maybe "Red-brick
  building 'The Cut' in Halesworth, Suffolk, UK" or "Halesworth's Red Brick
  Building 'The Cut'". Wait, the title hint mentions "Town centre, Halesworth,
  England, UK, GBR, Europe" so including that. Maybe "Red-brick building 'The
  Cut' in Halesworth town centre, Suffolk, UK".  For the description: Need to
  combine context and visible details. The building is a large red-brick
  structure with multiple windows, a white section with "THE CUT" sign, brick
  walls, gravel ground, bushes, blue sky with clouds. So something like "A
  large red-brick building with multiple windows and a white section
  displaying 'THE CUT' stands in Halesworth town centre, surrounded by brick
  walls, gravel, and greenery under a partly cloudy blue sky."
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.005325934
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1308
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords. 

Starting with the title: The image shows a large red-brick building with "THE CUT" on it, located in Halesworth, Suffolk, UK. The setting is a town centre, so maybe something like "Red-brick building 'The Cut' in Halesworth, Suffolk, UK" but needs to be concise. Maybe "Red-brick building 'The Cut' in Halesworth, Suffolk, UK" or "Halesworth's Red Brick Building 'The Cut'". Wait, the title hint mentions "Town centre, Halesworth, England, UK, GBR, Europe" so including that. Maybe "Red-brick building 'The Cut' in Halesworth town centre, Suffolk, UK".

For the description: Need to combine context and visible details. The building is a large red-brick structure with multiple windows, a white section with "THE CUT" sign, brick walls, gravel ground, bushes, blue sky with clouds. So something like "A large red-brick building with multiple windows and a white section displaying 'THE CUT' stands in Halesworth town centre, surrounded by brick walls, gravel, and greenery under a partly cloudy blue sky."

Keywords: Let's list the visible elements. Red-brick building, Halesworth, England, UK, Europe, Town centre, The Cut, brickwork, brick walls, gravel, bushes, blue sky, clouds, industrial, locations, mill (maybe, since it's a large brick building which could be a mill), red brick building, sky, trees, satellite dish, satellite dish on the building, white section, sign, windows, industrial architecture. Wait, the keyword hints include "Industrial" and "Mill" as hints. So include those. Also, "Adobe Stock" and "Any Vision" are hints but maybe not visible. The user said to omit uncertain details. So focus on visible elements. Let's check the image: brickwork, brick walls, gravel, bushes, blue sky, clouds, red brick building, The Cut sign, industrial, Halesworth, England, UK, Europe, town centre, windows, satellite dish, trees, satellite dish on building, white section, sign. That's 15 keywords.
```

</details>

## Indeterminate Attempts

<a id="diagnostic-mlx-community-smolvlm2-22b-instruct-mlx"></a>

<details>
<summary>mlx-community/SmolVLM2-2.2B-Instruct-mlx — not_evaluated — indeterminate</summary>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Execution and provenance

- *Execution:* indeterminate
- *Usability:* not_evaluated
- *Maintainer status:* none
- *Observations:* none
- *Phase:* model_load
- *Stage:* Network Error
- *Package:* unknown
- *Error type:* ValueError
- *Error message:* Model loading failed: [Errno 54] Connection reset by peer
- *Root error type:* ReadError
- *Root error message:* [Errno 54] Connection reset by peer
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.014517394
- *Post-cleanup cache memory (GB):* 0.0

#### Captured stdout/stderr

```text
=== STDERR ===
[01:10:15] DEBUG    HF Cache Info for mlx-community/SmolVLM2-2.2B-Instruct-mlx: size=4290.2 MB, files=12
```

</details>

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                        | Performance                                                                                 |
|-------------------------------------------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | rev 6c33f49ebc0b; Lfm2VlProcessor; stop completed       | 380 prompt / 95 generated; 511 tok/s; 1.3 GB peak; cleanup 0.000247/0.0 GB active/cache     |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | rev 16a710cf8afc; Lfm2VlProcessor; stop completed       | 580 prompt / 127 generated; 187 tok/s; 4.1 GB peak; cleanup 0.00624/0.0 GB active/cache     |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed       | 609 prompt / 105 generated; 120 tok/s; 16 GB peak; cleanup 0.0175/0.0 GB active/cache       |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed     | 2808 prompt / 141 generated; 187 tok/s; 7.3 GB peak; cleanup 0.00808/0.0 GB active/cache    |
| microsoft/Phi-3.5-vision-instruct                     | rev 12b77fb40b63; Phi3VProcessor; stop completed        | 1123 prompt / 94 generated; 55.8 tok/s; 9.4 GB peak; cleanup 0.0018/0.0 GB active/cache     |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed        | 1123 prompt / 94 generated; 53.7 tok/s; 9.4 GB peak; cleanup 0.0099/0.0 GB active/cache     |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed | 2110 prompt / 84 generated; 34.4 tok/s; 18 GB peak; cleanup 0.00467/0.0 GB active/cache     |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed     | 2809 prompt / 140 generated; 67.0 tok/s; 13 GB peak; cleanup 0.00755/0.0 GB active/cache    |
| mlx-community/InternVL3-14B-8bit                      | rev 50efc568c7df; InternVLChatProcessor; stop completed | 2110 prompt / 89 generated; 32.5 tok/s; 18 GB peak; cleanup 0.00436/0.0 GB active/cache     |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed      | 2916 prompt / 96 generated; 36.9 tok/s; 15 GB peak; cleanup 0.0236/0.0 GB active/cache      |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed     | 2809 prompt / 187 generated; 63.9 tok/s; 13 GB peak; cleanup 0.00782/0.0 GB active/cache    |
| mlx-community/Idefics3-8B-Llama3-bf16                 | rev 8c2a30c48864; Idefics3Processor; stop completed     | 2609 prompt / 122 generated; 32.4 tok/s; 19 GB peak; cleanup 0.00405/0.0 GB active/cache    |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed       | 609 prompt / 102 generated; 25.9 tok/s; 20 GB peak; cleanup 0.0186/0.0 GB active/cache      |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed       | 591 prompt / 132 generated; 29.5 tok/s; 18 GB peak; cleanup 0.0154/0.0 GB active/cache      |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | rev 8451adc50203; MllamaProcessor; stop completed       | 298 prompt / 105 generated; 21.7 tok/s; 15 GB peak; cleanup 0.00677/0.0 GB active/cache     |
| mlx-community/pixtral-12b-bf16                        | rev 378cf65efd79; PixtralProcessor; stop completed      | 2916 prompt / 94 generated; 19.8 tok/s; 27 GB peak; cleanup 0.0239/0.0 GB active/cache      |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed     | 2276 prompt / 109 generated; 27.1 tok/s; 23 GB peak; cleanup 0.00233/0.0 GB active/cache    |
| mlx-community/gemma-3-27b-it-qat-8bit                 | rev c408904bc9a0; Gemma3Processor; stop completed       | 591 prompt / 116 generated; 17.1 tok/s; 32 GB peak; cleanup 0.0159/0.0 GB active/cache      |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | rev 773591fa7388; Glm46VProcessor; stop completed       | 6342 prompt / 110 generated; 75.4 tok/s; 8.4 GB peak; cleanup 0.00347/0.0 GB active/cache   |
| Qwen/Qwen3-VL-2B-Instruct                             | rev 89644892e4d8; Qwen3VLProcessor; stop completed      | 16581 prompt / 110 generated; 93.9 tok/s; 8.6 GB peak; cleanup 0.000558/0.0 GB active/cache |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | rev c8a67a843274; Qwen3VLProcessor; stop completed      | 16581 prompt / 110 generated; 89.4 tok/s; 8.6 GB peak; cleanup 0.0105/0.0 GB active/cache   |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | rev 9eb2daaa8597; MllamaProcessor; stop completed       | 299 prompt / 114 generated; 5.02 tok/s; 25 GB peak; cleanup 0.00174/0.0 GB active/cache     |
| mlx-community/Ornith-1.0-35B-bf16                     | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed      | 16596 prompt / 126 generated; 59.8 tok/s; 76 GB peak; cleanup 0.00983/0.0 GB active/cache   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed      | 16596 prompt / 109 generated; 90.7 tok/s; 11 GB peak; cleanup 0.0139/0.0 GB active/cache    |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | rev b729d115bb2c; Qwen3VLProcessor; stop completed      | 16596 prompt / 114 generated; 65.0 tok/s; 35 GB peak; cleanup 0.0129/0.0 GB active/cache    |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed      | 16596 prompt / 122 generated; 107 tok/s; 26 GB peak; cleanup 0.0124/0.0 GB active/cache     |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | rev 731d09ba3597; Qwen3VLProcessor; stop completed      | 16596 prompt / 110 generated; 65.6 tok/s; 76 GB peak; cleanup 0.0134/0.0 GB active/cache    |
| mlx-community/Qwen3.5-27B-4bit                        | rev 45797d2985a1; Qwen3VLProcessor; stop completed      | 16596 prompt / 121 generated; 22.6 tok/s; 26 GB peak; cleanup 0.0113/0.0 GB active/cache    |
| mlx-community/Qwen3.5-27B-mxfp8                       | rev 2d6caf2325c2; Qwen3VLProcessor; stop completed      | 16596 prompt / 131 generated; 14.9 tok/s; 38 GB peak; cleanup 0.0118/0.0 GB active/cache    |
| mlx-community/Qwen3.6-27B-mxfp8                       | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed      | 16596 prompt / 120 generated; 14.2 tok/s; 38 GB peak; cleanup 0.0144/0.0 GB active/cache    |

</details>

## Shared Reproduction and Provenance

### Prompt

Save this exact prompt as prompt.txt.

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

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/Step-3.7-Flash-oQ2e                | 3dacb46f724ac89725bcd922fb779c7ed1499fe7 |
| mlx-community/nanoLLaVA-1.5-4bit                 | 5240204744963d72823e5de933c528c4aa82dfca |
| HuggingFaceTB/SmolVLM-Instruct                   | 81cd9a775a4d644f2faf4e7becff4559b46b14c7 |
| mlx-community/SmolVLM-Instruct-bf16              | cae61cdedd0602419b43b6102dc33cd9f1e929a6 |
| qnguyen3/nanoLLaVA                               | 13d60cec183a86755afed64da495fcc2c382ea80 |
| mlx-community/MiniCPM-V-4.6-8bit                 | 03721395f6b82cd000cc74cde28fcff8abd9a04c |
| mlx-community/FastVLM-0.5B-bf16                  | 81ffe929046666c43de53691147b1669ba0f3a4c |
| mlx-community/paligemma2-10b-ft-docci-448-6bit   | 1485fa9b3c7adb360cd354a29a401f0d441ec728 |
| mlx-community/LFM2-VL-1.6B-8bit                  | 294b90e5ae2389ecb61a9427b4572975eef614fe |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/gemma-3n-E4B-it-bf16               | d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a |
| mlx-community/paligemma2-10b-ft-docci-448-bf16   | 7c412694b919432784c730b62fadafb1c2e15d0d |
| mlx-community/gemma-3n-E2B-4bit                  | ec68dc186276e20e4bed30b96a2b5c667e0a81e3 |
| mlx-community/llava-v1.6-mistral-7b-8bit         | b8df5f329d95a7abe6429ed46093f9b84e8e6396 |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit            | 4b3b11ce0874c36a99e13e17e355049042f8620a |
| mlx-community/gemma-4-31b-bf16                   | 19f0f1af698c51edaf1e93b3a3a5435b282de30f |
| mlx-community/Kimi-VL-A3B-Thinking-8bit          | 85daf3dc2490c0f824143338f08ba45f475c9ce4 |
| mlx-community/paligemma2-3b-ft-docci-448-bf16    | f66333527ce75342b09d4df81873f65272ec2f30 |
| mlx-community/GLM-4.6V-Flash-6bit                | df9464782d3452e0dfd86afe0984f1c9eca75ca1 |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX    | 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2 |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/Molmo-7B-D-0924-bf16               | d871cbdb87a49b8071003098d6dbfd2a0f5a5b84 |
| mlx-community/Molmo-7B-D-0924-8bit               | 90a14ed7a230088904c7556fbe6d67b295c33f5f |
| mlx-community/MolmoPoint-8B-fp16                 | 0a60033b4e4813fb53df4c7523857d2ec972c7d9 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Qwen3-VL-2B-Thinking-bf16          | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/Qwen2-VL-2B-Instruct-4bit          | 01af461cdb9574acc09084a0ef94e216e142b085 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx         | 844516024a1c4400d34489b89ee067d794e432ed |

### Canonical parameterised Python reproduction

Run one model per process to avoid sequential Metal-state interactions.

```bash
python reproduce.py MODEL_ID --revision RESOLVED_REVISION --image 20260801-162840_DSC01220.jpg --prompt-file prompt.txt
```

```python
import argparse
from pathlib import Path

from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

LOAD_KWARGS = {
    "trust_remote_code": True,
}
TEMPLATE_KWARGS = {}
GENERATE_KWARGS = {
    "max_tokens": 500,
    "temperature": 0.0,
    "prefill_step_size": 4096,
}

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--revision")
parser.add_argument("--image", required=True)
parser.add_argument("--prompt-file", required=True)
args = parser.parse_args()
prompt = Path(args.prompt_file).read_text(encoding="utf-8")
load_kwargs = LOAD_KWARGS.copy()
if args.revision:
    load_kwargs["revision"] = args.revision
model, processor = load(args.model, **load_kwargs)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    prompt,
    num_images=1,
    **TEMPLATE_KWARGS,
)
if isinstance(formatted_prompt, list):
    formatted_prompt = "\n".join(str(message) for message in formatted_prompt)
result = generate(
    model,
    processor,
    formatted_prompt,
    image=args.image,
    **GENERATE_KWARGS,
)
print(result.text)

# Defaults used in the report: prompt.txt, 20260801-162840_DSC01220.jpg
```

### Components and system

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
