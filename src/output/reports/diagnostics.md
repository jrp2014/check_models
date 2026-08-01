# Diagnostics

## Run Summary

Outcome counts

| Outcome             |   Count |
|---------------------|---------|
| Attempted           |      62 |
| Conclusive outcomes |      62 |
| Completed           |      61 |
| Crashed             |       1 |
| Indeterminate       |       0 |

Maintainer status counts

| Maintainer status              |   Count |
|--------------------------------|---------|
| actionable failure             |       1 |
| none                           |      35 |
| observation needs reproduction |      26 |

Usability counts

| Usability     |   Count |
|---------------|---------|
| not evaluated |       1 |
| unusable      |      26 |
| usable        |      35 |

Observation counts

| Observation                                                                           |   Count |
|---------------------------------------------------------------------------------------|---------|
| Conversation-role control tokens remain visible                                       |       1 |
| Extra text appears before the Title field                                             |      12 |
| Internal reasoning block appears incomplete                                           |       1 |
| Internal reasoning text remains visible                                               |       3 |
| No response text was returned                                                         |       1 |
| Required fields are missing or empty                                                  |      24 |
| Response appears cut off at the token limit                                           |      10 |
| Response repeats the same text                                                        |       4 |
| Response repeats the task instructions instead of only returning the requested fields |       4 |
| Unrecognised model control tokens remain visible                                      |       4 |

## Triage

| Model                                                                                                           | Execution   | Usability     | Maintainer status              | Observations                                                                                                                                                                                                                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------|-------------|---------------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/Step-3.7-Flash-oQ2e](#diagnostic-mlx-community-step-37-flash-oq2e)                               | crashed     | not_evaluated | actionable_failure             | none                                                                                                                                                                                                                                                                                                          |
| [mlx-community/FastVLM-0.5B-bf16](#diagnostic-mlx-community-fastvlm-05b-bf16)                                   | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/nanoLLaVA-1.5-4bit](#diagnostic-mlx-community-nanollava-15-4bit)                                 | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/MiniCPM-V-4.6-8bit](#diagnostic-mlx-community-minicpm-v-46-8bit)                                 | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                                                                                                               |
| [qnguyen3/nanoLLaVA](#diagnostic-qnguyen3-nanollava)                                                            | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/LFM2.5-VL-1.6B-bf16](#diagnostic-mlx-community-lfm25-vl-16b-bf16)                                | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/SmolVLM-Instruct-bf16](#diagnostic-mlx-community-smolvlm-instruct-bf16)                          | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [HuggingFaceTB/SmolVLM-Instruct](#diagnostic-huggingfacetb-smolvlm-instruct)                                    | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/GLM-4.6V-Flash-6bit](#diagnostic-mlx-community-glm-46v-flash-6bit)                               | completed   | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                                                             |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed   | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                                                             |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed   | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                                                             |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                  | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit                                                                                           |
| [mlx-community/gemma-4-31b-bf16](#diagnostic-mlx-community-gemma-4-31b-bf16)                                    | completed   | unusable      | observation_needs_reproduction | No response text was returned; Required fields are missing or empty                                                                                                                                                                                                                                           |
| [mlx-community/gemma-3n-E2B-4bit](#diagnostic-mlx-community-gemma-3n-e2b-4bit)                                  | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                                                                                                             |
| [mlx-community/paligemma2-10b-ft-docci-448-6bit](#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit)    | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed   | unusable      | observation_needs_reproduction | Unrecognised model control tokens remain visible; Required fields are missing or empty; Extra text appears before the Title field                                                                                                                                                                             |
| [mlx-community/Kimi-VL-A3B-Thinking-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit)                  | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Internal reasoning text remains visible |
| [mlx-community/paligemma2-3b-ft-docci-448-bf16](#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16)      | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [jqlive/Kimi-VL-A3B-Thinking-2506-6bit](#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit)                      | completed   | unusable      | observation_needs_reproduction | Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible                                                                                                                                  |
| [mlx-community/llava-v1.6-mistral-7b-8bit](#diagnostic-mlx-community-llava-v16-mistral-7b-8bit)                 | completed   | unusable      | observation_needs_reproduction | Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                                                                             |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed   | unusable      | observation_needs_reproduction | Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit                                                           |
| [mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX](#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit                                                                                           |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty; Response appears cut off at the token limit                                                                                                                                                                                                                             |
| [mlx-community/paligemma2-10b-ft-docci-448-bf16](#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16)    | completed   | unusable      | observation_needs_reproduction | Required fields are missing or empty                                                                                                                                                                                                                                                                          |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed   | unusable      | observation_needs_reproduction | Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit                                                                                                                                                                                        |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed   | unusable      | observation_needs_reproduction | Response repeats the same text; Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning text remains visible                                                                                                         |

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
- *Post-cleanup active memory (GB):* 0.01459931
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
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 3397.46it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[00:24:59] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-fastvlm-05b-bf16"></a>

<details>
<summary>mlx-community/FastVLM-0.5B-bf16 — unusable — Required fields are missing or empty</summary>

### mlx-community/FastVLM-0.5B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Processor class:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002851938
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 120
- *Generation tokens:* 32
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
A serene moment of feline tranquility on a vibrant pink couch, with two cats in peaceful slumber, surrounded by the soft glow of ambient lighting.
```

</details>

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
- *Post-cleanup active memory (GB):* 0.01933436
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 116
- *Generation tokens:* 97
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: "A Study in Stripes"
Description: A close-up of a person wearing a striped shirt, with a focus on the pattern and color of the fabric. The image is set against a pink background, and the person is lying on a bed covered with a pink blanket. The lighting is soft, and the person is facing away from the camera, giving a sense of intimacy. The image is a straightforward representation of the subject, with no additional details or context provided.
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
- *Missing sections:* ["title"]
- *Unexpected text before Title:* Two cats sleeping on pink blanket with
  remote controls
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.007275636
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 330
- *Generation tokens:* 58
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Two cats sleeping on pink blanket with remote controls

Description: Two tabby cats rest on a pink fabric, with remote controls nearby; soft lighting and cozy setting.

Keywords: cats, sleeping, pink blanket, remote controls, tabby, relaxation, comfort, domestic, soft lighting
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
- *Post-cleanup active memory (GB):* 0.024610036
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 116
- *Generation tokens:* 40
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: A Cat Laying on a Couch
Description: A cat is laying on a couch, with its tail and paws visible. The cat is wearing a green and white striped shirt.
```

</details>

<a id="diagnostic-mlx-community-lfm25-vl-16b-bf16"></a>

<details>
<summary>mlx-community/LFM2.5-VL-1.6B-bf16 — unusable — Required fields are missing or empty</summary>

### mlx-community/LFM2.5-VL-1.6B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.006243442
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 368
- *Generation tokens:* 69
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Two cats sleeping on a pink blanket, one with a remote control nearby, the other with a collar.

The cats are lying on a pink blanket, one with a remote control nearby, the other with a collar.

Cats, blanket, remote control, collar, sleeping, pink, indoor, cozy, furry, relaxed
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
- *Post-cleanup active memory (GB):* 0.014484622
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1304
- *Generation tokens:* 9
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 Two Cats on a Pink Couch.
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
- *Prompt tokens:* 1304
- *Generation tokens:* 9
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 Two Cats on a Pink Couch.
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
- *Unexpected text before Title:* &lt;|begin_of_box|&gt;Title: Two tabby cats
  on pink couch
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* df9464782d3452e0dfd86afe0984f1c9eca75ca1
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003163234
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 499
- *Generation tokens:* 47
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: Two tabby cats on pink couch
Description: Two tabby cats rest on a pink couch, with remote controls nearby.
Keywords: tabby cats, pink couch, remote controls, resting cats, domestic cats<|end_of_box|>
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
  &lt;channel|&gt;Title: Two tabby cats sleeping on a pink blanket
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014910628
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 386
- *Generation tokens:* 74
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Two tabby cats sleeping on a pink blanket
Description: Two striped tabby cats are lying on a pink fabric surface on a sofa. Two remote controls are visible resting near them.
Keywords: cats, tabby, sleeping, lying, pink, blanket, sofa, indoor, remote, controls, striped, fur, resting, domestic, animals
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — unusable — Required fields are missing or empty</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.004047978
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2418
- *Generation tokens:* 13
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 Two cats laying on a pink blanket with remotes.<end_of_utterance>
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
  &lt;channel|&gt;Title: Two tabby cats sleeping on a pink blanket
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014910628
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 386
- *Generation tokens:* 75
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Two tabby cats sleeping on a pink blanket 
Description: Two tabby cats rest on a bright pink blanket draped over a red sofa. Two remote controls are visible near the animals.
Keywords: cats, tabby, sleeping, pink, blanket, sofa, remote, controls, indoor, domestic, feline, resting, fur, striped, lounging
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Thinking-bf16 — unusable — Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit</summary>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo, unexpected_catalog_preamble
- *Missing sections:* ["title", "keywords"]
- *Echoed instruction fragments:* ["omit uncertain details and unsupported
  identity location event"]
- *Unexpected text before Title:* Got it, let's tackle this. First, the title
  needs to be concrete and 5-10 words. The image has two cats on a pink couch
  with remotes. So maybe "Two tabby cats resting on pink couch with remotes"
  but that's 10 words. Wait, "Two tabby cats resting on pink couch with remote
  controls" but need to check. Wait, the problem says "British-English
  catalogue metadata", so maybe "Two tabby cats resting on pink couch with
  remote controls" but let's count: Two tabby cats resting on pink couch with
  remote controls is 10 words. That's good.  Next, the description. Need 1-2
  sentences. Main subject: two cats. Setting: pink couch. Action: resting.
  Lighting: bright (since it's a photo, probably natural light). Distinctive
  details: tabby patterns, remotes. So "Two tabby cats rest on a bright pink
  couch, one with a green collar, while two remote controls lie nearby."
  Wait, but the problem says "omit uncertain details and unsupported identity,
  location, event, brand, species, period, or intent." So species is cat,
  which is known, so okay. The remotes are visible, so "two remote controls"
  is okay. Lighting: the image is bright, so "bright lighting" or "natural
  light". Let's make it concise.
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.010814604
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 411
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this. First, the title needs to be concrete and 5-10 words. The image has two cats on a pink couch with remotes. So maybe "Two tabby cats resting on pink couch with remotes" but that's 10 words. Wait, "Two tabby cats resting on pink couch with remote controls" but need to check. Wait, the problem says "British-English catalogue metadata", so maybe "Two tabby cats resting on pink couch with remote controls" but let's count: Two tabby cats resting on pink couch with remote controls is 10 words. That's good.

Next, the description. Need 1-2 sentences. Main subject: two cats. Setting: pink couch. Action: resting. Lighting: bright (since it's a photo, probably natural light). Distinctive details: tabby patterns, remotes. So "Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby."

Wait, but the problem says "omit uncertain details and unsupported identity, location, event, brand, species, period, or intent." So species is cat, which is known, so okay. The remotes are visible, so "two remote controls" is okay. Lighting: the image is bright, so "bright lighting" or "natural light". Let's make it concise.

Description: "Two tabby cats rest on a pink couch, one with a green collar, while two remote controls lie nearby under the couch."

Wait, but the problem says "1-2 sentences". Let's check: "Two tabby cats rest on a bright pink couch, one with a green collar, while two remote controls lie nearby." That's two sentences. Wait, no, it's one sentence. Wait, the example in the problem says "1-2 sentences". So maybe "Two tabby cats rest on a pink couch, one with a green collar, while two remote controls lie nearby." That's one sentence. Wait, but the problem says "1-2 sentences", so that's okay.

Now keywords: 10-18 unique, comma-separated. Need to list distinct things. Let's see: cats, tabby, pink couch, remote controls, resting, two, couch, bright, collar, green, natural light. Wait, but need to be unique. Let's list: cats, tab
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
- *Post-cleanup active memory (GB):* 0.01812194
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 378
- *Generation tokens:* 1
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
(empty)
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
- *Post-cleanup active memory (GB):* 0.01653267
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 370
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text

- a 1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-21-22-23-24-25-26-27-28-29-30-31-32-33-34-35-36-37-38-39-40-41-42-43-44-45-46-47-48-49-50-51-52-53-54-55-56-57-58-59-60-61-62-63-64-65-66-67-68-69-70-71-72-73-74-75-76-77-76-78-79-80-81-82-83-84-85-86-87-88-89-90-91-92-93-94-95-96-97-98-99-100-101-102-103-104-105-106-107-108-109-110-111-112-113-114-115-116-117-118-119-120-121-122-123-124-125-126-127-128-129-130-131-132-133-134-135-136-137-138-139-140-141-142-143-144-145-146-147-148-149-150-1
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
- *Post-cleanup active memory (GB):* 0.020366572
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1135
- *Generation tokens:* 155
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
A top-down view of two cats lying on a pink blanket. The cat on the left is lying on its side with its head facing the top left corner of the image. Its body is facing the top right corner of the image. Its head is turned slightly to the left. Its tail is sticking out of the bottom left corner of the image. The cat on the right is lying on its side with its head facing the top right corner of the image. Its body is facing the bottom left corner of the image. Its head is turned slightly to the left. Its tail is sticking out of the bottom right corner of the image. There is a gray remote control on the left side of the image and a gray remote control on the right side of the image.
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
- *Unexpected text before Title:* &lt;|begin_of_box|&gt;Title: Two cats on a
  pink couch
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003785834
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 499
- *Generation tokens:* 89
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: Two cats on a pink couch
Description: Two cats lie on a pink couch, each near a remote control, with one cat’s head resting on the fabric and the other stretched out with eyes closed.
Keywords: cats, pink couch, remote controls, sleeping, relaxed, striped fur, feline, domestic, indoor, pets, lounging, two animals, soft fabric, calm, cozy, household, companionship<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-8bit — unusable — Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Internal reasoning text remains visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo, unexpected_catalog_preamble,
  thinking_trace_present, role_boundary_token_present
- *Missing sections:* ["title"]
- *Echoed instruction fragments:* ["identity location event brand species
  period or intent"]
- *Unexpected text before Title:* ◁think▷Okay, let's tackle this query. The
  user wants me to create British-English catalogue metadata based on the
  provided image. The instructions are clear: use only visible facts, omit
  uncertain details, and avoid things like identity, location, event, brand,
  species, period, or intent.  First, I need to analyze the image. There are
  two cats on a pink surface. The cats are lying down, one is stretching, the
  other is curled up. There are remote controls nearby. The image is highly
  saturated with vibrant colors. The main subject is the cats, so the title
  should reflect that. Maybe "Two cats resting on pink fabric with remote
  controls."  For the description, I need to mention the main elements:
  setting (pink fabric), action (resting, stretching), lighting (vibrant
  colors), and distinctive details (striped patterns, relaxed posture). Keep
  it concise, 1-2 sentences.  Keywords should be unique and relevant. The main
  elements are cats, pink fabric, remote controls, vibrant colors, striped
  patterns, relaxed posture, domestic setting, indoor environment, leisure
  activity, domestic animals, and soft lighting. Need to make sure they are
  comma-separated and within the count range. Check for any uncertain details.
  Since the image doesn't show the cats' breed or age, I shouldn't include
  those. Avoid any assumptions. Make sure all keywords are directly
  observable.◁/think▷Title: Two cats resting on pink fabric with remote
  controls
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;",
  "&lt;|im_user|&gt;"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* 85daf3dc2490c0f824143338f08ba45f475c9ce4
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.005981298
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 489
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let's tackle this query. The user wants me to create British-English catalogue metadata based on the provided image. The instructions are clear: use only visible facts, omit uncertain details, and avoid things like identity, location, event, brand, species, period, or intent. 

First, I need to analyze the image. There are two cats on a pink surface. The cats are lying down, one is stretching, the other is curled up. There are remote controls nearby. The image is highly saturated with vibrant colors. The main subject is the cats, so the title should reflect that. Maybe "Two cats resting on pink fabric with remote controls."

For the description, I need to mention the main elements: setting (pink fabric), action (resting, stretching), lighting (vibrant colors), and distinctive details (striped patterns, relaxed posture). Keep it concise, 1-2 sentences.

Keywords should be unique and relevant. The main elements are cats, pink fabric, remote controls, vibrant colors, striped patterns, relaxed posture, domestic setting, indoor environment, leisure activity, domestic animals, and soft lighting. Need to make sure they are comma-separated and within the count range. Check for any uncertain details. Since the image doesn't show the cats' breed or age, I shouldn't include those. Avoid any assumptions. Make sure all keywords are directly observable.◁/think▷Title: Two cats resting on pink fabric with remote controls  
Description: Two tabby cats lie relaxed on a vibrant pink fabric surface, one stretching while the other curls into a ball, with remote controls nearby.  
Keywords: cats, pink fabric, remote controls, vibrant colors, striped patterns, relaxed posture, domestic setting, indoor environment, leisure activity, domestic animals, soft lighting<|im_assistant|><|im_user|>>=x^2-2x-5, g(x)=f'(x)=2x-2.  Find the integral of g from 0 to 2x^2-5 using substitution u=2x^2-5, du=4x dx. Then the integral of g from 0 to 2x^2-5 is equal to [u] from 0 to 2x^2-5 multiplied by (4x)/2. Wait, no, substitution steps: Let u=2x²-5, then du/dx=4x, so du=4x dx. The integral of g(x
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
- *Post-cleanup active memory (GB):* 0.022430956
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1135
- *Generation tokens:* 151
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
A top-down view of two cats sleeping on a red couch. The cat on the left is a gray tabby with black stripes. It is lying on its side, with its head facing the bottom left corner of the image. Its front left leg is stretched out in front of its body. Its tail is sticking out behind it on the left side of the image. The cat on the right is lying on its side, with its head facing the right side of the image. Its front right leg is stretched out in front of its body. Its tail is sticking out behind it on the right side of the image. There is a white and gray remote control between the two cats. The couch is casting a shadow over the cats and the remote.
```

</details>

<a id="diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

<details>
<summary>jqlive/Kimi-VL-A3B-Thinking-2506-6bit — unusable — Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Internal reasoning text remains visible</summary>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, unexpected_catalog_preamble,
  thinking_trace_present, thinking_trace_incomplete
- *Unexpected text before Title:* ◁think▷So, let's analyze the image. The main
  subject is two tabby cats on a pink fabric surface. There are two remote
  controls (one blue, one silver) near them. The cats are lying down, one
  facing left, the other facing down. The lighting is bright, making colors
  vivid. Distinctive details: tabby patterns, pink fabric, remotes. Need to
  create a title, description, and keywords.
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["abrupt_tail"]
- *Resolved model revision:* 4b3b11ce0874c36a99e13e17e355049042f8620a
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001213516
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 489
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷So, let's analyze the image. The main subject is two tabby cats on a pink fabric surface. There are two remote controls (one blue, one silver) near them. The cats are lying down, one facing left, the other facing down. The lighting is bright, making colors vivid. Distinctive details: tabby patterns, pink fabric, remotes. Need to create a title, description, and keywords.

Title: "Two tabby cats rest on pink fabric with remotes nearby"

Description: Two tabby cats lie on a bright pink fabric surface, one facing left, the other facing down, with two remote controls beside them.

Keywords: tabby cats, pink fabric, remote controls, lying down, bright lighting, distinctive stripes, relaxed posture, domestic setting, soft surface, relaxed cats, close proximity, domestic environment.

Wait, check if all elements are included. The image has two cats, pink fabric, remotes. The description should be 1-2 sentences. Let's refine:

Title: "Two tabby cats rest on a pink fabric with remote controls nearby"

Description: Two tabby cats lie on a bright pink fabric surface, one facing left and the other facing down, with two remote controls beside them.

Keywords: tabby cats, pink fabric, remote controls, bright lighting, relaxed posture, domestic setting, striped patterns, close proximity, soft surface, calm cats, indoor environment.

Wait, need 10-18 unique keywords. Let's list them clearly:

Title: "Two tabby cats rest on pink fabric with remotes"

Description: Two tabby cats lie on a bright pink fabric surface, one facing left, the other facing down, with two remote controls beside them.

Keywords: tabby cats, pink fabric, remote controls, bright lighting, relaxed posture, domestic setting, striped patterns, close proximity, soft surface, calm cats, indoor environment.

Check if all elements are covered. The image shows two cats, pink fabric, remotes, their positions (left and down), lighting. Keywords should be unique and separated by commas. Let's finalize:

Title: "Two tabby cats rest on pink fabric with remotes"

Description: Two tabby cats lie on a bright pink fabric surface, one facing left, the other facing down, with two remote controls beside them.

Keywords: tabby cats, pink fabric, remote controls, bright lighting, relaxed posture, domestic setting, striped patterns, close proximity
```

</details>

<a id="diagnostic-mlx-community-llava-v16-mistral-7b-8bit"></a>

<details>
<summary>mlx-community/llava-v1.6-mistral-7b-8bit — unusable — Response repeats the same text; Required fields are missing or empty; Response appears cut off at the token limit</summary>

### mlx-community/llava-v1.6-mistral-7b-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* feline,
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail"]
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Processor class:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.018711768
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2469
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
cat, sleeping, pink blanket, remote control, striped, tabby, kitten, cozy, domestic, indoor, relaxation, comfort, home, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline,
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — Response repeats the same text; Required fields are missing or empty; Response repeats the task instructions instead of only returning the requested fields; Extra text appears before the Title field; Response appears cut off at the token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, prompt_instruction_echo, unexpected_catalog_preamble
- *Missing sections:* ["title"]
- *Repeated fragment:* phrase: "- 10-18 unique, comma-separate..."
- *Echoed instruction fragments:* ["return exactly these three sections",
  "return exactly these three sections and nothing else"]
- *Unexpected text before Title:* Action: Lighting: Brand: Species: Period:
  Event: Date: Name: Intent: Brand: Date: Name:
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.023463148
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4207
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
Action:
Lighting:
Brand:
Species:
Period:
Event:
Date:
Name:
Intent:
Brand:
Date:
Name:
Description:
Action:
Lighting:
Keywords:
Return exactly these three sections and nothing else:
Title:
Description:
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
-
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
- *Missing sections:* ["title", "description"]
- *Echoed instruction fragments:* ["return exactly these three sections",
  "create british english catalogue metadata using only clearly"]
- *Unexpected text before Title:* Here are my reasoning steps: The user asks:
  "Create British-English catalogue metadata using only clearly visible facts.
  Omit uncertain details and unsupported identity, location, event, brand,
  species, period, or intent.  Write: - a concrete 5-10-word title; - a
  1-2-sentence factual description of the main subject, setting, action,
  lighting, and distinctive details; - 10-18 unique, comma-separated keywords.
  Return exactly these three sections and nothing else:
- *Token-cap degradation evidence:* ["missing_sections"]
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Processor class:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002065504
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1426
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Here are my reasoning steps:
The user asks: "Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:"

We need to produce a title (5-10 words), a description (1-2 sentences), and a list of 10-18 unique keywords, separated by commas.

We must use only clearly visible facts from the image. The image shows two cats lying on a pink couch, sleeping. There's a remote control on the couch near them. The cats are tabby, with brown, black, and tan stripes. They appear to be sleeping, curled up. The couch is pink. The remote is a typical TV remote with buttons. The lighting appears to be natural or indoor, but we can say "soft indoor lighting" maybe. The cats are lying on their sides, one is curled up with its head on the couch, the other is lying on its side with its head near the remote. The remote is a standard TV remote, but we cannot assume brand. The couch is a pink fabric. The cats are domestic short-haired tabby. The image is presumably taken from above, showing the cats from a top-down perspective.

We need to avoid uncertain details: we cannot assume location (like "living room") unless it's clearly visible. The couch is pink, but we cannot be sure it's a living room. We can say "couch" or "sofa". The remote is present, but we cannot assume it's a TV remote? It looks like a remote control, but we can say "remote control". The cats are sleeping. The lighting is soft, maybe "ambient indoor lighting". The cats are tabby. The couch is pink. The remote is white with colored buttons. The cats are curled up. The cats are domestic. The cats are lying on a pink surface. The remote is placed near them.

We need to produce British English catalogue metadata. So we need to use British spelling, e.g., "colour" vs "color".
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — Required fields are missing or empty; Response appears cut off at the token limit</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation
- *Missing sections:* ["title", "description", "keywords"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00254064
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 545
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Alright, let's get to work. I need to generate some metadata for this image, and the instructions are very specific. First, I need a title, a description, and a list of keywords. The title has to be between five and ten words, and the description needs to be a short, factual paragraph. The keywords are crucial; they have to be unique and relevant, and I can only use what's *clearly* visible. No guessing, no assumptions.

Okay, let's break down the image. I see two cats, definitely. One is a calico, and the other is a tabby. They're on a red couch, and there's a remote control nearby. The lighting seems to be indoor, probably from a lamp or the sun. The cats are sleeping, and the image is a bit blurry.

Now, the title. I need something short and descriptive. "Two cats sleeping on a red couch" is a good start, but I can do better. "Calico and tabby cats resting on a red couch" is more specific, but maybe too long. "Two cats sleeping on a red couch with a remote" is also good. I'll go with "Two cats sleeping on a red couch with a remote." It's concise and accurate.

Next, the description. I need to cover the main subject, setting, action, lighting, and any distinctive details. The main subjects are the two cats. The setting is a red couch. They're sleeping. The lighting is soft, and the image is a bit blurry. The distinctive details are the calico and tabby patterns, the red couch, and the remote. I'll write: "A calico cat and a tabby cat are sleeping side-by-side on a red couch. A remote control lies nearby. The image is slightly blurry, capturing a quiet moment indoors." That's two sentences, factual, and covers everything.

Finally, the keywords. I need to be very careful here. I'll list the obvious things: cat, cats, calico, tabby, couch, red couch, pet, animal, sleeping, resting, couch, sofa, remote, television, television remote, blurry, indoor, home, furniture, pet cat, domestic cat, animal sleeping, and couch. I'll make sure they are all
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
- *Post-cleanup active memory (GB):* 0.021398764
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1135
- *Generation tokens:* 159
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
A top-down view of two cats lying on a pink blanket. The cat on the left is lying on its side, and its head is facing the top left corner of the image. Its body is facing the top right corner of the image. Its tail is sticking out to the left of the image. The cat on the right is lying on its side, and its head is facing the top right corner of the image. Its body is facing the top left corner of the Two remote controls are on the pink blanket, one on the left side of the image and one on the right side of the image. The cat on the left is lying between the two remote controls. The cat on the right is lying to the right of the remote control on the right side of the image.
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — unusable — Response repeats the same text; Extra text appears before the Title field; Response appears cut off at the token limit</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  unexpected_catalog_preamble
- *Repeated fragment:* keyword: "household frames"
- *Unexpected text before Title:* Here is the information in the format you
  requested:
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00676773
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 107
- *Generation tokens:* 500
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;

#### Complete output

```text
Here is the information in the format you requested:

**Title:** Two Cats on a Pink Blanket

**Description:** Two tabby cats are sleeping on a pink blanket, with two remote controls nearby. The cats are lying on their backs, with their paws stretched out and their eyes closed.

**Keywords:** Tabby cats, pink blanket, remote controls, sleeping, feline, domestic, household, comfort, relaxation, domesticated, household pets, household items, household appliances, household furniture, household decor, household textiles, household linens, household blankets, household comforters, household pillows, household throws, household rugs, household mats, household carpets, household flooring, household walls, household ceilings, household windows, household doors, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — Response repeats the same text; Required fields are missing or empty; Extra text appears before the Title field; Response appears cut off at the token limit; Internal reasoning text remains visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, unexpected_catalog_preamble, thinking_trace_present
- *Missing sections:* ["title"]
- *Repeated fragment:* keyword: "pink fabric"
- *Unexpected text before Title:* ◁think▷So, let's analyze the image. The
  image shows two tabby cats on a pink fabric surface. There are two remote
  controls, one near each cat. The cats are lying down, one facing left, the
  other facing down. The lighting is bright, making colors vivid. The scene is
  indoors, likely a living room.  First, the title: need a 5-10 word title.
  Maybe "Two tabby cats resting on pink fabric with remotes." Wait, check word
  count. "Two tabby cats rest on pink fabric with remotes." That's 8 words.
  Good.
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00532593
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 489
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷So, let's analyze the image. The image shows two tabby cats on a pink fabric surface. There are two remote controls, one near each cat. The cats are lying down, one facing left, the other facing down. The lighting is bright, making colors vivid. The scene is indoors, likely a living room. 

First, the title: need a 5-10 word title. Maybe "Two tabby cats resting on pink fabric with remotes." Wait, check word count. "Two tabby cats rest on pink fabric with remotes." That's 8 words. Good.

Description: 1-2 sentences. Describe main subject (two tabby cats), setting (pink fabric, likely couch), action (resting), lighting (bright, vivid colors), distinctive details (remotes, their positions). So: "Two tabby cats lie on a pink fabric surface, with two remote controls nearby, under bright lighting that enhances vivid colors." Wait, maybe shorter. "Two tabby cats rest on pink fabric; two remotes nearby, bright lighting highlights their fur patterns." Hmm, need to be concise. Let's refine: "Two tabby cats lie on pink fabric with two remote controls, under bright lighting emphasizing their striped fur." That's 2 sentences. Wait, the instruction says 1-2 sentences. Let's make it better: "Two tabby cats rest on a pink fabric surface, with two remote controls nearby, under bright lighting that highlights their striped patterns." That's 2 sentences.

Keywords: 10-18 unique, comma-separated. Keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment, relaxed cats. Let's count: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Maybe add "two cats" but that's redundant. Wait, "two tabby cats" is already mentioned. So keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Check if all are unique and visible. Yes.◁/think▷Title: Two tabby cats rest on pink fabric with remotes nearby  
Description: Two tabby cats
```

</details>

## Indeterminate Attempts

None.

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                        | Performance                                                                              |
|-------------------------------------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | rev 6c33f49ebc0b; Lfm2VlProcessor; stop completed       | 179 prompt / 48 generated; 496 tok/s; 1.1 GB peak; cleanup 0.000247/0.0 GB active/cache  |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | rev 01af461cdb95; Qwen2VLProcessor; stop completed      | 511 prompt / 40 generated; 280 tok/s; 2.5 GB peak; cleanup 0.0102/0.0 GB active/cache    |
| mlx-community/LFM2-VL-1.6B-8bit                       | rev 294b90e5ae23; Lfm2VlProcessor; stop completed       | 368 prompt / 86 generated; 324 tok/s; 3.0 GB peak; cleanup 0.00611/0.0 GB active/cache   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed     | 1069 prompt / 85 generated; 184 tok/s; 4.5 GB peak; cleanup 0.00806/0.0 GB active/cache  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | rev 844516024a1c; SmolVLMProcessor; stop completed      | 205 prompt / 55 generated; 131 tok/s; 5.5 GB peak; cleanup 0.0146/0.0 GB active/cache    |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | rev c8a67a843274; Qwen3VLProcessor; stop completed      | 409 prompt / 82 generated; 122 tok/s; 5.3 GB peak; cleanup 0.0105/0.0 GB active/cache    |
| Qwen/Qwen3-VL-2B-Instruct                             | rev 89644892e4d8; Qwen3VLProcessor; stop completed      | 409 prompt / 82 generated; 135 tok/s; 4.6 GB peak; cleanup 0.000558/0.0 GB active/cache  |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | rev 773591fa7388; Glm46VProcessor; stop completed       | 499 prompt / 61 generated; 87.1 tok/s; 7.8 GB peak; cleanup 0.00347/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed      | 421 prompt / 89 generated; 120 tok/s; 21 GB peak; cleanup 0.0123/0.0 GB active/cache     |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed       | 390 prompt / 73 generated; 108 tok/s; 16 GB peak; cleanup 0.0176/0.0 GB active/cache     |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed      | 421 prompt / 86 generated; 97.4 tok/s; 7.1 GB peak; cleanup 0.0139/0.0 GB active/cache   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed        | 883 prompt / 68 generated; 57.1 tok/s; 9.4 GB peak; cleanup 0.00988/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | rev b729d115bb2c; Qwen3VLProcessor; stop completed      | 421 prompt / 109 generated; 101 tok/s; 30 GB peak; cleanup 0.0128/0.0 GB active/cache    |
| microsoft/Phi-3.5-vision-instruct                     | rev 12b77fb40b63; Phi3VProcessor; stop completed        | 883 prompt / 68 generated; 50.8 tok/s; 9.4 GB peak; cleanup 0.0018/0.0 GB active/cache   |
| mlx-community/gemma-3n-E4B-it-bf16                    | rev d9c02d0b2fa8; Gemma3nProcessor; stop completed      | 378 prompt / 61 generated; 40.8 tok/s; 17 GB peak; cleanup 0.0171/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed     | 1070 prompt / 81 generated; 67.0 tok/s; 9.8 GB peak; cleanup 0.00754/0.0 GB active/cache |
| mlx-community/X-Reasoner-7B-8bit                      | rev 21732e74613b; Qwen2_5_VLProcessor; stop completed   | 511 prompt / 106 generated; 63.9 tok/s; 10 GB peak; cleanup 0.0149/0.0 GB active/cache   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed     | 1070 prompt / 106 generated; 63.6 tok/s; 10 GB peak; cleanup 0.0078/0.0 GB active/cache  |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed      | 1335 prompt / 89 generated; 40.3 tok/s; 15 GB peak; cleanup 0.0237/0.0 GB active/cache   |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed       | 379 prompt / 90 generated; 30.9 tok/s; 18 GB peak; cleanup 0.0155/0.0 GB active/cache    |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed | 3435 prompt / 59 generated; 34.7 tok/s; 18 GB peak; cleanup 0.00467/0.0 GB active/cache  |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed       | 390 prompt / 80 generated; 25.6 tok/s; 20 GB peak; cleanup 0.0186/0.0 GB active/cache    |
| mlx-community/Qwen3.5-27B-4bit                        | rev 45797d2985a1; Qwen3VLProcessor; stop completed      | 421 prompt / 106 generated; 32.6 tok/s; 19 GB peak; cleanup 0.0113/0.0 GB active/cache   |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed     | 537 prompt / 100 generated; 30.2 tok/s; 20 GB peak; cleanup 0.00233/0.0 GB active/cache  |
| mlx-community/pixtral-12b-bf16                        | rev 378cf65efd79; PixtralProcessor; stop completed      | 1335 prompt / 83 generated; 20.6 tok/s; 27 GB peak; cleanup 0.024/0.0 GB active/cache    |
| mlx-community/gemma-3-27b-it-qat-8bit                 | rev c408904bc9a0; Gemma3Processor; stop completed       | 379 prompt / 87 generated; 16.7 tok/s; 32 GB peak; cleanup 0.016/0.0 GB active/cache     |
| mlx-community/Molmo-7B-D-0924-8bit                    | rev 90a14ed7a230; MolmoProcessor; stop completed        | 839 prompt / 90 generated; 44.4 tok/s; 20 GB peak; cleanup 0.00837/0.0 GB active/cache   |
| mlx-community/Qwen3.6-27B-mxfp8                       | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed      | 421 prompt / 107 generated; 18.4 tok/s; 30 GB peak; cleanup 0.0144/0.0 GB active/cache   |
| mlx-community/Qwen3.5-27B-mxfp8                       | rev 2d6caf2325c2; Qwen3VLProcessor; stop completed      | 421 prompt / 115 generated; 17.7 tok/s; 30 GB peak; cleanup 0.0118/0.0 GB active/cache   |
| mlx-community/Molmo-7B-D-0924-bf16                    | rev d871cbdb87a4; MolmoProcessor; stop completed        | 839 prompt / 96 generated; 26.8 tok/s; 27 GB peak; cleanup 0.00868/0.0 GB active/cache   |
| mlx-community/InternVL3-14B-8bit                      | rev 50efc568c7df; InternVLChatProcessor; stop completed | 3435 prompt / 75 generated; 23.4 tok/s; 19 GB peak; cleanup 0.00436/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | rev 731d09ba3597; Qwen3VLProcessor; stop completed      | 421 prompt / 107 generated; 66.8 tok/s; 71 GB peak; cleanup 0.0134/0.0 GB active/cache   |
| mlx-community/Ornith-1.0-35B-bf16                     | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed      | 421 prompt / 133 generated; 63.7 tok/s; 71 GB peak; cleanup 0.00982/0.0 GB active/cache  |
| mlx-community/MolmoPoint-8B-fp16                      | rev 0a60033b4e48; MolmoPointProcessor; stop completed   | 860 prompt / 78 generated; 5.29 tok/s; 23 GB peak; cleanup 0.00931/0.0 GB active/cache   |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | rev 9eb2daaa8597; MllamaProcessor; stop completed       | 108 prompt / 105 generated; 4.98 tok/s; 25 GB peak; cleanup 0.00174/0.0 GB active/cache  |

</details>

## Shared Reproduction and Provenance

### Prompt

Save this exact prompt as prompt.txt.

```text
Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/Step-3.7-Flash-oQ2e                | 3dacb46f724ac89725bcd922fb779c7ed1499fe7 |
| mlx-community/FastVLM-0.5B-bf16                  | 81ffe929046666c43de53691147b1669ba0f3a4c |
| mlx-community/nanoLLaVA-1.5-4bit                 | 5240204744963d72823e5de933c528c4aa82dfca |
| mlx-community/MiniCPM-V-4.6-8bit                 | 03721395f6b82cd000cc74cde28fcff8abd9a04c |
| qnguyen3/nanoLLaVA                               | 13d60cec183a86755afed64da495fcc2c382ea80 |
| mlx-community/LFM2.5-VL-1.6B-bf16                | 16a710cf8afca206ff16a95a4ad6fe657f876ce1 |
| mlx-community/SmolVLM-Instruct-bf16              | cae61cdedd0602419b43b6102dc33cd9f1e929a6 |
| HuggingFaceTB/SmolVLM-Instruct                   | 81cd9a775a4d644f2faf4e7becff4559b46b14c7 |
| mlx-community/GLM-4.6V-Flash-6bit                | df9464782d3452e0dfd86afe0984f1c9eca75ca1 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/Qwen3-VL-2B-Thinking-bf16          | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/gemma-4-31b-bf16                   | 19f0f1af698c51edaf1e93b3a3a5435b282de30f |
| mlx-community/gemma-3n-E2B-4bit                  | ec68dc186276e20e4bed30b96a2b5c667e0a81e3 |
| mlx-community/paligemma2-10b-ft-docci-448-6bit   | 1485fa9b3c7adb360cd354a29a401f0d441ec728 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Kimi-VL-A3B-Thinking-8bit          | 85daf3dc2490c0f824143338f08ba45f475c9ce4 |
| mlx-community/paligemma2-3b-ft-docci-448-bf16    | f66333527ce75342b09d4df81873f65272ec2f30 |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit            | 4b3b11ce0874c36a99e13e17e355049042f8620a |
| mlx-community/llava-v1.6-mistral-7b-8bit         | b8df5f329d95a7abe6429ed46093f9b84e8e6396 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX    | 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2 |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/paligemma2-10b-ft-docci-448-bf16   | 7c412694b919432784c730b62fadafb1c2e15d0d |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |

### Canonical parameterised Python reproduction

Run one model per process to avoid sequential Metal-state interactions.

```bash
python reproduce.py MODEL_ID --revision RESOLVED_REVISION --image cats.jpg --prompt-file prompt.txt
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

# Defaults used in the report: prompt.txt, cats.jpg
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
