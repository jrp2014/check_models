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
| none                           |      24 |
| observation needs reproduction |      37 |

Usability counts

| Usability           |   Count |
|---------------------|---------|
| not evaluated       |       1 |
| unusable            |      32 |
| usable              |      24 |
| usable with caveats |       5 |

Observation counts

| Observation                 |   Count |
|-----------------------------|---------|
| draft returned unchanged    |       5 |
| minimal output              |       1 |
| missing requested sections  |      20 |
| prompt instruction echo     |       8 |
| repeated output             |      11 |
| thinking trace incomplete   |       3 |
| thinking trace present      |       3 |
| token cap truncation        |      14 |
| unexpected catalog preamble |      10 |
| unexpected special token    |       4 |

## Triage

| Model                                                                                                           | Execution   | Usability           | Maintainer status              | Observations                                                                                                                     |
|-----------------------------------------------------------------------------------------------------------------|-------------|---------------------|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/Step-3.7-Flash-oQ2e](#diagnostic-mlx-community-step-37-flash-oq2e)                               | crashed     | not_evaluated       | actionable_failure             | none                                                                                                                             |
| [mlx-community/nanoLLaVA-1.5-4bit](#diagnostic-mlx-community-nanollava-15-4bit)                                 | completed   | unusable            | observation_needs_reproduction | missing requested sections                                                                                                       |
| [qnguyen3/nanoLLaVA](#diagnostic-qnguyen3-nanollava)                                                            | completed   | unusable            | observation_needs_reproduction | missing requested sections                                                                                                       |
| [HuggingFaceTB/SmolVLM-Instruct](#diagnostic-huggingfacetb-smolvlm-instruct)                                    | completed   | usable_with_caveats | observation_needs_reproduction | draft returned unchanged                                                                                                         |
| [mlx-community/FastVLM-0.5B-bf16](#diagnostic-mlx-community-fastvlm-05b-bf16)                                   | completed   | unusable            | observation_needs_reproduction | repeated output, prompt instruction echo                                                                                         |
| [mlx-community/paligemma2-10b-ft-docci-448-6bit](#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit)    | completed   | unusable            | observation_needs_reproduction | missing requested sections, prompt instruction echo                                                                              |
| [mlx-community/SmolVLM2-2.2B-Instruct-mlx](#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx)                 | completed   | usable_with_caveats | observation_needs_reproduction | draft returned unchanged                                                                                                         |
| [mlx-community/SmolVLM-Instruct-bf16](#diagnostic-mlx-community-smolvlm-instruct-bf16)                          | completed   | usable_with_caveats | observation_needs_reproduction | draft returned unchanged                                                                                                         |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed   | unusable            | observation_needs_reproduction | missing requested sections, unexpected catalog preamble, unexpected special token                                                |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed   | unusable            | observation_needs_reproduction | missing requested sections, unexpected catalog preamble, unexpected special token                                                |
| [mlx-community/LFM2.5-VL-1.6B-bf16](#diagnostic-mlx-community-lfm25-vl-16b-bf16)                                | completed   | unusable            | observation_needs_reproduction | repeated output, token cap truncation                                                                                            |
| [mlx-community/gemma-4-31b-bf16](#diagnostic-mlx-community-gemma-4-31b-bf16)                                    | completed   | unusable            | observation_needs_reproduction | minimal output, missing requested sections                                                                                       |
| [mlx-community/Phi-3.5-vision-instruct-bf16](#diagnostic-mlx-community-phi-35-vision-instruct-bf16)             | completed   | unusable            | observation_needs_reproduction | repeated output                                                                                                                  |
| [mlx-community/paligemma2-10b-ft-docci-448-bf16](#diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16)    | completed   | unusable            | observation_needs_reproduction | missing requested sections, prompt instruction echo                                                                              |
| [microsoft/Phi-3.5-vision-instruct](#diagnostic-microsoft-phi-35-vision-instruct)                               | completed   | unusable            | observation_needs_reproduction | repeated output                                                                                                                  |
| [mlx-community/gemma-3n-E2B-4bit](#diagnostic-mlx-community-gemma-3n-e2b-4bit)                                  | completed   | unusable            | observation_needs_reproduction | repeated output, missing requested sections, token cap truncation                                                                |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed   | usable_with_caveats | observation_needs_reproduction | draft returned unchanged                                                                                                         |
| [mlx-community/gemma-3n-E4B-it-bf16](#diagnostic-mlx-community-gemma-3n-e4b-it-bf16)                            | completed   | unusable            | observation_needs_reproduction | missing requested sections                                                                                                       |
| [jqlive/Kimi-VL-A3B-Thinking-2506-6bit](#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit)                      | completed   | unusable            | observation_needs_reproduction | missing requested sections, token cap truncation, thinking trace present, thinking trace incomplete                              |
| [mlx-community/Kimi-VL-A3B-Thinking-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit)                  | completed   | unusable            | observation_needs_reproduction | missing requested sections, token cap truncation, unexpected catalog preamble, thinking trace present, thinking trace incomplete |
| [mlx-community/GLM-4.6V-Flash-6bit](#diagnostic-mlx-community-glm-46v-flash-6bit)                               | completed   | unusable            | observation_needs_reproduction | missing requested sections, unexpected catalog preamble, unexpected special token                                                |
| [mlx-community/llava-v1.6-mistral-7b-8bit](#diagnostic-mlx-community-llava-v16-mistral-7b-8bit)                 | completed   | unusable            | observation_needs_reproduction | prompt instruction echo                                                                                                          |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed   | unusable            | observation_needs_reproduction | repeated output, missing requested sections, token cap truncation, prompt instruction echo                                       |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed   | unusable            | observation_needs_reproduction | missing requested sections, token cap truncation, prompt instruction echo                                                        |
| [mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX](#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | completed   | unusable            | observation_needs_reproduction | prompt instruction echo, unexpected catalog preamble                                                                             |
| [mlx-community/GLM-4.6V-Flash-mxfp4](#diagnostic-mlx-community-glm-46v-flash-mxfp4)                             | completed   | unusable            | observation_needs_reproduction | repeated output, token cap truncation                                                                                            |
| [meta-llama/Llama-3.2-11B-Vision-Instruct](#diagnostic-meta-llama-llama-32-11b-vision-instruct)                 | completed   | unusable            | observation_needs_reproduction | missing requested sections                                                                                                       |
| [Qwen/Qwen3-VL-2B-Instruct](#diagnostic-qwen-qwen3-vl-2b-instruct)                                              | completed   | unusable            | observation_needs_reproduction | repeated output, token cap truncation                                                                                            |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed   | unusable            | observation_needs_reproduction | missing requested sections, unexpected catalog preamble, unexpected special token                                                |
| [mlx-community/Molmo-7B-D-0924-8bit](#diagnostic-mlx-community-molmo-7b-d-0924-8bit)                            | completed   | unusable            | observation_needs_reproduction | unexpected catalog preamble                                                                                                      |
| [mlx-community/Molmo-7B-D-0924-bf16](#diagnostic-mlx-community-molmo-7b-d-0924-bf16)                            | completed   | unusable            | observation_needs_reproduction | unexpected catalog preamble                                                                                                      |
| [mlx-community/Qwen3-VL-2B-Instruct-bf16](#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16)                  | completed   | unusable            | observation_needs_reproduction | repeated output, token cap truncation                                                                                            |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                  | completed   | unusable            | observation_needs_reproduction | missing requested sections, token cap truncation, unexpected catalog preamble                                                    |
| [mlx-community/paligemma2-3b-ft-docci-448-bf16](#diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16)      | completed   | unusable            | observation_needs_reproduction | missing requested sections, token cap truncation, prompt instruction echo                                                        |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed   | unusable            | observation_needs_reproduction | repeated output, missing requested sections, token cap truncation                                                                |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed   | unusable            | observation_needs_reproduction | repeated output, token cap truncation                                                                                            |
| [mlx-community/Qwen2-VL-2B-Instruct-4bit](#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit)                  | completed   | usable_with_caveats | observation_needs_reproduction | draft returned unchanged                                                                                                         |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed   | unusable            | observation_needs_reproduction | missing requested sections, token cap truncation, unexpected catalog preamble, thinking trace present, thinking trace incomplete |

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
- *Post-cleanup active memory (GB):* 0.014632082
- *Post-cleanup cache memory (GB):* 0.0

#### Complete traceback

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

## Completed Runs with Observations

<a id="diagnostic-mlx-community-nanollava-15-4bit"></a>

<details>
<summary>mlx-community/nanoLLaVA-1.5-4bit — unusable — missing requested sections</summary>

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
- *Post-cleanup active memory (GB):* 0.0193999
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 614
- *Generation tokens:* 73
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Viking Bay, Broadstairs, England, UK, GBR, Europe
Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
```

</details>

<a id="diagnostic-qnguyen3-nanollava"></a>

<details>
<summary>qnguyen3/nanoLLaVA — unusable — missing requested sections</summary>

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
- *Post-cleanup active memory (GB):* 0.024675576
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 614
- *Generation tokens:* 75
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Viking Bay, Broadstairs, England, UK, GBR, Europe
Description: A picturesque coastal town scene in Broadstairs, Kent, featuring a wide shot of the town, with a beautiful sky, a castle, and a sandy beach. The town's skyline and wind turbines are visible in the distance, and a crowd of people is enjoying the day.
```

</details>

<a id="diagnostic-huggingfacetb-smolvlm-instruct"></a>

<details>
<summary>HuggingFaceTB/SmolVLM-Instruct — usable_with_caveats — draft returned unchanged</summary>

### HuggingFaceTB/SmolVLM-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* draft_returned_unchanged
- *Draft fields returned unchanged:* ["title", "description", "keywords"]
- *Resolved model revision:* 81cd9a775a4d644f2faf4e7becff4559b46b14c7
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00011573
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1830
- *Generation tokens:* 122
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 Title: Viking Bay, Broadstairs, England, UK, GBR, Europe
Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
Keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-fastvlm-05b-bf16"></a>

<details>
<summary>mlx-community/FastVLM-0.5B-bf16 — unusable — repeated output, prompt instruction echo</summary>

### mlx-community/FastVLM-0.5B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, prompt_instruction_echo
- *Repeated fragment:* keyword: "england"
- *Echoed instruction fragments:* ["treat this draft as fallible retain
  supported details"]
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Processor class:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002851938
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 618
- *Generation tokens:* 270
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Viking Bay, Broadstairs, England, UK, GBR, Europe
Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
Keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
Draft metadata:
- Existing title: Viking Bay, Broadstairs, England, UK, GBR, Europe
- Existing description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
- Existing keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.
```

</details>

<a id="diagnostic-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

<details>
<summary>mlx-community/paligemma2-10b-ft-docci-448-6bit — unusable — missing requested sections, prompt instruction echo</summary>

### mlx-community/paligemma2-10b-ft-docci-448-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Echoed instruction fragments:* ["prompt instructions", "do not copy prompt
  instructions into the title"]
- *Resolved model revision:* 1485fa9b3c7adb360cd354a29a401f0d441ec728
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.020432112
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1646
- *Generation tokens:* 17
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
```

</details>

<a id="diagnostic-mlx-community-smolvlm2-22b-instruct-mlx"></a>

<details>
<summary>mlx-community/SmolVLM2-2.2B-Instruct-mlx — usable_with_caveats — draft returned unchanged</summary>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* draft_returned_unchanged
- *Draft fields returned unchanged:* ["title", "description", "keywords"]
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Processor class:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014632082
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 731
- *Generation tokens:* 124
- *Configured EOS token ID:* 49279
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 Title: Viking Bay, Broadstairs, England, UK, GBR, Europe

Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.

Keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-smolvlm-instruct-bf16"></a>

<details>
<summary>mlx-community/SmolVLM-Instruct-bf16 — usable_with_caveats — draft returned unchanged</summary>

### mlx-community/SmolVLM-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* draft_returned_unchanged
- *Draft fields returned unchanged:* ["title", "description", "keywords"]
- *Resolved model revision:* cae61cdedd0602419b43b6102dc33cd9f1e929a6
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014517394
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1830
- *Generation tokens:* 122
- *Configured EOS token ID:* 49154
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 Title: Viking Bay, Broadstairs, England, UK, GBR, Europe
Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
Keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-8bit — unusable — missing requested sections, unexpected catalog preamble, unexpected special token</summary>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|channel&gt;thought
  &lt;channel|&gt;Title: - Viking Bay in Broadstairs, Kent, England
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.0149434
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 895
- *Generation tokens:* 108
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title:
- Viking Bay in Broadstairs, Kent, England

Description:
- A high-angle view of holidaymakers relaxing on the sandy beach of Viking Bay on a sunny summer day. The scene features coastal town buildings, colourful beach huts, and offshore wind turbines visible on the distant sea horizon.

Keywords:
- Broadstairs, Kent, England, Viking Bay, beach, coastal town, summer, holiday, people, beach huts, sand, sea, horizon, wind turbines, architecture, tourism
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-mxfp8 — unusable — missing requested sections, unexpected catalog preamble, unexpected special token</summary>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|channel&gt;thought
  &lt;channel|&gt;Title: Viking Bay beach in Broadstairs, Kent, England
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.0149434
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 895
- *Generation tokens:* 101
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Viking Bay beach in Broadstairs, Kent, England

Description: A high-angle view of Viking Bay on a sunny summer day where beachgoers relax on the sandy beach. The background features coastal town buildings, a large stone structure, and wind turbines visible on the sea horizon.

Keywords: Broadstairs, Kent, England, Viking Bay, beach, sand, summer, sunny, people, buildings, coastal, sea, horizon, sky, holiday, leisure, tourism
```

</details>

<a id="diagnostic-mlx-community-lfm25-vl-16b-bf16"></a>

<details>
<summary>mlx-community/LFM2.5-VL-1.6B-bf16 — unusable — repeated output, token cap truncation</summary>

### mlx-community/LFM2.5-VL-1.6B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* keyword: "coastal scenery"
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.006243446
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 874
- *Generation tokens:* 500
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Viking Bay, Broadstairs, Kent, England, UK, GBR, Europe

Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summer's day, holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.

Keywords: Broadstairs, Kent, England, Europe, beach, holiday, town, castle, wind turbines, sky, summer, relaxed, people, buildings, coastline, seaside, summer, coastal town, seaside holiday, summer beach, coastal scenery, seaside resort, coastal landscape, seaside view, coastal townscape, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery, coastal view, coastal scenery,
```

</details>

<a id="diagnostic-mlx-community-gemma-4-31b-bf16"></a>

<details>
<summary>mlx-community/gemma-4-31b-bf16 — unusable — minimal output, missing requested sections</summary>

### mlx-community/gemma-4-31b-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* minimal_output, missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 19f0f1af698c51edaf1e93b3a3a5435b282de30f
- *Processor class:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.01818748
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 887
- *Generation tokens:* 4
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text


Image:
```

</details>

<a id="diagnostic-mlx-community-phi-35-vision-instruct-bf16"></a>

<details>
<summary>mlx-community/Phi-3.5-vision-instruct-bf16 — unusable — repeated output</summary>

### mlx-community/Phi-3.5-vision-instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output
- *Repeated fragment:* keyword: "relaxing"
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Processor class:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009897096
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1469
- *Generation tokens:* 244
- *Configured EOS token ID:* 32000
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
Title: Broadstairs Beach, Kent, England

Description: A vibrant scene from Broadstairs beach in Kent, England, captured on a sunny day with people enjoying the sandy beach and the town's skyline in the background. The beach is bustling with activity, with families and individuals relaxing under the clear blue sky.

Keywords: Broadstairs, Kent, England, beach, coastal town, sunny day, holiday makers, town skyline, castle, wind turbines, blue sky, relaxing, families, individuals, sandy beach, coast, town, skyline, castle, wind turbines, coastal town, Broadstairs, England, Europe, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Relaxing, Sitting, Horizon, Coast, Crowd, Relaxing, Sitting, Sky, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Relaxing, Sitting, Horizon, Coast, Crowd, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

<details>
<summary>mlx-community/paligemma2-10b-ft-docci-448-bf16 — unusable — missing requested sections, prompt instruction echo</summary>

### mlx-community/paligemma2-10b-ft-docci-448-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Echoed instruction fragments:* ["prompt instructions", "do not copy prompt
  instructions into the title"]
- *Resolved model revision:* 7c412694b919432784c730b62fadafb1c2e15d0d
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.021464304
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1646
- *Generation tokens:* 17
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
```

</details>

<a id="diagnostic-microsoft-phi-35-vision-instruct"></a>

<details>
<summary>microsoft/Phi-3.5-vision-instruct — unusable — repeated output</summary>

### microsoft/Phi-3.5-vision-instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output
- *Repeated fragment:* keyword: "relaxing"
- *Resolved model revision:* 12b77fb40b63a2c73c68243d3f767aab688a1b2a
- *Processor class:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00180336
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1469
- *Generation tokens:* 244
- *Configured EOS token ID:* 32000
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
Title: Broadstairs Beach, Kent, England

Description: A vibrant scene from Broadstairs beach in Kent, England, captured on a sunny day with people enjoying the sandy beach and the town's skyline in the background. The beach is bustling with activity, with families and individuals relaxing under the clear blue sky.

Keywords: Broadstairs, Kent, England, beach, coastal town, sunny day, holiday makers, town skyline, castle, wind turbines, blue sky, relaxing, families, individuals, sandy beach, coast, town, skyline, castle, wind turbines, coastal town, Broadstairs, England, Europe, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Relaxing, Sitting, Horizon, Coast, Crowd, Relaxing, Sitting, Sky, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Relaxing, Sitting, Horizon, Coast, Crowd, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-gemma-3n-e2b-4bit"></a>

<details>
<summary>mlx-community/gemma-3n-E2B-4bit — unusable — repeated output, missing requested sections, token cap truncation</summary>

### mlx-community/gemma-3n-E2B-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "- do not take..."
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* ec68dc186276e20e4bed30b96a2b5c667e0a81e3
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.01659821
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 881
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text

- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take it out of the time.
- Do not take
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — usable_with_caveats — draft returned unchanged</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* draft_returned_unchanged
- *Draft fields returned unchanged:* ["title", "description", "keywords"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.004047982
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2896
- *Generation tokens:* 121
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 Title: Viking Bay, Broadstairs, England, UK, GBR, Europe.
Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
Keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky.<end_of_utterance>
```

</details>

<a id="diagnostic-mlx-community-gemma-3n-e4b-it-bf16"></a>

<details>
<summary>mlx-community/gemma-3n-E4B-it-bf16 — unusable — missing requested sections</summary>

### mlx-community/gemma-3n-E4B-it-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.017138882
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 889
- *Generation tokens:* 293
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
## A Sunny Day at Viking Bay, Broadstairs

**A wide shot captures a vibrant scene at Viking Bay in Broadstairs, Kent, on a bright summer's day.** The sandy beach is bustling with people enjoying the sunshine. Many are relaxing on the sand, some under colourful beach umbrellas and tents. Others are strolling along the shoreline, while some are gathered near the water's edge. 

The town of Broadstairs forms a picturesque backdrop, with its charming white houses and buildings lining the coast. The clear blue sea stretches out to the horizon, dotted with wind turbines in the distance. The sky is a clear, pale blue, adding to the idyllic atmosphere. 

In the foreground, out-of-focus green foliage frames the scene, creating a natural border and drawing the viewer's eye towards the lively beach and town. The overall impression is one of a relaxed and enjoyable day at the seaside. 

**Key visual elements:**

* **Vibrant beach scene:** People relaxing, strolling, and enjoying the sun.
* **Charming coastal town:** White houses and buildings lining the shore.
* **Clear blue sea:** Stretching to the horizon with distant wind turbines.
* **Bright blue sky:** Contributing to the sunny atmosphere.
* **Green foliage:** Framing the scene in the foreground. 

This image perfectly captures the quintessential British seaside experience, with its blend of natural beauty and charming coastal town.
```

</details>

<a id="diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

<details>
<summary>jqlive/Kimi-VL-A3B-Thinking-2506-6bit — unusable — missing requested sections, token cap truncation, thinking trace present, thinking trace incomplete</summary>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_present, thinking_trace_incomplete
- *Missing sections:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections"]
- *Resolved model revision:* 4b3b11ce0874c36a99e13e17e355049042f8620a
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001213516
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1608
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let's tackle this metadata analysis. First, I need to create a title, description, and keywords based on the image. The image shows a sunny beach scene in Broadstairs, Kent, England. The beach is sandy with people relaxing, colorful beach huts, buildings, and the sea with wind turbines in the distance. The lighting is bright, indicating a sunny day. 

For the title, I should keep it concise. The existing draft was "Viking Bay, Broadstairs, England, UK, GBR, Europe" but maybe "Viking Bay, Broadstairs, Kent, UK" is more specific. The image has a wide shot, so "Wide shot of Viking Bay, Broadstairs, Kent" works.

Description needs to combine context and visible details. The scene is a sunny day with people relaxing on the beach, colorful beach huts, buildings, sea, and wind turbines. So: "A sunny day at Viking Bay, Broadstairs, Kent, with people relaxing on the sandy beach, colorful beach huts, historic buildings, and wind turbines visible in the distance."

Keywords should include elements from the image: beach, people, sand, buildings, beach huts, sea, wind turbines, sunny, England, UK, Europe, blue sky, trees, holiday, etc. Need to ensure they're unique and relevant. Check the existing keywords but add more specific ones like "beach huts" and "wind turbines" which were in the draft but maybe not in the original. Also include "blue sky" and "trees" from the foreground.

Wait, the existing draft had "Viking Bay, Broadstairs, England, UK, GBR, Europe" as title. The image shows the beach huts, so include that. Also, the buildings are historic, so "historic buildings" is good. The sea is visible, so "sea" and "blue sky". The wind turbines are in the distance, so "wind turbines". The foreground has trees, so "trees". People are relaxing, so "people" and "relaxing". Colors like "blue sky" and "sandy beach" are visible. So the keywords would be: beach, people, sand, beach huts, buildings, sea, wind turbines, sunny, England, UK, Europe, blue sky, trees, holiday, historic buildings, relaxing, blue, sand.

Check if all are visible: beach huts are there,
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-8bit — unusable — missing requested sections, token cap truncation, unexpected catalog preamble, thinking trace present, thinking trace incomplete</summary>

### mlx-community/Kimi-VL-A3B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_present,
  thinking_trace_incomplete
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* ◁think▷Okay, let's tackle this metadata
  cataloguing task. First, I need to analyze the image and the provided
  context. The image shows a beach scene with people relaxing, buildings along
  the coast, wind turbines in the distance, and a clear blue sky. The context
  mentions it's Viking Bay in Broadstairs, Kent, England, UK. The existing
  draft metadata has a title, description, and keywords, but I need to refine
  them based on the image.  Starting with the Title: The existing title is
  "Viking Bay, Broadstairs, England, UK, GBR". The image clearly shows the
  beach and the town, so the title should be concise. Maybe "Viking Bay Beach,
  Broadstairs, Kent, England" captures the main elements. "GBR" is the country
  code, but since the location is already specified, maybe omitting it would
  make the title more natural.  For the Description: The draft mentions a wide
  shot looking down at the town and beach. The image is a high-angle view, so
  "A high-angle view of Viking Bay beach in Broadstairs, Kent, with people
  relaxing on the sandy shore and the town skyline and wind turbines visible
  in the distance." That includes the key elements: the angle, location,
  people, beach, town, and wind turbines. The original description mentions
  "sunny summer day" but the image shows clear skies, so maybe "sunny" is
  accurate. Also, the original mentions "holidaymakers" but "people" is more
  general and fits better.
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections"]
- *Resolved model revision:* 85daf3dc2490c0f824143338f08ba45f475c9ce4
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.005981302
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1608
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let's tackle this metadata cataloguing task. First, I need to analyze the image and the provided context. The image shows a beach scene with people relaxing, buildings along the coast, wind turbines in the distance, and a clear blue sky. The context mentions it's Viking Bay in Broadstairs, Kent, England, UK. The existing draft metadata has a title, description, and keywords, but I need to refine them based on the image.

Starting with the Title: The existing title is "Viking Bay, Broadstairs, England, UK, GBR". The image clearly shows the beach and the town, so the title should be concise. Maybe "Viking Bay Beach, Broadstairs, Kent, England" captures the main elements. "GBR" is the country code, but since the location is already specified, maybe omitting it would make the title more natural.

For the Description: The draft mentions a wide shot looking down at the town and beach. The image is a high-angle view, so "A high-angle view of Viking Bay beach in Broadstairs, Kent, with people relaxing on the sandy shore and the town skyline and wind turbines visible in the distance." That includes the key elements: the angle, location, people, beach, town, and wind turbines. The original description mentions "sunny summer day" but the image shows clear skies, so maybe "sunny" is accurate. Also, the original mentions "holidaymakers" but "people" is more general and fits better.

Keywords: The existing keywords include "Coast, England, Europe" which are important. The image has "wind turbines" in the distance, so adding that. Also, "beach huts" might be visible, but they're not prominent. The original keywords have "Buildings, Bushes, Children, Crowd, Horizon, Objects, Relaxing, Sitting, Sky." The image shows colorful beach huts along the promenade, so adding "Beach huts" would be good. Also, "Wind turbines" from the context. Maybe "Promenade" instead of "Objects" for specificity. "Holidaymakers" could be replaced with "People" as in the draft. Let me check the existing keywords again. The draft has "Holiday" but the image doesn't show holidays specifically, just people relaxing. So maybe remove "Holiday" and add "Wind turbines" and "Beach huts." Also, "
```

</details>

<a id="diagnostic-mlx-community-glm-46v-flash-6bit"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-6bit — unusable — missing requested sections, unexpected catalog preamble, unexpected special token</summary>

### mlx-community/GLM-4.6V-Flash-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|begin_of_box|&gt;Title: Viking Bay
  beach, Broadstairs, Kent
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* df9464782d3452e0dfd86afe0984f1c9eca75ca1
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003163234
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6623
- *Generation tokens:* 129
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title:
Viking Bay beach, Broadstairs, Kent

Description:
A sunny view of Viking Bay beach in Broadstairs, Kent, with holidaymakers relaxing on the sandy shore. The scene includes colorful beach huts, a town skyline with a prominent castle-like building, and wind turbines visible on the horizon over the blue sea.

Keywords:
Viking Bay beach, Broadstairs, Kent, England, UK, GBR, Europe, beach, sand, holidaymakers, people, relaxing, beach huts, colorful, town skyline, castle-like building, wind turbines, sea, blue sky, sunny, coastal town, summer day<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-llava-v16-mistral-7b-8bit"></a>

<details>
<summary>mlx-community/llava-v1.6-mistral-7b-8bit — unusable — prompt instruction echo</summary>

### mlx-community/llava-v1.6-mistral-7b-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* prompt_instruction_echo
- *Echoed instruction fragments:* ["prompt instructions", "do not output
  reasoning", "distinguish supplied authoritative facts from visible details
  do"]
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Processor class:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.018777308
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2653
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Title: Viking Bay, Broadstairs, England, UK, GBR, Europe

Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.

Keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky

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

Draft descriptive metadata:
- Existing title: Viking Bay, Broadstairs, England, UK, GBR, Europe
- Existing description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
- Existing keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People,
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated output, missing requested sections, token cap truncation, prompt instruction echo</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "text after the label...."
- *Echoed instruction fragments:* ["output only the title text after the
  label"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.023528688
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4718
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Output only the title text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the label.
- Do not output the description text after the
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — missing requested sections, token cap truncation, prompt instruction echo</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Echoed instruction fragments:* ["prompt instructions", "treat this draft as
  fallible retain supported details"]
- *Token-cap degradation evidence:* ["missing_sections", "unfinished_section",
  "abrupt_tail"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00254064
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1927
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context gives the location as Viking Bay, Broadstairs, Kent, England, UK, GBR, Europe. The existing title is "Viking Bay, Broadstairs, England, UK, GBR, Europe" which is 7 words, so that's good.

Next, the description. The existing one says it's a wide shot of a coastal town in Broadstairs, Kent, sunny summer day, holidaymakers on Viking Bay beach, town skyline, castle, wind turbines in hazy distance. Wait, but let's check the image details: the user's prompt mentions "bustling seaside town of Broadstairs, Kent, England, UK, with a sandy beach, colorful beach huts, families enjoying the day, and wind turbines in the distance." Wait no, wait the user's prompt is a bit jumbled, but the authoritative context is Viking Bay, Broadstairs, Kent. The existing description is good, but let's make sure it's factual. The existing description says "sunny summers day" (should be "summer's" but wait no, the user's existing description is "sunny summers day" but let's check the rules: don't copy prompt instructions, but the existing description is a draft. Wait no, the task says "treat this draft as fallible. Retain supported details, correct errors, and add important visible information." Wait the existing description says "sunny summers day" which is a typo (should be "summer's"), but let's see: the user's prompt has "sunny summers day" but maybe it's a mistake. Wait no, the existing description is "A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance." Let's correct "summers" to "summer's" because it's a possessive (the day of summer). Also, add any visible details: the prompt mentions "bustling seaside town" but the existing description
```

</details>

<a id="diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

<details>
<summary>mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX — unusable — prompt instruction echo, unexpected catalog preamble</summary>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* prompt_instruction_echo, unexpected_catalog_preamble
- *Echoed instruction fragments:* ["5 10 words concrete and factual
  authoritative context"]
- *Unexpected text before Title:* Here are my reasoning steps: We need to
  produce a catalog record with Title, Description, Keywords sections. The
  image is a view of a beach with a town, likely Viking Bay in Broadstairs,
  Kent, England. The authoritative context includes capture date/time and GPS
  coordinates. The draft metadata includes title, description, keywords. We
  need to analyze the image for visible details, and produce a revised
  metadata.  We must follow the rules:
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Processor class:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002065504
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3038
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Here are my reasoning steps:
We need to produce a catalog record with Title, Description, Keywords sections. The image is a view of a beach with a town, likely Viking Bay in Broadstairs, Kent, England. The authoritative context includes capture date/time and GPS coordinates. The draft metadata includes title, description, keywords. We need to analyze the image for visible details, and produce a revised metadata.

We must follow the rules:

- Title: 5-10 words, concrete and factual; authoritative context may supply identity and location. Output only the title text after the label.

- Description: 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details. Output only the description text after the label.

- Keywords: 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.

We must distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image. So we can incorporate the capture date/time and GPS coordinates in the description if they improve the record, but we must not claim they are visible. We can mention that the image was captured on 2026-07-25 at 18:33:16 UTC+01:00, but we must not say "the image shows the date/time". We can just include that as part of the description.

We need to correct errors in the draft metadata. Let's examine the draft:

- Title: "Viking Bay, Broadstairs, England, UK, GBR, Europe". That's 7 words? Actually it's many words: "Viking Bay," (2?), "Broadstairs," (1), "England," (1), "UK," (1), "GBR," (1), "Europe" (1). That's 7 words? Let's count: "Viking" (1), "Bay," (2), "Broadstairs," (3), "England," (4), "UK," (5), "GBR," (6), "Europe" (7). That's 7 words, within 5-10. It's okay. But we might want to simplify to "Viking Bay, Broadstairs, Kent, England". The draft includes "UK, GBR, Europe" which are redundant.
```

</details>

<a id="diagnostic-mlx-community-glm-46v-flash-mxfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-mxfp4 — unusable — repeated output, token cap truncation</summary>

### mlx-community/GLM-4.6V-Flash-mxfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* keyword: "seaside resort"
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00347453
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6623
- *Generation tokens:* 500
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
Title: Viking Bay, Broadstairs, Kent, England
Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline, a large historic building, and wind turbines out to sea in the distance.
Keywords: Viking Bay, Broadstairs, Kent, England, UK, coastal town, sandy beach, holiday makers, town skyline, historic building, wind turbines, sea, blue sky, sunny weather, people relaxing, beach huts, colorful beach huts, coastal landscape, British seaside resort, summer holiday, tourism, coastal architecture, wind farm, distant horizon, beach umbrellas, sandy shore, coastal vegetation, blue sea, white buildings, historic architecture, coastal development, seaside promenade, beachgoers, leisure activities, coastal tourism, British coastline, seaside town, beach landscape, coastal scenery, summer beach, coastal view, seaside resort, beach scene, coastal architecture, British seaside, beach tourism, coastal development, seaside view, beachgoers, seaside resort, British coastline, coastal tourism, seaside town, beach landscape, coastal scenery, summer beach, coastal view, seaside resort, beach scene, coastal architecture, British seaside, beach tourism, coastal development, seaside view, beachgoers, seaside resort, British coastline, coastal tourism, seaside town, beach landscape, coastal scenery, summer beach, coastal view, seaside resort, beach scene, coastal architecture, British seaside, beach tourism, coastal development, seaside view, beachgoers, seaside resort, British coastline, coastal tourism, seaside town, beach landscape, coastal scenery, summer beach, coastal view, seaside resort, beach scene, coastal architecture, British seaside, beach tourism, coastal development, seaside view, beachgoers, seaside resort, British coastline, coastal tourism, seaside town, beach landscape, coastal scenery, summer beach, coastal view, seaside resort, beach scene, coastal architecture, British seaside, beach tourism, coastal development, seaside view, beachgoers, seaside resort, British coastline, coastal tourism, seaside town, beach landscape, coastal scenery, summer beach, coastal view, seaside resort, beach scene, coastal architecture, British seaside, beach tourism, coastal development, seaside view, beachgoers, seaside resort, British coastline, coastal tourism, seaside town, beach landscape, coastal scenery, summer beach, coastal view, seaside resort, beach scene, coastal architecture, British seaside, beach
```

</details>

<a id="diagnostic-meta-llama-llama-32-11b-vision-instruct"></a>

<details>
<summary>meta-llama/Llama-3.2-11B-Vision-Instruct — unusable — missing requested sections</summary>

### meta-llama/Llama-3.2-11B-Vision-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 9eb2daaa8597bf192a8b0e73f848f3a102794df5
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.001737816
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 586
- *Generation tokens:* 70
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;

#### Complete output

```text
Viking Bay, Broadstairs, England, UK, GBR, Europe

A wide shot of the sandy beach at Viking Bay in Broadstairs, England, with the town's skyline and castle visible in the background, and a hazy view of wind turbines in the distance, capturing a sunny summer day with people relaxing and enjoying the scenery.
```

</details>

<a id="diagnostic-qwen-qwen3-vl-2b-instruct"></a>

<details>
<summary>Qwen/Qwen3-VL-2B-Instruct — unusable — repeated output, token cap truncation</summary>

### Qwen/Qwen3-VL-2B-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* keyword: "england"
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00055812
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16927
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Viking Bay, Broadstairs, Kent, England, UK, Europe

Description:
A wide-angle view of the bustling Viking Bay beach in Broadstairs, Kent, on a sunny summer day. The sandy shore is filled with people relaxing, sunbathing, and playing, with colorful beach huts and a lively promenade. The town's buildings, including a prominent castle-like structure, line the hillside, and wind turbines are visible on the horizon over the sea.

Keywords:
Adults, Beach, Blue sky, Buildings, Children, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town,
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — unusable — missing requested sections, unexpected catalog preamble, unexpected special token</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|begin_of_box|&gt;Title: Viking Bay,
  Broadstairs, England, UK, GBR, Europe
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003785838
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6623
- *Generation tokens:* 121
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title:
Viking Bay, Broadstairs, England, UK, GBR, Europe

Description:
A wide shot looking down at a sunny English coastal town scene in Broadstairs, Kent, on 25 July 2026. Holiday makers relax on the sandy Viking Bay beach with the town skyline, castle, and wind turbines out to sea in the distance.

Keywords:
Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-molmo-7b-d-0924-8bit"></a>

<details>
<summary>mlx-community/Molmo-7B-D-0924-8bit — unusable — unexpected catalog preamble</summary>

### mlx-community/Molmo-7B-D-0924-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble
- *Unexpected text before Title:* Remove non-visual information.
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Processor class:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.008389754
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1793
- *Generation tokens:* 126
- *Configured EOS token ID:* 151643
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
 Remove non-visual information.

Title:
Viking Bay, Broadstairs, England, UK, GBR, Europe

Description:
A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holidaymakers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.

Keywords:
Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-molmo-7b-d-0924-bf16"></a>

<details>
<summary>mlx-community/Molmo-7B-D-0924-bf16 — unusable — unexpected catalog preamble</summary>

### mlx-community/Molmo-7B-D-0924-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble
- *Unexpected text before Title:* Remove non-visual information.
- *Resolved model revision:* d871cbdb87a49b8071003098d6dbfd2a0f5a5b84
- *Processor class:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00870105
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1793
- *Generation tokens:* 126
- *Configured EOS token ID:* 151643
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
 Remove non-visual information.

Title:
Viking Bay, Broadstairs, England, UK, GBR, Europe

Description:
A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day holidaymakers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.

Keywords:
Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Instruct-bf16 — unusable — repeated output, token cap truncation</summary>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* keyword: "england"
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.010519696
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16927
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Viking Bay, Broadstairs, Kent, England, UK, Europe

Description:
A wide-angle view of the bustling Viking Bay beach in Broadstairs, Kent, on a sunny summer day. The sandy shore is filled with people relaxing, sunbathing, and playing, with colorful beach huts and a lively promenade. The town's buildings, including a prominent castle-like structure, line the hillside, and wind turbines are visible on the horizon over the sea.

Keywords:
Adults, Beach, Blue sky, Buildings, Children, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town, Wind turbines, Viking Bay, Broadstairs, England, UK, Europe, People, Relaxing, Sitting, Sky, Sand, Sea, Town,
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Thinking-bf16 — unusable — missing requested sections, token cap truncation, unexpected catalog preamble</summary>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble
- *Missing sections:* ["keywords"]
- *Unexpected text before Title:* Got it, let's tackle this step by step.
  First, I need to analyze the image for metadata using the given context. The
  context provides a capture date, GPS coordinates, and some draft metadata.
  Let's start with the Title.
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.010830992
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16929
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, I need to analyze the image for metadata using the given context. The context provides a capture date, GPS coordinates, and some draft metadata. Let's start with the Title.

Title: Needs 5-10 words, concrete and factual. The draft title is "Viking Bay, Broadstairs, England, UK, GBR, Europe". But wait, the image shows a beach with buildings, so maybe "Viking Bay Beach, Broadstairs, Kent, England" or something. Wait, the context says GPS is 51.358240°N, 1.432820°E. Let's check the location. Broadstairs is in Kent, UK. The image has a beach with buildings, so the title should be specific. Let's see: the draft title is a bit off. Maybe "Viking Bay Beach, Broadstairs, Kent, England" but need to check if the image matches. Wait, the context says "Authoritative context: Capture date/time: 2026-07-25 18:33:16 UTC+01:00; GPS: 51.358240°N, 1.432820°E". So the location is Broadstairs, Kent, UK. The title should be concrete. Let's go with "Viking Bay Beach, Broadstairs, Kent, England".

Description: Needs 1-2 sentences. Combine authoritative context with visible details. The context says capture date is 2026-07-25, which is a summer day (July 25 is summer in the UK). The image shows a busy beach with people, buildings, blue sky, wind turbines in the distance. The draft description mentions "sunny summers day holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance." Wait, the image has a town skyline, but does it have a castle? The image shows a coastal town with buildings, but maybe not a castle. Wait, the context says "Authoritative context: Capture date/time: 2026-07-25 18:33:16 UTC+01:00; GPS: 51.3
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

<details>
<summary>mlx-community/paligemma2-3b-ft-docci-448-bf16 — unusable — missing requested sections, token cap truncation, prompt instruction echo</summary>

### mlx-community/paligemma2-3b-ft-docci-448-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Echoed instruction fragments:* ["prompt instructions", "do not copy prompt
  instructions into the title"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* f66333527ce75342b09d4df81873f65272ec2f30
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.022496496
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1646
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy-prompt instructions into the title, description, or keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent.
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.
- Do not copy
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — unusable — repeated output, missing requested sections, token cap truncation</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "worldly views of the..."
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.006767734
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 585
- *Generation tokens:* 500
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;

#### Complete output

```text
**Title**
Viking Bay, Broadstairs, England, UK, GBR, Europe

**Description**
A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent, with a sandy beach and a town skyline in the background, featuring a castle and wind turbines in the distance, on a sunny summer's day. Holidaymakers are relaxing on the beach.

**Keywords**
Adventurous, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Objects, People, Relaxing, Sitting, Sky, Summer, Sun, Sunbathing, Sunlight, Sunny, Trees, UK, Wind turbines, Windy, World, Worldly, Worldly view, Worldly views, Worldly views of nature, Worldly views of the sea, Worldly views of the sky, Worldly views of the sun, Worldly views of the wind, Worldly views of the wind turbines, Worldly views of the world, Worldly views of the world's nature, Worldly views of the world's sea, Worldly views of the world's sky, Worldly views of the world's sun, Worldly views of the world's wind, Worldly views of the world's wind turbines, Worldly views of the world's world, Worldly views of the world's world's nature, Worldly views of the world's world's sea, Worldly views of the world's world's sky, Worldly views of the world's world's sun, Worldly views of the world's world's wind, Worldly views of the world's world's wind turbines, Worldly views of the world's world's world, Worldly views of the world's world's world's nature, Worldly views of the world's world's world's sea, Worldly views of the world's world's world's sky, Worldly views of the world's world's world's sun, Worldly views of the world's world's world's wind, Worldly views of the world's world's world's wind turbines, Worldly views of the world's world's world's world, Worldly views of the world's world's world's world's nature, Worldly views of the world's world's world's world's sea, Worldly views of the world's world's world's world's sky, Worldly views of the world's world's world's world
```

</details>

<a id="diagnostic-mlx-community-x-reasoner-7b-8bit"></a>

<details>
<summary>mlx-community/X-Reasoner-7B-8bit — unusable — repeated output, token cap truncation</summary>

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* keyword: "england"
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.014943378
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16938
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Viking Bay, Broadstairs, England, UK, GBR, Europe

Description:
A vibrant scene at Viking Bay in Broadstairs, Kent, on a sunny summer day. Holidaymakers enjoy the sandy beach, with colorful beach huts and umbrellas dotting the shore, while the town's historic buildings and a distant horizon of wind turbines create a picturesque backdrop.

Keywords:
Broadstairs, England, UK, GBR, Europe, Viking Bay, Kent, Beach, Sandy, Holidaymakers, Colorful huts, Umbrellas, Town skyline, Wind turbines, Horizon, Summer, Sunlight, Relaxing, Buildings, Castle, Castle ruins, Windmills, Coastal town, England, Europe, Beach scene, People, Architecture, Wind energy, Tourism, Summer vacation, Historical site, Seaside resort, British coast, English countryside, Wind farm, Coastal landscape, English heritage, Beach activities, Family-friendly, Summer day, Clear sky, Blue sea, Sandy beach, Beach huts, Town buildings, Castle ruins, Wind turbines, Horizon line, Coastal town, England, UK, GBR, Europe, Viking Bay, Broadstairs, Kent, England, UK, GBR, Europe, Beach, Sandy, Holidaymakers, Colorful huts, Umbrellas, Town skyline, Wind turbines, Horizon, Summer, Sunlight, Relaxing, Buildings, Castle, Castle ruins, Windmills, Horizon, Wind energy, Coastal town, England, Europe, Beach scene, People, Architecture, Wind energy, Tourism, Summer vacation, Historical site, Seaside resort, English heritage, Beach activities, Family-friendly, Summer day, Clear sky, Blue sea, Sandy beach, Beach huts, Town buildings, Castle ruins, Wind turbines, Horizon line, Coastal town, England, UK, GBR, Europe, Viking Bay, Broadstairs, Kent, England, UK, GBR, Europe, Beach, Sandy, Holidaymakers, Colorful huts, Umbrellas, Town skyline, Wind turbines, Horizon, Summer, Sunlight, Relaxing, Buildings, Castle, Castle ruins, Windmills, Horizon, Wind energy, Coastal town, England, Europe, Beach scene, People, Architecture, Wind energy, Tourism, Summer vacation, Historical site, Seaside resort, English heritage, Beach activities, Family-friendly, Summer day, Clear sky, Blue sea, Sandy beach, Beach huts, Town buildings, Castle ruins, Wind turbines, Horizon line, Coastal
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-Instruct-4bit — usable_with_caveats — draft returned unchanged</summary>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* draft_returned_unchanged
- *Draft fields returned unchanged:* ["title", "description", "keywords"]
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.0102084
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16938
- *Generation tokens:* 120
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Viking Bay, Broadstairs, England, UK, GBR, Europe
Description: A wide shot looking down at a beautiful English coastal town scene in Broadstairs, Kent. On a sunny summers day, holiday makers relax on the sandy Viking Bay beach with the town skyline and castle in the background and wind turbines out to sea in the hazy distance.
Keywords: Adobe Stock, Adults, Any Vision, Blue sky, Britain, Buildings, Bushes, Children, Coast, Crowd, England, Europe, Holiday, Horizon, Kent, Objects, People, Relaxing, Sitting, Sky
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — missing requested sections, token cap truncation, unexpected catalog preamble, thinking trace present, thinking trace incomplete</summary>

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
  need to analyze the image and create a catalog record following the given
  instructions. The user wants a title, description, and keywords based on the
  image, using British English. The context provided includes the location
  (Viking Bay, Broadstairs, England, UK, GBR, Europe) and the date/time
  (2026-07-25 18:33:16 UTC+01:00), GPS coordinates (51.358240°N, 1.432820°E).
  Looking at the image: it's a sunny day at a beach with people relaxing. The
  beach is sandy, buildings (residential and possibly commercial) line the
  shore, the sea is visible with wind turbines in the distance. The foreground
  has greenery, maybe bushes. The sky is clear blue.  For the title, I need a
  5-10 word title. The existing draft was "Viking Bay, Broadstairs, England,
  UK, GBR, Europe" but maybe expand a bit. Maybe "Sunny Viking Bay Beach,
  Broadstairs, Kent" since it's more descriptive.  Description should be 1-2
  sentences. Mention the beach, people relaxing, buildings, sea, wind
  turbines, sunny day. Use British English terms like "sandy beach", "coastal
  town scene".
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.005325934
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1608
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to analyze the image and create a catalog record following the given instructions. The user wants a title, description, and keywords based on the image, using British English. The context provided includes the location (Viking Bay, Broadstairs, England, UK, GBR, Europe) and the date/time (2026-07-25 18:33:16 UTC+01:00), GPS coordinates (51.358240°N, 1.432820°E). 

Looking at the image: it's a sunny day at a beach with people relaxing. The beach is sandy, buildings (residential and possibly commercial) line the shore, the sea is visible with wind turbines in the distance. The foreground has greenery, maybe bushes. The sky is clear blue. 

For the title, I need a 5-10 word title. The existing draft was "Viking Bay, Broadstairs, England, UK, GBR, Europe" but maybe expand a bit. Maybe "Sunny Viking Bay Beach, Broadstairs, Kent" since it's more descriptive. 

Description should be 1-2 sentences. Mention the beach, people relaxing, buildings, sea, wind turbines, sunny day. Use British English terms like "sandy beach", "coastal town scene". 

Keywords: Need 10-18 terms. Include location (Viking Bay, Broadstairs, Kent, England, UK, GBR, Europe), people (adults, children, crowd), activities (relaxing, sitting, sunbathing), elements (beach, sea, buildings, wind turbines, sky, bushes, clear blue, sunny, holiday, horizon). Maybe add "coastal town", "beach huts", "tents", "coastal townscape".

Check if any details are uncertain. The existing draft had "Holiday makers" but maybe "people" is better. Also, the wind turbines are visible in the distance. The buildings are a mix of residential and maybe some commercial. The foreground has greenery, so "bushes" is visible. The sky is clear blue. 

So, title: "Sunny Viking Bay Beach, Broadstairs, Kent". Description: "A sunny day at Viking Bay beach in Broadstairs, Kent, with people relaxing on the sandy shore, colorful beach huts, a coastal town skyline, and wind turbines visible over the sea." Keywords: Viking
```

</details>

## Indeterminate Attempts

None.

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                        | Performance                                                                               |
|-------------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | rev 6c33f49ebc0b; Lfm2VlProcessor; stop completed       | 688 prompt / 127 generated; 506 tok/s; 1.5 GB peak; cleanup 0.000247/0.0 GB active/cache  |
| mlx-community/LFM2-VL-1.6B-8bit                       | rev 294b90e5ae23; Lfm2VlProcessor; stop completed       | 874 prompt / 126 generated; 322 tok/s; 3.0 GB peak; cleanup 0.00611/0.0 GB active/cache   |
| mlx-community/MiniCPM-V-4.6-8bit                      | rev 03721395f6b8; MiniCPMVProcessor; stop completed     | 1234 prompt / 85 generated; 266 tok/s; 4.1 GB peak; cleanup 0.00729/0.0 GB active/cache   |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed       | 899 prompt / 104 generated; 112 tok/s; 17 GB peak; cleanup 0.0177/0.0 GB active/cache     |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed     | 2875 prompt / 135 generated; 179 tok/s; 6.4 GB peak; cleanup 0.00808/0.0 GB active/cache  |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed     | 2876 prompt / 126 generated; 65.4 tok/s; 12 GB peak; cleanup 0.00755/0.0 GB active/cache  |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed      | 2947 prompt / 79 generated; 39.1 tok/s; 15 GB peak; cleanup 0.0238/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed     | 2876 prompt / 129 generated; 61.7 tok/s; 12 GB peak; cleanup 0.00782/0.0 GB active/cache  |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed | 2909 prompt / 111 generated; 31.3 tok/s; 18 GB peak; cleanup 0.00467/0.0 GB active/cache  |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed       | 899 prompt / 105 generated; 26.5 tok/s; 20 GB peak; cleanup 0.0187/0.0 GB active/cache    |
| mlx-community/pixtral-12b-bf16                        | rev 378cf65efd79; PixtralProcessor; stop completed      | 2947 prompt / 79 generated; 20.3 tok/s; 27 GB peak; cleanup 0.0241/0.0 GB active/cache    |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed       | 890 prompt / 115 generated; 23.2 tok/s; 18 GB peak; cleanup 0.0155/0.0 GB active/cache    |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed     | 2343 prompt / 111 generated; 24.7 tok/s; 22 GB peak; cleanup 0.00233/0.0 GB active/cache  |
| mlx-community/InternVL3-14B-8bit                      | rev 50efc568c7df; InternVLChatProcessor; stop completed | 2909 prompt / 124 generated; 23.5 tok/s; 19 GB peak; cleanup 0.00436/0.0 GB active/cache  |
| mlx-community/gemma-3-27b-it-qat-8bit                 | rev c408904bc9a0; Gemma3Processor; stop completed       | 890 prompt / 118 generated; 11.6 tok/s; 32 GB peak; cleanup 0.0161/0.0 GB active/cache    |
| mlx-community/MolmoPoint-8B-fp16                      | rev 0a60033b4e48; MolmoPointProcessor; stop completed   | 3410 prompt / 121 generated; 5.68 tok/s; 27 GB peak; cleanup 0.00932/0.0 GB active/cache  |
| mlx-community/Ornith-1.0-35B-bf16                     | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed      | 16957 prompt / 126 generated; 64.4 tok/s; 76 GB peak; cleanup 0.00983/0.0 GB active/cache |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed      | 16957 prompt / 227 generated; 94.2 tok/s; 26 GB peak; cleanup 0.0124/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | rev b729d115bb2c; Qwen3VLProcessor; stop completed      | 16957 prompt / 114 generated; 90.7 tok/s; 35 GB peak; cleanup 0.0129/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | rev 731d09ba3597; Qwen3VLProcessor; stop completed      | 16957 prompt / 123 generated; 60.0 tok/s; 76 GB peak; cleanup 0.0134/0.0 GB active/cache  |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed      | 16957 prompt / 127 generated; 84.5 tok/s; 11 GB peak; cleanup 0.0139/0.0 GB active/cache  |
| mlx-community/Qwen3.5-27B-4bit                        | rev 45797d2985a1; Qwen3VLProcessor; stop completed      | 16957 prompt / 142 generated; 28.6 tok/s; 26 GB peak; cleanup 0.0113/0.0 GB active/cache  |
| mlx-community/Qwen3.5-27B-mxfp8                       | rev 2d6caf2325c2; Qwen3VLProcessor; stop completed      | 16957 prompt / 132 generated; 16.0 tok/s; 38 GB peak; cleanup 0.0118/0.0 GB active/cache  |
| mlx-community/Qwen3.6-27B-mxfp8                       | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed      | 16957 prompt / 122 generated; 12.2 tok/s; 38 GB peak; cleanup 0.0144/0.0 GB active/cache  |

</details>

## Shared Reproduction and Provenance

### Prompt

Save this exact prompt as prompt.txt.

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

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/Step-3.7-Flash-oQ2e                | 3dacb46f724ac89725bcd922fb779c7ed1499fe7 |
| mlx-community/nanoLLaVA-1.5-4bit                 | 5240204744963d72823e5de933c528c4aa82dfca |
| qnguyen3/nanoLLaVA                               | 13d60cec183a86755afed64da495fcc2c382ea80 |
| HuggingFaceTB/SmolVLM-Instruct                   | 81cd9a775a4d644f2faf4e7becff4559b46b14c7 |
| mlx-community/FastVLM-0.5B-bf16                  | 81ffe929046666c43de53691147b1669ba0f3a4c |
| mlx-community/paligemma2-10b-ft-docci-448-6bit   | 1485fa9b3c7adb360cd354a29a401f0d441ec728 |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx         | 844516024a1c4400d34489b89ee067d794e432ed |
| mlx-community/SmolVLM-Instruct-bf16              | cae61cdedd0602419b43b6102dc33cd9f1e929a6 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/LFM2.5-VL-1.6B-bf16                | 16a710cf8afca206ff16a95a4ad6fe657f876ce1 |
| mlx-community/gemma-4-31b-bf16                   | 19f0f1af698c51edaf1e93b3a3a5435b282de30f |
| mlx-community/Phi-3.5-vision-instruct-bf16       | d8da684308c275a86659e2b36a9189b2f4aec8ea |
| mlx-community/paligemma2-10b-ft-docci-448-bf16   | 7c412694b919432784c730b62fadafb1c2e15d0d |
| microsoft/Phi-3.5-vision-instruct                | 12b77fb40b63a2c73c68243d3f767aab688a1b2a |
| mlx-community/gemma-3n-E2B-4bit                  | ec68dc186276e20e4bed30b96a2b5c667e0a81e3 |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |
| mlx-community/gemma-3n-E4B-it-bf16               | d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit            | 4b3b11ce0874c36a99e13e17e355049042f8620a |
| mlx-community/Kimi-VL-A3B-Thinking-8bit          | 85daf3dc2490c0f824143338f08ba45f475c9ce4 |
| mlx-community/GLM-4.6V-Flash-6bit                | df9464782d3452e0dfd86afe0984f1c9eca75ca1 |
| mlx-community/llava-v1.6-mistral-7b-8bit         | b8df5f329d95a7abe6429ed46093f9b84e8e6396 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX    | 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2 |
| mlx-community/GLM-4.6V-Flash-mxfp4               | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b |
| meta-llama/Llama-3.2-11B-Vision-Instruct         | 9eb2daaa8597bf192a8b0e73f848f3a102794df5 |
| Qwen/Qwen3-VL-2B-Instruct                        | 89644892e4d85e24eaac8bacfd4f463576704203 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Molmo-7B-D-0924-8bit               | 90a14ed7a230088904c7556fbe6d67b295c33f5f |
| mlx-community/Molmo-7B-D-0924-bf16               | d871cbdb87a49b8071003098d6dbfd2a0f5a5b84 |
| mlx-community/Qwen3-VL-2B-Instruct-bf16          | c8a67a84327484ba87f5ec4f8fb927cdafd791aa |
| mlx-community/Qwen3-VL-2B-Thinking-bf16          | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/paligemma2-3b-ft-docci-448-bf16    | f66333527ce75342b09d4df81873f65272ec2f30 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/Qwen2-VL-2B-Instruct-4bit          | 01af461cdb9574acc09084a0ef94e216e142b085 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |

### Canonical parameterised Python reproduction

Run one model per process to avoid sequential Metal-state interactions.

```bash
python reproduce.py MODEL_ID --revision RESOLVED_REVISION --image 20260725-183316_DSC01175.jpg --prompt-file prompt.txt
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

# Defaults used in the report: prompt.txt, 20260725-183316_DSC01175.jpg
```

### Components and system

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
