# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

These models serve many purposes; this run probes exactly one narrow task:
producing catalogue metadata for a single photograph from the assisted-lane
prompt and whatever context it supplies — here, camera-recorded capture
context plus draft descriptive hints previously produced by a more capable
model. Results say nothing about a model's fitness for other uses.

## Run Summary

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 41    |
| Conclusive outcomes | 41    |
| Completed           | 40    |
| Crashed             | 1     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 1     |
| none                           | 29    |
| observation needs reproduction | 11    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 11    |
| usable              | 10    |
| usable with caveats | 19    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| No response text was returned                                                         | 1     |
| Response repeats the same text                                                        | 6     |
| Generation was stopped early after sustained repeated output                          | 4     |
| Unrecognised model control tokens remain visible                                      | 3     |
| Required fields are missing or empty                                                  | 5     |
| Response repeats the task instructions instead of only returning the requested fields | 1     |
| Extra text appears before the Title field                                             | 3     |
| Response appears cut off at the token limit                                           | 4     |
| Internal reasoning block appears incomplete                                           | 2     |
| Title or keywords do not meet requested constraints                                   | 23    |
| Title, Description and Keywords copy all supplied hints unchanged                     | 1     |

## Triage

| Model                                                                                                      | Execution | Usability           | Maintainer status              | Observations                                                                                                                |
|------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [tencent/Youtu-VL-4B-Instruct](#diagnostic-tencent-youtu-vl-4b-instruct)                                   | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                        |
| [mlx-community/gemma-3n-E4B-it-bf16](#diagnostic-mlx-community-gemma-3n-e4b-it-bf16)                       | completed | unusable            | observation_needs_reproduction | empty response; missing required fields                                                                                     |
| [jinaai/jina-vlm-mlx](#diagnostic-jinaai-jina-vlm-mlx)                                                     | completed | unusable            | observation_needs_reproduction | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [LiquidAI/LFM2.5-VL-450M-MLX-bf16](#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16)                            | completed | unusable            | observation_needs_reproduction | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [mlx-community/GLM-4.6V-Flash-mxfp4](#diagnostic-mlx-community-glm-46v-flash-mxfp4)                        | completed | unusable            | observation_needs_reproduction | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                 | completed | unusable            | observation_needs_reproduction | repeated text; stopped early: repeating; title/keyword constraints failed                                                   |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)   | completed | unusable            | observation_needs_reproduction | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [mlx-community/Molmo2-8B-4bit](#diagnostic-mlx-community-molmo2-8b-4bit)                                   | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; cut off at token limit                                                              |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)   | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                      |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                      |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                    | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible; title/keyword constraints failed                                                                    |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)              | completed | unusable            | observation_needs_reproduction | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                |

## Crashes requiring action

<a id="diagnostic-tencent-youtu-vl-4b-instruct"></a>

### tencent/Youtu-VL-4B-Instruct

#### Root exception and chain

```text
builtins.ImportError: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)
builtins.ValueError: Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type youtu_vl)
- *Phase:* model_load
- *Stage:* Lib Version
- *Package:* model-repo-code
- *Error type:* ValueError
- *Error message:* Model loading failed: cannot import name
  'DefaultFastImageProcessorKwargs' from
  'transformers.image_processing_utils_fast' (unknown location)
- *Root error type:* ImportError
- *Root error message:* cannot import name 'DefaultFastImageProcessorKwargs'
  from 'transformers.image_processing_utils_fast' (unknown location)
- *Resolved model revision:* 8d30a0e49662a1d628a472b12df264dbcd768753
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.013059304
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 10.68
- *Parameter count:* 4.00B (name-estimate)
- *Declared context length:* 32,768 (max_position_embeddings)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13593, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12726, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 818, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1217, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1362, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 9 more times]
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo2/processing.py", line 768, in _patched_auto_processor_from_pretrained_molmo2
    return _original_auto_processor_from_pretrained_molmo2.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 644, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 11 more times]
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/auto/processing_auto.py", line 326, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/processing_utils.py", line 1735, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/processing_utils.py", line 1875, in _get_arguments_from_pretrained
    sub_processor = auto_processor_class.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/auto/image_processing_auto.py", line 672, in from_pretrained
    image_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/dynamic_module_utils.py", line 623, in get_class_from_dynamic_module
    return get_class_in_module(class_name, final_module, force_reload=force_download)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/dynamic_module_utils.py", line 309, in get_class_in_module
    module_spec.loader.exec_module(module)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
  File "<frozen importlib._bootstrap_external>", line 759, in exec_module
  File "<frozen importlib._bootstrap>", line 491, in _call_with_frames_removed
  File "~/.cache/huggingface/modules/transformers_modules/_8d30a0e49662a1d628a472b12df264dbcd768753/71940f1d5c3bbe2f/image_processing_siglip2_fast.py", line 7, in <module>
    from transformers.image_processing_utils_fast import (
    ...<3 lines>...
    )
ImportError: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14606, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13608, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: cannot import name 'DefaultFastImageProcessorKwargs' from 'transformers.image_processing_utils_fast' (unknown location)

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 19 files:   0%|          | 0/19 [00:00<?, ?it/s]
Fetching 19 files: 100%|##########| 19/19 [00:00<00:00, 2785.16it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[01:47:22] DEBUG    HF Cache Info for tencent/Youtu-VL-4B-Instruct: size=10229.6 MB, files=27
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-gemma-3n-e4b-it-bf16"></a>

<details>
<summary>mlx-community/gemma-3n-E4B-it-bf16 — unusable — empty response; missing required fields</summary>

### mlx-community/gemma-3n-E4B-it-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* empty_output, missing_requested_sections
- *Arch supported by installed mlx-vlm:* yes (model_type gemma3n)
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.011519164
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 569
- *Prompt composition:* 569 = 311 text/template (tokenizer-exact) + 258
  non-text (45%, image/audio expansion)
- *Checkpoint weights (GB):* 15.70
- *Parameter count:* 4.00B (name-estimate)
- *Declared context length:* 32,768 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (15.71 GB vs 15.70 GB on disk)
- *Generation tokens:* 1
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
(empty)
```

</details>

<a id="diagnostic-jinaai-jina-vlm-mlx"></a>

<details>
<summary>jinaai/jina-vlm-mlx — unusable — repeated text; stopped early: repeating; title/keyword constraints failed</summary>

### jinaai/jina-vlm-mlx

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type jvlm via jina_vlm)
- *Repeated fragment:* keyword: "outboard motor"
- *Title word count:* 13
- *Requested title word range:* [5, 10]
- *Keyword count:* 54
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["outboard motor", "passenger", "railing", "river",
  "riverbank", "shrubs", "trees", "leisure", "cruising", "nautical"]
- *Resolved model revision:* a987631a01dc554a787d87a45fb01fb48f8aaca4
- *Processor class:* mlx_vlm.models.jina_vlm.processing_jinavlm.JinaVLMProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Post-cleanup active memory (GB):* 0.000754728
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2047
- *Prompt composition:* 2,047 = 304 text/template (tokenizer-exact) + 1,743
  non-text (85%, image/audio expansion)
- *Checkpoint weights (GB):* 2.11
- *Quantization:* 4-bit, group 64
- *Declared context length:* 40,960 (text_config.max_sequence_length)
- *Load active memory vs checkpoint:* 1.00x (2.11 GB vs 2.11 GB on disk)
- *Generation tokens:* 275
- *Configured EOS token ID:* 151643
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
A White Motor Cruiser named 'Wavey Katey II' Cruises Along a Calm Waterway

Description:
A white motor cruiser named 'Wavey Katey II' is seen cruising along a calm waterway, flying a British maritime flag. The boat is passing by a rustic wooden riverside house and lush green trees. The boat driver is visible inside the cabin, enjoying the peaceful surroundings. The boat is equipped with a canopy, fenders, and an outboard motor. The scene is set during the daytime under clear skies.

Keywords:
Boat, Wavey Katey II, Motor Cruiser, British Maritime Flag, Calm Waterway, Rustic Wooden Riverbank, Lush Green Trees, Boat Driver, Cabin Cruiser, Canopy, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Leisure, Cruising, Nautical, Outboard Motor, Passenger, Railing, River,
```

</details>

<a id="diagnostic-liquidai-lfm25-vl-450m-mlx-bf16"></a>

<details>
<summary>LiquidAI/LFM2.5-VL-450M-MLX-bf16 — unusable — repeated text; stopped early: repeating; title/keyword constraints failed</summary>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Repeated fragment:* keyword: "boat fender"
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 42
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["boat fender", "boat driver"]
- *Resolved model revision:* ed71acdae0799c98cb450787d049d1343b65cd52
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Post-cleanup active memory (GB):* 0.00013212
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2627
- *Prompt composition:* 2,627 = 318 text/template (tokenizer-exact) + 2,309
  non-text (88%, image/audio expansion)
- *Checkpoint weights (GB):* 0.90
- *Parameter count:* 0.45B (name-estimate)
- *Declared context length:* 128,000 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (0.90 GB vs 0.90 GB on disk)
- *Generation tokens:* 200
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Boat Cruising on a Calm River

Description:
A white motor cruiser boat named 'Wavey Katey II' cruises along a calm river, with a British maritime flag flying and lush green trees in the background.

Keywords:
Boat, Cruising, River, Riverbank, Foliage, Nautical, Outboard motor, Passenger, Cabin cruiser, Canopy, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver, Boat fender, Boat driver,
```

</details>

<a id="diagnostic-mlx-community-glm-46v-flash-mxfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-mxfp4 — unusable — repeated text; stopped early: repeating; title/keyword constraints failed</summary>

### mlx-community/GLM-4.6V-Flash-mxfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Repeated fragment:* keyword: "boat"
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 50
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["boat", "boat driver", "boat fender"]
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Post-cleanup active memory (GB):* 0.002425904
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6325
- *Prompt composition:* 6,325 = 294 text/template (tokenizer-exact) + 6,031
  non-text (95%, image/audio expansion)
- *Checkpoint weights (GB):* 6.78
- *Quantization:* 4-bit, group 32, mxfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (6.78 GB vs 6.78 GB on disk)
- *Generation tokens:* 200
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: Wavey Katey II Motor Cruiser
Description: The white motor cruiser 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.
Keywords: Wavey Katey II, motor cruiser, British maritime flag, calm waterway, rustic wooden riverside house, lush green trees, boat, boating, leisure, motorboat, nautical, outboard motor, river, riverbank, shrubs, trees, canopy, cabin cruiser, fender, passenger, railing, flag, foliage, cottage, cruising, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat driver, boat fender, boat, boat
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — unusable — repeated text; stopped early: repeating; title/keyword constraints failed</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Repeated fragment:* keyword: "boat"
- *Title word count:* 7
- *Requested title word range:* [5, 10]
- *Keyword count:* 61
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["boat", "waterway", "flag", "motor", "cruiser",
  "maritime", "british"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Post-cleanup active memory (GB):* 0.002999356
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2603
- *Prompt composition:* 2,603 = 290 text/template (tokenizer-exact) + 2,313
  non-text (89%, image/audio expansion)
- *Checkpoint weights (GB):* 16.92
- *Parameter count:* 8.00B (name-estimate)
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (16.93 GB vs 16.92 GB on disk)
- *Generation tokens:* 250
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
 Title: "Wavey Katey II Cruises Along Calm Waterway".

Description: "The 'Wavey Katey II' motor cruiser boat, adorned with a British maritime flag, glides smoothly along a serene waterway. The boat, painted white with a blue canopy, is accompanied by a rustic wooden riverside house and lush green trees, creating a picturesque scene. The boat's outboard motor propels it forward, and the calm water reflects the surrounding foliage, adding to the tranquil atmosphere."

Keywords: motor cruiser, boat, Wavey Katey II, British maritime flag, calm waterway, rustic wooden riverside house, lush green trees, serene, outboard motor, picturesque, tranquil, waterway, boat, flag, canopy, motor, boat, motor, cruiser, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British, boat, cruiser, motor, boat, waterway, boat, flag, maritime, British,
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_incomplete,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Repeated fragment:* keyword: "motorboat"
- *Unexpected text before Title:* ◁think▷Got it, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  context provided. Let's start by analyzing the image.  The image shows a
  white motor cruiser named "Wavey Katey II" on a calm waterway. There's a
  rustic wooden riverside house in the background, lush green trees, and the
  boat has a British maritime flag. The capture date/time is 2026-08-28
  16:17:42 UTC+01:00.  For the title, I need a 5-10 word British-English
  title. Maybe "Wavey Katey II Cruising on a Calm River". That's concise and
  descriptive.  Next, the description: combine context and main subject. The
  boat is a motor cruiser, name, on a calm river, with a rustic house and
  trees. So: "A white motor cruiser named 'Wavey Katey II' cruises along a
  calm river, passing a rustic wooden riverside house surrounded by lush green
  trees, with a British maritime flag flying."
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Title word count:* 8
- *Requested title word range:* [5, 10]
- *Keyword count:* 73
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["motorboat", "cruising", "river", "calm water",
  "rustic house", "lush foliage", "british flag", "fenders", "wait",
  "boating", "cabin cruiser", "leisure", "nautical"]
- *Token-cap degradation evidence:* ["repetitive_tail",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003966048
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1307
- *Prompt composition:* 1,307 = 300 text/template (tokenizer-exact) + 1,007
  non-text (77%, image/audio expansion)
- *Checkpoint weights (GB):* 32.82
- *Parameter count:* 3.00B (name-estimate)
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (32.82 GB vs 32.82 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and context provided. Let's start by analyzing the image.

The image shows a white motor cruiser named "Wavey Katey II" on a calm waterway. There's a rustic wooden riverside house in the background, lush green trees, and the boat has a British maritime flag. The capture date/time is 2026-08-28 16:17:42 UTC+01:00. 

For the title, I need a 5-10 word British-English title. Maybe "Wavey Katey II Cruising on a Calm River". That's concise and descriptive.

Next, the description: combine context and main subject. The boat is a motor cruiser, name, on a calm river, with a rustic house and trees. So: "A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying."

Keywords: Need 10-18 unique keywords. Let's list them: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Motor Cruiser, Fenders, Cabin Cruiser. That's 10 keywords.

Wait, check if all hints are covered. The hints include Boat, Boat driver (but no driver visible), Boat fender (yes, the fenders are there), Boating, Cabin cruiser, Canopy (the blue canopy), Cottage (the house), Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor (maybe the engine), Passenger (no visible passenger), Railing (the house's railing), River, Riverbank, Shrubs, Trees. But the image might not show all. Let's pick the most visible: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical. That's 10. Maybe add "Leisure" as well. Let's adjust.

Wait, the hints mention "Boating" and "Leisure" as keywords. So include those. Let's check again. The image has the boat, motorboat, cruising, river, calm water, rustic house, lush foliage, British flag, fenders, cabin cruiser, nautical, boating, leisure. That's 12. Maybe "Outboard motor" if visible. The boat has an outboard? The image shows the engine area, so maybe. Let's include that. So: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Outboard Motor. That's 12. Maybe that's enough.

Now, structure:

Title: Wavey Katey II Cruising on a Calm River

Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying.

Keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Outboard Motor

Wait, the hints also mention "Passenger" but no passenger is visible. So maybe omit. Let's stick to what's visible. Let's check the image again. The boat has a canopy (blue), so include Canopy. So add Canopy. Now keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Canopy, Outboard Motor. That's 13. Maybe that's okay.

Wait, the hints also have "Shrubs" and "Trees" but the image has trees in the background. Maybe "Trees" is covered under Lush Foliage. So maybe not necessary. Let's finalize.

Title: Wavey Katey II Cruising on a Calm River

Description: A white motor cruiser named 'Wavey Katey II' cruises along a calm river, passing a rustic wooden riverside house surrounded by lush green trees, with a British maritime flag flying.

Keywords: Boat, Motorboat, Cruising, River, Calm Water, Rustic House, Lush Foliage, British Flag, Fenders, Cabin Cruiser, Nautical, Boating, Leisure, Canopy, Outboard Motor

Wait, the hints mention "Boating" and "Leisure" as keywords. Let's check if all are covered. Yes. The boat is a cabin cruiser, so Cabin Cruiser is good. The outboard
```

</details>

<a id="diagnostic-mlx-community-molmo2-8b-4bit"></a>

<details>
<summary>mlx-community/Molmo2-8B-4bit — unusable — repeated text; missing required fields; cut off at token limit</summary>

### mlx-community/Molmo2-8B-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "boat side identification text,..."
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "unfinished_list"]
- *Resolved model revision:* 4fcbe926577641af12aa5086cf5c292db9dd9730
- *Processor class:* mlx_vlm.models.molmo2.processing.Molmo2Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.005981284
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1504
- *Prompt composition:* 1,504 = 302 text/template (tokenizer-exact) + 1,202
  non-text (80%, image/audio expansion)
- *Checkpoint weights (GB):* 6.44
- *Parameter count:* 8.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 36,864 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (6.45 GB vs 6.44 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Wavey Katey II Cruises Calm River Past Rustic Cottage

A white motor cruiser named Wavey Katey II glides smoothly along a tranquil waterway, its blue canopy and red flag visible against the backdrop of a rustic wooden riverside house and lush green trees. The boat's name is clearly displayed on its side, and the scene captures a peaceful moment of leisure boating.

Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, Waterway, Rustic house, Blue canopy, Red flag, White boat, Calm water, Rustic wooden house, Lush green trees, Boat name, Boat side, Boat side name, Boat side display, Boat side text, Boat side writing, Boat side lettering, Boat side branding, Boat side logo, Boat side emblem, Boat side insignia, Boat side marking, Boat side identification, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text, Boat side identification writing, Boat side identification lettering, Boat side identification branding, Boat side identification emblem, Boat side identification insignia, Boat side identification marking, Boat side identification text,
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
- *Post-cleanup active memory (GB):* 0.010437794
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 575
- *Prompt composition:* 575 = 310 text/template (tokenizer-exact) + 265
  non-text (46%, image/audio expansion)
- *Checkpoint weights (GB):* 27.97
- *Parameter count:* 26.00B (name-estimate)
- *Quantization:* 8-bit, group 64, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (27.99 GB vs 27.97 GB on disk)
- *Generation tokens:* 88
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: White motor cruiser Wavey Katey II cruising on a river
Description: A white cabin motor cruiser named Wavey Katey II cruises along a calm waterway, past a British maritime flag and a rustic wooden cottage.
Keywords: Boat, cabin cruiser, motorboat, river, outboard motor, canopy, flag, fender, railing, riverbank, nautical, leisure, foliage, trees, shrubs, cottage
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
- *Post-cleanup active memory (GB):* 0.010437794
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 575
- *Prompt composition:* 575 = 310 text/template (tokenizer-exact) + 265
  non-text (46%, image/audio expansion)
- *Checkpoint weights (GB):* 27.20
- *Parameter count:* 26.00B (name-estimate)
- *Quantization:* 8-bit, group 32, mxfp8
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (27.21 GB vs 27.20 GB on disk)
- *Generation tokens:* 89
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: White motor cruiser Wavey Katey II cruising on a river
Description: A white cabin motor cruiser named 'Wavey Katey II' travels along a calm waterway past a rustic wooden riverside cottage and lush green foliage.
Keywords: Boat, cabin cruiser, motorboat, river, riverbank, cruising, outboard motor, canopy, fender, flag, railing, foliage, trees, shrubs, leisure, nautical
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — usable_with_caveats — control tokens visible; title/keyword constraints failed</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Title word count:* 7
- *Requested title word range:* [5, 10]
- *Keyword count:* 21
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002737212
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6325
- *Prompt composition:* 6,325 = 294 text/template (tokenizer-exact) + 6,031
  non-text (95%, image/audio expansion)
- *Checkpoint weights (GB):* 61.86
- *Quantization:* 4-bit, group 16, nvfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (61.86 GB vs 61.86 GB on disk)
- *Generation tokens:* 126
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Wavey Katey II Motor Cruiser on River
Description: A white motor cruiser named "Wavey Katey II" cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees, with a blue canopy and visible fenders.
Keywords: Boat, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees, White boat, British flag, Blue canopy, Rustic house, Lush greenery<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, unexpected_catalog_preamble,
  thinking_trace_incomplete, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this task
  step by step. First, the title needs to be 5-10 words, concrete. The boat is
  named "Wavey Katey II" and it's a motor cruiser. So maybe "Wavey Katey II
  Motor Cruiser on Waterway".  Next, the description: need to combine context
  (capture date, maybe) with main subject, setting, action, lighting,
  distinctive details. The boat is cruising, past a wooden riverside house,
  flying a British flag, calm waterway, etc. Let's make it a 1-2 sentence
  factual description.  Then keywords: need 10-18 unique ones from the hints,
  covering context and visible details. Let's list them out: Boat, Cabin
  cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure,
  Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising,
  Riverbank, Shrubs. Wait, let's check the hints. The hints include Boat, Boat
  driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising,
  Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger,
  Railing, River, Riverbank, Shrubs, Trees. So pick unique ones, maybe 12-15.
  Now, let's structure each part:
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 11
- *Requested title word range:* [5, 10]
- *Keyword count:* 29
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["trees"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002114608
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6321
- *Prompt composition:* 6,321 = 290 text/template (tokenizer-exact) + 6,031
  non-text (95%, image/audio expansion)
- *Checkpoint weights (GB):* 11.77
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 8-bit, group 64, affine
- *Declared context length:* 65,536 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (11.78 GB vs 11.77 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The boat is named "Wavey Katey II" and it's a motor cruiser. So maybe "Wavey Katey II Motor Cruiser on Waterway". 

Next, the description: need to combine context (capture date, maybe) with main subject, setting, action, lighting, distinctive details. The boat is cruising, past a wooden riverside house, flying a British flag, calm waterway, etc. Let's make it a 1-2 sentence factual description. 

Then keywords: need 10-18 unique ones from the hints, covering context and visible details. Let's list them out: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs. Wait, let's check the hints. The hints include Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees. So pick unique ones, maybe 12-15. 

Now, let's structure each part:

Title: "Wavey Katey II Motor Cruiser on Waterway" (about 8 words, concrete)

Description: Combine context (capture date) with subject, setting, action. So "The motor cruiser 'Wavey Katey II' cruises along a calm waterway on 2026-08-28, passing a rustic wooden riverside house with lush greenery, flying a British maritime flag." Wait, need to make sure it's 1-2 sentences. Let's make it: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, with a British maritime flag visible." 

Keywords: Let's list them comma-separated. Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs. That's 17, which is within 10-18. 

Wait, let's check the image details. The boat has "4PLUS2" on it, "Wavey Katey II" on the side, Yamaha outboard motor, blue canopy, fenders, British flag. The setting is a river with a wooden cottage, trees, etc. 

So finalizing:

Title: "Wavey Katey II Motor Cruiser"

Description: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, flying a British maritime flag."

Keywords: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical

Wait, let's count keywords: 19, maybe trim one. Let's do Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical. That's 19, maybe remove "Nautical" to make 18. 

Wait, the hints include "Nautical" as a keyword, so maybe include it. Let's check the image: the boat is on a river, has maritime elements, so Nautical is appropriate. 

Now, let's make sure the description is factual, combining context (capture date) with subject, setting, action, lighting (calm waterway, maybe daylight), distinctive details (British flag, wooden house, lush trees). 

Final check:

Title: "Wavey Katey II Motor Cruiser" (5 words, concrete)

Description: "The motor cruiser 'Wavey Katey II' cruises along a calm waterway past a rustic wooden riverside house and lush green trees on 2026-08-28, flying a British maritime flag." (1 sentence, combines context, subject, setting, action, distinctive details)

Keywords: Boat, Cabin cruiser, Motorboat, River, Flag, Cottage, Trees, Foliage, Boating, Leisure, Outboard motor, Passenger, Railing, Canopy, Boat fender, Cruising, Riverbank, Shrubs, Nautical (18
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Usability           | Observations                                                         |
|-------------------------------------------------------|---------------------|----------------------------------------------------------------------|
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | unusable            | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16                       | unusable            | missing required fields                                              |
| mlx-community/MiniCPM-V-4.6-8bit                      | unusable            | missing required fields; extra text before Title                     |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/gemma-3-27b-it-qat-4bit                 | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/granite-4.0-3b-vision-4bit              | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/InternVL3-8B-bf16                       | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/North-Micro-Vision-Instruct-4bit        | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Phi-3.5-vision-instruct-bf16            | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/pixtral-12b-8bit                        | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit          | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3.8-27B-4bit                        | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | usable_with_caveats | title/keyword constraints failed; draft hints copied unchanged       |
| mlx-community/Step-3.7-Flash-oQ2e                     | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/X-Reasoner-7B-8bit                      | usable_with_caveats | title/keyword constraints failed                                     |
| Qwen/Qwen3-VL-2B-Instruct                             | usable_with_caveats | title/keyword constraints failed                                     |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                             | Runtime identity                                    | Performance                                                                                |
|---------------------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------|
| mlx-community/gemma-4-26b-a4b-it-4bit             | rev 0d77464eeb23; Gemma4Processor; stop completed   | 579 prompt / 101 generated; 129 tok/s; 16 GB peak; cleanup 0.012/0.0 GB active/cache       |
| mlx-community/gemma-4-31b-it-4bit                 | rev 696d436c4047; Gemma4Processor; stop completed   | 579 prompt / 107 generated; 26.1 tok/s; 20 GB peak; cleanup 0.0126/0.0 GB active/cache     |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit             | rev 12c5ae493041; Lfm2VlProcessor; stop completed   | 2619 prompt / 106 generated; 208 tok/s; 4.0 GB peak; cleanup 0.00436/0.0 GB active/cache   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | rev 7c992876448f; Mistral3Processor; stop completed | 2577 prompt / 157 generated; 67.7 tok/s; 12 GB peak; cleanup 0.00515/0.0 GB active/cache   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit   | rev a962dcb09eee; Mistral3Processor; stop completed | 2576 prompt / 138 generated; 192 tok/s; 6.4 GB peak; cleanup 0.00567/0.0 GB active/cache   |
| mlx-community/Ornith-1.0-35B-bf16                 | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed  | 16639 prompt / 112 generated; 66.0 tok/s; 74 GB peak; cleanup 0.00701/0.0 GB active/cache  |
| mlx-community/Qwen3-VL-2B-Thinking-bf16           | rev c325e5ea14c2; Qwen3VLProcessor; stop completed  | 16625 prompt / 903 generated; 90.3 tok/s; 8.4 GB peak; cleanup 0.00739/0.0 GB active/cache |
| mlx-community/Qwen3.5-35B-A3B-4bit                | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed  | 16639 prompt / 150 generated; 111 tok/s; 24 GB peak; cleanup 0.00821/0.0 GB active/cache   |
| mlx-community/Qwen3.5-9B-MLX-4bit                 | rev 938d8919941c; Qwen3VLProcessor; stop completed  | 16639 prompt / 115 generated; 91.3 tok/s; 10 GB peak; cleanup 0.00873/0.0 GB active/cache  |
| mlx-community/Qwen3.6-27B-mxfp8                   | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed  | 16639 prompt / 106 generated; 17.5 tok/s; 33 GB peak; cleanup 0.00924/0.0 GB active/cache  |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 5,616 pixels
- *Image size:* 33,850,802 bytes
- *Image SHA-256:* b318746396941f675647ccf9ebdf8652161618926a84538fac85170096a7f92c

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-28 16:17:42 UTC+01:00

Descriptive hints:
- Description hint: A white motor cruiser boat named 'Wavey Katey II' cruises along a calm waterway, flying a British maritime flag, past a rustic wooden riverside house and lush green trees.
- Keyword hints: Boat, Boat driver, Boat fender, Boating, Cabin cruiser, Canopy, Cottage, Cruising, Flag, Foliage, Leisure, Motorboat, Nautical, Outboard motor, Passenger, Railing, River, Riverbank, Shrubs, Trees

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

| Model                                         | Resolved revision                        |
|-----------------------------------------------|------------------------------------------|
| tencent/Youtu-VL-4B-Instruct                  | 8d30a0e49662a1d628a472b12df264dbcd768753 |
| mlx-community/gemma-3n-E4B-it-bf16            | d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a |
| jinaai/jina-vlm-mlx                           | a987631a01dc554a787d87a45fb01fb48f8aaca4 |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16              | ed71acdae0799c98cb450787d049d1343b65cd52 |
| mlx-community/GLM-4.6V-Flash-mxfp4            | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b |
| mlx-community/Idefics3-8B-Llama3-bf16         | 8c2a30c48864f3251701b7bde40f601d25535098 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16  | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/Molmo2-8B-4bit                  | 4fcbe926577641af12aa5086cf5c292db9dd9730 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit  | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/GLM-4.6V-nvfp4                  | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/GLM-4.1V-9B-Thinking-8bit       | 9677807f106500eb7690391c27645d59f6855cfb |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.0rc0                                                                                                                                        |
| mlx-vlm source revision    | f5d9533a912e3769b21fa646252469149530fc55                                                                                                        |
| mlx                        | 0.32.3.dev20260829+052e77db9                                                                                                                    |
| mlx source revision        | 052e77db9ddd5b4389f701a1bae046e9f73e8c24                                                                                                        |
| mlx-lm                     | 0.32.0                                                                                                                                          |
| mlx-lm source revision     | 77c33b14373ac70d7abd6f82af15962852adadbb                                                                                                        |
| mlx-audio                  | 0.5.0                                                                                                                                           |
| transformers               | 5.16.1                                                                                                                                          |
| tokenizers                 | 0.23.1                                                                                                                                          |
| huggingface-hub            | 1.29.0                                                                                                                                          |
| Python Version             | 3.14.7                                                                                                                                          |
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
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-314-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (182,433,408 bytes, sha256=612fa3487b6372fdf8e64e4f30f98a8403ca3f1e5ba118be7a94bfbd9fc3335c) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,941,920 bytes, sha256=fa7a63ded63b7daac385c8976f6c1afb39ab60338277fafb9c6c86feb24e9814)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
