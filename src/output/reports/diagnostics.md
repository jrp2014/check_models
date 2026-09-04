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
| Attempted           | 42    |
| Conclusive outcomes | 42    |
| Completed           | 41    |
| Crashed             | 1     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 1     |
| none                           | 36    |
| observation needs reproduction | 5     |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 8     |
| usable              | 15    |
| usable with caveats | 18    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 2     |
| Unrecognised model control tokens remain visible                                      | 1     |
| Required fields are missing or empty                                                  | 5     |
| Response repeats the task instructions instead of only returning the requested fields | 2     |
| Extra text appears before the Title field                                             | 4     |
| Response appears cut off at the token limit                                           | 4     |
| Internal reasoning block appears incomplete                                           | 1     |
| Conversation-role control tokens remain visible                                       | 1     |
| Title or keywords do not meet requested constraints                                   | 19    |

## Triage

| Model                                                                                                    | Execution | Usability           | Maintainer status              | Observations                                                                                                                      |
|----------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [tencent/Youtu-VL-4B-Instruct](#diagnostic-tencent-youtu-vl-4b-instruct)                                 | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                              |
| [mlx-community/GLM-4.6V-Flash-mxfp4](#diagnostic-mlx-community-glm-46v-flash-mxfp4)                      | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [Qwen/Qwen3-VL-2B-Instruct](#diagnostic-qwen-qwen3-vl-2b-instruct)                                       | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                  | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                            |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)            | completed | unusable            | observation_needs_reproduction | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16) | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                                                               |

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
- *Post-cleanup active memory (GB):* 0.013436274
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13601, in _run_model_generation
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 817, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1298, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1443, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 10 more times]
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/molmo2/processing.py", line 768, in _patched_auto_processor_from_pretrained_molmo2
    return _original_auto_processor_from_pretrained_molmo2.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 652, in _patched_auto_processor_from_pretrained
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14614, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13616, in _run_model_generation
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
Fetching 19 files: 100%|##########| 19/19 [00:00<00:00, 3199.06it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[15:03:13] DEBUG    HF Cache Info for tencent/Youtu-VL-4B-Instruct: size=10229.6 MB, files=27
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-glm-46v-flash-mxfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-mxfp4 — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/GLM-4.6V-Flash-mxfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Repeated fragment:* keyword: "hampshire"
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 230
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["hampshire", "england", "rapids", "arched doorway",
  "tourist attraction", "historic watermill", "diamond paned windows",
  "weathered roof tiles", "red brick chimney", "stone archway over river",
  "mossy stone wall", "green ivy", "traditional english mill architecture",
  "historic watermill on river itchen", "winchester", "national trust
  property", "scenic riverfront", "historic mill with water flow",
  "traditional english architecture", "stone and brick building"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002425904
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6430
- *Prompt composition:* 6,430 = 287 text/template (tokenizer-exact) + 6,143
  non-text (96%, image/audio expansion)
- *Checkpoint weights (GB):* 6.78
- *Quantization:* 4-bit, group 32, mxfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (6.78 GB vs 6.78 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: Winchester City Mill
Description: The historic Winchester City Mill, an ancient watermill over the River Itchen, is shown with visitors walking along the pathway and an elderly woman sitting nearby.
Keywords: Winchester City Mill, River Itchen, watermill, historic building, pedestrians, elderly woman, stone wall, clay tiles, blue sky, chimney, Hampshire, England, National Trust, riverbank, scenery, rapids, sitting, walking, pathway, water, brick building, arched doorway, window with diamond panes, outdoor lamp, greenery, moss, stone bridge, flowing water, reflection, cloudy sky, historic architecture, traditional building, riverfront, tourist attraction, English countryside, historic mill, water-powered mill, river crossing, stone archway, brick chimney, tiled roof, weathered stone, mossy stone, water flow, rapids, riverbank scenery, historic watermill, traditional English mill, riverfront historic building, stone and brick architecture, water-powered historic mill, river Itchen watermill, Winchester historic mill, National Trust managed historic site, traditional English watermill, historic riverfront architecture, stone bridge over river, water flow over weir, historic mill building with arched doorway, diamond-paned windows, brick and stone construction, weathered roof tiles, red brick chimney, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill on River Itchen, Winchester, Hampshire, England, National Trust property, tourist attraction, scenic riverfront, historic mill with water flow, traditional English architecture, stone and brick building, arched doorway, diamond-paned windows, red brick chimney, weathered roof tiles, stone archway over river, mossy stone wall, green ivy, traditional English mill architecture, historic watermill
```

</details>

<a id="diagnostic-qwen-qwen3-vl-2b-instruct"></a>

<details>
<summary>Qwen/Qwen3-VL-2B-Instruct — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### Qwen/Qwen3-VL-2B-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Repeated fragment:* keyword: "national trust"
- *Title word count:* 4
- *Requested title word range:* [5, 10]
- *Keyword count:* 327
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["national trust", "pedestrians", "riverbank", "stone
  wall", "brick building", "blue sky", "chimney", "clay tiles", "elderly
  woman", "girls", "rapids", "scenery", "sitting", "people", "river"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.000443432
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16520
- *Prompt composition:* 16,520 = 297 text/template (tokenizer-exact) + 16,223
  non-text (98%, image/audio expansion)
- *Checkpoint weights (GB):* 4.26
- *Parameter count:* 2.00B (name-estimate)
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (4.26 GB vs 4.26 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: River Itchen, Hampshire, England

Description: A historic watermill in Hampshire, England, with a stone wall and brick building, is situated over the River Itchen. Pedestrians walk along a bridge, and an elderly woman sits on a bench in front of the mill. The scene is set under a blue sky with white clouds, and the river flows with rapids near the bridge.

Keywords: River Itchen, Hampshire, England, Mill, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Rapids, Scenery, Sitting, People, River, National Trust, Pedestrians, Riverbank, Stone wall, Brick building, Blue sky, Chimney, Clay tiles
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — usable_with_caveats — control tokens visible</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002737212
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6430
- *Prompt composition:* 6,430 = 287 text/template (tokenizer-exact) + 6,143
  non-text (96%, image/audio expansion)
- *Checkpoint weights (GB):* 61.86
- *Quantization:* 4-bit, group 16, nvfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (61.86 GB vs 61.86 GB on disk)
- *Generation tokens:* 108
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Historic Winchester City Mill by River Itchen

Description: Two girls walk along the pathway in front of the historic Winchester City Mill, a brick and stone watermill with clay tiles, situated over the River Itchen in Hampshire, England, under a blue sky with white clouds.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, watermill, brick, stone, clay tiles, chimney, blue sky, girls, pedestrians, elderly woman, National Trust, riverbank, scenery, sitting, rapids<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, prompt_instruction_echo,
  unexpected_catalog_preamble, thinking_trace_incomplete,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Echoed instruction fragments:* ["retain details supported by the image
  correct conflicts"]
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this task
  step by step. First, the title needs to be 5-10 words, concrete. The image
  shows a historic mill by a river with people, so maybe "Winchester City Mill
  by River Itchen".  Next, the description: need to combine context (capture
  date, location) with main subject (mill), setting (river, historic), action
  (visitors walk, relax), lighting (blue sky), distinctive details (brick,
  stone, water). So something like "Historic Winchester City Mill by River
  Itchen with visitors on a pathway, under a blue sky, showcasing brick and
  stone architecture." Wait, need to make sure it's 1-2 sentences. Let's check
  the context: capture date is 2026-08-31, location is Winchester, Hampshire,
  England, mill, people walking, sitting, river. So description: "Visitors
  walk and relax near the historic Winchester City Mill, a brick and stone
  structure over the River Itchen, under a bright blue sky."  Then keywords:
  need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but
  maybe not visible, but context says National Trust? Wait, the hints include
  National Trust, so maybe include. Also, the elements: arch, blue sky,
  chimney, clay tiles, elderly woman, girls, Hampshire, mill, National Trust,
  pedestrians, people, river Itchen, riverbank, scenery, sitting, stone wall.
  Let's list them: "Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls,
  Hampshire, Mill, National Trust, Pedestrians, People, River Itchen,
  Riverbank, Scenery, Sitting, Stone wall" – that's 16, which is within 10-18.
  Wait, let's check each part. Title: "Winchester City Mill by River Itchen"
  (7 words, concrete). Description: "Visitors walk and relax near the historic
  Winchester City Mill, a brick and stone structure over the River Itchen,
  under a bright blue sky, with distinctive architectural features like a
  chimney and stone arches." Wait, need to make sure it's 1-2 sentences. Let's
  make it concise. "Visitors walk and relax near the historic Winchester City
  Mill, a brick and stone structure over the River Itchen, under a bright blue
  sky." That's one sentence. Then keywords: combine all relevant hints. Let's
  list all hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay
  tiles, Elderly woman, Girls, England, Hampshire, Mill, National Trust,
  Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting,
  Stone wall. Remove duplicates: England is already in Hampshire, so keep
  Hampshire. Remove Adobe Stock/Any Vision if not visible, but context says
  "National Trust" is a hint, so include. So keywords: Arch, Blue sky,
  Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust,
  Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall.
  That's 16, which is good.  Now, let's finalize each section:
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 9
- *Requested title word range:* [5, 10]
- *Keyword count:* 42
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["blue sky", "chimney", "clay tiles", "elderly woman",
  "girls", "hampshire", "mill", "national trust", "pedestrians", "people",
  "river itchen", "riverbank", "scenery", "sitting"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002114608
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6426
- *Prompt composition:* 6,426 = 283 text/template (tokenizer-exact) + 6,143
  non-text (96%, image/audio expansion)
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
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image shows a historic mill by a river with people, so maybe "Winchester City Mill by River Itchen". 

Next, the description: need to combine context (capture date, location) with main subject (mill), setting (river, historic), action (visitors walk, relax), lighting (blue sky), distinctive details (brick, stone, water). So something like "Historic Winchester City Mill by River Itchen with visitors on a pathway, under a blue sky, showcasing brick and stone architecture." Wait, need to make sure it's 1-2 sentences. Let's check the context: capture date is 2026-08-31, location is Winchester, Hampshire, England, mill, people walking, sitting, river. So description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." 

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not visible, but context says National Trust? Wait, the hints include National Trust, so maybe include. Also, the elements: arch, blue sky, chimney, clay tiles, elderly woman, girls, Hampshire, mill, National Trust, pedestrians, people, river Itchen, riverbank, scenery, sitting, stone wall. Let's list them: "Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall" – that's 16, which is within 10-18. 

Wait, let's check each part. Title: "Winchester City Mill by River Itchen" (7 words, concrete). Description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky, with distinctive architectural features like a chimney and stone arches." Wait, need to make sure it's 1-2 sentences. Let's make it concise. "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." That's one sentence. Then keywords: combine all relevant hints. Let's list all hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, England, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall. Remove duplicates: England is already in Hampshire, so keep Hampshire. Remove Adobe Stock/Any Vision if not visible, but context says "National Trust" is a hint, so include. So keywords: Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall. That's 16, which is good. 

Now, let's finalize each section:

Title: "Winchester City Mill by River Itchen" (5 words, concrete)

Description: "Visitors walk and relax near the historic Winchester City Mill, a brick and stone structure over the River Itchen, under a bright blue sky." (1 sentence, combines context and visible details)

Keywords: Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Scenery, Sitting, Stone wall (comma-separated, unique, 16 words)

Wait, check if "Rapids" is visible? The water has some movement, maybe rapids? The description hint says "Rapids" but the image might have water flow. But the description hint says "Rapids" as a hint, but maybe the image shows water flow. But the user says "retain details supported by the image, correct conflicts, and add important visible details." The image has water flowing under the mill, so maybe "Rapids" is a detail. But let's check the hints again. The hints include "Rapids" as a keyword. So maybe include "Rapids" if visible. The water under the mill has some movement, so maybe "Rapids" is a detail. So add "Rapids" to keywords. Then keywords would be Arch, Blue sky, Chimney, Clay tiles, Elderly woman, Girls, Hampshire, Mill, National Trust, Pedestrians, People, River Itchen, Riverbank, Rapids, Scenery, Sitting, Stone wall. That's 
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — usable_with_caveats — role tokens visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003966048
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1307
- *Prompt composition:* 1,307 = 294 text/template (tokenizer-exact) + 1,013
  non-text (78%, image/audio expansion)
- *Checkpoint weights (GB):* 32.82
- *Parameter count:* 3.00B (name-estimate)
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (32.82 GB vs 32.82 GB on disk)
- *Generation tokens:* 710
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date/time is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City Mill, River Itchen, Hampshire, England, people walking, stone wall, chimney, etc. The image shows a historic brick building with a tiled roof, a bridge over a river, people walking, and the river flowing. 

For the title, something concise like "Winchester City Mill and River Itchen Scene" works. The description should combine the context and visible elements: "Historic Winchester City Mill by River Itchen with people walking on a stone bridge, brick building, and flowing water." Keywords need to be 10-18 unique terms. Let's list what's visible: Winchester City Mill, River Itchen, England, Hampshire, stone bridge, brick building, tiled roof, chimney, pedestrians, elderly woman, girls, blue sky, Adobe Stock, National Trust, Rapids, Scenery. Wait, the hints mention "Elderly woman" and "Girls" but in the image, there are two women, one might be elderly. Also, "Adobe Stock" and "National Trust" are hints but maybe not visible. Check the image: the building is a National Trust property, so include that. "Adobe Stock" might be a watermark, but the user said not to claim visible details. So maybe omit that. So keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust. That's 10 keywords. Let me check the hints again. The hints include "Clay tiles" but the roof is tiled, so "Tiled roof" is better. "Rapids" might not be visible, the water is flowing but not necessarily rapids. Maybe remove that. "Elderly woman" and "Girls" – the image has two people, one might be elderly. So include "Elderly woman" and "Girls". So keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls. That's 12. Maybe too many? Wait, the user said 10-18, so 12 is okay. Let me adjust. Maybe "Riverbank" instead of "Scenery". Wait, the hints have "Scenery" as a keyword. So include that. Let me finalize: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls. That's 12. Okay.◁/think▷Title: Winchester City Mill and River Itchen Scene  
Description: Historic Winchester City Mill by River Itchen with people walking on a stone bridge, featuring a brick building, tiled roof, and flowing water under a blue sky.  
Keywords: Winchester City Mill, River Itchen, England, Hampshire, Stone bridge, Brick building, Tiled roof, Chimney, Pedestrians, Riverbank, Scenery, National Trust, Elderly woman, Girls<|im_assistant|>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Usability           | Observations                                                                                  |
|-------------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------|
| jinaai/jina-vlm-mlx                                   | unusable            | missing required fields                                                                       |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | unusable            | missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16                       | unusable            | missing required fields                                                                       |
| mlx-community/gemma-3n-E4B-it-bf16                    | unusable            | missing required fields; extra text before Title                                              |
| mlx-community/MiniCPM-V-4.6-8bit                      | unusable            | missing required fields; extra text before Title                                              |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/gemma-3-27b-it-qat-4bit                 | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Molmo2-8B-4bit                          | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/North-Micro-Vision-Instruct-4bit        | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit           | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Phi-3.5-vision-instruct-bf16            | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/pixtral-12b-8bit                        | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit          | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Step-3.7-Flash-oQ2e                     | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/X-Reasoner-7B-8bit                      | usable_with_caveats | title/keyword constraints failed                                                              |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                             | Runtime identity                                           | Performance                                                                                |
|---------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| mlx-community/diffusiongemma-26B-A4B-it-8bit      | rev 7b95e3887078; DiffusionGemma4Processor; stop completed | 570 prompt / 84 generated; 81.9 tok/s; 29 GB peak; cleanup 0.0108/0.0 GB active/cache      |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8     | rev ded389e478f8; DiffusionGemma4Processor; stop completed | 570 prompt / 85 generated; 56.2 tok/s; 28 GB peak; cleanup 0.0108/0.0 GB active/cache      |
| mlx-community/gemma-4-26b-a4b-it-4bit             | rev 0d77464eeb23; Gemma4Processor; stop completed          | 574 prompt / 105 generated; 129 tok/s; 16 GB peak; cleanup 0.0124/0.0 GB active/cache      |
| mlx-community/gemma-4-31b-it-4bit                 | rev 696d436c4047; Gemma4Processor; stop completed          | 574 prompt / 90 generated; 25.1 tok/s; 20 GB peak; cleanup 0.013/0.0 GB active/cache       |
| mlx-community/granite-4.0-3b-vision-4bit          | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed  | 1371 prompt / 88 generated; 176 tok/s; 4.7 GB peak; cleanup 0.0132/0.0 GB active/cache     |
| mlx-community/Idefics3-8B-Llama3-bf16             | rev 8c2a30c48864; Idefics3Processor; stop completed        | 2597 prompt / 162 generated; 32.3 tok/s; 18 GB peak; cleanup 0.003/0.0 GB active/cache     |
| mlx-community/InternVL3-8B-bf16                   | rev e0df3dd79263; InternVLChatProcessor; stop completed    | 2086 prompt / 88 generated; 34.3 tok/s; 17 GB peak; cleanup 0.00331/0.0 GB active/cache    |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit             | rev 12c5ae493041; Lfm2VlProcessor; stop completed          | 2086 prompt / 83 generated; 210 tok/s; 4.0 GB peak; cleanup 0.00436/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | rev 7c992876448f; Mistral3Processor; stop completed        | 2918 prompt / 180 generated; 67.1 tok/s; 13 GB peak; cleanup 0.00513/0.0 GB active/cache   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit   | rev a962dcb09eee; Mistral3Processor; stop completed        | 2917 prompt / 139 generated; 190 tok/s; 7.8 GB peak; cleanup 0.00565/0.0 GB active/cache   |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit         | rev b4a74fa6001f; MuseGlimmerProcessor; stop completed     | 4388 prompt / 992 generated; 25.4 tok/s; 25 GB peak; cleanup 0.00637/0.0 GB active/cache   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16           | rev c325e5ea14c2; Qwen3VLProcessor; stop completed         | 16522 prompt / 907 generated; 91.5 tok/s; 8.4 GB peak; cleanup 0.00778/0.0 GB active/cache |
| mlx-community/Qwen3.5-35B-A3B-4bit                | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed         | 16536 prompt / 143 generated; 110 tok/s; 24 GB peak; cleanup 0.0086/0.0 GB active/cache    |
| mlx-community/Qwen3.6-27B-mxfp8                   | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed         | 16536 prompt / 139 generated; 17.8 tok/s; 33 GB peak; cleanup 0.00964/0.0 GB active/cache  |
| mlx-community/Qwen3.8-27B-4bit                    | rev 3e6447f082e8; Qwen3VLProcessor; stop completed         | 16536 prompt / 131 generated; 30.5 tok/s; 21 GB peak; cleanup 0.0101/0.0 GB active/cache   |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 9,984 pixels
- *Image size:* 66,295,254 bytes
- *Image SHA-256:* 168b4850b1427394bbe84a99ffd05533ffaf7e995e4213d9ce2d36c959e70c7b

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-31 15:02:20 UTC+01:00

Descriptive hints:
- Description hint: Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England.
- Keyword hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall

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

| Model                                        | Resolved revision                        |
|----------------------------------------------|------------------------------------------|
| tencent/Youtu-VL-4B-Instruct                 | 8d30a0e49662a1d628a472b12df264dbcd768753 |
| mlx-community/GLM-4.6V-Flash-mxfp4           | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b |
| Qwen/Qwen3-VL-2B-Instruct                    | 89644892e4d85e24eaac8bacfd4f463576704203 |
| mlx-community/GLM-4.6V-nvfp4                 | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/GLM-4.1V-9B-Thinking-8bit      | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 | fb254434d4026bee7aa840dea1c5d59feea8fd48 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.0rc0                                                                                                                                        |
| mlx-vlm source revision    | 5c9b5f52adfeab35b5ece0bb2d6e4d44541d9e32                                                                                                        |
| mlx                        | 0.32.3.dev20260904+b6368984b                                                                                                                    |
| mlx source revision        | b6368984b8e02a3fb3ee7986846c0fb85e1fccf7                                                                                                        |
| mlx-lm                     | 0.32.0                                                                                                                                          |
| mlx-lm source revision     | 4343e96fa72e0ed458d02d08e2d220a1b22061cb                                                                                                        |
| mlx-audio                  | 0.5.1                                                                                                                                           |
| transformers               | 5.16.1                                                                                                                                          |
| tokenizers                 | 0.23.2                                                                                                                                          |
| huggingface-hub            | 1.30.0                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (182,474,576 bytes, sha256=42ffcb402b5e7c7023c96051d078f0101c582b70aed3cf61cc70e394bd6e036c) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,958,976 bytes, sha256=c134b2fe7fa611db39c82cf3a3358114c5e390334ea9e4ddc7c973f414f57c98)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
