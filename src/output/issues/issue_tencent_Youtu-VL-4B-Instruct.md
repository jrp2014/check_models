# Crash: tencent/Youtu-VL-4B-Instruct

## Maintainer evidence

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

## Reproduction inputs

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

The crash occurred during model load, before image decoding, so the exact
input image is not required: substitute any local image for the placeholder
path and run one native mlx-vlm process.

```bash
python -m mlx_vlm.generate --model tencent/Youtu-VL-4B-Instruct --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision 8d30a0e49662a1d628a472b12df264dbcd768753 --trust-remote-code
```

## Provenance and Environment

### Components

| Component       | Value                                                            |
|-----------------|------------------------------------------------------------------|
| mlx-vlm         | 0.7.0rc0                                                         |
| mlx             | 0.32.3.dev20260829+052e77db9                                     |
| mlx-lm          | 0.32.0                                                           |
| transformers    | 5.16.1                                                           |
| tokenizers      | 0.23.1                                                           |
| huggingface-hub | 1.29.0                                                           |
| Pillow          | 12.3.0                                                           |
| Python Version  | 3.14.7                                                           |
| macOS Version   | 26.6.2                                                           |
| GPU/Chip        | Apple M5 Max                                                     |
| check_models    | 0.16.5; revision 7299db1db9c3863984bb90b7ee4129779c30877b; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
