# Crash: mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit

## Before filing: rule out the checkpoint

- The crash happened while loading. A load failure can come from the
  checkpoint (weight names or config that do not match mlx-vlm's model
  definition) as well as from mlx-vlm: try another conversion of the same
  model with a native `load()` before filing.
- Reproduce with the native command below, outside check_models, and say in
  the issue which of these checks you ran.

## Maintainer evidence

### mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit

#### Root exception and chain

```text
builtins.TypeError: Field 'attn_temperature_tuning' expected bool, got int (value: 4)
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'attn_temperature_tuning':
    TypeError: Field 'attn_temperature_tuning' expected bool, got int (value: 4)
builtins.ValueError: Model loading failed: Validation error for field 'attn_temperature_tuning':
    TypeError: Field 'attn_temperature_tuning' expected bool, got int (value: 4)
```

#### Execution and provenance

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type llama4)
- *Family README in installed mlx-vlm:* none
- *Phase:* model_load
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model loading failed: Validation error for field
  'attn_temperature_tuning':     TypeError: Field 'attn_temperature_tuning'
  expected bool, got int (value: 4)
- *Root error type:* TypeError
- *Root error message:* Field 'attn_temperature_tuning' expected bool, got int
  (value: 4)
- *Resolved model revision:* f89d3ebf0bb8f9b512d6c732aff83937deb55c3e
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.004
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 61.12
- *Parameter count:* 17.00B (name-estimate)
- *Quantization:* 4-bit, group 64
- *Declared context length:* 10,485,760 (text_config.max_position_embeddings)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/huggingface_hub/dataclasses.py", line 144, in __strict_setattr__
    validator(value)
    ~~~~~~~~~^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/huggingface_hub/dataclasses.py", line 625, in validator
    type_validator(field.name, value, field.type)
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/huggingface_hub/dataclasses.py", line 472, in type_validator
    _validate_simple_type(name, value, expected_type)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/huggingface_hub/dataclasses.py", line 615, in _validate_simple_type
    raise TypeError(
        f"Field '{name}' expected {expected_type.__name__}, got {type(value).__name__} (value: {repr(value)})"
    )
TypeError: Field 'attn_temperature_tuning' expected bool, got int (value: 4)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15156, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14012, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 864, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1323, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1468, in load_processor
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 657, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 657, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 657, in _patched_auto_processor_from_pretrained
    return previous_from_pretrained.__func__(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        cls, pretrained_model_name_or_path, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  [Previous line repeated 10 more times]
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/auto/processing_auto.py", line 346, in from_pretrained
    return processor_class.from_pretrained(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/processing_utils.py", line 1754, in from_pretrained
    args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/processing_utils.py", line 1879, in _get_arguments_from_pretrained
    tokenizer = cls._load_tokenizer_from_pretrained(
        sub_processor_type, pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/processing_utils.py", line 1815, in _load_tokenizer_from_pretrained
    tokenizer = auto_processor_class.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/auto/tokenization_auto.py", line 807, in from_pretrained
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
    )
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/auto/configuration_auto.py", line 440, in from_pretrained
    return config_class.from_dict(config_dict, **unused_kwargs)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/configuration_utils.py", line 946, in from_dict
    config = cls(**config_dict)
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/huggingface_hub/dataclasses.py", line 275, in init_with_validate
    initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/configuration_utils.py", line 172, in __init__
    self.__post_init__(**additional_kwargs)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/models/llama4/configuration_llama4.py", line 258, in __post_init__
    self.text_config = Llama4TextConfig(**self.text_config)
                       ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/huggingface_hub/dataclasses.py", line 275, in init_with_validate
    initial_init(self, *args, **kwargs)  # type: ignore [call-arg]
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/transformers/configuration_utils.py", line 157, in __init__
    setattr(self, f.name, standard_kwargs[f.name])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages/huggingface_hub/dataclasses.py", line 146, in __strict_setattr__
    raise StrictDataclassFieldValidationError(field=name, cause=e) from e
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'attn_temperature_tuning':
    TypeError: Field 'attn_temperature_tuning' expected bool, got int (value: 4)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 16460, in process_image_with_model
    generation_run = _run_model_generation(
        params=params,
        phase_callback=_update_phase,
        phase_timer=phase_timer,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15171, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Validation error for field 'attn_temperature_tuning':
    TypeError: Field 'attn_temperature_tuning' expected bool, got int (value: 4)

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
[00:24:21] INFO     Loading model weights and processor...
Fetching 21 files:   0%|          | 0/21 [00:00<?, ?it/s]
Fetching 21 files: 100%|██████████| 21/21 [00:00<00:00, 5545.23it/s]
[00:24:27] DEBUG    HF Cache Info for mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit: size=58311.1 MB, files=23
```

## Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,805 x 6,538 pixels
- *Image size:* 54,173,041 bytes
- *Image SHA-256:* bd1c0e60ad08718e500069551a1ac0d3697010373ec1509d4f76cd6e1d79b11a

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-26 17:36:29 UTC+01:00
- GPS: 50.682100°N, 3.466600°W

Descriptive hints:
- Description hint: A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light.
- Keyword hints: Boat, Boat canopy, Boat fender, Boating, Cabin cruiser, Calm Water, Dock, Harbor, Marina, Mast, Mooring, Motorboat, Nautical, Reflection, Rope, Sailboat, Sailing, Water reflection, Watercraft, Yacht

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
python -m mlx_vlm.generate --verbose --model mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision f89d3ebf0bb8f9b512d6c732aff83937deb55c3e --trust-remote-code
```

## Provenance and Environment

### Components

| Component       | Value                                                             |
|-----------------|-------------------------------------------------------------------|
| mlx-vlm         | 0.7.4                                                             |
| mlx             | 0.32.4.dev20261003+0e3ff3643                                      |
| transformers    | 5.18.0                                                            |
| tokenizers      | 0.23.2                                                            |
| huggingface-hub | 1.33.0                                                            |
| Pillow          | 12.3.0                                                            |
| Python Version  | 3.14.7                                                            |
| macOS Version   | 27.0.1                                                            |
| GPU/Chip        | Apple M5 Max                                                      |
| check_models    | 0.17.39; revision 7859973d56b25a11aee138533438ff8f4d2ffad1; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
