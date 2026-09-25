# Crash: mlx-community/InternVL3_5-1B-4bit

## Maintainer evidence

### mlx-community/InternVL3_5-1B-4bit

#### Root exception and chain

```text
builtins.ValueError: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
builtins.ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
```

#### Execution and provenance

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* no (model_type internvl)
- *Phase:* model_load
- *Stage:* Unsupported Arch
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model loading failed: Model type internvl not supported.
  Error: No module named 'mlx_vlm.speculative.drafters.internvl'
- *Root error type:* ValueError
- *Root error message:* Model type internvl not supported. Error: No module
  named 'mlx_vlm.speculative.drafters.internvl'
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.002443844
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 1.08
- *Parameter count:* 1.00B (name-estimate)
- *Quantization:* 4-bit, group 32, affine
- *Declared context length:* 40,960 (text_config.max_position_embeddings)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14490, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13446, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 842, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1307, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 965, in load_model
    model_class, _ = get_model_and_args(config=config, model_path=model_path)
                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 785, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15608, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14505, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
[23:54:09] INFO     Loading model weights and processor...
Fetching 14 files:   0%|          | 0/14 [00:00<?, ?it/s]
Fetching 14 files: 100%|██████████| 14/14 [00:00<00:00, 3749.22it/s]
ERROR:root:Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
[23:54:09] DEBUG    HF Cache Info for mlx-community/InternVL3_5-1B-4bit: size=1046.5 MB, files=16
```

## Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 49,407,373 bytes
- *Image SHA-256:* 97f53d5eeb6a63e6e685321bc95e66807a87db2e01f63f48a48a99f9342c7d12

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-19 17:12:46 UTC+01:00

Descriptive hints:
- Description hint: Two sailors steer small dinghies—a Vortex catamaran (sail number 1067) on the left and a Laser dinghy (sail number GBR 188572) on the right—across calm coastal or river waters against a backdrop of dense green woodland.
- Keyword hints: Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water

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
python -m mlx_vlm.generate --model mlx-community/InternVL3_5-1B-4bit --image any-local-image.jpg --prompt x --max-tokens 8 --temperature 0.0 --revision f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 --trust-remote-code
```

## Provenance and Environment

### Components

| Component       | Value                                                             |
|-----------------|-------------------------------------------------------------------|
| mlx-vlm         | 0.7.3                                                             |
| mlx             | 0.32.3.dev20260925+073d2252c                                      |
| transformers    | 5.17.0                                                            |
| tokenizers      | 0.23.2                                                            |
| huggingface-hub | 1.33.0                                                            |
| Pillow          | 12.3.0                                                            |
| Python Version  | 3.14.7                                                            |
| macOS Version   | 27.0                                                              |
| GPU/Chip        | Apple M5 Max                                                      |
| check_models    | 0.17.38; revision 29bae6c868c9e1e58221eb34da8336fa58581ed1; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
