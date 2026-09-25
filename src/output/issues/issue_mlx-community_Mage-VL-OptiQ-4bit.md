# Crash: mlx-community/Mage-VL-OptiQ-4bit

## Maintainer evidence

### mlx-community/Mage-VL-OptiQ-4bit

#### Root exception and chain

```text
builtins.ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]
builtins.ValueError: Model generation failed for mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]
```

#### Execution and provenance

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mage_vl)
- *Phase:* generation_before_first_token
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model generation failed for
  mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600
  calculated=3800 grid=[(1, 50, 76)]
- *Root error type:* ValueError
- *Root error message:* cu_seqlens mismatch: total_patches=7600
  calculated=3800 grid=[(1, 50, 76)]
- *Resolved model revision:* c98dad5f92f13334cc679c93fb9f185ab49ed626
- *Processor class:* mlx_vlm.models.mage_vl.processing_mage_vl.MageVLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* exception
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.0038693
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 3.92
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Snapshot notes (neutral):* processor config missing from snapshot
  (preprocessor_config.json, processor_config.json)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13761, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14392, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<6 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14296, in _generate_with_repetition_guard
    for chunk in stream_generate(
                 ~~~~~~~~~~~~~~~^
        model=model, processor=processor, prompt=prompt, image=image, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1083, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 424, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/mage_vl.py", line 66, in get_input_embeddings
    hidden_states = self.vision_tower(
        pixel_values.astype(inputs_embeds.dtype), patch_positions, grid_thw
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 381, in __call__
    cu = build_cu_seqlens(grid_thw, total_patches, self.config.frame_windows_size)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 309, in build_cu_seqlens
    raise ValueError(
        f"cu_seqlens mismatch: total_patches={total_patches} calculated={cu[-1]} grid={grid_thw}"
    )
ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15569, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14497, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14413, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13770, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 2093.91it/s]
Download complete:           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete:           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:09:39] DEBUG    Legacy snapshot note for mlx-community/Mage-VL-OptiQ-4bit: processor config missing from snapshot
                    (preprocessor_config.json, processor_config.json).
[22:09:40] ERROR    Generation error for mlx-community/Mage-VL-OptiQ-4bit
                    ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]
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

The original local input is not published, so this report does not claim a
complete reproduction command. Use a shareable equivalent image or add the
original image before filing.

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-b32a199300908fe5.jpg>
- *Preview dimensions:* 1,024 x 683 pixels
- *Preview size:* 142,804 bytes
- *Preview SHA-256:* b32a199300908fe503b50bbf79a097af655b5c3c31d1b7ada8ea97c88cc62c91

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-b32a199300908fe5.jpg
printf '%s\n' 'b32a199300908fe503b50bbf79a097af655b5c3c31d1b7ada8ea97c88cc62c91  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --model mlx-community/Mage-VL-OptiQ-4bit --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

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
Keywords:' --max-tokens 1000 --temperature 0.0 --revision c98dad5f92f13334cc679c93fb9f185ab49ed626 --trust-remote-code --seed 0 --prefill-step-size 2048
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
| check_models    | 0.17.37; revision a44e522108f41f3d5688057f27b50e93c14f468d; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
