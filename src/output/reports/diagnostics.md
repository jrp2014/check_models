# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run Summary

- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 44.7 MB

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 49    |
| Conclusive outcomes | 49    |
| Completed           | 46    |
| Crashed             | 3     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 3     |
| none                           | 41    |
| observation needs reproduction | 5     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| not assessed         | 3     |
| major concerns       | 11    |
| no concerns detected | 27    |
| concerns detected    | 8     |

Observation counts

| Observation                                                               | Count |
|---------------------------------------------------------------------------|-------|
| Response repeats the same text                                            | 4     |
| Generation was stopped early after sustained repeated output              | 4     |
| Unrecognised model control tokens remain visible                          | 1     |
| Required labelled fields not detected                                     | 5     |
| Response appears cut off at the token limit                               | 3     |
| Repeated keyword entries                                                  | 5     |
| Output repeats the prompt's own hint text instead of describing the image | 6     |
| Names a place the prompt did not supply                                   | 2     |

## Triage

| Model                                                                                                                        | Execution | Mechanical checks | Maintainer status              | Observations                                                                                  |
|------------------------------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|-----------------------------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)                                            | crashed   | not assessed      | actionable_failure             | none                                                                                          |
| [mlx-community/Mage-VL-OptiQ-4bit](#diagnostic-mlx-community-mage-vl-optiq-4bit)                                             | crashed   | not assessed      | actionable_failure             | none                                                                                          |
| [mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit](#diagnostic-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit) | crashed   | not assessed      | actionable_failure             | none                                                                                          |
| [LiquidAI/LFM2.5-VL-450M-MLX-bf16](#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16)                                              | completed | major concerns    | observation_needs_reproduction | repeated text; stopped early: repeating; duplicate keywords; prompt hint repeated             |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-4bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-4bit)              | completed | major concerns    | observation_needs_reproduction | repeated text; stopped early: repeating; duplicate keywords                                   |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit)              | completed | major concerns    | observation_needs_reproduction | repeated text; cut off at token limit                                                         |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                                   | completed | major concerns    | observation_needs_reproduction | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected |
| [mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit](#diagnostic-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)                     | completed | major concerns    | observation_needs_reproduction | stopped early: repeating; duplicate keywords                                                  |

## Crashes requiring action

<a id="diagnostic-mlx-community-internvl35-1b-4bit"></a>

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
- *Post-cleanup active memory (GB):* 0.002116162
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14457, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13369, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 826, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15585, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14472, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Fetching 14 files:   0%|          | 0/14 [00:00<?, ?it/s]
Fetching 14 files: 100%|##########| 14/14 [00:00<00:00, 4711.94it/s]
ERROR:root:Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
```

<a id="diagnostic-mlx-community-mage-vl-optiq-4bit"></a>

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
- *Post-cleanup active memory (GB):* 0.00408229
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13703, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14380, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<5 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14275, in _generate_with_repetition_guard
    for chunk in stream_generate(
                 ~~~~~~~~~~~~~~~^
        model=model, processor=processor, prompt=prompt, image=image, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1083, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 434, in generate_step
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15585, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14484, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14400, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13712, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 3074.10it/s]
[23:33:20] Generation error for mlx-community/Mage-VL-OptiQ-4bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13703, in
           _run_generation_guarded
               return generate_once()
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14380, in
           _generate_once
               return _generate_with_repetition_guard(
                   model=prepared.model,
               ...<5 lines>...
                   **prepared.generate_kwargs,
               )
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14275, in
           _generate_with_repetition_guard
               for chunk in stream_generate(
                            ~~~~~~~~~~~~~~~^
                   model=model, processor=processor, prompt=prompt, image=image, **kwargs
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               ):
               ^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1083, in
           stream_generate
               for n, (token, logprobs) in enumerate(gen):
                                           ~~~~~~~~~^^^^^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 434, in
           generate_step
               embedding_output = model.get_input_embeddings(
                   input_ids, pixel_values, mask=mask, **kwargs
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/mage_vl.py", line 66,
           in get_input_embeddings
               hidden_states = self.vision_tower(
                   pixel_values.astype(inputs_embeds.dtype), patch_positions, grid_thw
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 381,
           in __call__
               cu = build_cu_seqlens(grid_thw, total_patches, self.config.frame_windows_size)
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mage_vl/vision.py", line 309,
           in build_cu_seqlens
               raise ValueError(
                   f"cu_seqlens mismatch: total_patches={total_patches} calculated={cu[-1]}
           grid={grid_thw}"
               )
           ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]
```

<a id="diagnostic-mlx-community-nvidia-nemotron-3-nano-omni-30b-a3b-4bit"></a>

### mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit

#### Root exception and chain

```text
builtins.ValueError: Received 729 parameters not in model; families: backbone, lm_head; representative parameters: backbone.embeddings.biases, backbone.embeddings.scales, backbone.embeddings.weight.
builtins.ValueError: Model loading failed: Received 729 parameters not in model; families: backbone, lm_head; representative parameters: backbone.embeddings.biases, backbone.embeddings.scales, backbone.embeddings.weight.
```

#### Execution and provenance

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type
  nemotronh_nano_omni_reasoning_v3 via nemotron_h_nano_omni)
- *Phase:* model_load
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model loading failed: Received 729 parameters not in model;
  families: backbone, lm_head; representative parameters:
  backbone.embeddings.biases, backbone.embeddings.scales,
  backbone.embeddings.weight.
- *Root error type:* ValueError
- *Root error message:* Received 729 parameters not in model; families:
  backbone, lm_head; representative parameters: backbone.embeddings.biases,
  backbone.embeddings.scales, backbone.embeddings.weight.
- *Resolved model revision:* 8f86ad8f279ce1ec3b8b65970f32da1e89ab7a45
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.005901056
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 20.65
- *Parameter count:* 30.00B total, 3.00B active (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 131,072 (max_sequence_length)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14457, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13369, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 826, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1172, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 729 parameters not in model: 
backbone.embeddings.biases,
backbone.embeddings.scales,
backbone.embeddings.weight,
backbone.layers.0.mixer.A_log,
backbone.layers.0.mixer.D,
backbone.layers.0.mixer.conv1d.bias,
backbone.layers.0.mixer.conv1d.weight,
backbone.layers.0.mixer.dt_bias,
backbone.layers.0.mixer.in_proj.biases,
backbone.layers.0.mixer.in_proj.scales,
backbone.layers.0.mixer.in_proj.weight,
backbone.layers.0.mixer.norm.weight,
backbone.layers.0.mixer.out_proj.biases,
backbone.layers.0.mixer.out_proj.scales,
backbone.layers.0.mixer.out_proj.weight,
backbone.layers.0.norm.weight,
backbone.layers.1.mixer.gate.e_score_correction_bias,
backbone.layers.1.mixer.gate.weight,
backbone.layers.1.mixer.shared_experts.down_proj.biases,
backbone.layers.1.mixer.shared_experts.down_proj.scales,
backbone.layers.1.mixer.shared_experts.down_proj.weight,
backbone.layers.1.mixer.shared_experts.up_proj.biases,
backbone.layers.1.mixer.shared_experts.up_proj.scales,
backbone.layers.1.mixer.shared_experts.up_proj.weight,
backbone.layers.1.mixer.switch_mlp.fc1.biases,
backbone.layers.1.mixer.switch_mlp.fc1.scales,
backbone.layers.1.mixer.switch_mlp.fc1.weight,
backbone.layers.1.mixer.switch_mlp.fc2.biases,
backbone.layers.1.mixer.switch_mlp.fc2.scales,
backbone.layers.1.mixer.switch_mlp.fc2.weight,
backbone.layers.1.norm.weight,
backbone.layers.10.mixer.gate.e_score_correction_bias,
backbone.layers.10.mixer.gate.weight,
backbone.layers.10.mixer.shared_experts.down_proj.biases,
backbone.layers.10.mixer.shared_experts.down_proj.scales,
backbone.layers.10.mixer.shared_experts.down_proj.weight,
backbone.layers.10.mixer.shared_experts.up_proj.biases,
backbone.layers.10.mixer.shared_experts.up_proj.scales,
backbone.layers.10.mixer.shared_experts.up_proj.weight,
backbone.layers.10.mixer.switch_mlp.fc1.biases,
backbone.layers.10.mixer.switch_mlp.fc1.scales,
backbone.layers.10.mixer.switch_mlp.fc1.weight,
backbone.layers.10.mixer.switch_mlp.fc2.biases,
backbone.layers.10.mixer.switch_mlp.fc2.scales,
backbone.layers.10.mixer.switch_mlp.fc2.weight,
backbone.layers.10.norm.weight,
backbone.layers.11.mixer.A_log,
backbone.layers.11.mixer.D,
backbone.layers.11.mixer.conv1d.bias,
backbone.layers.11.mixer.conv1d.weight,
backbone.layers.11.mixer.dt_bias,
backbone.layers.11.mixer.in_proj.biases,
backbone.layers.11.mixer.in_proj.scales,
backbone.layers.11.mixer.in_proj.weight,
backbone.layers.11.mixer.norm.weight,
backbone.layers.11.mixer.out_proj.biases,
backbone.layers.11.mixer.out_proj.scales,
backbone.layers.11.mixer.out_proj.weight,
backbone.layers.11.norm.weight,
backbone.layers.12.mixer.k_proj.biases,
backbone.layers.12.mixer.k_proj.scales,
backbone.layers.12.mixer.k_proj.weight,
backbone.layers.12.mixer.o_proj.biases,
backbone.layers.12.mixer.o_proj.scales,
backbone.layers.12.mixer.o_proj.weight,
backbone.layers.12.mixer.q_proj.biases,
backbone.layers.12.mixer.q_proj.scales,
backbone.layers.12.mixer.q_proj.weight,
backbone.layers.12.mixer.v_proj.biases,
backbone.layers.12.mixer.v_proj.scales,
backbone.layers.12.mixer.v_proj.weight,
backbone.layers.12.norm.weight,
backbone.layers.13.mixer.gate.e_score_correction_bias,
backbone.layers.13.mixer.gate.weight,
backbone.layers.13.mixer.shared_experts.down_proj.biases,
backbone.layers.13.mixer.shared_experts.down_proj.scales,
backbone.layers.13.mixer.shared_experts.down_proj.weight,
backbone.layers.13.mixer.shared_experts.up_proj.biases,
backbone.layers.13.mixer.shared_experts.up_proj.scales,
backbone.layers.13.mixer.shared_experts.up_proj.weight,
backbone.layers.13.mixer.switch_mlp.fc1.biases,
backbone.layers.13.mixer.switch_mlp.fc1.scales,
backbone.layers.13.mixer.switch_mlp.fc1.weight,
backbone.layers.13.mixer.switch_mlp.fc2.biases,
backbone.layers.13.mixer.switch_mlp.fc2.scales,
backbone.layers.13.mixer.switch_mlp.fc2.weight,
backbone.layers.13.norm.weight,
backbone.layers.14.mixer.A_log,
backbone.layers.14.mixer.D,
backbone.layers.14.mixer.conv1d.bias,
backbone.layers.14.mixer.conv1d.weight,
backbone.layers.14.mixer.dt_bias,
backbone.layers.14.mixer.in_proj.biases,
backbone.layers.14.mixer.in_proj.scales,
backbone.layers.14.mixer.in_proj.weight,
backbone.layers.14.mixer.norm.weight,
backbone.layers.14.mixer.out_proj.biases,
backbone.layers.14.mixer.out_proj.scales,
backbone.layers.14.mixer.out_proj.weight,
backbone.layers.14.norm.weight,
backbone.layers.15.mixer.gate.e_score_correction_bias,
backbone.layers.15.mixer.gate.weight,
backbone.layers.15.mixer.shared_experts.down_proj.biases,
backbone.layers.15.mixer.shared_experts.down_proj.scales,
backbone.layers.15.mixer.shared_experts.down_proj.weight,
backbone.layers.15.mixer.shared_experts.up_proj.biases,
backbone.layers.15.mixer.shared_experts.up_proj.scales,
backbone.layers.15.mixer.shared_experts.up_proj.weight,
backbone.layers.15.mixer.switch_mlp.fc1.biases,
backbone.layers.15.mixer.switch_mlp.fc1.scales,
backbone.layers.15.mixer.switch_mlp.fc1.weight,
backbone.layers.15.mixer.switch_mlp.fc2.biases,
backbone.layers.15.mixer.switch_mlp.fc2.scales,
backbone.layers.15.mixer.switch_mlp.fc2.weight,
backbone.layers.15.norm.weight,
backbone.layers.16.mixer.A_log,
backbone.layers.16.mixer.D,
backbone.layers.16.mixer.conv1d.bias,
backbone.layers.16.mixer.conv1d.weight,
backbone.layers.16.mixer.dt_bias,
backbone.layers.16.mixer.in_proj.biases,
backbone.layers.16.mixer.in_proj.scales,
backbone.layers.16.mixer.in_proj.weight,
backbone.layers.16.mixer.norm.weight,
backbone.layers.16.mixer.out_proj.biases,
backbone.layers.16.mixer.out_proj.scales,
backbone.layers.16.mixer.out_proj.weight,
backbone.layers.16.norm.weight,
backbone.layers.17.mixer.gate.e_score_correction_bias,
backbone.layers.17.mixer.gate.weight,
backbone.layers.17.mixer.shared_experts.down_proj.biases,
backbone.layers.17.mixer.shared_experts.down_proj.scales,
backbone.layers.17.mixer.shared_experts.down_proj.weight,
backbone.layers.17.mixer.shared_experts.up_proj.biases,
backbone.layers.17.mixer.shared_experts.up_proj.scales,
backbone.layers.17.mixer.shared_experts.up_proj.weight,
backbone.layers.17.mixer.switch_mlp.fc1.biases,
backbone.layers.17.mixer.switch_mlp.fc1.scales,
backbone.layers.17.mixer.switch_mlp.fc1.weight,
backbone.layers.17.mixer.switch_mlp.fc2.biases,
backbone.layers.17.mixer.switch_mlp.fc2.scales,
backbone.layers.17.mixer.switch_mlp.fc2.weight,
backbone.layers.17.norm.weight,
backbone.layers.18.mixer.A_log,
backbone.layers.18.mixer.D,
backbone.layers.18.mixer.conv1d.bias,
backbone.layers.18.mixer.conv1d.weight,
backbone.layers.18.mixer.dt_bias,
backbone.layers.18.mixer.in_proj.biases,
backbone.layers.18.mixer.in_proj.scales,
backbone.layers.18.mixer.in_proj.weight,
backbone.layers.18.mixer.norm.weight,
backbone.layers.18.mixer.out_proj.biases,
backbone.layers.18.mixer.out_proj.scales,
backbone.layers.18.mixer.out_proj.weight,
backbone.layers.18.norm.weight,
backbone.layers.19.mixer.k_proj.biases,
backbone.layers.19.mixer.k_proj.scales,
backbone.layers.19.mixer.k_proj.weight,
backbone.layers.19.mixer.o_proj.biases,
backbone.layers.19.mixer.o_proj.scales,
backbone.layers.19.mixer.o_proj.weight,
backbone.layers.19.mixer.q_proj.biases,
backbone.layers.19.mixer.q_proj.scales,
backbone.layers.19.mixer.q_proj.weight,
backbone.layers.19.mixer.v_proj.biases,
backbone.layers.19.mixer.v_proj.scales,
backbone.layers.19.mixer.v_proj.weight,
backbone.layers.19.norm.weight,
backbone.layers.2.mixer.A_log,
backbone.layers.2.mixer.D,
backbone.layers.2.mixer.conv1d.bias,
backbone.layers.2.mixer.conv1d.weight,
backbone.layers.2.mixer.dt_bias,
backbone.layers.2.mixer.in_proj.biases,
backbone.layers.2.mixer.in_proj.scales,
backbone.layers.2.mixer.in_proj.weight,
backbone.layers.2.mixer.norm.weight,
backbone.layers.2.mixer.out_proj.biases,
backbone.layers.2.mixer.out_proj.scales,
backbone.layers.2.mixer.out_proj.weight,
backbone.layers.2.norm.weight,
backbone.layers.20.mixer.gate.e_score_correction_bias,
backbone.layers.20.mixer.gate.weight,
backbone.layers.20.mixer.shared_experts.down_proj.biases,
backbone.layers.20.mixer.shared_experts.down_proj.scales,
backbone.layers.20.mixer.shared_experts.down_proj.weight,
backbone.layers.20.mixer.shared_experts.up_proj.biases,
backbone.layers.20.mixer.shared_experts.up_proj.scales,
backbone.layers.20.mixer.shared_experts.up_proj.weight,
backbone.layers.20.mixer.switch_mlp.fc1.biases,
backbone.layers.20.mixer.switch_mlp.fc1.scales,
backbone.layers.20.mixer.switch_mlp.fc1.weight,
backbone.layers.20.mixer.switch_mlp.fc2.biases,
backbone.layers.20.mixer.switch_mlp.fc2.scales,
backbone.layers.20.mixer.switch_mlp.fc2.weight,
backbone.layers.20.norm.weight,
backbone.layers.21.mixer.A_log,
backbone.layers.21.mixer.D,
backbone.layers.21.mixer.conv1d.bias,
backbone.layers.21.mixer.conv1d.weight,
backbone.layers.21.mixer.dt_bias,
backbone.layers.21.mixer.in_proj.biases,
backbone.layers.21.mixer.in_proj.scales,
backbone.layers.21.mixer.in_proj.weight,
backbone.layers.21.mixer.norm.weight,
backbone.layers.21.mixer.out_proj.biases,
backbone.layers.21.mixer.out_proj.scales,
backbone.layers.21.mixer.out_proj.weight,
backbone.layers.21.norm.weight,
backbone.layers.22.mixer.gate.e_score_correction_bias,
backbone.layers.22.mixer.gate.weight,
backbone.layers.22.mixer.shared_experts.down_proj.biases,
backbone.layers.22.mixer.shared_experts.down_proj.scales,
backbone.layers.22.mixer.shared_experts.down_proj.weight,
backbone.layers.22.mixer.shared_experts.up_proj.biases,
backbone.layers.22.mixer.shared_experts.up_proj.scales,
backbone.layers.22.mixer.shared_experts.up_proj.weight,
backbone.layers.22.mixer.switch_mlp.fc1.biases,
backbone.layers.22.mixer.switch_mlp.fc1.scales,
backbone.layers.22.mixer.switch_mlp.fc1.weight,
backbone.layers.22.mixer.switch_mlp.fc2.biases,
backbone.layers.22.mixer.switch_mlp.fc2.scales,
backbone.layers.22.mixer.switch_mlp.fc2.weight,
backbone.layers.22.norm.weight,
backbone.layers.23.mixer.A_log,
backbone.layers.23.mixer.D,
backbone.layers.23.mixer.conv1d.bias,
backbone.layers.23.mixer.conv1d.weight,
backbone.layers.23.mixer.dt_bias,
backbone.layers.23.mixer.in_proj.biases,
backbone.layers.23.mixer.in_proj.scales,
backbone.layers.23.mixer.in_proj.weight,
backbone.layers.23.mixer.norm.weight,
backbone.layers.23.mixer.out_proj.biases,
backbone.layers.23.mixer.out_proj.scales,
backbone.layers.23.mixer.out_proj.weight,
backbone.layers.23.norm.weight,
backbone.layers.24.mixer.gate.e_score_correction_bias,
backbone.layers.24.mixer.gate.weight,
backbone.layers.24.mixer.shared_experts.down_proj.biases,
backbone.layers.24.mixer.shared_experts.down_proj.scales,
backbone.layers.24.mixer.shared_experts.down_proj.weight,
backbone.layers.24.mixer.shared_experts.up_proj.biases,
backbone.layers.24.mixer.shared_experts.up_proj.scales,
backbone.layers.24.mixer.shared_experts.up_proj.weight,
backbone.layers.24.mixer.switch_mlp.fc1.biases,
backbone.layers.24.mixer.switch_mlp.fc1.scales,
backbone.layers.24.mixer.switch_mlp.fc1.weight,
backbone.layers.24.mixer.switch_mlp.fc2.biases,
backbone.layers.24.mixer.switch_mlp.fc2.scales,
backbone.layers.24.mixer.switch_mlp.fc2.weight,
backbone.layers.24.norm.weight,
backbone.layers.25.mixer.A_log,
backbone.layers.25.mixer.D,
backbone.layers.25.mixer.conv1d.bias,
backbone.layers.25.mixer.conv1d.weight,
backbone.layers.25.mixer.dt_bias,
backbone.layers.25.mixer.in_proj.biases,
backbone.layers.25.mixer.in_proj.scales,
backbone.layers.25.mixer.in_proj.weight,
backbone.layers.25.mixer.norm.weight,
backbone.layers.25.mixer.out_proj.biases,
backbone.layers.25.mixer.out_proj.scales,
backbone.layers.25.mixer.out_proj.weight,
backbone.layers.25.norm.weight,
backbone.layers.26.mixer.k_proj.biases,
backbone.layers.26.mixer.k_proj.scales,
backbone.layers.26.mixer.k_proj.weight,
backbone.layers.26.mixer.o_proj.biases,
backbone.layers.26.mixer.o_proj.scales,
backbone.layers.26.mixer.o_proj.weight,
backbone.layers.26.mixer.q_proj.biases,
backbone.layers.26.mixer.q_proj.scales,
backbone.layers.26.mixer.q_proj.weight,
backbone.layers.26.mixer.v_proj.biases,
backbone.layers.26.mixer.v_proj.scales,
backbone.layers.26.mixer.v_proj.weight,
backbone.layers.26.norm.weight,
backbone.layers.27.mixer.gate.e_score_correction_bias,
backbone.layers.27.mixer.gate.weight,
backbone.layers.27.mixer.shared_experts.down_proj.biases,
backbone.layers.27.mixer.shared_experts.down_proj.scales,
backbone.layers.27.mixer.shared_experts.down_proj.weight,
backbone.layers.27.mixer.shared_experts.up_proj.biases,
backbone.layers.27.mixer.shared_experts.up_proj.scales,
backbone.layers.27.mixer.shared_experts.up_proj.weight,
backbone.layers.27.mixer.switch_mlp.fc1.biases,
backbone.layers.27.mixer.switch_mlp.fc1.scales,
backbone.layers.27.mixer.switch_mlp.fc1.weight,
backbone.layers.27.mixer.switch_mlp.fc2.biases,
backbone.layers.27.mixer.switch_mlp.fc2.scales,
backbone.layers.27.mixer.switch_mlp.fc2.weight,
backbone.layers.27.norm.weight,
backbone.layers.28.mixer.A_log,
backbone.layers.28.mixer.D,
backbone.layers.28.mixer.conv1d.bias,
backbone.layers.28.mixer.conv1d.weight,
backbone.layers.28.mixer.dt_bias,
backbone.layers.28.mixer.in_proj.biases,
backbone.layers.28.mixer.in_proj.scales,
backbone.layers.28.mixer.in_proj.weight,
backbone.layers.28.mixer.norm.weight,
backbone.layers.28.mixer.out_proj.biases,
backbone.layers.28.mixer.out_proj.scales,
backbone.layers.28.mixer.out_proj.weight,
backbone.layers.28.norm.weight,
backbone.layers.29.mixer.gate.e_score_correction_bias,
backbone.layers.29.mixer.gate.weight,
backbone.layers.29.mixer.shared_experts.down_proj.biases,
backbone.layers.29.mixer.shared_experts.down_proj.scales,
backbone.layers.29.mixer.shared_experts.down_proj.weight,
backbone.layers.29.mixer.shared_experts.up_proj.biases,
backbone.layers.29.mixer.shared_experts.up_proj.scales,
backbone.layers.29.mixer.shared_experts.up_proj.weight,
backbone.layers.29.mixer.switch_mlp.fc1.biases,
backbone.layers.29.mixer.switch_mlp.fc1.scales,
backbone.layers.29.mixer.switch_mlp.fc1.weight,
backbone.layers.29.mixer.switch_mlp.fc2.biases,
backbone.layers.29.mixer.switch_mlp.fc2.scales,
backbone.layers.29.mixer.switch_mlp.fc2.weight,
backbone.layers.29.norm.weight,
backbone.layers.3.mixer.gate.e_score_correction_bias,
backbone.layers.3.mixer.gate.weight,
backbone.layers.3.mixer.shared_experts.down_proj.biases,
backbone.layers.3.mixer.shared_experts.down_proj.scales,
backbone.layers.3.mixer.shared_experts.down_proj.weight,
backbone.layers.3.mixer.shared_experts.up_proj.biases,
backbone.layers.3.mixer.shared_experts.up_proj.scales,
backbone.layers.3.mixer.shared_experts.up_proj.weight,
backbone.layers.3.mixer.switch_mlp.fc1.biases,
backbone.layers.3.mixer.switch_mlp.fc1.scales,
backbone.layers.3.mixer.switch_mlp.fc1.weight,
backbone.layers.3.mixer.switch_mlp.fc2.biases,
backbone.layers.3.mixer.switch_mlp.fc2.scales,
backbone.layers.3.mixer.switch_mlp.fc2.weight,
backbone.layers.3.norm.weight,
backbone.layers.30.mixer.A_log,
backbone.layers.30.mixer.D,
backbone.layers.30.mixer.conv1d.bias,
backbone.layers.30.mixer.conv1d.weight,
backbone.layers.30.mixer.dt_bias,
backbone.layers.30.mixer.in_proj.biases,
backbone.layers.30.mixer.in_proj.scales,
backbone.layers.30.mixer.in_proj.weight,
backbone.layers.30.mixer.norm.weight,
backbone.layers.30.mixer.out_proj.biases,
backbone.layers.30.mixer.out_proj.scales,
backbone.layers.30.mixer.out_proj.weight,
backbone.layers.30.norm.weight,
backbone.layers.31.mixer.gate.e_score_correction_bias,
backbone.layers.31.mixer.gate.weight,
backbone.layers.31.mixer.shared_experts.down_proj.biases,
backbone.layers.31.mixer.shared_experts.down_proj.scales,
backbone.layers.31.mixer.shared_experts.down_proj.weight,
backbone.layers.31.mixer.shared_experts.up_proj.biases,
backbone.layers.31.mixer.shared_experts.up_proj.scales,
backbone.layers.31.mixer.shared_experts.up_proj.weight,
backbone.layers.31.mixer.switch_mlp.fc1.biases,
backbone.layers.31.mixer.switch_mlp.fc1.scales,
backbone.layers.31.mixer.switch_mlp.fc1.weight,
backbone.layers.31.mixer.switch_mlp.fc2.biases,
backbone.layers.31.mixer.switch_mlp.fc2.scales,
backbone.layers.31.mixer.switch_mlp.fc2.weight,
backbone.layers.31.norm.weight,
backbone.layers.32.mixer.A_log,
backbone.layers.32.mixer.D,
backbone.layers.32.mixer.conv1d.bias,
backbone.layers.32.mixer.conv1d.weight,
backbone.layers.32.mixer.dt_bias,
backbone.layers.32.mixer.in_proj.biases,
backbone.layers.32.mixer.in_proj.scales,
backbone.layers.32.mixer.in_proj.weight,
backbone.layers.32.mixer.norm.weight,
backbone.layers.32.mixer.out_proj.biases,
backbone.layers.32.mixer.out_proj.scales,
backbone.layers.32.mixer.out_proj.weight,
backbone.layers.32.norm.weight,
backbone.layers.33.mixer.k_proj.biases,
backbone.layers.33.mixer.k_proj.scales,
backbone.layers.33.mixer.k_proj.weight,
backbone.layers.33.mixer.o_proj.biases,
backbone.layers.33.mixer.o_proj.scales,
backbone.layers.33.mixer.o_proj.weight,
backbone.layers.33.mixer.q_proj.biases,
backbone.layers.33.mixer.q_proj.scales,
backbone.layers.33.mixer.q_proj.weight,
backbone.layers.33.mixer.v_proj.biases,
backbone.layers.33.mixer.v_proj.scales,
backbone.layers.33.mixer.v_proj.weight,
backbone.layers.33.norm.weight,
backbone.layers.34.mixer.gate.e_score_correction_bias,
backbone.layers.34.mixer.gate.weight,
backbone.layers.34.mixer.shared_experts.down_proj.biases,
backbone.layers.34.mixer.shared_experts.down_proj.scales,
backbone.layers.34.mixer.shared_experts.down_proj.weight,
backbone.layers.34.mixer.shared_experts.up_proj.biases,
backbone.layers.34.mixer.shared_experts.up_proj.scales,
backbone.layers.34.mixer.shared_experts.up_proj.weight,
backbone.layers.34.mixer.switch_mlp.fc1.biases,
backbone.layers.34.mixer.switch_mlp.fc1.scales,
backbone.layers.34.mixer.switch_mlp.fc1.weight,
backbone.layers.34.mixer.switch_mlp.fc2.biases,
backbone.layers.34.mixer.switch_mlp.fc2.scales,
backbone.layers.34.mixer.switch_mlp.fc2.weight,
backbone.layers.34.norm.weight,
backbone.layers.35.mixer.A_log,
backbone.layers.35.mixer.D,
backbone.layers.35.mixer.conv1d.bias,
backbone.layers.35.mixer.conv1d.weight,
backbone.layers.35.mixer.dt_bias,
backbone.layers.35.mixer.in_proj.biases,
backbone.layers.35.mixer.in_proj.scales,
backbone.layers.35.mixer.in_proj.weight,
backbone.layers.35.mixer.norm.weight,
backbone.layers.35.mixer.out_proj.biases,
backbone.layers.35.mixer.out_proj.scales,
backbone.layers.35.mixer.out_proj.weight,
backbone.layers.35.norm.weight,
backbone.layers.36.mixer.gate.e_score_correction_bias,
backbone.layers.36.mixer.gate.weight,
backbone.layers.36.mixer.shared_experts.down_proj.biases,
backbone.layers.36.mixer.shared_experts.down_proj.scales,
backbone.layers.36.mixer.shared_experts.down_proj.weight,
backbone.layers.36.mixer.shared_experts.up_proj.biases,
backbone.layers.36.mixer.shared_experts.up_proj.scales,
backbone.layers.36.mixer.shared_experts.up_proj.weight,
backbone.layers.36.mixer.switch_mlp.fc1.biases,
backbone.layers.36.mixer.switch_mlp.fc1.scales,
backbone.layers.36.mixer.switch_mlp.fc1.weight,
backbone.layers.36.mixer.switch_mlp.fc2.biases,
backbone.layers.36.mixer.switch_mlp.fc2.scales,
backbone.layers.36.mixer.switch_mlp.fc2.weight,
backbone.layers.36.norm.weight,
backbone.layers.37.mixer.A_log,
backbone.layers.37.mixer.D,
backbone.layers.37.mixer.conv1d.bias,
backbone.layers.37.mixer.conv1d.weight,
backbone.layers.37.mixer.dt_bias,
backbone.layers.37.mixer.in_proj.biases,
backbone.layers.37.mixer.in_proj.scales,
backbone.layers.37.mixer.in_proj.weight,
backbone.layers.37.mixer.norm.weight,
backbone.layers.37.mixer.out_proj.biases,
backbone.layers.37.mixer.out_proj.scales,
backbone.layers.37.mixer.out_proj.weight,
backbone.layers.37.norm.weight,
backbone.layers.38.mixer.gate.e_score_correction_bias,
backbone.layers.38.mixer.gate.weight,
backbone.layers.38.mixer.shared_experts.down_proj.biases,
backbone.layers.38.mixer.shared_experts.down_proj.scales,
backbone.layers.38.mixer.shared_experts.down_proj.weight,
backbone.layers.38.mixer.shared_experts.up_proj.biases,
backbone.layers.38.mixer.shared_experts.up_proj.scales,
backbone.layers.38.mixer.shared_experts.up_proj.weight,
backbone.layers.38.mixer.switch_mlp.fc1.biases,
backbone.layers.38.mixer.switch_mlp.fc1.scales,
backbone.layers.38.mixer.switch_mlp.fc1.weight,
backbone.layers.38.mixer.switch_mlp.fc2.biases,
backbone.layers.38.mixer.switch_mlp.fc2.scales,
backbone.layers.38.mixer.switch_mlp.fc2.weight,
backbone.layers.38.norm.weight,
backbone.layers.39.mixer.A_log,
backbone.layers.39.mixer.D,
backbone.layers.39.mixer.conv1d.bias,
backbone.layers.39.mixer.conv1d.weight,
backbone.layers.39.mixer.dt_bias,
backbone.layers.39.mixer.in_proj.biases,
backbone.layers.39.mixer.in_proj.scales,
backbone.layers.39.mixer.in_proj.weight,
backbone.layers.39.mixer.norm.weight,
backbone.layers.39.mixer.out_proj.biases,
backbone.layers.39.mixer.out_proj.scales,
backbone.layers.39.mixer.out_proj.weight,
backbone.layers.39.norm.weight,
backbone.layers.4.mixer.A_log,
backbone.layers.4.mixer.D,
backbone.layers.4.mixer.conv1d.bias,
backbone.layers.4.mixer.conv1d.weight,
backbone.layers.4.mixer.dt_bias,
backbone.layers.4.mixer.in_proj.biases,
backbone.layers.4.mixer.in_proj.scales,
backbone.layers.4.mixer.in_proj.weight,
backbone.layers.4.mixer.norm.weight,
backbone.layers.4.mixer.out_proj.biases,
backbone.layers.4.mixer.out_proj.scales,
backbone.layers.4.mixer.out_proj.weight,
backbone.layers.4.norm.weight,
backbone.layers.40.mixer.gate.e_score_correction_bias,
backbone.layers.40.mixer.gate.weight,
backbone.layers.40.mixer.shared_experts.down_proj.biases,
backbone.layers.40.mixer.shared_experts.down_proj.scales,
backbone.layers.40.mixer.shared_experts.down_proj.weight,
backbone.layers.40.mixer.shared_experts.up_proj.biases,
backbone.layers.40.mixer.shared_experts.up_proj.scales,
backbone.layers.40.mixer.shared_experts.up_proj.weight,
backbone.layers.40.mixer.switch_mlp.fc1.biases,
backbone.layers.40.mixer.switch_mlp.fc1.scales,
backbone.layers.40.mixer.switch_mlp.fc1.weight,
backbone.layers.40.mixer.switch_mlp.fc2.biases,
backbone.layers.40.mixer.switch_mlp.fc2.scales,
backbone.layers.40.mixer.switch_mlp.fc2.weight,
backbone.layers.40.norm.weight,
backbone.layers.41.mixer.A_log,
backbone.layers.41.mixer.D,
backbone.layers.41.mixer.conv1d.bias,
backbone.layers.41.mixer.conv1d.weight,
backbone.layers.41.mixer.dt_bias,
backbone.layers.41.mixer.in_proj.biases,
backbone.layers.41.mixer.in_proj.scales,
backbone.layers.41.mixer.in_proj.weight,
backbone.layers.41.mixer.norm.weight,
backbone.layers.41.mixer.out_proj.biases,
backbone.layers.41.mixer.out_proj.scales,
backbone.layers.41.mixer.out_proj.weight,
backbone.layers.41.norm.weight,
backbone.layers.42.mixer.k_proj.biases,
backbone.layers.42.mixer.k_proj.scales,
backbone.layers.42.mixer.k_proj.weight,
backbone.layers.42.mixer.o_proj.biases,
backbone.layers.42.mixer.o_proj.scales,
backbone.layers.42.mixer.o_proj.weight,
backbone.layers.42.mixer.q_proj.biases,
backbone.layers.42.mixer.q_proj.scales,
backbone.layers.42.mixer.q_proj.weight,
backbone.layers.42.mixer.v_proj.biases,
backbone.layers.42.mixer.v_proj.scales,
backbone.layers.42.mixer.v_proj.weight,
backbone.layers.42.norm.weight,
backbone.layers.43.mixer.gate.e_score_correction_bias,
backbone.layers.43.mixer.gate.weight,
backbone.layers.43.mixer.shared_experts.down_proj.biases,
backbone.layers.43.mixer.shared_experts.down_proj.scales,
backbone.layers.43.mixer.shared_experts.down_proj.weight,
backbone.layers.43.mixer.shared_experts.up_proj.biases,
backbone.layers.43.mixer.shared_experts.up_proj.scales,
backbone.layers.43.mixer.shared_experts.up_proj.weight,
backbone.layers.43.mixer.switch_mlp.fc1.biases,
backbone.layers.43.mixer.switch_mlp.fc1.scales,
backbone.layers.43.mixer.switch_mlp.fc1.weight,
backbone.layers.43.mixer.switch_mlp.fc2.biases,
backbone.layers.43.mixer.switch_mlp.fc2.scales,
backbone.layers.43.mixer.switch_mlp.fc2.weight,
backbone.layers.43.norm.weight,
backbone.layers.44.mixer.A_log,
backbone.layers.44.mixer.D,
backbone.layers.44.mixer.conv1d.bias,
backbone.layers.44.mixer.conv1d.weight,
backbone.layers.44.mixer.dt_bias,
backbone.layers.44.mixer.in_proj.biases,
backbone.layers.44.mixer.in_proj.scales,
backbone.layers.44.mixer.in_proj.weight,
backbone.layers.44.mixer.norm.weight,
backbone.layers.44.mixer.out_proj.biases,
backbone.layers.44.mixer.out_proj.scales,
backbone.layers.44.mixer.out_proj.weight,
backbone.layers.44.norm.weight,
backbone.layers.45.mixer.gate.e_score_correction_bias,
backbone.layers.45.mixer.gate.weight,
backbone.layers.45.mixer.shared_experts.down_proj.biases,
backbone.layers.45.mixer.shared_experts.down_proj.scales,
backbone.layers.45.mixer.shared_experts.down_proj.weight,
backbone.layers.45.mixer.shared_experts.up_proj.biases,
backbone.layers.45.mixer.shared_experts.up_proj.scales,
backbone.layers.45.mixer.shared_experts.up_proj.weight,
backbone.layers.45.mixer.switch_mlp.fc1.biases,
backbone.layers.45.mixer.switch_mlp.fc1.scales,
backbone.layers.45.mixer.switch_mlp.fc1.weight,
backbone.layers.45.mixer.switch_mlp.fc2.biases,
backbone.layers.45.mixer.switch_mlp.fc2.scales,
backbone.layers.45.mixer.switch_mlp.fc2.weight,
backbone.layers.45.norm.weight,
backbone.layers.46.mixer.A_log,
backbone.layers.46.mixer.D,
backbone.layers.46.mixer.conv1d.bias,
backbone.layers.46.mixer.conv1d.weight,
backbone.layers.46.mixer.dt_bias,
backbone.layers.46.mixer.in_proj.biases,
backbone.layers.46.mixer.in_proj.scales,
backbone.layers.46.mixer.in_proj.weight,
backbone.layers.46.mixer.norm.weight,
backbone.layers.46.mixer.out_proj.biases,
backbone.layers.46.mixer.out_proj.scales,
backbone.layers.46.mixer.out_proj.weight,
backbone.layers.46.norm.weight,
backbone.layers.47.mixer.gate.e_score_correction_bias,
backbone.layers.47.mixer.gate.weight,
backbone.layers.47.mixer.shared_experts.down_proj.biases,
backbone.layers.47.mixer.shared_experts.down_proj.scales,
backbone.layers.47.mixer.shared_experts.down_proj.weight,
backbone.layers.47.mixer.shared_experts.up_proj.biases,
backbone.layers.47.mixer.shared_experts.up_proj.scales,
backbone.layers.47.mixer.shared_experts.up_proj.weight,
backbone.layers.47.mixer.switch_mlp.fc1.biases,
backbone.layers.47.mixer.switch_mlp.fc1.scales,
backbone.layers.47.mixer.switch_mlp.fc1.weight,
backbone.layers.47.mixer.switch_mlp.fc2.biases,
backbone.layers.47.mixer.switch_mlp.fc2.scales,
backbone.layers.47.mixer.switch_mlp.fc2.weight,
backbone.layers.47.norm.weight,
backbone.layers.48.mixer.A_log,
backbone.layers.48.mixer.D,
backbone.layers.48.mixer.conv1d.bias,
backbone.layers.48.mixer.conv1d.weight,
backbone.layers.48.mixer.dt_bias,
backbone.layers.48.mixer.in_proj.biases,
backbone.layers.48.mixer.in_proj.scales,
backbone.layers.48.mixer.in_proj.weight,
backbone.layers.48.mixer.norm.weight,
backbone.layers.48.mixer.out_proj.biases,
backbone.layers.48.mixer.out_proj.scales,
backbone.layers.48.mixer.out_proj.weight,
backbone.layers.48.norm.weight,
backbone.layers.49.mixer.gate.e_score_correction_bias,
backbone.layers.49.mixer.gate.weight,
backbone.layers.49.mixer.shared_experts.down_proj.biases,
backbone.layers.49.mixer.shared_experts.down_proj.scales,
backbone.layers.49.mixer.shared_experts.down_proj.weight,
backbone.layers.49.mixer.shared_experts.up_proj.biases,
backbone.layers.49.mixer.shared_experts.up_proj.scales,
backbone.layers.49.mixer.shared_experts.up_proj.weight,
backbone.layers.49.mixer.switch_mlp.fc1.biases,
backbone.layers.49.mixer.switch_mlp.fc1.scales,
backbone.layers.49.mixer.switch_mlp.fc1.weight,
backbone.layers.49.mixer.switch_mlp.fc2.biases,
backbone.layers.49.mixer.switch_mlp.fc2.scales,
backbone.layers.49.mixer.switch_mlp.fc2.weight,
backbone.layers.49.norm.weight,
backbone.layers.5.mixer.k_proj.biases,
backbone.layers.5.mixer.k_proj.scales,
backbone.layers.5.mixer.k_proj.weight,
backbone.layers.5.mixer.o_proj.biases,
backbone.layers.5.mixer.o_proj.scales,
backbone.layers.5.mixer.o_proj.weight,
backbone.layers.5.mixer.q_proj.biases,
backbone.layers.5.mixer.q_proj.scales,
backbone.layers.5.mixer.q_proj.weight,
backbone.layers.5.mixer.v_proj.biases,
backbone.layers.5.mixer.v_proj.scales,
backbone.layers.5.mixer.v_proj.weight,
backbone.layers.5.norm.weight,
backbone.layers.50.mixer.A_log,
backbone.layers.50.mixer.D,
backbone.layers.50.mixer.conv1d.bias,
backbone.layers.50.mixer.conv1d.weight,
backbone.layers.50.mixer.dt_bias,
backbone.layers.50.mixer.in_proj.biases,
backbone.layers.50.mixer.in_proj.scales,
backbone.layers.50.mixer.in_proj.weight,
backbone.layers.50.mixer.norm.weight,
backbone.layers.50.mixer.out_proj.biases,
backbone.layers.50.mixer.out_proj.scales,
backbone.layers.50.mixer.out_proj.weight,
backbone.layers.50.norm.weight,
backbone.layers.51.mixer.gate.e_score_correction_bias,
backbone.layers.51.mixer.gate.weight,
backbone.layers.51.mixer.shared_experts.down_proj.biases,
backbone.layers.51.mixer.shared_experts.down_proj.scales,
backbone.layers.51.mixer.shared_experts.down_proj.weight,
backbone.layers.51.mixer.shared_experts.up_proj.biases,
backbone.layers.51.mixer.shared_experts.up_proj.scales,
backbone.layers.51.mixer.shared_experts.up_proj.weight,
backbone.layers.51.mixer.switch_mlp.fc1.biases,
backbone.layers.51.mixer.switch_mlp.fc1.scales,
backbone.layers.51.mixer.switch_mlp.fc1.weight,
backbone.layers.51.mixer.switch_mlp.fc2.biases,
backbone.layers.51.mixer.switch_mlp.fc2.scales,
backbone.layers.51.mixer.switch_mlp.fc2.weight,
backbone.layers.51.norm.weight,
backbone.layers.6.mixer.gate.e_score_correction_bias,
backbone.layers.6.mixer.gate.weight,
backbone.layers.6.mixer.shared_experts.down_proj.biases,
backbone.layers.6.mixer.shared_experts.down_proj.scales,
backbone.layers.6.mixer.shared_experts.down_proj.weight,
backbone.layers.6.mixer.shared_experts.up_proj.biases,
backbone.layers.6.mixer.shared_experts.up_proj.scales,
backbone.layers.6.mixer.shared_experts.up_proj.weight,
backbone.layers.6.mixer.switch_mlp.fc1.biases,
backbone.layers.6.mixer.switch_mlp.fc1.scales,
backbone.layers.6.mixer.switch_mlp.fc1.weight,
backbone.layers.6.mixer.switch_mlp.fc2.biases,
backbone.layers.6.mixer.switch_mlp.fc2.scales,
backbone.layers.6.mixer.switch_mlp.fc2.weight,
backbone.layers.6.norm.weight,
backbone.layers.7.mixer.A_log,
backbone.layers.7.mixer.D,
backbone.layers.7.mixer.conv1d.bias,
backbone.layers.7.mixer.conv1d.weight,
backbone.layers.7.mixer.dt_bias,
backbone.layers.7.mixer.in_proj.biases,
backbone.layers.7.mixer.in_proj.scales,
backbone.layers.7.mixer.in_proj.weight,
backbone.layers.7.mixer.norm.weight,
backbone.layers.7.mixer.out_proj.biases,
backbone.layers.7.mixer.out_proj.scales,
backbone.layers.7.mixer.out_proj.weight,
backbone.layers.7.norm.weight,
backbone.layers.8.mixer.gate.e_score_correction_bias,
backbone.layers.8.mixer.gate.weight,
backbone.layers.8.mixer.shared_experts.down_proj.biases,
backbone.layers.8.mixer.shared_experts.down_proj.scales,
backbone.layers.8.mixer.shared_experts.down_proj.weight,
backbone.layers.8.mixer.shared_experts.up_proj.biases,
backbone.layers.8.mixer.shared_experts.up_proj.scales,
backbone.layers.8.mixer.shared_experts.up_proj.weight,
backbone.layers.8.mixer.switch_mlp.fc1.biases,
backbone.layers.8.mixer.switch_mlp.fc1.scales,
backbone.layers.8.mixer.switch_mlp.fc1.weight,
backbone.layers.8.mixer.switch_mlp.fc2.biases,
backbone.layers.8.mixer.switch_mlp.fc2.scales,
backbone.layers.8.mixer.switch_mlp.fc2.weight,
backbone.layers.8.norm.weight,
backbone.layers.9.mixer.A_log,
backbone.layers.9.mixer.D,
backbone.layers.9.mixer.conv1d.bias,
backbone.layers.9.mixer.conv1d.weight,
backbone.layers.9.mixer.dt_bias,
backbone.layers.9.mixer.in_proj.biases,
backbone.layers.9.mixer.in_proj.scales,
backbone.layers.9.mixer.in_proj.weight,
backbone.layers.9.mixer.norm.weight,
backbone.layers.9.mixer.out_proj.biases,
backbone.layers.9.mixer.out_proj.scales,
backbone.layers.9.mixer.out_proj.weight,
backbone.layers.9.norm.weight,
backbone.norm_f.weight,
lm_head.biases,
lm_head.scales,
lm_head.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15585, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14472, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 729 parameters not in model: 
backbone.embeddings.biases,
backbone.embeddings.scales,
backbone.embeddings.weight,
backbone.layers.0.mixer.A_log,
backbone.layers.0.mixer.D,
backbone.layers.0.mixer.conv1d.bias,
backbone.layers.0.mixer.conv1d.weight,
backbone.layers.0.mixer.dt_bias,
backbone.layers.0.mixer.in_proj.biases,
backbone.layers.0.mixer.in_proj.scales,
backbone.layers.0.mixer.in_proj.weight,
backbone.layers.0.mixer.norm.weight,
backbone.layers.0.mixer.out_proj.biases,
backbone.layers.0.mixer.out_proj.scales,
backbone.layers.0.mixer.out_proj.weight,
backbone.layers.0.norm.weight,
backbone.layers.1.mixer.gate.e_score_correction_bias,
backbone.layers.1.mixer.gate.weight,
backbone.layers.1.mixer.shared_experts.down_proj.biases,
backbone.layers.1.mixer.shared_experts.down_proj.scales,
backbone.layers.1.mixer.shared_experts.down_proj.weight,
backbone.layers.1.mixer.shared_experts.up_proj.biases,
backbone.layers.1.mixer.shared_experts.up_proj.scales,
backbone.layers.1.mixer.shared_experts.up_proj.weight,
backbone.layers.1.mixer.switch_mlp.fc1.biases,
backbone.layers.1.mixer.switch_mlp.fc1.scales,
backbone.layers.1.mixer.switch_mlp.fc1.weight,
backbone.layers.1.mixer.switch_mlp.fc2.biases,
backbone.layers.1.mixer.switch_mlp.fc2.scales,
backbone.layers.1.mixer.switch_mlp.fc2.weight,
backbone.layers.1.norm.weight,
backbone.layers.10.mixer.gate.e_score_correction_bias,
backbone.layers.10.mixer.gate.weight,
backbone.layers.10.mixer.shared_experts.down_proj.biases,
backbone.layers.10.mixer.shared_experts.down_proj.scales,
backbone.layers.10.mixer.shared_experts.down_proj.weight,
backbone.layers.10.mixer.shared_experts.up_proj.biases,
backbone.layers.10.mixer.shared_experts.up_proj.scales,
backbone.layers.10.mixer.shared_experts.up_proj.weight,
backbone.layers.10.mixer.switch_mlp.fc1.biases,
backbone.layers.10.mixer.switch_mlp.fc1.scales,
backbone.layers.10.mixer.switch_mlp.fc1.weight,
backbone.layers.10.mixer.switch_mlp.fc2.biases,
backbone.layers.10.mixer.switch_mlp.fc2.scales,
backbone.layers.10.mixer.switch_mlp.fc2.weight,
backbone.layers.10.norm.weight,
backbone.layers.11.mixer.A_log,
backbone.layers.11.mixer.D,
backbone.layers.11.mixer.conv1d.bias,
backbone.layers.11.mixer.conv1d.weight,
backbone.layers.11.mixer.dt_bias,
backbone.layers.11.mixer.in_proj.biases,
backbone.layers.11.mixer.in_proj.scales,
backbone.layers.11.mixer.in_proj.weight,
backbone.layers.11.mixer.norm.weight,
backbone.layers.11.mixer.out_proj.biases,
backbone.layers.11.mixer.out_proj.scales,
backbone.layers.11.mixer.out_proj.weight,
backbone.layers.11.norm.weight,
backbone.layers.12.mixer.k_proj.biases,
backbone.layers.12.mixer.k_proj.scales,
backbone.layers.12.mixer.k_proj.weight,
backbone.layers.12.mixer.o_proj.biases,
backbone.layers.12.mixer.o_proj.scales,
backbone.layers.12.mixer.o_proj.weight,
backbone.layers.12.mixer.q_proj.biases,
backbone.layers.12.mixer.q_proj.scales,
backbone.layers.12.mixer.q_proj.weight,
backbone.layers.12.mixer.v_proj.biases,
backbone.layers.12.mixer.v_proj.scales,
backbone.layers.12.mixer.v_proj.weight,
backbone.layers.12.norm.weight,
backbone.layers.13.mixer.gate.e_score_correction_bias,
backbone.layers.13.mixer.gate.weight,
backbone.layers.13.mixer.shared_experts.down_proj.biases,
backbone.layers.13.mixer.shared_experts.down_proj.scales,
backbone.layers.13.mixer.shared_experts.down_proj.weight,
backbone.layers.13.mixer.shared_experts.up_proj.biases,
backbone.layers.13.mixer.shared_experts.up_proj.scales,
backbone.layers.13.mixer.shared_experts.up_proj.weight,
backbone.layers.13.mixer.switch_mlp.fc1.biases,
backbone.layers.13.mixer.switch_mlp.fc1.scales,
backbone.layers.13.mixer.switch_mlp.fc1.weight,
backbone.layers.13.mixer.switch_mlp.fc2.biases,
backbone.layers.13.mixer.switch_mlp.fc2.scales,
backbone.layers.13.mixer.switch_mlp.fc2.weight,
backbone.layers.13.norm.weight,
backbone.layers.14.mixer.A_log,
backbone.layers.14.mixer.D,
backbone.layers.14.mixer.conv1d.bias,
backbone.layers.14.mixer.conv1d.weight,
backbone.layers.14.mixer.dt_bias,
backbone.layers.14.mixer.in_proj.biases,
backbone.layers.14.mixer.in_proj.scales,
backbone.layers.14.mixer.in_proj.weight,
backbone.layers.14.mixer.norm.weight,
backbone.layers.14.mixer.out_proj.biases,
backbone.layers.14.mixer.out_proj.scales,
backbone.layers.14.mixer.out_proj.weight,
backbone.layers.14.norm.weight,
backbone.layers.15.mixer.gate.e_score_correction_bias,
backbone.layers.15.mixer.gate.weight,
backbone.layers.15.mixer.shared_experts.down_proj.biases,
backbone.layers.15.mixer.shared_experts.down_proj.scales,
backbone.layers.15.mixer.shared_experts.down_proj.weight,
backbone.layers.15.mixer.shared_experts.up_proj.biases,
backbone.layers.15.mixer.shared_experts.up_proj.scales,
backbone.layers.15.mixer.shared_experts.up_proj.weight,
backbone.layers.15.mixer.switch_mlp.fc1.biases,
backbone.layers.15.mixer.switch_mlp.fc1.scales,
backbone.layers.15.mixer.switch_mlp.fc1.weight,
backbone.layers.15.mixer.switch_mlp.fc2.biases,
backbone.layers.15.mixer.switch_mlp.fc2.scales,
backbone.layers.15.mixer.switch_mlp.fc2.weight,
backbone.layers.15.norm.weight,
backbone.layers.16.mixer.A_log,
backbone.layers.16.mixer.D,
backbone.layers.16.mixer.conv1d.bias,
backbone.layers.16.mixer.conv1d.weight,
backbone.layers.16.mixer.dt_bias,
backbone.layers.16.mixer.in_proj.biases,
backbone.layers.16.mixer.in_proj.scales,
backbone.layers.16.mixer.in_proj.weight,
backbone.layers.16.mixer.norm.weight,
backbone.layers.16.mixer.out_proj.biases,
backbone.layers.16.mixer.out_proj.scales,
backbone.layers.16.mixer.out_proj.weight,
backbone.layers.16.norm.weight,
backbone.layers.17.mixer.gate.e_score_correction_bias,
backbone.layers.17.mixer.gate.weight,
backbone.layers.17.mixer.shared_experts.down_proj.biases,
backbone.layers.17.mixer.shared_experts.down_proj.scales,
backbone.layers.17.mixer.shared_experts.down_proj.weight,
backbone.layers.17.mixer.shared_experts.up_proj.biases,
backbone.layers.17.mixer.shared_experts.up_proj.scales,
backbone.layers.17.mixer.shared_experts.up_proj.weight,
backbone.layers.17.mixer.switch_mlp.fc1.biases,
backbone.layers.17.mixer.switch_mlp.fc1.scales,
backbone.layers.17.mixer.switch_mlp.fc1.weight,
backbone.layers.17.mixer.switch_mlp.fc2.biases,
backbone.layers.17.mixer.switch_mlp.fc2.scales,
backbone.layers.17.mixer.switch_mlp.fc2.weight,
backbone.layers.17.norm.weight,
backbone.layers.18.mixer.A_log,
backbone.layers.18.mixer.D,
backbone.layers.18.mixer.conv1d.bias,
backbone.layers.18.mixer.conv1d.weight,
backbone.layers.18.mixer.dt_bias,
backbone.layers.18.mixer.in_proj.biases,
backbone.layers.18.mixer.in_proj.scales,
backbone.layers.18.mixer.in_proj.weight,
backbone.layers.18.mixer.norm.weight,
backbone.layers.18.mixer.out_proj.biases,
backbone.layers.18.mixer.out_proj.scales,
backbone.layers.18.mixer.out_proj.weight,
backbone.layers.18.norm.weight,
backbone.layers.19.mixer.k_proj.biases,
backbone.layers.19.mixer.k_proj.scales,
backbone.layers.19.mixer.k_proj.weight,
backbone.layers.19.mixer.o_proj.biases,
backbone.layers.19.mixer.o_proj.scales,
backbone.layers.19.mixer.o_proj.weight,
backbone.layers.19.mixer.q_proj.biases,
backbone.layers.19.mixer.q_proj.scales,
backbone.layers.19.mixer.q_proj.weight,
backbone.layers.19.mixer.v_proj.biases,
backbone.layers.19.mixer.v_proj.scales,
backbone.layers.19.mixer.v_proj.weight,
backbone.layers.19.norm.weight,
backbone.layers.2.mixer.A_log,
backbone.layers.2.mixer.D,
backbone.layers.2.mixer.conv1d.bias,
backbone.layers.2.mixer.conv1d.weight,
backbone.layers.2.mixer.dt_bias,
backbone.layers.2.mixer.in_proj.biases,
backbone.layers.2.mixer.in_proj.scales,
backbone.layers.2.mixer.in_proj.weight,
backbone.layers.2.mixer.norm.weight,
backbone.layers.2.mixer.out_proj.biases,
backbone.layers.2.mixer.out_proj.scales,
backbone.layers.2.mixer.out_proj.weight,
backbone.layers.2.norm.weight,
backbone.layers.20.mixer.gate.e_score_correction_bias,
backbone.layers.20.mixer.gate.weight,
backbone.layers.20.mixer.shared_experts.down_proj.biases,
backbone.layers.20.mixer.shared_experts.down_proj.scales,
backbone.layers.20.mixer.shared_experts.down_proj.weight,
backbone.layers.20.mixer.shared_experts.up_proj.biases,
backbone.layers.20.mixer.shared_experts.up_proj.scales,
backbone.layers.20.mixer.shared_experts.up_proj.weight,
backbone.layers.20.mixer.switch_mlp.fc1.biases,
backbone.layers.20.mixer.switch_mlp.fc1.scales,
backbone.layers.20.mixer.switch_mlp.fc1.weight,
backbone.layers.20.mixer.switch_mlp.fc2.biases,
backbone.layers.20.mixer.switch_mlp.fc2.scales,
backbone.layers.20.mixer.switch_mlp.fc2.weight,
backbone.layers.20.norm.weight,
backbone.layers.21.mixer.A_log,
backbone.layers.21.mixer.D,
backbone.layers.21.mixer.conv1d.bias,
backbone.layers.21.mixer.conv1d.weight,
backbone.layers.21.mixer.dt_bias,
backbone.layers.21.mixer.in_proj.biases,
backbone.layers.21.mixer.in_proj.scales,
backbone.layers.21.mixer.in_proj.weight,
backbone.layers.21.mixer.norm.weight,
backbone.layers.21.mixer.out_proj.biases,
backbone.layers.21.mixer.out_proj.scales,
backbone.layers.21.mixer.out_proj.weight,
backbone.layers.21.norm.weight,
backbone.layers.22.mixer.gate.e_score_correction_bias,
backbone.layers.22.mixer.gate.weight,
backbone.layers.22.mixer.shared_experts.down_proj.biases,
backbone.layers.22.mixer.shared_experts.down_proj.scales,
backbone.layers.22.mixer.shared_experts.down_proj.weight,
backbone.layers.22.mixer.shared_experts.up_proj.biases,
backbone.layers.22.mixer.shared_experts.up_proj.scales,
backbone.layers.22.mixer.shared_experts.up_proj.weight,
backbone.layers.22.mixer.switch_mlp.fc1.biases,
backbone.layers.22.mixer.switch_mlp.fc1.scales,
backbone.layers.22.mixer.switch_mlp.fc1.weight,
backbone.layers.22.mixer.switch_mlp.fc2.biases,
backbone.layers.22.mixer.switch_mlp.fc2.scales,
backbone.layers.22.mixer.switch_mlp.fc2.weight,
backbone.layers.22.norm.weight,
backbone.layers.23.mixer.A_log,
backbone.layers.23.mixer.D,
backbone.layers.23.mixer.conv1d.bias,
backbone.layers.23.mixer.conv1d.weight,
backbone.layers.23.mixer.dt_bias,
backbone.layers.23.mixer.in_proj.biases,
backbone.layers.23.mixer.in_proj.scales,
backbone.layers.23.mixer.in_proj.weight,
backbone.layers.23.mixer.norm.weight,
backbone.layers.23.mixer.out_proj.biases,
backbone.layers.23.mixer.out_proj.scales,
backbone.layers.23.mixer.out_proj.weight,
backbone.layers.23.norm.weight,
backbone.layers.24.mixer.gate.e_score_correction_bias,
backbone.layers.24.mixer.gate.weight,
backbone.layers.24.mixer.shared_experts.down_proj.biases,
backbone.layers.24.mixer.shared_experts.down_proj.scales,
backbone.layers.24.mixer.shared_experts.down_proj.weight,
backbone.layers.24.mixer.shared_experts.up_proj.biases,
backbone.layers.24.mixer.shared_experts.up_proj.scales,
backbone.layers.24.mixer.shared_experts.up_proj.weight,
backbone.layers.24.mixer.switch_mlp.fc1.biases,
backbone.layers.24.mixer.switch_mlp.fc1.scales,
backbone.layers.24.mixer.switch_mlp.fc1.weight,
backbone.layers.24.mixer.switch_mlp.fc2.biases,
backbone.layers.24.mixer.switch_mlp.fc2.scales,
backbone.layers.24.mixer.switch_mlp.fc2.weight,
backbone.layers.24.norm.weight,
backbone.layers.25.mixer.A_log,
backbone.layers.25.mixer.D,
backbone.layers.25.mixer.conv1d.bias,
backbone.layers.25.mixer.conv1d.weight,
backbone.layers.25.mixer.dt_bias,
backbone.layers.25.mixer.in_proj.biases,
backbone.layers.25.mixer.in_proj.scales,
backbone.layers.25.mixer.in_proj.weight,
backbone.layers.25.mixer.norm.weight,
backbone.layers.25.mixer.out_proj.biases,
backbone.layers.25.mixer.out_proj.scales,
backbone.layers.25.mixer.out_proj.weight,
backbone.layers.25.norm.weight,
backbone.layers.26.mixer.k_proj.biases,
backbone.layers.26.mixer.k_proj.scales,
backbone.layers.26.mixer.k_proj.weight,
backbone.layers.26.mixer.o_proj.biases,
backbone.layers.26.mixer.o_proj.scales,
backbone.layers.26.mixer.o_proj.weight,
backbone.layers.26.mixer.q_proj.biases,
backbone.layers.26.mixer.q_proj.scales,
backbone.layers.26.mixer.q_proj.weight,
backbone.layers.26.mixer.v_proj.biases,
backbone.layers.26.mixer.v_proj.scales,
backbone.layers.26.mixer.v_proj.weight,
backbone.layers.26.norm.weight,
backbone.layers.27.mixer.gate.e_score_correction_bias,
backbone.layers.27.mixer.gate.weight,
backbone.layers.27.mixer.shared_experts.down_proj.biases,
backbone.layers.27.mixer.shared_experts.down_proj.scales,
backbone.layers.27.mixer.shared_experts.down_proj.weight,
backbone.layers.27.mixer.shared_experts.up_proj.biases,
backbone.layers.27.mixer.shared_experts.up_proj.scales,
backbone.layers.27.mixer.shared_experts.up_proj.weight,
backbone.layers.27.mixer.switch_mlp.fc1.biases,
backbone.layers.27.mixer.switch_mlp.fc1.scales,
backbone.layers.27.mixer.switch_mlp.fc1.weight,
backbone.layers.27.mixer.switch_mlp.fc2.biases,
backbone.layers.27.mixer.switch_mlp.fc2.scales,
backbone.layers.27.mixer.switch_mlp.fc2.weight,
backbone.layers.27.norm.weight,
backbone.layers.28.mixer.A_log,
backbone.layers.28.mixer.D,
backbone.layers.28.mixer.conv1d.bias,
backbone.layers.28.mixer.conv1d.weight,
backbone.layers.28.mixer.dt_bias,
backbone.layers.28.mixer.in_proj.biases,
backbone.layers.28.mixer.in_proj.scales,
backbone.layers.28.mixer.in_proj.weight,
backbone.layers.28.mixer.norm.weight,
backbone.layers.28.mixer.out_proj.biases,
backbone.layers.28.mixer.out_proj.scales,
backbone.layers.28.mixer.out_proj.weight,
backbone.layers.28.norm.weight,
backbone.layers.29.mixer.gate.e_score_correction_bias,
backbone.layers.29.mixer.gate.weight,
backbone.layers.29.mixer.shared_experts.down_proj.biases,
backbone.layers.29.mixer.shared_experts.down_proj.scales,
backbone.layers.29.mixer.shared_experts.down_proj.weight,
backbone.layers.29.mixer.shared_experts.up_proj.biases,
backbone.layers.29.mixer.shared_experts.up_proj.scales,
backbone.layers.29.mixer.shared_experts.up_proj.weight,
backbone.layers.29.mixer.switch_mlp.fc1.biases,
backbone.layers.29.mixer.switch_mlp.fc1.scales,
backbone.layers.29.mixer.switch_mlp.fc1.weight,
backbone.layers.29.mixer.switch_mlp.fc2.biases,
backbone.layers.29.mixer.switch_mlp.fc2.scales,
backbone.layers.29.mixer.switch_mlp.fc2.weight,
backbone.layers.29.norm.weight,
backbone.layers.3.mixer.gate.e_score_correction_bias,
backbone.layers.3.mixer.gate.weight,
backbone.layers.3.mixer.shared_experts.down_proj.biases,
backbone.layers.3.mixer.shared_experts.down_proj.scales,
backbone.layers.3.mixer.shared_experts.down_proj.weight,
backbone.layers.3.mixer.shared_experts.up_proj.biases,
backbone.layers.3.mixer.shared_experts.up_proj.scales,
backbone.layers.3.mixer.shared_experts.up_proj.weight,
backbone.layers.3.mixer.switch_mlp.fc1.biases,
backbone.layers.3.mixer.switch_mlp.fc1.scales,
backbone.layers.3.mixer.switch_mlp.fc1.weight,
backbone.layers.3.mixer.switch_mlp.fc2.biases,
backbone.layers.3.mixer.switch_mlp.fc2.scales,
backbone.layers.3.mixer.switch_mlp.fc2.weight,
backbone.layers.3.norm.weight,
backbone.layers.30.mixer.A_log,
backbone.layers.30.mixer.D,
backbone.layers.30.mixer.conv1d.bias,
backbone.layers.30.mixer.conv1d.weight,
backbone.layers.30.mixer.dt_bias,
backbone.layers.30.mixer.in_proj.biases,
backbone.layers.30.mixer.in_proj.scales,
backbone.layers.30.mixer.in_proj.weight,
backbone.layers.30.mixer.norm.weight,
backbone.layers.30.mixer.out_proj.biases,
backbone.layers.30.mixer.out_proj.scales,
backbone.layers.30.mixer.out_proj.weight,
backbone.layers.30.norm.weight,
backbone.layers.31.mixer.gate.e_score_correction_bias,
backbone.layers.31.mixer.gate.weight,
backbone.layers.31.mixer.shared_experts.down_proj.biases,
backbone.layers.31.mixer.shared_experts.down_proj.scales,
backbone.layers.31.mixer.shared_experts.down_proj.weight,
backbone.layers.31.mixer.shared_experts.up_proj.biases,
backbone.layers.31.mixer.shared_experts.up_proj.scales,
backbone.layers.31.mixer.shared_experts.up_proj.weight,
backbone.layers.31.mixer.switch_mlp.fc1.biases,
backbone.layers.31.mixer.switch_mlp.fc1.scales,
backbone.layers.31.mixer.switch_mlp.fc1.weight,
backbone.layers.31.mixer.switch_mlp.fc2.biases,
backbone.layers.31.mixer.switch_mlp.fc2.scales,
backbone.layers.31.mixer.switch_mlp.fc2.weight,
backbone.layers.31.norm.weight,
backbone.layers.32.mixer.A_log,
backbone.layers.32.mixer.D,
backbone.layers.32.mixer.conv1d.bias,
backbone.layers.32.mixer.conv1d.weight,
backbone.layers.32.mixer.dt_bias,
backbone.layers.32.mixer.in_proj.biases,
backbone.layers.32.mixer.in_proj.scales,
backbone.layers.32.mixer.in_proj.weight,
backbone.layers.32.mixer.norm.weight,
backbone.layers.32.mixer.out_proj.biases,
backbone.layers.32.mixer.out_proj.scales,
backbone.layers.32.mixer.out_proj.weight,
backbone.layers.32.norm.weight,
backbone.layers.33.mixer.k_proj.biases,
backbone.layers.33.mixer.k_proj.scales,
backbone.layers.33.mixer.k_proj.weight,
backbone.layers.33.mixer.o_proj.biases,
backbone.layers.33.mixer.o_proj.scales,
backbone.layers.33.mixer.o_proj.weight,
backbone.layers.33.mixer.q_proj.biases,
backbone.layers.33.mixer.q_proj.scales,
backbone.layers.33.mixer.q_proj.weight,
backbone.layers.33.mixer.v_proj.biases,
backbone.layers.33.mixer.v_proj.scales,
backbone.layers.33.mixer.v_proj.weight,
backbone.layers.33.norm.weight,
backbone.layers.34.mixer.gate.e_score_correction_bias,
backbone.layers.34.mixer.gate.weight,
backbone.layers.34.mixer.shared_experts.down_proj.biases,
backbone.layers.34.mixer.shared_experts.down_proj.scales,
backbone.layers.34.mixer.shared_experts.down_proj.weight,
backbone.layers.34.mixer.shared_experts.up_proj.biases,
backbone.layers.34.mixer.shared_experts.up_proj.scales,
backbone.layers.34.mixer.shared_experts.up_proj.weight,
backbone.layers.34.mixer.switch_mlp.fc1.biases,
backbone.layers.34.mixer.switch_mlp.fc1.scales,
backbone.layers.34.mixer.switch_mlp.fc1.weight,
backbone.layers.34.mixer.switch_mlp.fc2.biases,
backbone.layers.34.mixer.switch_mlp.fc2.scales,
backbone.layers.34.mixer.switch_mlp.fc2.weight,
backbone.layers.34.norm.weight,
backbone.layers.35.mixer.A_log,
backbone.layers.35.mixer.D,
backbone.layers.35.mixer.conv1d.bias,
backbone.layers.35.mixer.conv1d.weight,
backbone.layers.35.mixer.dt_bias,
backbone.layers.35.mixer.in_proj.biases,
backbone.layers.35.mixer.in_proj.scales,
backbone.layers.35.mixer.in_proj.weight,
backbone.layers.35.mixer.norm.weight,
backbone.layers.35.mixer.out_proj.biases,
backbone.layers.35.mixer.out_proj.scales,
backbone.layers.35.mixer.out_proj.weight,
backbone.layers.35.norm.weight,
backbone.layers.36.mixer.gate.e_score_correction_bias,
backbone.layers.36.mixer.gate.weight,
backbone.layers.36.mixer.shared_experts.down_proj.biases,
backbone.layers.36.mixer.shared_experts.down_proj.scales,
backbone.layers.36.mixer.shared_experts.down_proj.weight,
backbone.layers.36.mixer.shared_experts.up_proj.biases,
backbone.layers.36.mixer.shared_experts.up_proj.scales,
backbone.layers.36.mixer.shared_experts.up_proj.weight,
backbone.layers.36.mixer.switch_mlp.fc1.biases,
backbone.layers.36.mixer.switch_mlp.fc1.scales,
backbone.layers.36.mixer.switch_mlp.fc1.weight,
backbone.layers.36.mixer.switch_mlp.fc2.biases,
backbone.layers.36.mixer.switch_mlp.fc2.scales,
backbone.layers.36.mixer.switch_mlp.fc2.weight,
backbone.layers.36.norm.weight,
backbone.layers.37.mixer.A_log,
backbone.layers.37.mixer.D,
backbone.layers.37.mixer.conv1d.bias,
backbone.layers.37.mixer.conv1d.weight,
backbone.layers.37.mixer.dt_bias,
backbone.layers.37.mixer.in_proj.biases,
backbone.layers.37.mixer.in_proj.scales,
backbone.layers.37.mixer.in_proj.weight,
backbone.layers.37.mixer.norm.weight,
backbone.layers.37.mixer.out_proj.biases,
backbone.layers.37.mixer.out_proj.scales,
backbone.layers.37.mixer.out_proj.weight,
backbone.layers.37.norm.weight,
backbone.layers.38.mixer.gate.e_score_correction_bias,
backbone.layers.38.mixer.gate.weight,
backbone.layers.38.mixer.shared_experts.down_proj.biases,
backbone.layers.38.mixer.shared_experts.down_proj.scales,
backbone.layers.38.mixer.shared_experts.down_proj.weight,
backbone.layers.38.mixer.shared_experts.up_proj.biases,
backbone.layers.38.mixer.shared_experts.up_proj.scales,
backbone.layers.38.mixer.shared_experts.up_proj.weight,
backbone.layers.38.mixer.switch_mlp.fc1.biases,
backbone.layers.38.mixer.switch_mlp.fc1.scales,
backbone.layers.38.mixer.switch_mlp.fc1.weight,
backbone.layers.38.mixer.switch_mlp.fc2.biases,
backbone.layers.38.mixer.switch_mlp.fc2.scales,
backbone.layers.38.mixer.switch_mlp.fc2.weight,
backbone.layers.38.norm.weight,
backbone.layers.39.mixer.A_log,
backbone.layers.39.mixer.D,
backbone.layers.39.mixer.conv1d.bias,
backbone.layers.39.mixer.conv1d.weight,
backbone.layers.39.mixer.dt_bias,
backbone.layers.39.mixer.in_proj.biases,
backbone.layers.39.mixer.in_proj.scales,
backbone.layers.39.mixer.in_proj.weight,
backbone.layers.39.mixer.norm.weight,
backbone.layers.39.mixer.out_proj.biases,
backbone.layers.39.mixer.out_proj.scales,
backbone.layers.39.mixer.out_proj.weight,
backbone.layers.39.norm.weight,
backbone.layers.4.mixer.A_log,
backbone.layers.4.mixer.D,
backbone.layers.4.mixer.conv1d.bias,
backbone.layers.4.mixer.conv1d.weight,
backbone.layers.4.mixer.dt_bias,
backbone.layers.4.mixer.in_proj.biases,
backbone.layers.4.mixer.in_proj.scales,
backbone.layers.4.mixer.in_proj.weight,
backbone.layers.4.mixer.norm.weight,
backbone.layers.4.mixer.out_proj.biases,
backbone.layers.4.mixer.out_proj.scales,
backbone.layers.4.mixer.out_proj.weight,
backbone.layers.4.norm.weight,
backbone.layers.40.mixer.gate.e_score_correction_bias,
backbone.layers.40.mixer.gate.weight,
backbone.layers.40.mixer.shared_experts.down_proj.biases,
backbone.layers.40.mixer.shared_experts.down_proj.scales,
backbone.layers.40.mixer.shared_experts.down_proj.weight,
backbone.layers.40.mixer.shared_experts.up_proj.biases,
backbone.layers.40.mixer.shared_experts.up_proj.scales,
backbone.layers.40.mixer.shared_experts.up_proj.weight,
backbone.layers.40.mixer.switch_mlp.fc1.biases,
backbone.layers.40.mixer.switch_mlp.fc1.scales,
backbone.layers.40.mixer.switch_mlp.fc1.weight,
backbone.layers.40.mixer.switch_mlp.fc2.biases,
backbone.layers.40.mixer.switch_mlp.fc2.scales,
backbone.layers.40.mixer.switch_mlp.fc2.weight,
backbone.layers.40.norm.weight,
backbone.layers.41.mixer.A_log,
backbone.layers.41.mixer.D,
backbone.layers.41.mixer.conv1d.bias,
backbone.layers.41.mixer.conv1d.weight,
backbone.layers.41.mixer.dt_bias,
backbone.layers.41.mixer.in_proj.biases,
backbone.layers.41.mixer.in_proj.scales,
backbone.layers.41.mixer.in_proj.weight,
backbone.layers.41.mixer.norm.weight,
backbone.layers.41.mixer.out_proj.biases,
backbone.layers.41.mixer.out_proj.scales,
backbone.layers.41.mixer.out_proj.weight,
backbone.layers.41.norm.weight,
backbone.layers.42.mixer.k_proj.biases,
backbone.layers.42.mixer.k_proj.scales,
backbone.layers.42.mixer.k_proj.weight,
backbone.layers.42.mixer.o_proj.biases,
backbone.layers.42.mixer.o_proj.scales,
backbone.layers.42.mixer.o_proj.weight,
backbone.layers.42.mixer.q_proj.biases,
backbone.layers.42.mixer.q_proj.scales,
backbone.layers.42.mixer.q_proj.weight,
backbone.layers.42.mixer.v_proj.biases,
backbone.layers.42.mixer.v_proj.scales,
backbone.layers.42.mixer.v_proj.weight,
backbone.layers.42.norm.weight,
backbone.layers.43.mixer.gate.e_score_correction_bias,
backbone.layers.43.mixer.gate.weight,
backbone.layers.43.mixer.shared_experts.down_proj.biases,
backbone.layers.43.mixer.shared_experts.down_proj.scales,
backbone.layers.43.mixer.shared_experts.down_proj.weight,
backbone.layers.43.mixer.shared_experts.up_proj.biases,
backbone.layers.43.mixer.shared_experts.up_proj.scales,
backbone.layers.43.mixer.shared_experts.up_proj.weight,
backbone.layers.43.mixer.switch_mlp.fc1.biases,
backbone.layers.43.mixer.switch_mlp.fc1.scales,
backbone.layers.43.mixer.switch_mlp.fc1.weight,
backbone.layers.43.mixer.switch_mlp.fc2.biases,
backbone.layers.43.mixer.switch_mlp.fc2.scales,
backbone.layers.43.mixer.switch_mlp.fc2.weight,
backbone.layers.43.norm.weight,
backbone.layers.44.mixer.A_log,
backbone.layers.44.mixer.D,
backbone.layers.44.mixer.conv1d.bias,
backbone.layers.44.mixer.conv1d.weight,
backbone.layers.44.mixer.dt_bias,
backbone.layers.44.mixer.in_proj.biases,
backbone.layers.44.mixer.in_proj.scales,
backbone.layers.44.mixer.in_proj.weight,
backbone.layers.44.mixer.norm.weight,
backbone.layers.44.mixer.out_proj.biases,
backbone.layers.44.mixer.out_proj.scales,
backbone.layers.44.mixer.out_proj.weight,
backbone.layers.44.norm.weight,
backbone.layers.45.mixer.gate.e_score_correction_bias,
backbone.layers.45.mixer.gate.weight,
backbone.layers.45.mixer.shared_experts.down_proj.biases,
backbone.layers.45.mixer.shared_experts.down_proj.scales,
backbone.layers.45.mixer.shared_experts.down_proj.weight,
backbone.layers.45.mixer.shared_experts.up_proj.biases,
backbone.layers.45.mixer.shared_experts.up_proj.scales,
backbone.layers.45.mixer.shared_experts.up_proj.weight,
backbone.layers.45.mixer.switch_mlp.fc1.biases,
backbone.layers.45.mixer.switch_mlp.fc1.scales,
backbone.layers.45.mixer.switch_mlp.fc1.weight,
backbone.layers.45.mixer.switch_mlp.fc2.biases,
backbone.layers.45.mixer.switch_mlp.fc2.scales,
backbone.layers.45.mixer.switch_mlp.fc2.weight,
backbone.layers.45.norm.weight,
backbone.layers.46.mixer.A_log,
backbone.layers.46.mixer.D,
backbone.layers.46.mixer.conv1d.bias,
backbone.layers.46.mixer.conv1d.weight,
backbone.layers.46.mixer.dt_bias,
backbone.layers.46.mixer.in_proj.biases,
backbone.layers.46.mixer.in_proj.scales,
backbone.layers.46.mixer.in_proj.weight,
backbone.layers.46.mixer.norm.weight,
backbone.layers.46.mixer.out_proj.biases,
backbone.layers.46.mixer.out_proj.scales,
backbone.layers.46.mixer.out_proj.weight,
backbone.layers.46.norm.weight,
backbone.layers.47.mixer.gate.e_score_correction_bias,
backbone.layers.47.mixer.gate.weight,
backbone.layers.47.mixer.shared_experts.down_proj.biases,
backbone.layers.47.mixer.shared_experts.down_proj.scales,
backbone.layers.47.mixer.shared_experts.down_proj.weight,
backbone.layers.47.mixer.shared_experts.up_proj.biases,
backbone.layers.47.mixer.shared_experts.up_proj.scales,
backbone.layers.47.mixer.shared_experts.up_proj.weight,
backbone.layers.47.mixer.switch_mlp.fc1.biases,
backbone.layers.47.mixer.switch_mlp.fc1.scales,
backbone.layers.47.mixer.switch_mlp.fc1.weight,
backbone.layers.47.mixer.switch_mlp.fc2.biases,
backbone.layers.47.mixer.switch_mlp.fc2.scales,
backbone.layers.47.mixer.switch_mlp.fc2.weight,
backbone.layers.47.norm.weight,
backbone.layers.48.mixer.A_log,
backbone.layers.48.mixer.D,
backbone.layers.48.mixer.conv1d.bias,
backbone.layers.48.mixer.conv1d.weight,
backbone.layers.48.mixer.dt_bias,
backbone.layers.48.mixer.in_proj.biases,
backbone.layers.48.mixer.in_proj.scales,
backbone.layers.48.mixer.in_proj.weight,
backbone.layers.48.mixer.norm.weight,
backbone.layers.48.mixer.out_proj.biases,
backbone.layers.48.mixer.out_proj.scales,
backbone.layers.48.mixer.out_proj.weight,
backbone.layers.48.norm.weight,
backbone.layers.49.mixer.gate.e_score_correction_bias,
backbone.layers.49.mixer.gate.weight,
backbone.layers.49.mixer.shared_experts.down_proj.biases,
backbone.layers.49.mixer.shared_experts.down_proj.scales,
backbone.layers.49.mixer.shared_experts.down_proj.weight,
backbone.layers.49.mixer.shared_experts.up_proj.biases,
backbone.layers.49.mixer.shared_experts.up_proj.scales,
backbone.layers.49.mixer.shared_experts.up_proj.weight,
backbone.layers.49.mixer.switch_mlp.fc1.biases,
backbone.layers.49.mixer.switch_mlp.fc1.scales,
backbone.layers.49.mixer.switch_mlp.fc1.weight,
backbone.layers.49.mixer.switch_mlp.fc2.biases,
backbone.layers.49.mixer.switch_mlp.fc2.scales,
backbone.layers.49.mixer.switch_mlp.fc2.weight,
backbone.layers.49.norm.weight,
backbone.layers.5.mixer.k_proj.biases,
backbone.layers.5.mixer.k_proj.scales,
backbone.layers.5.mixer.k_proj.weight,
backbone.layers.5.mixer.o_proj.biases,
backbone.layers.5.mixer.o_proj.scales,
backbone.layers.5.mixer.o_proj.weight,
backbone.layers.5.mixer.q_proj.biases,
backbone.layers.5.mixer.q_proj.scales,
backbone.layers.5.mixer.q_proj.weight,
backbone.layers.5.mixer.v_proj.biases,
backbone.layers.5.mixer.v_proj.scales,
backbone.layers.5.mixer.v_proj.weight,
backbone.layers.5.norm.weight,
backbone.layers.50.mixer.A_log,
backbone.layers.50.mixer.D,
backbone.layers.50.mixer.conv1d.bias,
backbone.layers.50.mixer.conv1d.weight,
backbone.layers.50.mixer.dt_bias,
backbone.layers.50.mixer.in_proj.biases,
backbone.layers.50.mixer.in_proj.scales,
backbone.layers.50.mixer.in_proj.weight,
backbone.layers.50.mixer.norm.weight,
backbone.layers.50.mixer.out_proj.biases,
backbone.layers.50.mixer.out_proj.scales,
backbone.layers.50.mixer.out_proj.weight,
backbone.layers.50.norm.weight,
backbone.layers.51.mixer.gate.e_score_correction_bias,
backbone.layers.51.mixer.gate.weight,
backbone.layers.51.mixer.shared_experts.down_proj.biases,
backbone.layers.51.mixer.shared_experts.down_proj.scales,
backbone.layers.51.mixer.shared_experts.down_proj.weight,
backbone.layers.51.mixer.shared_experts.up_proj.biases,
backbone.layers.51.mixer.shared_experts.up_proj.scales,
backbone.layers.51.mixer.shared_experts.up_proj.weight,
backbone.layers.51.mixer.switch_mlp.fc1.biases,
backbone.layers.51.mixer.switch_mlp.fc1.scales,
backbone.layers.51.mixer.switch_mlp.fc1.weight,
backbone.layers.51.mixer.switch_mlp.fc2.biases,
backbone.layers.51.mixer.switch_mlp.fc2.scales,
backbone.layers.51.mixer.switch_mlp.fc2.weight,
backbone.layers.51.norm.weight,
backbone.layers.6.mixer.gate.e_score_correction_bias,
backbone.layers.6.mixer.gate.weight,
backbone.layers.6.mixer.shared_experts.down_proj.biases,
backbone.layers.6.mixer.shared_experts.down_proj.scales,
backbone.layers.6.mixer.shared_experts.down_proj.weight,
backbone.layers.6.mixer.shared_experts.up_proj.biases,
backbone.layers.6.mixer.shared_experts.up_proj.scales,
backbone.layers.6.mixer.shared_experts.up_proj.weight,
backbone.layers.6.mixer.switch_mlp.fc1.biases,
backbone.layers.6.mixer.switch_mlp.fc1.scales,
backbone.layers.6.mixer.switch_mlp.fc1.weight,
backbone.layers.6.mixer.switch_mlp.fc2.biases,
backbone.layers.6.mixer.switch_mlp.fc2.scales,
backbone.layers.6.mixer.switch_mlp.fc2.weight,
backbone.layers.6.norm.weight,
backbone.layers.7.mixer.A_log,
backbone.layers.7.mixer.D,
backbone.layers.7.mixer.conv1d.bias,
backbone.layers.7.mixer.conv1d.weight,
backbone.layers.7.mixer.dt_bias,
backbone.layers.7.mixer.in_proj.biases,
backbone.layers.7.mixer.in_proj.scales,
backbone.layers.7.mixer.in_proj.weight,
backbone.layers.7.mixer.norm.weight,
backbone.layers.7.mixer.out_proj.biases,
backbone.layers.7.mixer.out_proj.scales,
backbone.layers.7.mixer.out_proj.weight,
backbone.layers.7.norm.weight,
backbone.layers.8.mixer.gate.e_score_correction_bias,
backbone.layers.8.mixer.gate.weight,
backbone.layers.8.mixer.shared_experts.down_proj.biases,
backbone.layers.8.mixer.shared_experts.down_proj.scales,
backbone.layers.8.mixer.shared_experts.down_proj.weight,
backbone.layers.8.mixer.shared_experts.up_proj.biases,
backbone.layers.8.mixer.shared_experts.up_proj.scales,
backbone.layers.8.mixer.shared_experts.up_proj.weight,
backbone.layers.8.mixer.switch_mlp.fc1.biases,
backbone.layers.8.mixer.switch_mlp.fc1.scales,
backbone.layers.8.mixer.switch_mlp.fc1.weight,
backbone.layers.8.mixer.switch_mlp.fc2.biases,
backbone.layers.8.mixer.switch_mlp.fc2.scales,
backbone.layers.8.mixer.switch_mlp.fc2.weight,
backbone.layers.8.norm.weight,
backbone.layers.9.mixer.A_log,
backbone.layers.9.mixer.D,
backbone.layers.9.mixer.conv1d.bias,
backbone.layers.9.mixer.conv1d.weight,
backbone.layers.9.mixer.dt_bias,
backbone.layers.9.mixer.in_proj.biases,
backbone.layers.9.mixer.in_proj.scales,
backbone.layers.9.mixer.in_proj.weight,
backbone.layers.9.mixer.norm.weight,
backbone.layers.9.mixer.out_proj.biases,
backbone.layers.9.mixer.out_proj.scales,
backbone.layers.9.mixer.out_proj.weight,
backbone.layers.9.norm.weight,
backbone.norm_f.weight,
lm_head.biases,
lm_head.scales,
lm_head.weight.

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Fetching 25 files:   0%|          | 0/25 [00:00<?, ?it/s]
Fetching 25 files: 100%|##########| 25/25 [00:00<00:00, 4387.35it/s]
```

## Completed Runs with Observations

<a id="diagnostic-liquidai-lfm25-vl-450m-mlx-bf16"></a>

<details>
<summary>LiquidAI/LFM2.5-VL-450M-MLX-bf16 — unusable — repeated text; stopped early: repeating; duplicate keywords; prompt hint repeated</summary>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords,
  prompt_hint_echoed
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Repeated fragment:* keyword: "water"
- *Fields repeating the prompt's hint:* ["description"]
- *Title word count:* 11
- *Keyword count:* 63
- *Duplicate keywords:* ["canal", "water", "architecture", "waterway",
  "aquatic bird", "waterfront"]
- *Resolved model revision:* ed71acdae0799c98cb450787d049d1343b65cd52
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.6015436669986229
- *Peak memory at first token (GB):* 1.939804848
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.000133656
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2112
- *Prompt composition:* 2,112 = 329 text/template + 1,783 image tokens (84%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 0.90
- *Parameter count:* 0.45B (name-estimate)
- *Declared context length:* 128,000 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (0.90 GB vs 0.90 GB on disk)
- *Generation tokens:* 200
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Title:
Swans glide across calm river waters in a serene canal setting.

Description:
A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.

Keywords:
Swans, canal, water, serene, leisure, boats, water, architecture, greenery, water, architecture, waterway, aquatic bird, waterfront, waterfront, waterway, aquatic bird, canal, water, architecture, water, waterway, waterway, water, aquatic bird, waterfront, water, waterway, waterway, water, water, waterway, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water, water
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-4bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-4bit — unusable — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Repeated fragment:* keyword: "riverside scenery"
- *Title word count:* 8
- *Keyword count:* 48
- *Duplicate keywords:* ["riverside", "riverside scenery", "riverside beauty",
  "riverside charm", "riverside tranquility", "riverside serenity", "riverside
  calmness", "riverside peacefulness"]
- *Resolved model revision:* 82f31be9840fa0d4c7e99257fe2e28b59a46df97
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.997180541002308
- *Peak memory at first token (GB):* 9.821530684
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.003558002
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 301
- *Prompt composition:* 301 = 300 text/template + 1 image tokens (0%; exact,
  counted by token id in the prepared input)
- *Checkpoint weights (GB):* 6.01
- *Parameter count:* 11.00B (name-estimate)
- *Quantization:* 4-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (6.02 GB vs 6.01 GB on disk)
- *Generation tokens:* 275
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Title: Serene River Scene with a Majestic White Swan

Description: A tranquil river scene unfolds with a solitary white swan gliding effortlessly across the calm waters, surrounded by lush foliage and the serene atmosphere of a riverside residential area, where leisure boats and cruisers are moored, creating a picturesque scene with a sense of peace and harmony.

Keywords: riverside residential area, leisure boats, calm waters, serene atmosphere, lush foliage, white swan, riverside, moored boats, peaceful scene, picturesque scene, tranquil scene, riverside scene, riverside life, riverside living, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside calmness, riverside peacefulness, riverside scenery, riverside beauty, riverside charm, riverside tranquility, riverside serenity, riverside
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — unusable — repeated text; cut off at token limit</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Repeated fragment:* phrase: "riverbank riverbank riverbank ..."
- *Title word count:* 8
- *Keyword count:* 120
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 2.200358417001553
- *Peak memory at first token (GB):* 14.975330028
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.00408229
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 301
- *Prompt composition:* 301 = 300 text/template + 1 image tokens (0%; exact,
  counted by token id in the prepared input)
- *Checkpoint weights (GB):* 11.34
- *Parameter count:* 11.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (11.35 GB vs 11.34 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Title: Serene River Scene with a Majestic White Swan

Description: A tranquil river scene unfolds on September 12, 2026, at 17:41:02 UTC+01:00, near 52.393850°N, 0.270830°E, featuring a solitary white swan gliding across the calm waters, surrounded by lush greenery and a backdrop of moored boats and riverside residential buildings.

Keywords: River, White Swan, Lush Greenery, Moored Boats, Riversid Residential Buildings, Tranquil Atmosphere, September Day, 2026, 17:41:02 UTC+01:00, 52.393850°N, 0.270830°E, Water Reflection, Waterfront, Waterway, Aquatic Bird, Architecture, Boat, Nature, Wildlife, Scenic View, Peaceful Scene, Calm Waters, Serene Environment, Outdoor Activity, Water Sports, Bird Watching, Wildlife Photography, Nature Photography, Scenic Photography, Landscape Photography, Riverbank, Riverbank Scene, Riverbank View, Riverbank Photography, Riverbank Landscape, Riverbank Nature, Riverbank Wildlife, Riverbank Scenic, Riverbank Peaceful, Riverbank Calm, Riverbank Serene, Riverbank Outdoor, Riverbank Water Sports, Riverbank Bird Watching, Riverbank Wildlife Photography, Riverbank Nature Photography, Riverbank Scenic Photography, Riverbank Landscape Photography, Riverbank Riverbank Scene, Riverbank Riverbank View, Riverbank Riverbank Photography, Riverbank Riverbank Landscape, Riverbank Riverbank Nature, Riverbank Riverbank Wildlife, Riverbank Riverbank Scenic, Riverbank Riverbank Peaceful, Riverbank Riverbank Calm, Riverbank Riverbank Serene, Riverbank Riverbank Outdoor, Riverbank Riverbank Water Sports, Riverbank Riverbank Bird Watching, Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Nature Photography, Riverbank Riverbank Scenic Photography, Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Landscape, Riverbank Riverbank Riverbank Nature, Riverbank Riverbank Riverbank Wildlife, Riverbank Riverbank Riverbank Scenic, Riverbank Riverbank Riverbank Peaceful, Riverbank Riverbank Riverbank Calm, Riverbank Riverbank Riverbank Serene, Riverbank Riverbank Riverbank Outdoor, Riverbank Riverbank Riverbank Water Sports, Riverbank Riverbank Riverbank Bird Watching, Riverbank Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Riverbank Nature Photography, Riverbank Riverbank Riverbank Scenic Photography, Riverbank Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Riverbank Landscape, Riverbank Riverbank Riverbank Riverbank Nature, Riverbank Riverbank Riverbank Riverbank Wildlife, Riverbank Riverbank Riverbank Riverbank Scenic, Riverbank Riverbank Riverbank Riverbank Peaceful, Riverbank Riverbank Riverbank Riverbank Calm, Riverbank Riverbank Riverbank Riverbank Serene, Riverbank Riverbank Riverbank Riverbank Outdoor, Riverbank Riverbank Riverbank Riverbank Water Sports, Riverbank Riverbank Riverbank Riverbank Bird Watching, Riverbank Riverbank Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Riverbank Riverbank Nature Photography, Riverbank Riverbank Riverbank Riverbank Scenic Photography, Riverbank Riverbank Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Landscape, Riverbank Riverbank Riverbank Riverbank Riverbank Nature, Riverbank Riverbank Riverbank Riverbank Riverbank Wildlife, Riverbank Riverbank Riverbank Riverbank Riverbank Scenic, Riverbank Riverbank Riverbank Riverbank Riverbank Peaceful, Riverbank Riverbank Riverbank Riverbank Riverbank Calm, Riverbank Riverbank Riverbank Riverbank Riverbank Serene, Riverbank Riverbank Riverbank Riverbank Riverbank Outdoor, Riverbank Riverbank Riverbank Riverbank Riverbank Water Sports, Riverbank Riverbank Riverbank Riverbank Riverbank Bird Watching, Riverbank Riverbank Riverbank Riverbank Riverbank Wildlife Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Nature Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Scenic Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Landscape Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank Scene, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank View, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank Photography, Riverbank Riverbank Riverbank Riverbank Riverbank Riverbank Landscape
```

</details>

<a id="diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit"></a>

<details>
<summary>mlx-community/llm-jp-4-vl-9b-mlx-4bit — unusable — repeated text; stopped early: repeating; control tokens visible; labelled fields not detected</summary>

### mlx-community/llm-jp-4-vl-9b-mlx-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort,
  missing_requested_sections, unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "a white swan swimming..."
- *Unexpected special tokens:* ["&lt;|channel|&gt;", "&lt;|message|&gt;"]
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Processor class:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.5114584160037339
- *Peak memory at first token (GB):* 6.72918174
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.015223696
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2195
- *Prompt composition:* 2,195 = 403 text/template + 1,792 image tokens (82%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.70 GB vs 5.68 GB on disk)
- *Generation tokens:* 200
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white swan swimming in a river with a white
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-30b-a3b-instruct-4bit"></a>

<details>
<summary>mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit — unusable — stopped early: repeating; duplicate keywords</summary>

### mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl_moe)
- *Title word count:* 9
- *Keyword count:* 57
- *Duplicate keywords:* ["wildlife", "animal", "bird"]
- *Resolved model revision:* 0555d34cb1ed80c0e61a5635194c70027b4c2ff3
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 35.70896762499615
- *Peak memory at first token (GB):* 23.303763298
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.7; top_p 0.8; top_k 20; repetition_penalty 1.0
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: generation_config
- *Post-cleanup active memory (GB):* 0.007932704
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16547
- *Prompt composition:* 16,547 = 323 text/template + 16,224 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 18.25
- *Parameter count:* 30.00B total, 3.00B active (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (18.26 GB vs 18.25 GB on disk)
- *Generation tokens:* 200
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Title: Solitary swan on a calm river with moored boats
Description: A white swan glides across the calm waters of a river, framed by green foliage in the foreground. In the background, a row of moored motorboats and cruisers are docked beside a residential building with balconies, under a soft, overcast sky.
Keywords: swan, river, boats, mooring, motorboat, residential, building, balcony, trees, foliage, greenery, water, reflection, calm, wildlife, waterfowl, aquatic bird, architecture, canal, waterfront, marina, pier, waterway, leisure, nature, outdoor, serene, tranquil, peaceful, dusk, evening, overcast, cloudy, landscape, scene, view, perspective, foreground, background, composition, natural, wildlife, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal,
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                         | Mechanical checks | Observations                             |
|-----------------------------------------------|-------------------|------------------------------------------|
| mlx-community/gemma-3n-E4B-it-4bit            | major concerns    | labelled fields not detected             |
| mlx-community/nanoLLaVA-1.5-4bit              | major concerns    | labelled fields not detected             |
| mlx-community/paligemma2-10b-mix-448-4bit     | major concerns    | labelled fields not detected             |
| mlx-community/SmolVLM-256M-Instruct-4bit      | major concerns    | labelled fields not detected             |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit  | major concerns    | cut off at token limit                   |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit     | major concerns    | cut off at token limit                   |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | duplicate keywords; prompt hint repeated |
| mlx-community/Molmo2-8B-4bit                  | concerns detected | duplicate keywords                       |
| mlx-community/GLM-4.6V-nvfp4                  | concerns detected | prompt hint repeated                     |
| mlx-community/pixtral-12b-8bit                | concerns detected | prompt hint repeated                     |
| mlx-community/Qwen2-VL-7B-Instruct-4bit       | concerns detected | prompt hint repeated                     |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx      | concerns detected | prompt hint repeated                     |
| mlx-community/gemma-3-27b-it-qat-4bit         | concerns detected | unsupplied place name                    |
| mlx-community/Qwen3.8-27B-4bit                | concerns detected | unsupplied place name                    |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                 | Runtime identity                                           | Performance                                           |
|-------------------------------------------------------|------------------------------------------------------------|-------------------------------------------------------|
| mlx-community/aya-vision-8b-4bit                      | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed | 2090 prompt / 96 generated; 100 tok/s; 6.5 GB peak    |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed        | 2393 prompt / 151 generated; 29.6 tok/s; 23 GB peak   |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit      | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed     | 1634 prompt / 997 generated; 77.8 tok/s; 19 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                     | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed   | 595 prompt / 101 generated; 60.4 tok/s; 7.6 GB peak   |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed          | 595 prompt / 96 generated; 108 tok/s; 16 GB peak      |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed          | 595 prompt / 89 generated; 26.3 tok/s; 20 GB peak     |
| mlx-community/gemma-4-e4b-it-4bit                     | rev 475b9088d297; Gemma4Processor; stop completed          | 591 prompt / 73 generated; 125 tok/s; 6.0 GB peak     |
| mlx-community/GLM-4.6V-Flash-4bit                     | rev bd7b20686e8c; Glm46VProcessor; stop completed          | 6450 prompt / 140 generated; 74.1 tok/s; 8.8 GB peak  |
| mlx-community/granite-4.0-3b-vision-4bit              | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed  | 1379 prompt / 65 generated; 176 tok/s; 4.7 GB peak    |
| mlx-community/Idefics3-8B-Llama3-bf16                 | rev 8c2a30c48864; Idefics3Processor; stop completed        | 2612 prompt / 138 generated; 35.5 tok/s; 18 GB peak   |
| mlx-community/InternVL3-14B-4bit                      | rev 26328eaab82c; InternVLChatProcessor; stop completed    | 2113 prompt / 126 generated; 57.4 tok/s; 10 GB peak   |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed    | 2113 prompt / 77 generated; 37.7 tok/s; 17 GB peak    |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                 | rev 7886c0b4a4b5; Lfm2VlProcessor; stop completed          | 2103 prompt / 81 generated; 214 tok/s; 4.0 GB peak    |
| mlx-community/MiniCPM-o-4_5-4bit                      | rev 592c09d85e7b; MiniCPMOProcessor; stop completed        | 391 prompt / 101 generated; 105 tok/s; 7.0 GB peak    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed        | 2926 prompt / 109 generated; 67.2 tok/s; 13 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed        | 2926 prompt / 187 generated; 62.3 tok/s; 13 GB peak   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed        | 2925 prompt / 102 generated; 191 tok/s; 7.8 GB peak   |
| mlx-community/North-Micro-Vision-Instruct-4bit        | rev 87466363e6c5; CohereCompassProcessor; stop completed   | 4083 prompt / 84 generated; 154 tok/s; 3.9 GB peak    |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit           | rev 4620fdbbd1e7; Qwen3VLProcessor; stop completed         | 1289 prompt / 155 generated; 71.8 tok/s; 24 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed           | 1133 prompt / 107 generated; 54.9 tok/s; 9.3 GB peak  |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | rev c325e5ea14c2; Qwen3VLProcessor; stop completed         | 16549 prompt / 912 generated; 86.3 tok/s; 8.4 GB peak |
| mlx-community/Qwen3-VL-32B-Instruct-4bit              | rev 6e5644d3ea4b; Qwen3VLProcessor; stop completed         | 16547 prompt / 177 generated; 20.4 tok/s; 26 GB peak  |
| mlx-community/Qwen3-VL-8B-Instruct-4bit               | rev defcdea7cc7a; Qwen3VLProcessor; stop completed         | 16547 prompt / 93 generated; 69.7 tok/s; 11 GB peak   |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed         | 16563 prompt / 97 generated; 72.4 tok/s; 25 GB peak   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed         | 16563 prompt / 86 generated; 91.4 tok/s; 11 GB peak   |
| mlx-community/Step-3.7-Flash-oQ3e                     | rev 41d17ee00e16; Step3VLProcessor; stop completed         | 3491 prompt / 121 generated; 47.4 tok/s; 92 GB peak   |
| mlx-community/X-Reasoner-7B-8bit                      | rev 21732e74613b; Qwen2_5_VLProcessor; stop completed      | 16558 prompt / 136 generated; 57.6 tok/s; 14 GB peak  |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 44,654,464 bytes
- *Image SHA-256:* c109700d51a838d36e8d56fc769534be3598a8af6310eed6d0afa19ecbe7dccf

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-57f7cdeb1b88952a.jpg>
- *Preview dimensions:* 1,024 x 683 pixels
- *Preview size:* 157,139 bytes
- *Preview SHA-256:* 57f7cdeb1b88952a3eeb6f5aa7ba4587981131b5841d7f8f82911704c242125b

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-57f7cdeb1b88952a.jpg
printf '%s\n' '57f7cdeb1b88952a3eeb6f5aa7ba4587981131b5841d7f8f82911704c242125b  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-12 17:41:02 UTC+01:00
- GPS: 52.393850°N, 0.270830°E

Descriptive hints:
- Description hint: A solitary white swan glides gracefully across calm river waters framed by lush foliage, with leisure boats and cruisers moored alongside riverside residential buildings in the background.
- Keyword hints: Adobe Stock, Any Vision, Bird, Canal, Greenery, Marina, Mooring, Motorboat, Pier, Riverbank, Swimming, Trees, Vegetation, Water reflection, Waterfowl, Waterfront, Waterway, aquatic bird, architecture, boat

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:' --max-tokens 1000 --temperature 0.0 --revision RESOLVED_REVISION --trust-remote-code --seed 0 --prefill-step-size 2048
```

### Highlighted model revisions

| Model                                                  | Resolved revision                        |
|--------------------------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-1B-4bit                      | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/Mage-VL-OptiQ-4bit                       | c98dad5f92f13334cc679c93fb9f185ab49ed626 |
| mlx-community/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-4bit | 8f86ad8f279ce1ec3b8b65970f32da1e89ab7a45 |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                       | ed71acdae0799c98cb450787d049d1343b65cd52 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit       | 82f31be9840fa0d4c7e99257fe2e28b59a46df97 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit       | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit                  | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit           | 0555d34cb1ed80c0e61a5635194c70027b4c2ff3 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.1                                                                                                                                           |
| mlx-vlm source revision    | e79b0e041677ec4ca5333ba750376bb4e8c434cb                                                                                                        |
| mlx                        | 0.32.3.dev20260912+229f5b430                                                                                                                    |
| mlx source revision        | 229f5b430                                                                                                                                       |
| mlx-audio                  | 0.5.4                                                                                                                                           |
| transformers               | 5.17.0                                                                                                                                          |
| tokenizers                 | 0.23.2                                                                                                                                          |
| huggingface-hub            | 1.32.0                                                                                                                                          |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 27.0.0                                                                                                                                   |
| macOS Version              | 27.0                                                                                                                                            |
| SDK Version                | 27.0                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk                                              |
| Xcode Version              | 27.0                                                                                                                                            |
| Xcode Build                | 27A266a                                                                                                                                         |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX27.0.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.921 (metalfe-32023.921.6)                                                                                             |
| Metallib Linker Version    | AIR-LLD 32023.921 (metalfe-32023.921.6) (compatible with legacy metallib linker)                                                                |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.3.34.2)                                                                                                  |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (189,439,120 bytes, sha256=3e940a8eda4eb7bc22bd545f335ba774b504258eb8817cc56b86d156a8b574df) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,488,288 bytes, sha256=ab77edac8a1ec78b4f4a24648706d4703110d3e401b64e74761dabb81bd7df83)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
