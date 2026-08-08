# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

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
| none                           | 25    |
| observation needs reproduction | 16    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 25    |
| usable              | 7     |
| usable with caveats | 9     |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 4     |
| Required fields are missing or empty                                                  | 9     |
| Response repeats the task instructions instead of only returning the requested fields | 7     |
| Extra text appears before the Title field                                             | 18    |
| Response appears cut off at the token limit                                           | 7     |
| Internal reasoning block appears incomplete                                           | 3     |
| Conversation-role control tokens remain visible                                       | 2     |
| Expected model wrapper tokens remain visible                                          | 8     |
| Title or keywords do not meet requested constraints                                   | 17    |

## Triage

| Model                                                                                                           | Execution | Usability           | Maintainer status              | Observations                                                                                                                      |
|-----------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/Inkling-Small-mlx-4bit](#diagnostic-mlx-community-inkling-small-mlx-4bit)                        | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                              |
| [LiquidAI/LFM2.5-VL-450M-MLX-bf16](#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16)                                 | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed | unusable            | observation_needs_reproduction | repeated text; echoes instructions; extra text before Title                                                                       |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed | unusable            | observation_needs_reproduction | repeated text; extra text before Title; cut off at token limit; wrapper tokens visible; title/keyword constraints failed          |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; echoes instructions; cut off at token limit                                               |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title; role tokens visible                                                             |
| [mlx-community/gemma-4-26b-a4b-it-4bit](#diagnostic-mlx-community-gemma-4-26b-a4b-it-4bit)                      | completed | unusable            | observation_needs_reproduction | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [mlx-community/MiniCPM-V-4.6-8bit](#diagnostic-mlx-community-minicpm-v-46-8bit)                                 | completed | unusable            | observation_needs_reproduction | echoes instructions; extra text before Title; wrapper tokens visible                                                              |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                   | completed | unusable            | observation_needs_reproduction | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                      |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed | unusable            | observation_needs_reproduction | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                      |
| [mlx-community/Ornith-1.0-35B-bf16](#diagnostic-mlx-community-ornith-10-35b-bf16)                               | completed | unusable            | observation_needs_reproduction | extra text before Title; wrapper tokens visible                                                                                   |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                  | completed | unusable            | observation_needs_reproduction | extra text before Title; wrapper tokens visible; title/keyword constraints failed                                                 |
| [mlx-community/Qwen3.5-35B-A3B-4bit](#diagnostic-mlx-community-qwen35-35b-a3b-4bit)                             | completed | unusable            | observation_needs_reproduction | extra text before Title; wrapper tokens visible                                                                                   |
| [mlx-community/Qwen3.5-9B-MLX-4bit](#diagnostic-mlx-community-qwen35-9b-mlx-4bit)                               | completed | unusable            | observation_needs_reproduction | extra text before Title; wrapper tokens visible                                                                                   |
| [mlx-community/Qwen3.6-27B-mxfp8](#diagnostic-mlx-community-qwen36-27b-mxfp8)                                   | completed | unusable            | observation_needs_reproduction | extra text before Title; wrapper tokens visible; title/keyword constraints failed                                                 |
| [mlx-community/Step-3.7-Flash-oQ2e](#diagnostic-mlx-community-step-37-flash-oq2e)                               | completed | unusable            | observation_needs_reproduction | extra text before Title; wrapper tokens visible                                                                                   |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                                                               |

## Crashes requiring action

<a id="diagnostic-mlx-community-inkling-small-mlx-4bit"></a>

### mlx-community/Inkling-Small-mlx-4bit

#### Root exception and chain

```text
builtins.ValueError: Received 362 parameters not in model; families: audio_tower, language_model; representative parameters: audio_tower.encoder.biases, audio_tower.encoder.scales, language_model.model.layers.10.mlp.experts.down_proj.biases.
builtins.ValueError: Model loading failed: Received 362 parameters not in model; families: audio_tower, language_model; representative parameters: audio_tower.encoder.biases, audio_tower.encoder.scales, language_model.model.layers.10.mlp.experts.down_proj.biases.
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type inkling_mm_model via
  inkling)
- *Phase:* model_load
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model loading failed: Received 362 parameters not in model;
  families: audio_tower, language_model; representative parameters:
  audio_tower.encoder.biases, audio_tower.encoder.scales,
  language_model.model.layers.10.mlp.experts.down_proj.biases.
- *Root error type:* ValueError
- *Root error message:* Received 362 parameters not in model; families:
  audio_tower, language_model; representative parameters:
  audio_tower.encoder.biases, audio_tower.encoder.scales,
  language_model.model.layers.10.mlp.experts.down_proj.biases.
- *Resolved model revision:* f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.00268806
- *Post-cleanup cache memory (GB):* 0.0

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11951, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11428, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 748, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 964, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 840, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 362 parameters not in model: 
audio_tower.encoder.biases,
audio_tower.encoder.scales,
language_model.model.layers.10.mlp.experts.down_proj.biases,
language_model.model.layers.10.mlp.experts.down_proj.scales,
language_model.model.layers.10.mlp.experts.down_proj.weight,
language_model.model.layers.10.mlp.experts.gate_proj.biases,
language_model.model.layers.10.mlp.experts.gate_proj.scales,
language_model.model.layers.10.mlp.experts.gate_proj.weight,
language_model.model.layers.10.mlp.experts.up_proj.biases,
language_model.model.layers.10.mlp.experts.up_proj.scales,
language_model.model.layers.10.mlp.experts.up_proj.weight,
language_model.model.layers.11.mlp.experts.down_proj.biases,
language_model.model.layers.11.mlp.experts.down_proj.scales,
language_model.model.layers.11.mlp.experts.down_proj.weight,
language_model.model.layers.11.mlp.experts.gate_proj.biases,
language_model.model.layers.11.mlp.experts.gate_proj.scales,
language_model.model.layers.11.mlp.experts.gate_proj.weight,
language_model.model.layers.11.mlp.experts.up_proj.biases,
language_model.model.layers.11.mlp.experts.up_proj.scales,
language_model.model.layers.11.mlp.experts.up_proj.weight,
language_model.model.layers.12.mlp.experts.down_proj.biases,
language_model.model.layers.12.mlp.experts.down_proj.scales,
language_model.model.layers.12.mlp.experts.down_proj.weight,
language_model.model.layers.12.mlp.experts.gate_proj.biases,
language_model.model.layers.12.mlp.experts.gate_proj.scales,
language_model.model.layers.12.mlp.experts.gate_proj.weight,
language_model.model.layers.12.mlp.experts.up_proj.biases,
language_model.model.layers.12.mlp.experts.up_proj.scales,
language_model.model.layers.12.mlp.experts.up_proj.weight,
language_model.model.layers.13.mlp.experts.down_proj.biases,
language_model.model.layers.13.mlp.experts.down_proj.scales,
language_model.model.layers.13.mlp.experts.down_proj.weight,
language_model.model.layers.13.mlp.experts.gate_proj.biases,
language_model.model.layers.13.mlp.experts.gate_proj.scales,
language_model.model.layers.13.mlp.experts.gate_proj.weight,
language_model.model.layers.13.mlp.experts.up_proj.biases,
language_model.model.layers.13.mlp.experts.up_proj.scales,
language_model.model.layers.13.mlp.experts.up_proj.weight,
language_model.model.layers.14.mlp.experts.down_proj.biases,
language_model.model.layers.14.mlp.experts.down_proj.scales,
language_model.model.layers.14.mlp.experts.down_proj.weight,
language_model.model.layers.14.mlp.experts.gate_proj.biases,
language_model.model.layers.14.mlp.experts.gate_proj.scales,
language_model.model.layers.14.mlp.experts.gate_proj.weight,
language_model.model.layers.14.mlp.experts.up_proj.biases,
language_model.model.layers.14.mlp.experts.up_proj.scales,
language_model.model.layers.14.mlp.experts.up_proj.weight,
language_model.model.layers.15.mlp.experts.down_proj.biases,
language_model.model.layers.15.mlp.experts.down_proj.scales,
language_model.model.layers.15.mlp.experts.down_proj.weight,
language_model.model.layers.15.mlp.experts.gate_proj.biases,
language_model.model.layers.15.mlp.experts.gate_proj.scales,
language_model.model.layers.15.mlp.experts.gate_proj.weight,
language_model.model.layers.15.mlp.experts.up_proj.biases,
language_model.model.layers.15.mlp.experts.up_proj.scales,
language_model.model.layers.15.mlp.experts.up_proj.weight,
language_model.model.layers.16.mlp.experts.down_proj.biases,
language_model.model.layers.16.mlp.experts.down_proj.scales,
language_model.model.layers.16.mlp.experts.down_proj.weight,
language_model.model.layers.16.mlp.experts.gate_proj.biases,
language_model.model.layers.16.mlp.experts.gate_proj.scales,
language_model.model.layers.16.mlp.experts.gate_proj.weight,
language_model.model.layers.16.mlp.experts.up_proj.biases,
language_model.model.layers.16.mlp.experts.up_proj.scales,
language_model.model.layers.16.mlp.experts.up_proj.weight,
language_model.model.layers.17.mlp.experts.down_proj.biases,
language_model.model.layers.17.mlp.experts.down_proj.scales,
language_model.model.layers.17.mlp.experts.down_proj.weight,
language_model.model.layers.17.mlp.experts.gate_proj.biases,
language_model.model.layers.17.mlp.experts.gate_proj.scales,
language_model.model.layers.17.mlp.experts.gate_proj.weight,
language_model.model.layers.17.mlp.experts.up_proj.biases,
language_model.model.layers.17.mlp.experts.up_proj.scales,
language_model.model.layers.17.mlp.experts.up_proj.weight,
language_model.model.layers.18.mlp.experts.down_proj.biases,
language_model.model.layers.18.mlp.experts.down_proj.scales,
language_model.model.layers.18.mlp.experts.down_proj.weight,
language_model.model.layers.18.mlp.experts.gate_proj.biases,
language_model.model.layers.18.mlp.experts.gate_proj.scales,
language_model.model.layers.18.mlp.experts.gate_proj.weight,
language_model.model.layers.18.mlp.experts.up_proj.biases,
language_model.model.layers.18.mlp.experts.up_proj.scales,
language_model.model.layers.18.mlp.experts.up_proj.weight,
language_model.model.layers.19.mlp.experts.down_proj.biases,
language_model.model.layers.19.mlp.experts.down_proj.scales,
language_model.model.layers.19.mlp.experts.down_proj.weight,
language_model.model.layers.19.mlp.experts.gate_proj.biases,
language_model.model.layers.19.mlp.experts.gate_proj.scales,
language_model.model.layers.19.mlp.experts.gate_proj.weight,
language_model.model.layers.19.mlp.experts.up_proj.biases,
language_model.model.layers.19.mlp.experts.up_proj.scales,
language_model.model.layers.19.mlp.experts.up_proj.weight,
language_model.model.layers.2.mlp.experts.down_proj.biases,
language_model.model.layers.2.mlp.experts.down_proj.scales,
language_model.model.layers.2.mlp.experts.down_proj.weight,
language_model.model.layers.2.mlp.experts.gate_proj.biases,
language_model.model.layers.2.mlp.experts.gate_proj.scales,
language_model.model.layers.2.mlp.experts.gate_proj.weight,
language_model.model.layers.2.mlp.experts.up_proj.biases,
language_model.model.layers.2.mlp.experts.up_proj.scales,
language_model.model.layers.2.mlp.experts.up_proj.weight,
language_model.model.layers.20.mlp.experts.down_proj.biases,
language_model.model.layers.20.mlp.experts.down_proj.scales,
language_model.model.layers.20.mlp.experts.down_proj.weight,
language_model.model.layers.20.mlp.experts.gate_proj.biases,
language_model.model.layers.20.mlp.experts.gate_proj.scales,
language_model.model.layers.20.mlp.experts.gate_proj.weight,
language_model.model.layers.20.mlp.experts.up_proj.biases,
language_model.model.layers.20.mlp.experts.up_proj.scales,
language_model.model.layers.20.mlp.experts.up_proj.weight,
language_model.model.layers.21.mlp.experts.down_proj.biases,
language_model.model.layers.21.mlp.experts.down_proj.scales,
language_model.model.layers.21.mlp.experts.down_proj.weight,
language_model.model.layers.21.mlp.experts.gate_proj.biases,
language_model.model.layers.21.mlp.experts.gate_proj.scales,
language_model.model.layers.21.mlp.experts.gate_proj.weight,
language_model.model.layers.21.mlp.experts.up_proj.biases,
language_model.model.layers.21.mlp.experts.up_proj.scales,
language_model.model.layers.21.mlp.experts.up_proj.weight,
language_model.model.layers.22.mlp.experts.down_proj.biases,
language_model.model.layers.22.mlp.experts.down_proj.scales,
language_model.model.layers.22.mlp.experts.down_proj.weight,
language_model.model.layers.22.mlp.experts.gate_proj.biases,
language_model.model.layers.22.mlp.experts.gate_proj.scales,
language_model.model.layers.22.mlp.experts.gate_proj.weight,
language_model.model.layers.22.mlp.experts.up_proj.biases,
language_model.model.layers.22.mlp.experts.up_proj.scales,
language_model.model.layers.22.mlp.experts.up_proj.weight,
language_model.model.layers.23.mlp.experts.down_proj.biases,
language_model.model.layers.23.mlp.experts.down_proj.scales,
language_model.model.layers.23.mlp.experts.down_proj.weight,
language_model.model.layers.23.mlp.experts.gate_proj.biases,
language_model.model.layers.23.mlp.experts.gate_proj.scales,
language_model.model.layers.23.mlp.experts.gate_proj.weight,
language_model.model.layers.23.mlp.experts.up_proj.biases,
language_model.model.layers.23.mlp.experts.up_proj.scales,
language_model.model.layers.23.mlp.experts.up_proj.weight,
language_model.model.layers.24.mlp.experts.down_proj.biases,
language_model.model.layers.24.mlp.experts.down_proj.scales,
language_model.model.layers.24.mlp.experts.down_proj.weight,
language_model.model.layers.24.mlp.experts.gate_proj.biases,
language_model.model.layers.24.mlp.experts.gate_proj.scales,
language_model.model.layers.24.mlp.experts.gate_proj.weight,
language_model.model.layers.24.mlp.experts.up_proj.biases,
language_model.model.layers.24.mlp.experts.up_proj.scales,
language_model.model.layers.24.mlp.experts.up_proj.weight,
language_model.model.layers.25.mlp.experts.down_proj.biases,
language_model.model.layers.25.mlp.experts.down_proj.scales,
language_model.model.layers.25.mlp.experts.down_proj.weight,
language_model.model.layers.25.mlp.experts.gate_proj.biases,
language_model.model.layers.25.mlp.experts.gate_proj.scales,
language_model.model.layers.25.mlp.experts.gate_proj.weight,
language_model.model.layers.25.mlp.experts.up_proj.biases,
language_model.model.layers.25.mlp.experts.up_proj.scales,
language_model.model.layers.25.mlp.experts.up_proj.weight,
language_model.model.layers.26.mlp.experts.down_proj.biases,
language_model.model.layers.26.mlp.experts.down_proj.scales,
language_model.model.layers.26.mlp.experts.down_proj.weight,
language_model.model.layers.26.mlp.experts.gate_proj.biases,
language_model.model.layers.26.mlp.experts.gate_proj.scales,
language_model.model.layers.26.mlp.experts.gate_proj.weight,
language_model.model.layers.26.mlp.experts.up_proj.biases,
language_model.model.layers.26.mlp.experts.up_proj.scales,
language_model.model.layers.26.mlp.experts.up_proj.weight,
language_model.model.layers.27.mlp.experts.down_proj.biases,
language_model.model.layers.27.mlp.experts.down_proj.scales,
language_model.model.layers.27.mlp.experts.down_proj.weight,
language_model.model.layers.27.mlp.experts.gate_proj.biases,
language_model.model.layers.27.mlp.experts.gate_proj.scales,
language_model.model.layers.27.mlp.experts.gate_proj.weight,
language_model.model.layers.27.mlp.experts.up_proj.biases,
language_model.model.layers.27.mlp.experts.up_proj.scales,
language_model.model.layers.27.mlp.experts.up_proj.weight,
language_model.model.layers.28.mlp.experts.down_proj.biases,
language_model.model.layers.28.mlp.experts.down_proj.scales,
language_model.model.layers.28.mlp.experts.down_proj.weight,
language_model.model.layers.28.mlp.experts.gate_proj.biases,
language_model.model.layers.28.mlp.experts.gate_proj.scales,
language_model.model.layers.28.mlp.experts.gate_proj.weight,
language_model.model.layers.28.mlp.experts.up_proj.biases,
language_model.model.layers.28.mlp.experts.up_proj.scales,
language_model.model.layers.28.mlp.experts.up_proj.weight,
language_model.model.layers.29.mlp.experts.down_proj.biases,
language_model.model.layers.29.mlp.experts.down_proj.scales,
language_model.model.layers.29.mlp.experts.down_proj.weight,
language_model.model.layers.29.mlp.experts.gate_proj.biases,
language_model.model.layers.29.mlp.experts.gate_proj.scales,
language_model.model.layers.29.mlp.experts.gate_proj.weight,
language_model.model.layers.29.mlp.experts.up_proj.biases,
language_model.model.layers.29.mlp.experts.up_proj.scales,
language_model.model.layers.29.mlp.experts.up_proj.weight,
language_model.model.layers.3.mlp.experts.down_proj.biases,
language_model.model.layers.3.mlp.experts.down_proj.scales,
language_model.model.layers.3.mlp.experts.down_proj.weight,
language_model.model.layers.3.mlp.experts.gate_proj.biases,
language_model.model.layers.3.mlp.experts.gate_proj.scales,
language_model.model.layers.3.mlp.experts.gate_proj.weight,
language_model.model.layers.3.mlp.experts.up_proj.biases,
language_model.model.layers.3.mlp.experts.up_proj.scales,
language_model.model.layers.3.mlp.experts.up_proj.weight,
language_model.model.layers.30.mlp.experts.down_proj.biases,
language_model.model.layers.30.mlp.experts.down_proj.scales,
language_model.model.layers.30.mlp.experts.down_proj.weight,
language_model.model.layers.30.mlp.experts.gate_proj.biases,
language_model.model.layers.30.mlp.experts.gate_proj.scales,
language_model.model.layers.30.mlp.experts.gate_proj.weight,
language_model.model.layers.30.mlp.experts.up_proj.biases,
language_model.model.layers.30.mlp.experts.up_proj.scales,
language_model.model.layers.30.mlp.experts.up_proj.weight,
language_model.model.layers.31.mlp.experts.down_proj.biases,
language_model.model.layers.31.mlp.experts.down_proj.scales,
language_model.model.layers.31.mlp.experts.down_proj.weight,
language_model.model.layers.31.mlp.experts.gate_proj.biases,
language_model.model.layers.31.mlp.experts.gate_proj.scales,
language_model.model.layers.31.mlp.experts.gate_proj.weight,
language_model.model.layers.31.mlp.experts.up_proj.biases,
language_model.model.layers.31.mlp.experts.up_proj.scales,
language_model.model.layers.31.mlp.experts.up_proj.weight,
language_model.model.layers.32.mlp.experts.down_proj.biases,
language_model.model.layers.32.mlp.experts.down_proj.scales,
language_model.model.layers.32.mlp.experts.down_proj.weight,
language_model.model.layers.32.mlp.experts.gate_proj.biases,
language_model.model.layers.32.mlp.experts.gate_proj.scales,
language_model.model.layers.32.mlp.experts.gate_proj.weight,
language_model.model.layers.32.mlp.experts.up_proj.biases,
language_model.model.layers.32.mlp.experts.up_proj.scales,
language_model.model.layers.32.mlp.experts.up_proj.weight,
language_model.model.layers.33.mlp.experts.down_proj.biases,
language_model.model.layers.33.mlp.experts.down_proj.scales,
language_model.model.layers.33.mlp.experts.down_proj.weight,
language_model.model.layers.33.mlp.experts.gate_proj.biases,
language_model.model.layers.33.mlp.experts.gate_proj.scales,
language_model.model.layers.33.mlp.experts.gate_proj.weight,
language_model.model.layers.33.mlp.experts.up_proj.biases,
language_model.model.layers.33.mlp.experts.up_proj.scales,
language_model.model.layers.33.mlp.experts.up_proj.weight,
language_model.model.layers.34.mlp.experts.down_proj.biases,
language_model.model.layers.34.mlp.experts.down_proj.scales,
language_model.model.layers.34.mlp.experts.down_proj.weight,
language_model.model.layers.34.mlp.experts.gate_proj.biases,
language_model.model.layers.34.mlp.experts.gate_proj.scales,
language_model.model.layers.34.mlp.experts.gate_proj.weight,
language_model.model.layers.34.mlp.experts.up_proj.biases,
language_model.model.layers.34.mlp.experts.up_proj.scales,
language_model.model.layers.34.mlp.experts.up_proj.weight,
language_model.model.layers.35.mlp.experts.down_proj.biases,
language_model.model.layers.35.mlp.experts.down_proj.scales,
language_model.model.layers.35.mlp.experts.down_proj.weight,
language_model.model.layers.35.mlp.experts.gate_proj.biases,
language_model.model.layers.35.mlp.experts.gate_proj.scales,
language_model.model.layers.35.mlp.experts.gate_proj.weight,
language_model.model.layers.35.mlp.experts.up_proj.biases,
language_model.model.layers.35.mlp.experts.up_proj.scales,
language_model.model.layers.35.mlp.experts.up_proj.weight,
language_model.model.layers.36.mlp.experts.down_proj.biases,
language_model.model.layers.36.mlp.experts.down_proj.scales,
language_model.model.layers.36.mlp.experts.down_proj.weight,
language_model.model.layers.36.mlp.experts.gate_proj.biases,
language_model.model.layers.36.mlp.experts.gate_proj.scales,
language_model.model.layers.36.mlp.experts.gate_proj.weight,
language_model.model.layers.36.mlp.experts.up_proj.biases,
language_model.model.layers.36.mlp.experts.up_proj.scales,
language_model.model.layers.36.mlp.experts.up_proj.weight,
language_model.model.layers.37.mlp.experts.down_proj.biases,
language_model.model.layers.37.mlp.experts.down_proj.scales,
language_model.model.layers.37.mlp.experts.down_proj.weight,
language_model.model.layers.37.mlp.experts.gate_proj.biases,
language_model.model.layers.37.mlp.experts.gate_proj.scales,
language_model.model.layers.37.mlp.experts.gate_proj.weight,
language_model.model.layers.37.mlp.experts.up_proj.biases,
language_model.model.layers.37.mlp.experts.up_proj.scales,
language_model.model.layers.37.mlp.experts.up_proj.weight,
language_model.model.layers.38.mlp.experts.down_proj.biases,
language_model.model.layers.38.mlp.experts.down_proj.scales,
language_model.model.layers.38.mlp.experts.down_proj.weight,
language_model.model.layers.38.mlp.experts.gate_proj.biases,
language_model.model.layers.38.mlp.experts.gate_proj.scales,
language_model.model.layers.38.mlp.experts.gate_proj.weight,
language_model.model.layers.38.mlp.experts.up_proj.biases,
language_model.model.layers.38.mlp.experts.up_proj.scales,
language_model.model.layers.38.mlp.experts.up_proj.weight,
language_model.model.layers.39.mlp.experts.down_proj.biases,
language_model.model.layers.39.mlp.experts.down_proj.scales,
language_model.model.layers.39.mlp.experts.down_proj.weight,
language_model.model.layers.39.mlp.experts.gate_proj.biases,
language_model.model.layers.39.mlp.experts.gate_proj.scales,
language_model.model.layers.39.mlp.experts.gate_proj.weight,
language_model.model.layers.39.mlp.experts.up_proj.biases,
language_model.model.layers.39.mlp.experts.up_proj.scales,
language_model.model.layers.39.mlp.experts.up_proj.weight,
language_model.model.layers.4.mlp.experts.down_proj.biases,
language_model.model.layers.4.mlp.experts.down_proj.scales,
language_model.model.layers.4.mlp.experts.down_proj.weight,
language_model.model.layers.4.mlp.experts.gate_proj.biases,
language_model.model.layers.4.mlp.experts.gate_proj.scales,
language_model.model.layers.4.mlp.experts.gate_proj.weight,
language_model.model.layers.4.mlp.experts.up_proj.biases,
language_model.model.layers.4.mlp.experts.up_proj.scales,
language_model.model.layers.4.mlp.experts.up_proj.weight,
language_model.model.layers.40.mlp.experts.down_proj.biases,
language_model.model.layers.40.mlp.experts.down_proj.scales,
language_model.model.layers.40.mlp.experts.down_proj.weight,
language_model.model.layers.40.mlp.experts.gate_proj.biases,
language_model.model.layers.40.mlp.experts.gate_proj.scales,
language_model.model.layers.40.mlp.experts.gate_proj.weight,
language_model.model.layers.40.mlp.experts.up_proj.biases,
language_model.model.layers.40.mlp.experts.up_proj.scales,
language_model.model.layers.40.mlp.experts.up_proj.weight,
language_model.model.layers.41.mlp.experts.down_proj.biases,
language_model.model.layers.41.mlp.experts.down_proj.scales,
language_model.model.layers.41.mlp.experts.down_proj.weight,
language_model.model.layers.41.mlp.experts.gate_proj.biases,
language_model.model.layers.41.mlp.experts.gate_proj.scales,
language_model.model.layers.41.mlp.experts.gate_proj.weight,
language_model.model.layers.41.mlp.experts.up_proj.biases,
language_model.model.layers.41.mlp.experts.up_proj.scales,
language_model.model.layers.41.mlp.experts.up_proj.weight,
language_model.model.layers.5.mlp.experts.down_proj.biases,
language_model.model.layers.5.mlp.experts.down_proj.scales,
language_model.model.layers.5.mlp.experts.down_proj.weight,
language_model.model.layers.5.mlp.experts.gate_proj.biases,
language_model.model.layers.5.mlp.experts.gate_proj.scales,
language_model.model.layers.5.mlp.experts.gate_proj.weight,
language_model.model.layers.5.mlp.experts.up_proj.biases,
language_model.model.layers.5.mlp.experts.up_proj.scales,
language_model.model.layers.5.mlp.experts.up_proj.weight,
language_model.model.layers.6.mlp.experts.down_proj.biases,
language_model.model.layers.6.mlp.experts.down_proj.scales,
language_model.model.layers.6.mlp.experts.down_proj.weight,
language_model.model.layers.6.mlp.experts.gate_proj.biases,
language_model.model.layers.6.mlp.experts.gate_proj.scales,
language_model.model.layers.6.mlp.experts.gate_proj.weight,
language_model.model.layers.6.mlp.experts.up_proj.biases,
language_model.model.layers.6.mlp.experts.up_proj.scales,
language_model.model.layers.6.mlp.experts.up_proj.weight,
language_model.model.layers.7.mlp.experts.down_proj.biases,
language_model.model.layers.7.mlp.experts.down_proj.scales,
language_model.model.layers.7.mlp.experts.down_proj.weight,
language_model.model.layers.7.mlp.experts.gate_proj.biases,
language_model.model.layers.7.mlp.experts.gate_proj.scales,
language_model.model.layers.7.mlp.experts.gate_proj.weight,
language_model.model.layers.7.mlp.experts.up_proj.biases,
language_model.model.layers.7.mlp.experts.up_proj.scales,
language_model.model.layers.7.mlp.experts.up_proj.weight,
language_model.model.layers.8.mlp.experts.down_proj.biases,
language_model.model.layers.8.mlp.experts.down_proj.scales,
language_model.model.layers.8.mlp.experts.down_proj.weight,
language_model.model.layers.8.mlp.experts.gate_proj.biases,
language_model.model.layers.8.mlp.experts.gate_proj.scales,
language_model.model.layers.8.mlp.experts.gate_proj.weight,
language_model.model.layers.8.mlp.experts.up_proj.biases,
language_model.model.layers.8.mlp.experts.up_proj.scales,
language_model.model.layers.8.mlp.experts.up_proj.weight,
language_model.model.layers.9.mlp.experts.down_proj.biases,
language_model.model.layers.9.mlp.experts.down_proj.scales,
language_model.model.layers.9.mlp.experts.down_proj.weight,
language_model.model.layers.9.mlp.experts.gate_proj.biases,
language_model.model.layers.9.mlp.experts.gate_proj.scales,
language_model.model.layers.9.mlp.experts.gate_proj.weight,
language_model.model.layers.9.mlp.experts.up_proj.biases,
language_model.model.layers.9.mlp.experts.up_proj.scales,
language_model.model.layers.9.mlp.experts.up_proj.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12381, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11966, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 362 parameters not in model: 
audio_tower.encoder.biases,
audio_tower.encoder.scales,
language_model.model.layers.10.mlp.experts.down_proj.biases,
language_model.model.layers.10.mlp.experts.down_proj.scales,
language_model.model.layers.10.mlp.experts.down_proj.weight,
language_model.model.layers.10.mlp.experts.gate_proj.biases,
language_model.model.layers.10.mlp.experts.gate_proj.scales,
language_model.model.layers.10.mlp.experts.gate_proj.weight,
language_model.model.layers.10.mlp.experts.up_proj.biases,
language_model.model.layers.10.mlp.experts.up_proj.scales,
language_model.model.layers.10.mlp.experts.up_proj.weight,
language_model.model.layers.11.mlp.experts.down_proj.biases,
language_model.model.layers.11.mlp.experts.down_proj.scales,
language_model.model.layers.11.mlp.experts.down_proj.weight,
language_model.model.layers.11.mlp.experts.gate_proj.biases,
language_model.model.layers.11.mlp.experts.gate_proj.scales,
language_model.model.layers.11.mlp.experts.gate_proj.weight,
language_model.model.layers.11.mlp.experts.up_proj.biases,
language_model.model.layers.11.mlp.experts.up_proj.scales,
language_model.model.layers.11.mlp.experts.up_proj.weight,
language_model.model.layers.12.mlp.experts.down_proj.biases,
language_model.model.layers.12.mlp.experts.down_proj.scales,
language_model.model.layers.12.mlp.experts.down_proj.weight,
language_model.model.layers.12.mlp.experts.gate_proj.biases,
language_model.model.layers.12.mlp.experts.gate_proj.scales,
language_model.model.layers.12.mlp.experts.gate_proj.weight,
language_model.model.layers.12.mlp.experts.up_proj.biases,
language_model.model.layers.12.mlp.experts.up_proj.scales,
language_model.model.layers.12.mlp.experts.up_proj.weight,
language_model.model.layers.13.mlp.experts.down_proj.biases,
language_model.model.layers.13.mlp.experts.down_proj.scales,
language_model.model.layers.13.mlp.experts.down_proj.weight,
language_model.model.layers.13.mlp.experts.gate_proj.biases,
language_model.model.layers.13.mlp.experts.gate_proj.scales,
language_model.model.layers.13.mlp.experts.gate_proj.weight,
language_model.model.layers.13.mlp.experts.up_proj.biases,
language_model.model.layers.13.mlp.experts.up_proj.scales,
language_model.model.layers.13.mlp.experts.up_proj.weight,
language_model.model.layers.14.mlp.experts.down_proj.biases,
language_model.model.layers.14.mlp.experts.down_proj.scales,
language_model.model.layers.14.mlp.experts.down_proj.weight,
language_model.model.layers.14.mlp.experts.gate_proj.biases,
language_model.model.layers.14.mlp.experts.gate_proj.scales,
language_model.model.layers.14.mlp.experts.gate_proj.weight,
language_model.model.layers.14.mlp.experts.up_proj.biases,
language_model.model.layers.14.mlp.experts.up_proj.scales,
language_model.model.layers.14.mlp.experts.up_proj.weight,
language_model.model.layers.15.mlp.experts.down_proj.biases,
language_model.model.layers.15.mlp.experts.down_proj.scales,
language_model.model.layers.15.mlp.experts.down_proj.weight,
language_model.model.layers.15.mlp.experts.gate_proj.biases,
language_model.model.layers.15.mlp.experts.gate_proj.scales,
language_model.model.layers.15.mlp.experts.gate_proj.weight,
language_model.model.layers.15.mlp.experts.up_proj.biases,
language_model.model.layers.15.mlp.experts.up_proj.scales,
language_model.model.layers.15.mlp.experts.up_proj.weight,
language_model.model.layers.16.mlp.experts.down_proj.biases,
language_model.model.layers.16.mlp.experts.down_proj.scales,
language_model.model.layers.16.mlp.experts.down_proj.weight,
language_model.model.layers.16.mlp.experts.gate_proj.biases,
language_model.model.layers.16.mlp.experts.gate_proj.scales,
language_model.model.layers.16.mlp.experts.gate_proj.weight,
language_model.model.layers.16.mlp.experts.up_proj.biases,
language_model.model.layers.16.mlp.experts.up_proj.scales,
language_model.model.layers.16.mlp.experts.up_proj.weight,
language_model.model.layers.17.mlp.experts.down_proj.biases,
language_model.model.layers.17.mlp.experts.down_proj.scales,
language_model.model.layers.17.mlp.experts.down_proj.weight,
language_model.model.layers.17.mlp.experts.gate_proj.biases,
language_model.model.layers.17.mlp.experts.gate_proj.scales,
language_model.model.layers.17.mlp.experts.gate_proj.weight,
language_model.model.layers.17.mlp.experts.up_proj.biases,
language_model.model.layers.17.mlp.experts.up_proj.scales,
language_model.model.layers.17.mlp.experts.up_proj.weight,
language_model.model.layers.18.mlp.experts.down_proj.biases,
language_model.model.layers.18.mlp.experts.down_proj.scales,
language_model.model.layers.18.mlp.experts.down_proj.weight,
language_model.model.layers.18.mlp.experts.gate_proj.biases,
language_model.model.layers.18.mlp.experts.gate_proj.scales,
language_model.model.layers.18.mlp.experts.gate_proj.weight,
language_model.model.layers.18.mlp.experts.up_proj.biases,
language_model.model.layers.18.mlp.experts.up_proj.scales,
language_model.model.layers.18.mlp.experts.up_proj.weight,
language_model.model.layers.19.mlp.experts.down_proj.biases,
language_model.model.layers.19.mlp.experts.down_proj.scales,
language_model.model.layers.19.mlp.experts.down_proj.weight,
language_model.model.layers.19.mlp.experts.gate_proj.biases,
language_model.model.layers.19.mlp.experts.gate_proj.scales,
language_model.model.layers.19.mlp.experts.gate_proj.weight,
language_model.model.layers.19.mlp.experts.up_proj.biases,
language_model.model.layers.19.mlp.experts.up_proj.scales,
language_model.model.layers.19.mlp.experts.up_proj.weight,
language_model.model.layers.2.mlp.experts.down_proj.biases,
language_model.model.layers.2.mlp.experts.down_proj.scales,
language_model.model.layers.2.mlp.experts.down_proj.weight,
language_model.model.layers.2.mlp.experts.gate_proj.biases,
language_model.model.layers.2.mlp.experts.gate_proj.scales,
language_model.model.layers.2.mlp.experts.gate_proj.weight,
language_model.model.layers.2.mlp.experts.up_proj.biases,
language_model.model.layers.2.mlp.experts.up_proj.scales,
language_model.model.layers.2.mlp.experts.up_proj.weight,
language_model.model.layers.20.mlp.experts.down_proj.biases,
language_model.model.layers.20.mlp.experts.down_proj.scales,
language_model.model.layers.20.mlp.experts.down_proj.weight,
language_model.model.layers.20.mlp.experts.gate_proj.biases,
language_model.model.layers.20.mlp.experts.gate_proj.scales,
language_model.model.layers.20.mlp.experts.gate_proj.weight,
language_model.model.layers.20.mlp.experts.up_proj.biases,
language_model.model.layers.20.mlp.experts.up_proj.scales,
language_model.model.layers.20.mlp.experts.up_proj.weight,
language_model.model.layers.21.mlp.experts.down_proj.biases,
language_model.model.layers.21.mlp.experts.down_proj.scales,
language_model.model.layers.21.mlp.experts.down_proj.weight,
language_model.model.layers.21.mlp.experts.gate_proj.biases,
language_model.model.layers.21.mlp.experts.gate_proj.scales,
language_model.model.layers.21.mlp.experts.gate_proj.weight,
language_model.model.layers.21.mlp.experts.up_proj.biases,
language_model.model.layers.21.mlp.experts.up_proj.scales,
language_model.model.layers.21.mlp.experts.up_proj.weight,
language_model.model.layers.22.mlp.experts.down_proj.biases,
language_model.model.layers.22.mlp.experts.down_proj.scales,
language_model.model.layers.22.mlp.experts.down_proj.weight,
language_model.model.layers.22.mlp.experts.gate_proj.biases,
language_model.model.layers.22.mlp.experts.gate_proj.scales,
language_model.model.layers.22.mlp.experts.gate_proj.weight,
language_model.model.layers.22.mlp.experts.up_proj.biases,
language_model.model.layers.22.mlp.experts.up_proj.scales,
language_model.model.layers.22.mlp.experts.up_proj.weight,
language_model.model.layers.23.mlp.experts.down_proj.biases,
language_model.model.layers.23.mlp.experts.down_proj.scales,
language_model.model.layers.23.mlp.experts.down_proj.weight,
language_model.model.layers.23.mlp.experts.gate_proj.biases,
language_model.model.layers.23.mlp.experts.gate_proj.scales,
language_model.model.layers.23.mlp.experts.gate_proj.weight,
language_model.model.layers.23.mlp.experts.up_proj.biases,
language_model.model.layers.23.mlp.experts.up_proj.scales,
language_model.model.layers.23.mlp.experts.up_proj.weight,
language_model.model.layers.24.mlp.experts.down_proj.biases,
language_model.model.layers.24.mlp.experts.down_proj.scales,
language_model.model.layers.24.mlp.experts.down_proj.weight,
language_model.model.layers.24.mlp.experts.gate_proj.biases,
language_model.model.layers.24.mlp.experts.gate_proj.scales,
language_model.model.layers.24.mlp.experts.gate_proj.weight,
language_model.model.layers.24.mlp.experts.up_proj.biases,
language_model.model.layers.24.mlp.experts.up_proj.scales,
language_model.model.layers.24.mlp.experts.up_proj.weight,
language_model.model.layers.25.mlp.experts.down_proj.biases,
language_model.model.layers.25.mlp.experts.down_proj.scales,
language_model.model.layers.25.mlp.experts.down_proj.weight,
language_model.model.layers.25.mlp.experts.gate_proj.biases,
language_model.model.layers.25.mlp.experts.gate_proj.scales,
language_model.model.layers.25.mlp.experts.gate_proj.weight,
language_model.model.layers.25.mlp.experts.up_proj.biases,
language_model.model.layers.25.mlp.experts.up_proj.scales,
language_model.model.layers.25.mlp.experts.up_proj.weight,
language_model.model.layers.26.mlp.experts.down_proj.biases,
language_model.model.layers.26.mlp.experts.down_proj.scales,
language_model.model.layers.26.mlp.experts.down_proj.weight,
language_model.model.layers.26.mlp.experts.gate_proj.biases,
language_model.model.layers.26.mlp.experts.gate_proj.scales,
language_model.model.layers.26.mlp.experts.gate_proj.weight,
language_model.model.layers.26.mlp.experts.up_proj.biases,
language_model.model.layers.26.mlp.experts.up_proj.scales,
language_model.model.layers.26.mlp.experts.up_proj.weight,
language_model.model.layers.27.mlp.experts.down_proj.biases,
language_model.model.layers.27.mlp.experts.down_proj.scales,
language_model.model.layers.27.mlp.experts.down_proj.weight,
language_model.model.layers.27.mlp.experts.gate_proj.biases,
language_model.model.layers.27.mlp.experts.gate_proj.scales,
language_model.model.layers.27.mlp.experts.gate_proj.weight,
language_model.model.layers.27.mlp.experts.up_proj.biases,
language_model.model.layers.27.mlp.experts.up_proj.scales,
language_model.model.layers.27.mlp.experts.up_proj.weight,
language_model.model.layers.28.mlp.experts.down_proj.biases,
language_model.model.layers.28.mlp.experts.down_proj.scales,
language_model.model.layers.28.mlp.experts.down_proj.weight,
language_model.model.layers.28.mlp.experts.gate_proj.biases,
language_model.model.layers.28.mlp.experts.gate_proj.scales,
language_model.model.layers.28.mlp.experts.gate_proj.weight,
language_model.model.layers.28.mlp.experts.up_proj.biases,
language_model.model.layers.28.mlp.experts.up_proj.scales,
language_model.model.layers.28.mlp.experts.up_proj.weight,
language_model.model.layers.29.mlp.experts.down_proj.biases,
language_model.model.layers.29.mlp.experts.down_proj.scales,
language_model.model.layers.29.mlp.experts.down_proj.weight,
language_model.model.layers.29.mlp.experts.gate_proj.biases,
language_model.model.layers.29.mlp.experts.gate_proj.scales,
language_model.model.layers.29.mlp.experts.gate_proj.weight,
language_model.model.layers.29.mlp.experts.up_proj.biases,
language_model.model.layers.29.mlp.experts.up_proj.scales,
language_model.model.layers.29.mlp.experts.up_proj.weight,
language_model.model.layers.3.mlp.experts.down_proj.biases,
language_model.model.layers.3.mlp.experts.down_proj.scales,
language_model.model.layers.3.mlp.experts.down_proj.weight,
language_model.model.layers.3.mlp.experts.gate_proj.biases,
language_model.model.layers.3.mlp.experts.gate_proj.scales,
language_model.model.layers.3.mlp.experts.gate_proj.weight,
language_model.model.layers.3.mlp.experts.up_proj.biases,
language_model.model.layers.3.mlp.experts.up_proj.scales,
language_model.model.layers.3.mlp.experts.up_proj.weight,
language_model.model.layers.30.mlp.experts.down_proj.biases,
language_model.model.layers.30.mlp.experts.down_proj.scales,
language_model.model.layers.30.mlp.experts.down_proj.weight,
language_model.model.layers.30.mlp.experts.gate_proj.biases,
language_model.model.layers.30.mlp.experts.gate_proj.scales,
language_model.model.layers.30.mlp.experts.gate_proj.weight,
language_model.model.layers.30.mlp.experts.up_proj.biases,
language_model.model.layers.30.mlp.experts.up_proj.scales,
language_model.model.layers.30.mlp.experts.up_proj.weight,
language_model.model.layers.31.mlp.experts.down_proj.biases,
language_model.model.layers.31.mlp.experts.down_proj.scales,
language_model.model.layers.31.mlp.experts.down_proj.weight,
language_model.model.layers.31.mlp.experts.gate_proj.biases,
language_model.model.layers.31.mlp.experts.gate_proj.scales,
language_model.model.layers.31.mlp.experts.gate_proj.weight,
language_model.model.layers.31.mlp.experts.up_proj.biases,
language_model.model.layers.31.mlp.experts.up_proj.scales,
language_model.model.layers.31.mlp.experts.up_proj.weight,
language_model.model.layers.32.mlp.experts.down_proj.biases,
language_model.model.layers.32.mlp.experts.down_proj.scales,
language_model.model.layers.32.mlp.experts.down_proj.weight,
language_model.model.layers.32.mlp.experts.gate_proj.biases,
language_model.model.layers.32.mlp.experts.gate_proj.scales,
language_model.model.layers.32.mlp.experts.gate_proj.weight,
language_model.model.layers.32.mlp.experts.up_proj.biases,
language_model.model.layers.32.mlp.experts.up_proj.scales,
language_model.model.layers.32.mlp.experts.up_proj.weight,
language_model.model.layers.33.mlp.experts.down_proj.biases,
language_model.model.layers.33.mlp.experts.down_proj.scales,
language_model.model.layers.33.mlp.experts.down_proj.weight,
language_model.model.layers.33.mlp.experts.gate_proj.biases,
language_model.model.layers.33.mlp.experts.gate_proj.scales,
language_model.model.layers.33.mlp.experts.gate_proj.weight,
language_model.model.layers.33.mlp.experts.up_proj.biases,
language_model.model.layers.33.mlp.experts.up_proj.scales,
language_model.model.layers.33.mlp.experts.up_proj.weight,
language_model.model.layers.34.mlp.experts.down_proj.biases,
language_model.model.layers.34.mlp.experts.down_proj.scales,
language_model.model.layers.34.mlp.experts.down_proj.weight,
language_model.model.layers.34.mlp.experts.gate_proj.biases,
language_model.model.layers.34.mlp.experts.gate_proj.scales,
language_model.model.layers.34.mlp.experts.gate_proj.weight,
language_model.model.layers.34.mlp.experts.up_proj.biases,
language_model.model.layers.34.mlp.experts.up_proj.scales,
language_model.model.layers.34.mlp.experts.up_proj.weight,
language_model.model.layers.35.mlp.experts.down_proj.biases,
language_model.model.layers.35.mlp.experts.down_proj.scales,
language_model.model.layers.35.mlp.experts.down_proj.weight,
language_model.model.layers.35.mlp.experts.gate_proj.biases,
language_model.model.layers.35.mlp.experts.gate_proj.scales,
language_model.model.layers.35.mlp.experts.gate_proj.weight,
language_model.model.layers.35.mlp.experts.up_proj.biases,
language_model.model.layers.35.mlp.experts.up_proj.scales,
language_model.model.layers.35.mlp.experts.up_proj.weight,
language_model.model.layers.36.mlp.experts.down_proj.biases,
language_model.model.layers.36.mlp.experts.down_proj.scales,
language_model.model.layers.36.mlp.experts.down_proj.weight,
language_model.model.layers.36.mlp.experts.gate_proj.biases,
language_model.model.layers.36.mlp.experts.gate_proj.scales,
language_model.model.layers.36.mlp.experts.gate_proj.weight,
language_model.model.layers.36.mlp.experts.up_proj.biases,
language_model.model.layers.36.mlp.experts.up_proj.scales,
language_model.model.layers.36.mlp.experts.up_proj.weight,
language_model.model.layers.37.mlp.experts.down_proj.biases,
language_model.model.layers.37.mlp.experts.down_proj.scales,
language_model.model.layers.37.mlp.experts.down_proj.weight,
language_model.model.layers.37.mlp.experts.gate_proj.biases,
language_model.model.layers.37.mlp.experts.gate_proj.scales,
language_model.model.layers.37.mlp.experts.gate_proj.weight,
language_model.model.layers.37.mlp.experts.up_proj.biases,
language_model.model.layers.37.mlp.experts.up_proj.scales,
language_model.model.layers.37.mlp.experts.up_proj.weight,
language_model.model.layers.38.mlp.experts.down_proj.biases,
language_model.model.layers.38.mlp.experts.down_proj.scales,
language_model.model.layers.38.mlp.experts.down_proj.weight,
language_model.model.layers.38.mlp.experts.gate_proj.biases,
language_model.model.layers.38.mlp.experts.gate_proj.scales,
language_model.model.layers.38.mlp.experts.gate_proj.weight,
language_model.model.layers.38.mlp.experts.up_proj.biases,
language_model.model.layers.38.mlp.experts.up_proj.scales,
language_model.model.layers.38.mlp.experts.up_proj.weight,
language_model.model.layers.39.mlp.experts.down_proj.biases,
language_model.model.layers.39.mlp.experts.down_proj.scales,
language_model.model.layers.39.mlp.experts.down_proj.weight,
language_model.model.layers.39.mlp.experts.gate_proj.biases,
language_model.model.layers.39.mlp.experts.gate_proj.scales,
language_model.model.layers.39.mlp.experts.gate_proj.weight,
language_model.model.layers.39.mlp.experts.up_proj.biases,
language_model.model.layers.39.mlp.experts.up_proj.scales,
language_model.model.layers.39.mlp.experts.up_proj.weight,
language_model.model.layers.4.mlp.experts.down_proj.biases,
language_model.model.layers.4.mlp.experts.down_proj.scales,
language_model.model.layers.4.mlp.experts.down_proj.weight,
language_model.model.layers.4.mlp.experts.gate_proj.biases,
language_model.model.layers.4.mlp.experts.gate_proj.scales,
language_model.model.layers.4.mlp.experts.gate_proj.weight,
language_model.model.layers.4.mlp.experts.up_proj.biases,
language_model.model.layers.4.mlp.experts.up_proj.scales,
language_model.model.layers.4.mlp.experts.up_proj.weight,
language_model.model.layers.40.mlp.experts.down_proj.biases,
language_model.model.layers.40.mlp.experts.down_proj.scales,
language_model.model.layers.40.mlp.experts.down_proj.weight,
language_model.model.layers.40.mlp.experts.gate_proj.biases,
language_model.model.layers.40.mlp.experts.gate_proj.scales,
language_model.model.layers.40.mlp.experts.gate_proj.weight,
language_model.model.layers.40.mlp.experts.up_proj.biases,
language_model.model.layers.40.mlp.experts.up_proj.scales,
language_model.model.layers.40.mlp.experts.up_proj.weight,
language_model.model.layers.41.mlp.experts.down_proj.biases,
language_model.model.layers.41.mlp.experts.down_proj.scales,
language_model.model.layers.41.mlp.experts.down_proj.weight,
language_model.model.layers.41.mlp.experts.gate_proj.biases,
language_model.model.layers.41.mlp.experts.gate_proj.scales,
language_model.model.layers.41.mlp.experts.gate_proj.weight,
language_model.model.layers.41.mlp.experts.up_proj.biases,
language_model.model.layers.41.mlp.experts.up_proj.scales,
language_model.model.layers.41.mlp.experts.up_proj.weight,
language_model.model.layers.5.mlp.experts.down_proj.biases,
language_model.model.layers.5.mlp.experts.down_proj.scales,
language_model.model.layers.5.mlp.experts.down_proj.weight,
language_model.model.layers.5.mlp.experts.gate_proj.biases,
language_model.model.layers.5.mlp.experts.gate_proj.scales,
language_model.model.layers.5.mlp.experts.gate_proj.weight,
language_model.model.layers.5.mlp.experts.up_proj.biases,
language_model.model.layers.5.mlp.experts.up_proj.scales,
language_model.model.layers.5.mlp.experts.up_proj.weight,
language_model.model.layers.6.mlp.experts.down_proj.biases,
language_model.model.layers.6.mlp.experts.down_proj.scales,
language_model.model.layers.6.mlp.experts.down_proj.weight,
language_model.model.layers.6.mlp.experts.gate_proj.biases,
language_model.model.layers.6.mlp.experts.gate_proj.scales,
language_model.model.layers.6.mlp.experts.gate_proj.weight,
language_model.model.layers.6.mlp.experts.up_proj.biases,
language_model.model.layers.6.mlp.experts.up_proj.scales,
language_model.model.layers.6.mlp.experts.up_proj.weight,
language_model.model.layers.7.mlp.experts.down_proj.biases,
language_model.model.layers.7.mlp.experts.down_proj.scales,
language_model.model.layers.7.mlp.experts.down_proj.weight,
language_model.model.layers.7.mlp.experts.gate_proj.biases,
language_model.model.layers.7.mlp.experts.gate_proj.scales,
language_model.model.layers.7.mlp.experts.gate_proj.weight,
language_model.model.layers.7.mlp.experts.up_proj.biases,
language_model.model.layers.7.mlp.experts.up_proj.scales,
language_model.model.layers.7.mlp.experts.up_proj.weight,
language_model.model.layers.8.mlp.experts.down_proj.biases,
language_model.model.layers.8.mlp.experts.down_proj.scales,
language_model.model.layers.8.mlp.experts.down_proj.weight,
language_model.model.layers.8.mlp.experts.gate_proj.biases,
language_model.model.layers.8.mlp.experts.gate_proj.scales,
language_model.model.layers.8.mlp.experts.gate_proj.weight,
language_model.model.layers.8.mlp.experts.up_proj.biases,
language_model.model.layers.8.mlp.experts.up_proj.scales,
language_model.model.layers.8.mlp.experts.up_proj.weight,
language_model.model.layers.9.mlp.experts.down_proj.biases,
language_model.model.layers.9.mlp.experts.down_proj.scales,
language_model.model.layers.9.mlp.experts.down_proj.weight,
language_model.model.layers.9.mlp.experts.gate_proj.biases,
language_model.model.layers.9.mlp.experts.gate_proj.scales,
language_model.model.layers.9.mlp.experts.gate_proj.weight,
language_model.model.layers.9.mlp.experts.up_proj.biases,
language_model.model.layers.9.mlp.experts.up_proj.scales,
language_model.model.layers.9.mlp.experts.up_proj.weight.

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 54 files:   0%|          | 0/54 [00:00<?, ?it/s]
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 4404.15it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[23:57:20] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

## Completed Runs with Observations

<a id="diagnostic-liquidai-lfm25-vl-450m-mlx-bf16"></a>

<details>
<summary>LiquidAI/LFM2.5-VL-450M-MLX-bf16 — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2-vl via lfm2_vl)
- *Repeated fragment:* Church,
- *Title word count:* 9
- *Requested title word range:* [5, 10]
- *Keyword count:* 469
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["church"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00013212
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 389
- *Generation tokens:* 1000
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
Title:
Akeman Street Baptist Church, Tring, England, UK, GBR, Europe

Description:
A brick-built church with a red door and arched windows stands in a serene town center setting, surrounded by lush greenery and a clear sky.

Keywords:
Akeman Street Baptist Church, Tring, England, UK, GBR, Europe, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church, Church
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-8bit — unusable — repeated text; echoes instructions; extra text before Title</summary>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, prompt_instruction_echo,
  unexpected_catalog_preamble
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Repeated fragment:* phrase: "akeman street baptist church,..."
- *Echoed instruction fragments:* ["title hint:", "description hint:",
  "keyword hints:"]
- *Unexpected text before Title:* thought I need to create British-English
  catalogue metadata for the provided image and context.  - Image capture
  date/time: 2026-08-08 15:43:55 UTC+01:00 - GPS: 51.815915°N, 0.638706°W -
  Title hint: Town centre, Tring, England, UK, GBR, Europe - Description hint:
  Akeman Street Baptist Church, Tring, Herts - Keyword hints: Adobe Stock,
  Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel,
  Chimney, Christian, Church, Clouds, England, Entrance, Europe,
  Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky  - The
  building is a red brick church. - It's a multi-story structure (likely a
  Baptist chapel). - Arched windows on the upper and lower floors. - A central
  arched doorway with dark red doors. - A sign on the left that says "Akeman
  Street Baptist Church". - There is a brick wall in front. - There are bushes
  and trees around the building. - The sky is cloudy with some sunlight. - The
  location is Tring, Hertfordshire (based on hints and GPS). Note: One hint
  says Buckinghamshire, but another says Herts. Tring is in Hertfordshire.
- *Thinking trace markers:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.010273988
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 598
- *Generation tokens:* 983
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
<|channel>thought
I need to create British-English catalogue metadata for the provided image and context.

- Image capture date/time: 2026-08-08 15:43:55 UTC+01:00
- GPS: 51.815915°N, 0.638706°W
- Title hint: Town centre, Tring, England, UK, GBR, Europe
- Description hint: Akeman Street Baptist Church, Tring, Herts
- Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

- The building is a red brick church.
- It's a multi-story structure (likely a Baptist chapel).
- Arched windows on the upper and lower floors.
- A central arched doorway with dark red doors.
- A sign on the left that says "Akeman Street Baptist Church".
- There is a brick wall in front.
- There are bushes and trees around the building.
- The sky is cloudy with some sunlight.
- The location is Tring, Hertfordshire (based on hints and GPS). Note: One hint says Buckinghamshire, but another says Herts. Tring is in Hertfordshire.

- Title: Akeman Street Baptist Church in Tring, Hertfordshire (8 words)
- Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
- Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre.

- Title: Akeman Street Baptist Church in Tring, Hertfordshire
- Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
- Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town

Wait, let me double-check the location. Tring is in Hertfordshire. The GPS 51.815915°N, 0.638706°W corresponds to Tring, Herts.

Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre

Let's refine the keywords to be 10-18.
1. Akeman Street Baptist Church
2. Tring
3. Hertfordshire
4. England
5. red brick
6. church
7. chapel
8. Christian
9. entrance
10. windows
11. clouds
12. roof
13. architecture
14. sign
15. bushes
16. town centre
Total 16 keywords.

Title: Akeman Street Baptist Church, Tring, Hertfordshire (6 words)
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky. (2 sentences)
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre

Wait, the prompt asks for "British-English". "Town centre" is correct.

Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel's... no.<channel|>Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central arched entrance under a cloudy sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, Christian, entrance, windows, clouds, roof, architecture, sign, bushes, town centre
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — repeated text; extra text before Title; cut off at token limit; wrapper tokens visible; title/keyword constraints failed</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  unexpected_catalog_preamble, configured_wrapper_present,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Repeated fragment:* keyword: "uk"
- *Unexpected text before Title:* Okay, let's break this down. I've got an
  image of a church and need to create some metadata for it. First, I need a
  title. The hints are a bit messy, but I see "Akeman Street Baptist Church"
  and "Tring, Herts" in there. That's a good start. I'll go with "Akeman
  Street Baptist Church, Tring, Herts."  Next, the description. I need to be
  factual and descriptive. The image shows a red brick church with a tower and
  a steeple. The sky is partly cloudy, and there's a stone wall and gate in
  the foreground. I'll combine the location from the hints with the visual
  details. I'll say: "A red brick Akeman Street Baptist Church in Tring,
  Hertfordshire, England, with a steeple, stone wall, and gate under a partly
  cloudy sky."  Finally, keywords. I need to be comprehensive. I'll pull from
  the hints and the image itself. I'll include the name of the church, the
  location, the architectural style, the materials, the sky, and some general
  terms. I'll make sure to include "Hertfordshire" and "England" and "UK" for
  the location. I'll also add "Tring" and "Baptist Church" for specificity.
  I'll also add "Red brick," "
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 223
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["tring", "hertfordshire", "england", "uk", "europe",
  "red brick", "steeple", "stone wall", "gate", "entrance", "sign", "sky",
  "clouds", "bushes", "herts", "baptist church", "church"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001180718
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1629
- *Generation tokens:* 1000
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
Okay, let's break this down. I've got an image of a church and need to create some metadata for it. First, I need a title. The hints are a bit messy, but I see "Akeman Street Baptist Church" and "Tring, Herts" in there. That's a good start. I'll go with "Akeman Street Baptist Church, Tring, Herts."

Next, the description. I need to be factual and descriptive. The image shows a red brick church with a tower and a steeple. The sky is partly cloudy, and there's a stone wall and gate in the foreground. I'll combine the location from the hints with the visual details. I'll say: "A red brick Akeman Street Baptist Church in Tring, Hertfordshire, England, with a steeple, stone wall, and gate under a partly cloudy sky."

Finally, keywords. I need to be comprehensive. I'll pull from the hints and the image itself. I'll include the name of the church, the location, the architectural style, the materials, the sky, and some general terms. I'll make sure to include "Hertfordshire" and "England" and "UK" for the location. I'll also add "Tring" and "Baptist Church" for specificity. I'll also add "Red brick," "
</think>

Title: Akeman Street Baptist Church, Tring, Herts

Description: A red brick Akeman Street Baptist Church in Tring, Hertfordshire, England, with a steeple, stone wall, and gate under a partly cloudy sky.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, GBR, Europe, Red brick, Roof, Steeple, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign, Sky, Clouds, Bushes, Hertfordshire, Tring, Herts, England, UK, UK, Europe, Baptist Church, Church, Steeple, Steeple, Red brick, Stone wall, Gate, Entrance, Sign
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated text; missing required fields; echoes instructions; cut off at token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, prompt_instruction_echo
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "adobe stock, adobe stock,..."
- *Echoed instruction fragments:* ["title hint:", "description hint:",
  "keyword hints:"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.0141243
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4416
- *Generation tokens:* 1000
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
- Title hint: Tring, England, UK, GBR, Europe
- Description hint: Tring, England, UK, GBR, Europe
- Keyword hints: Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe Stock, Adobe
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — missing required fields; extra text before Title; role tokens visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Missing sections:* ["title"]
- *Unexpected text before Title:* ◁think▷Okay, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the provided
  context and image. The capture date and GPS are authoritative, but I
  shouldn't claim they're visible. The title hint is "Town centre, Tring,
  England, UK, GBR, Europe" and the description hint is "Akeman Street Baptist
  Church, Tring, Herts". The keywords include Adobe Stock, Akeman Street
  Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney,
  Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire,
  Locations, Objects, Red brick, Roof, Sign, Sky.  Looking at the image, the
  main subject is the Akeman Street Baptist Church in Tring, a red-brick
  building with arched windows, a dark door, a sign, and surrounded by
  greenery. The sky is partly cloudy. The setting is a town centre in Tring,
  Hertfordshire, UK. The image is from Adobe Stock, so that's a keyword. The
  church is a chapel, has a chimney, and there's a sign. The lighting is
  bright, so "bright lighting" could be a detail. The keywords need to be
  unique and cover visible details. Let's structure the title first: "Akeman
  Street Baptist Church, Tring, Hertfordshire, UK". Then the description
  should mention the church, location, architectural details, surroundings,
  and lighting. Keywords should include all the hints and visible elements
  like red brick, chimney, sign, bushes, clouds, etc. Need to check for
  conflicts and ensure accuracy. Let's draft the title, description, and
  keywords.◁/think▷Title: Akeman Street Baptist Church, Tring, Hertfordshire,
  UK
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003654752
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1321
- *Generation tokens:* 463
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the provided context and image. The capture date and GPS are authoritative, but I shouldn't claim they're visible. The title hint is "Town centre, Tring, England, UK, GBR, Europe" and the description hint is "Akeman Street Baptist Church, Tring, Herts". The keywords include Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky.

Looking at the image, the main subject is the Akeman Street Baptist Church in Tring, a red-brick building with arched windows, a dark door, a sign, and surrounded by greenery. The sky is partly cloudy. The setting is a town centre in Tring, Hertfordshire, UK. The image is from Adobe Stock, so that's a keyword. The church is a chapel, has a chimney, and there's a sign. The lighting is bright, so "bright lighting" could be a detail. The keywords need to be unique and cover visible details. Let's structure the title first: "Akeman Street Baptist Church, Tring, Hertfordshire, UK". Then the description should mention the church, location, architectural details, surroundings, and lighting. Keywords should include all the hints and visible elements like red brick, chimney, sign, bushes, clouds, etc. Need to check for conflicts and ensure accuracy. Let's draft the title, description, and keywords.◁/think▷Title: Akeman Street Baptist Church, Tring, Hertfordshire, UK  
Description: Akeman Street Baptist Church in Tring, Hertfordshire, UK, a red-brick chapel with arched windows, a dark wooden door, and a sign, set against a partly cloudy sky with lush bushes and adjacent buildings in a town centre.  
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, UK, Red brick, Chapel, Christian, Clouds, Bushes, Chimney, Church, Entrance, England, Europe, Locations, Objects, Sky, Bright lighting, Adobe Stock<|im_assistant|>
```

</details>

<a id="diagnostic-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

<details>
<summary>mlx-community/gemma-4-26b-a4b-it-4bit — unusable — echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/gemma-4-26b-a4b-it-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, prompt_instruction_echo,
  unexpected_catalog_preamble, thinking_trace_incomplete,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type gemma4)
- *Echoed instruction fragments:* ["title hint:", "description hint:"]
- *Unexpected text before Title:* thought * Capture date/time: 2026-08-08
  15:43:55 UTC+01:00 * GPS: 51.815915°N, 0.638706°W * Title hint: Town centre,
  Tring, England, UK, GBR, Europe * Description hint: Akeman Street Baptist
  Church, Tring, Herts
- *Thinking trace markers:* ["&lt;|channel&gt;thought"]
- *Title word count:* 10
- *Requested title word range:* [5, 10]
- *Keyword count:* 36
- *Requested keyword count range:* [10, 18]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace"]
- *Resolved model revision:* 0d77464eeb233a2da68ebf9d7dc4edaac7db956d
- *Processor class:* mlx_vlm.models.gemma4.processing_gemma4.Gemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.011879668
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 598
- *Generation tokens:* 1000
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
<|channel>thought
*   Capture date/time: 2026-08-08 15:43:55 UTC+01:00
    *   GPS: 51.815915°N, 0.638706°W
    *   Title hint: Town centre, Tring, England, UK, GBR, Europe
    *   Description hint: Akeman Street Baptist Church, Tring, Herts
    *   Keywords: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

    *   Subject: A two-story red brick building with arched windows.
    *   Signage: A sign on the left says "Akeman Street Baptist Church".
     least one sign on the building itself (small plaque above the door).
    *   Architecture: Red brick, arched windows (three on top, two on bottom flanking the door), central arched doorway with dark red doors, hipped roof with a small finial/vent.
    *   Setting: Street level, stone/brick wall in front, some greenery/bushes, cloudy sky.
    *   Location: Tring, Hertfordshire (based on description hint and GPS).
    *   Lighting: Daylight, slightly overcast/cloudy sky.

    *   *Title:* Akeman Street Baptist Church in Tring, England. (7 words) - Good.
    *   *Description:* A two-story red brick building housing the Akeman Street Baptist Church in Tring, Hertfordshire, under a cloudy sky. The structure features arched windows and a central dark red door.
    *   *Keywords:* Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, street level, cloudy sky, brick wall, entrance.

    *   *Title:* Akeman Street Baptist Church in Tring, England
    *   *Description:* The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky.
    *   *Keywords:* Akeman Street Baptist Church, Tolog, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, brick wall, entrance, sky, clouds, street level.

    *Wait, let's refine the title and description to be more professional.*

    *Title:* Akeman Street Baptist Church in Tring, England
    *Description:* The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky.
    *Keywords:* Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, architecture, church, chapel, arched windows, historic building, Europe, UK, brick wall, entrance, sky, clouds, street level.

    *Wait, the prompt says "a concrete 5-10-word title".*
    "Akeman Street Baptist Church in Tring, England" is 7 words. Perfect.

    *Description: 1-2 sentences.*
    "The red brick Akeman Street Baptist Church in Tring, Hertfordshire, features arched windows and a central dark red door under a cloudy sky." (21 words). Perfect.

    *Keywords: 10-18 unique, comma-separated.*
    1. Akeman Street Baptist Church
    2. Tring
    3. Hertfordshire
    4. England
    5. red brick
    6. architecture
    7. church
    8. chapel
    9. arched windows
    10. historic building
    11. Europe
            12. UK
    13. brick wall
    14. entrance
    15. sky
    16. clouds
    17. street level

    *Let's check the keywords against the hints.*
    Hints: Adobe Stock (omit), Akeman Street Baptist Church (keep), Any Vision (omit), Buckinghamshire (the hint says Tring, Herts, so Buckinghamshire is likely a mistake in the hint or refers to a different context, I will stick to Herts/Tring), Bushes (can add), Chapel (keep), Chimney (not visible), Christian (can add), Church (keep), Clouds (keep), England (keep), Entrance (keep), Europe (keep), Hertfordshire (keep), Locations (omit), Objects (omit), Red brick (keep), Roof (can add),
```

</details>

<a id="diagnostic-mlx-community-minicpm-v-46-8bit"></a>

<details>
<summary>mlx-community/MiniCPM-V-4.6-8bit — unusable — echoes instructions; extra text before Title; wrapper tokens visible</summary>

### mlx-community/MiniCPM-V-4.6-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* prompt_instruction_echo, unexpected_catalog_preamble,
  configured_wrapper_present
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Echoed instruction fragments:* ["a 1 2 sentence factual description
  combining relevant"]
- *Unexpected text before Title:* We need to create the catalogue metadata.
  Let's break down the required sections:  1. Title: needs to be a concrete
  5-10 word title. 2. Description: needs to be a 1-2 sentence factual
  description combining relevant context with main visible subject, setting,
  action, lighting, and distinctive details. 3. Keywords: 10-18 unique,
  covering relevant context and visible details.  First, extract the key
  elements from the image:  - The building is Akeman Street Baptist Church,
  Tring, England, UK, Europe. - The building is a brick church with arched
  windows and a red door. - There are trees and bushes, a sign for Akeman
  Street Baptist Church. - The sky is partly cloudy. - The setting is a town
  center, with a brick wall and gate.  Now, craft the title: need 5-10 words.
  Something like "Akeman Street Baptist Church, Tring, England". But need to
  make it concrete and descriptive.
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00481803
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 930
- *Generation tokens:* 412
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
We need to create the catalogue metadata. Let's break down the required sections:

1. Title: needs to be a concrete 5-10 word title.
2. Description: needs to be a 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, and distinctive details.
3. Keywords: 10-18 unique, covering relevant context and visible details.

First, extract the key elements from the image:

- The building is Akeman Street Baptist Church, Tring, England, UK, Europe.
- The building is a brick church with arched windows and a red door.
- There are trees and bushes, a sign for Akeman Street Baptist Church.
- The sky is partly cloudy.
- The setting is a town center, with a brick wall and gate.

Now, craft the title: need 5-10 words. Something like "Akeman Street Baptist Church, Tring, England". But need to make it concrete and descriptive.

Description: Combine the context and visible details. The church is in a town, with brick architecture, surrounded by greenery, under partly cloudy skies.

Keywords: need 10-18 unique, covering relevant context and visible details. Let's list: Akeman Street, Baptist Church, Tring, England, UK, Europe, brick, red door, arched windows, trees, bushes, sign, sky
</think>

Title:
Akeman Street Baptist Church, Tring, England

Description:
Akeman Street Baptist Church is a brick church in Tring, featuring arched windows and a red door, set amidst greenery under partly cloudy skies.

Keywords: Akeman Street, Baptist Church, Tring, England, UK, Europe, brick, red door, arched windows, trees, bushes, sign, sky, Christian, church, entrance, Herts, Buckinghamshire, England, Europe, Tring, architecture, greenery
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
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this.
  First, the title needs to be 5-10 words. The main subject is the Akeman
  Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring". Then
  the description should combine context (date, location) with the main
  subject, setting, etc. Let's check the image: red brick building, entrance
  with steps, sign, greenery. So description should mention the church, its
  location, the setting, maybe the time of day (since it's sunny). Then
  keywords need to cover the hints and visible details. Let's make sure each
  part is correct.
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 39
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["hertfordshire", "red brick", "chapel", "entrance",
  "clouds", "church", "england", "sign", "bushes"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001803312
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6446
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is the Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring". Then the description should combine context (date, location) with the main subject, setting, etc. Let's check the image: red brick building, entrance with steps, sign, greenery. So description should mention the church, its location, the setting, maybe the time of day (since it's sunny). Then keywords need to cover the hints and visible details. Let's make sure each part is correct.

Title: "Akeman Street Baptist Church, Tring"
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, captured on 2026-08-08, shows a red brick chapel with a central entrance, steps, and surrounding greenery under a partly cloudy sky."
Keywords: "Akeman Street Baptist Church, Tring, Hertfordshire, red brick, chapel, entrance, steps, greenery, clouds, sky, church, England, Hertfordshire, location, brick, building, sign, bushes"

Wait, need to make sure keywords are unique and cover the hints. Let's check the hints: Adobe Stock, Any Vision, Buckinghamshire (wait, Hertfordshire is the county, not Buckinghamshire), Bushes, Chapel, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky. So combine those. Let's list them:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, need 10-18. Let's count. Let's make sure each is unique. Let's do:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, maybe better to check the image details. The building is red brick, has a sign, entrance with steps, bushes, clouds in sky, etc. So keywords should include those. Let's finalize:

Title: "Akeman Street Baptist Church, Tring"
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, photographed on 2026-08-08, features a red brick chapel with a central entrance, steps, and surrounding greenery under a partly cloudy sky."
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian

Wait, need to make sure the description is 1-2 sentences. Let's adjust:

Description: "Akeman Street Baptist Church in Tring, Hertfordshire, captured on 2026-08-08, is a red brick chapel with a central entrance, steps, and greenery, set under a partly cloudy sky."

Yes, that's 1-2 sentences. Now check the title is 5-10 words: "Akeman Street Baptist Church, Tring" is 6 words, good. Keywords need to be unique, comma-separated, 10-18. Let's list again:

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertfordshire, England, location, brick building, roof, Christian. Wait, that's 16, which is within 10-18. Let's check if all hints are covered. The hints included Adobe Stock, Any Vision, Buckinghamshire (but we have Hertfordshire), Bushes, Chapel, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky. So we have most of them. Maybe add "red brick" as a keyword. Let's confirm each part.

Title: "Akeman Street Baptist Church, Tring" (5 words, good)
Description: "Akeman Street Baptist Church in Tring, Hertfordshire, photographed on 2026-08-08, is a red brick chapel with a central entrance, steps, and greenery, set under a partly cloudy sky." (1 sentence, covers context, subject, setting, lighting, details)
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, chapel, church, entrance, red brick, bushes, clouds, sky, sign, Hertford
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — unusable — extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, unexpected_catalog_preamble,
  thinking_trace_incomplete, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this step
  by step. First, the title needs to be 5-10 words, concrete. The main subject
  is Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church,
  Tring" but wait, need to be 5-10 words. Let's see, "Akeman Street Baptist
  Church, Tring, England" is 7 words, that works.  Next, the description: need
  to combine context (Tring, Herts) with visible details. The church is red
  brick, has a steep roof, arched windows, a maroon door, steps, a sign,
  bushes around. Let's put that together: "Akeman Street Baptist Church in
  Tring, Hertfordshire, is a red-brick building with arched windows, a maroon
  door, and steps leading to its entrance, surrounded by greenery." Wait, does
  that cover the main points? Let's check: subject (church), setting (Tring,
  Herts), visible details (red brick, arched windows, maroon door, steps,
  greenery). Yes.
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 15
- *Requested title word range:* [5, 10]
- *Keyword count:* 66
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["hertfordshire", "england", "europe", "red brick",
  "sign", "bushes", "chapel", "church", "clouds", "entrance", "maroon door",
  "arched windows", "greenery"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "abrupt_tail"]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002425916
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6446
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
<think>Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The main subject is Akeman Street Baptist Church, so maybe "Akeman Street Baptist Church, Tring" but wait, need to be 5-10 words. Let's see, "Akeman Street Baptist Church, Tring, England" is 7 words, that works.

Next, the description: need to combine context (Tring, Herts) with visible details. The church is red brick, has a steep roof, arched windows, a maroon door, steps, a sign, bushes around. Let's put that together: "Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery." Wait, does that cover the main points? Let's check: subject (church), setting (Tring, Herts), visible details (red brick, arched windows, maroon door, steps, greenery). Yes.

Keywords: Need 10-18 unique, comma-separated. Let's list the hints and add visible ones. Hints include Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof. Also, visible details: steps, maroon door, arched windows, greenery, brick wall, chimney (wait, do we see a chimney? The image has a chimney on the left house, but the church itself—wait, the church has a roof with a spire, but chimney? Maybe not. Let's stick to visible: Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall. Let's count: 1. Akeman Street Baptist Church, 2. Hertfordshire, 3. England, 4. Europe, 5. Red brick, 6. Sign, 7. Bushes, 8. Chapel, 9. Church, 10. Clouds, 11. Entrance, 12. Roof, 13. Steps, 14. Maroon door, 15. Arched windows, 16. Greenery, 17. Brick wall. That's 17, which is within range.

Wait, let's check the description again. The context says "Akeman Street Baptist Church, Tring, Herts" so the description should mention Tring and Herts. Let's adjust the description to include that: "Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery." Yes, that's good.

Title: Let's make sure it's 5-10 words. "Akeman Street Baptist Church, Tring" is 6 words, perfect.

Keywords: Let's make sure they are unique and cover all. Let's list again: Akeman Street Baptist Church, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall. That's 15, but maybe add "Tring" since it's in the context. Wait, the title has Tring, but keywords can include it. Let's replace one with Tring: Akeman Street Baptist Church, Tring, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery. That's 15, still good.

Wait, let's check the image: the sign says "Akeman Street Baptist Church" so that's a keyword. The church is red brick, has a roof with a spire, arched windows, maroon door, steps, a brick wall with pillars, bushes around. So keywords should include those. Let's finalize:

Title: Akeman Street Baptist Church, Tring

Description: Akeman Street Baptist Church in Tring, Hertfordshire, is a red-brick building with arched windows, a maroon door, and steps leading to its entrance, surrounded by greenery.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, Europe, Red brick, Sign, Bushes, Chapel, Church, Clouds, Entrance, Roof, Steps, Maroon door, Arched windows, Greenery, Brick wall

Wait, let's count keywords: 1
```

</details>

<a id="diagnostic-mlx-community-ornith-10-35b-bf16"></a>

<details>
<summary>mlx-community/Ornith-1.0-35B-bf16 — unusable — extra text before Title; wrapper tokens visible</summary>

### mlx-community/Ornith-1.0-35B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble, configured_wrapper_present
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Unexpected text before Title:* The user wants metadata for an image of a
  church.  **1. Analyze the Image:** * **Subject:** A red brick building,
  clearly a church. * **Signage:** A sign on the left reads "Akeman Street
  Baptist Church" with service times "Sunday Services 10:30am &amp; 6:00pm"
  and a website "www.akemanstreet.org". * **Architecture:** Red brick facade,
  arched windows (three on the upper level, two on the lower level flanking
  the door), a central arched doorway with double red doors. There's a stone
  plaque above the door. The roof is a hipped roof with a small cupola/spire
  on top. * **Surroundings:** * Left: Another brick building (residential?),
  trees, bushes. * Right: Large green trees and bushes obscuring part of the
  building. * Foreground: A low brick wall with stone pillars and black metal
  gates. Steps leading up to the church entrance. A paved path. * Sky: Blue
  sky with scattered white clouds. * **Lighting:** Bright sunlight, casting
  shadows. The sun seems to be coming from the upper left/behind the camera,
  illuminating the front facade. * **Context clues:** The sign says "Akeman
  Street Baptist Church". The hint says "Tring, Herts".
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00704627
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16557
- *Generation tokens:* 435
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
The user wants metadata for an image of a church.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A sign on the left reads "Akeman Street Baptist Church" with service times "Sunday Services 10:30am & 6:00pm" and a website "www.akemanstreet.org".
*   **Architecture:** Red brick facade, arched windows (three on the upper level, two on the lower level flanking the door), a central arched doorway with double red doors. There's a stone plaque above the door. The roof is a hipped roof with a small cupola/spire on top.
*   **Surroundings:**
    *   Left: Another brick building (residential?), trees, bushes.
    *   Right: Large green trees and bushes obscuring part of the building.
    *   Foreground: A low brick wall with stone pillars and black metal gates. Steps leading up to the church entrance. A paved path.
    *   Sky: Blue sky with scattered white clouds.
*   **Lighting:** Bright sunlight, casting shadows. The sun seems to be coming from the upper left/behind the camera, illuminating the front facade.
*   **Context clues:** The sign says "Akeman Street Baptist Church". The hint says "Tring, Herts".
</think>

Title: Akeman Street Baptist Church in Tring, Hertfordshire
Description: A red brick Baptist church stands on Akeman Street in Tring, Hertfordshire, featuring arched windows, a central double door, and a stone plaque above the entrance. A low brick wall with stone pillars and black gates encloses the front garden, where a sign displays service times under a blue sky with scattered clouds.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Red brick, Church, Chapel, Sign, Entrance, Roof, Sky, Clouds, Gates, Steps, Garden, Architecture, Christian
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Thinking-bf16 — unusable — extra text before Title; wrapper tokens visible; title/keyword constraints failed</summary>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble, configured_wrapper_present,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Unexpected text before Title:* Got it, let's tackle this step by step.
  First, the title. The context says it's Akeman Street Baptist Church in
  Tring, Herts. The descriptive hints mention that. So a title should be
  concise. Maybe "Akeman Street Baptist Church, Tring, Herts, England" but
  need to be 5-10 words. Let's check: "Akeman Street Baptist Church, Tring,
  Herts, England" is 10 words. Wait, maybe shorter: "Akeman Street Baptist
  Church, Tring, Herts, England" is 10 words. Alternatively, "Tring Baptist
  Church, Akeman Street, Herts, England" but the context says "Akeman Street
  Baptist Church". So title: "Akeman Street Baptist Church, Tring, Herts,
  England" – that's 10 words.  Next, the description. Need to combine context
  with visible subject, setting, action, lighting, details. The image shows
  the church, red brick, arched windows, door, sign, trees, sky. Capture date
  is 2026-08-08, but we don't claim it's visible. The sign says "Akeman Street
  Baptist Church", so the main subject is the church. Setting is the town
  centre, Tring, Herts. Lighting: sunny, clouds in sky. Distinctive details:
  red brick, arched windows, dark red door, stone pillars,
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Title word count:* 7
- *Requested title word range:* [5, 10]
- *Keyword count:* 19
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.008045704
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16547
- *Generation tokens:* 428
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title. The context says it's Akeman Street Baptist Church in Tring, Herts. The descriptive hints mention that. So a title should be concise. Maybe "Akeman Street Baptist Church, Tring, Herts, England" but need to be 5-10 words. Let's check: "Akeman Street Baptist Church, Tring, Herts, England" is 10 words. Wait, maybe shorter: "Akeman Street Baptist Church, Tring, Herts, England" is 10 words. Alternatively, "Tring Baptist Church, Akeman Street, Herts, England" but the context says "Akeman Street Baptist Church". So title: "Akeman Street Baptist Church, Tring, Herts, England" – that's 10 words.

Next, the description. Need to combine context with visible subject, setting, action, lighting, details. The image shows the church, red brick, arched windows, door, sign, trees, sky. Capture date is 2026-08-08, but we don't claim it's visible. The sign says "Akeman Street Baptist Church", so the main subject is the church. Setting is the town centre, Tring, Herts. Lighting: sunny, clouds in sky. Distinctive details: red brick, arched windows, dark red door, stone pillars,
</think>

Title: Akeman Street Baptist Church, Tring, Herts, England

Description: Red-brick Akeman Street Baptist Church in Tring, Herts, England, featuring arched windows, a dark red door, stone pillars, and a sign, under a partly cloudy sky with sunlight illuminating the scene.

Keywords: Akeman Street Baptist Church, Tring, Herts, England, Red brick, Arched windows, Dark red door, Stone pillars, Sign, Clouds, Sunlight, Town centre, Church, Hertfordshire, Bushes, Chimney, Christian, Entrance, Europe
```

</details>

<a id="diagnostic-mlx-community-qwen35-35b-a3b-4bit"></a>

<details>
<summary>mlx-community/Qwen3.5-35B-A3B-4bit — unusable — extra text before Title; wrapper tokens visible</summary>

### mlx-community/Qwen3.5-35B-A3B-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble, configured_wrapper_present
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5_moe)
- *Unexpected text before Title:* The user wants metadata for an image of a
  church.  **1. Analyze the Image:** * **Subject:** A red brick building,
  clearly a church. * **Signage:** A red sign on the left reads "Akeman Street
  Baptist Church". It also lists Sunday Services times. * **Architecture:**
  Red brick facade, arched windows (some with white frames, some with leaded
  glass), a central arched doorway with double doors (dark red/maroon), a
  slate roof with a small cupola/spire on top. There are stone pillars and a
  low brick wall in the foreground. * **Setting:** Looks like a town street.
  There are other brick buildings to the left. Trees and bushes surround the
  church, especially on the right. * **Lighting:** Bright daylight, blue sky
  with scattered white clouds. Shadows suggest the sun is high but slightly to
  the left/behind the camera. * **Details:** A plaque above the door says
  "Erected 1808 Rebuilt 1852". There are steps leading up to the entrance. A
  black metal gate is visible on the left.  **2. Process Context:** *
  **Date/Time:** 2026-08-08 15:43:55 UTC+01:00 (This is future-dated relative
  to now, but I must treat it
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Resolved model revision:* 1e20fd8d42056f870933bf98ca6211024744f7ec
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00855361
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16557
- *Generation tokens:* 402
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
The user wants metadata for an image of a church.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A red sign on the left reads "Akeman Street Baptist Church". It also lists Sunday Services times.
*   **Architecture:** Red brick facade, arched windows (some with white frames, some with leaded glass), a central arched doorway with double doors (dark red/maroon), a slate roof with a small cupola/spire on top. There are stone pillars and a low brick wall in the foreground.
*   **Setting:** Looks like a town street. There are other brick buildings to the left. Trees and bushes surround the church, especially on the right.
*   **Lighting:** Bright daylight, blue sky with scattered white clouds. Shadows suggest the sun is high but slightly to the left/behind the camera.
*   **Details:** A plaque above the door says "Erected 1808 Rebuilt 1852". There are steps leading up to the entrance. A black metal gate is visible on the left.

**2. Process Context:**
*   **Date/Time:** 2026-08-08 15:43:55 UTC+01:00 (This is future-dated relative to now, but I must treat it
</think>

Title: Akeman Street Baptist Church Exterior
Description: A red brick church with arched windows and a slate roof stands under a blue sky with scattered clouds, featuring a sign for Akeman Street Baptist Church and a stone wall in the foreground.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Red brick, Church, Chapel, Sky, Clouds, Roof, Entrance, Sign, Bushes, Trees, Architecture, Street, Historic
```

</details>

<a id="diagnostic-mlx-community-qwen35-9b-mlx-4bit"></a>

<details>
<summary>mlx-community/Qwen3.5-9B-MLX-4bit — unusable — extra text before Title; wrapper tokens visible</summary>

### mlx-community/Qwen3.5-9B-MLX-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble, configured_wrapper_present
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Unexpected text before Title:* The user wants British-English catalogue
  metadata for the provided image.  **1. Analyze the Image:** * **Subject:** A
  red brick building, clearly a church. * **Signage:** A red sign on the left
  reads "Akeman Street Baptist Church". * **Architecture:** * Red brick
  facade. * Three arched windows on the upper floor (left, center, right). *
  Two arched windows on the lower floor (left, right). * A central arched
  doorway with a maroon double door. * A small cupola or lantern on the roof.
  * A plaque above the door: "ERECTED 4th May 1808 REBUILT 31st May 1832". *
  Two stone obelisks/monuments in front of the building. * **Setting:** *
  Surrounded by greenery/bushes/trees. * A brick wall with pillars and gates
  in the foreground. * Steps leading up to the entrance. * Another brick
  building visible to the left. * **Lighting/Weather:** * Daytime, partly
  cloudy sky. * Sunlight hitting the roof and upper part of the building. *
  **Context:** * Location: Tr
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Resolved model revision:* 938d8919941c6e7efd3c7150eff7fe9d12afa631
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009077898
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16557
- *Generation tokens:* 427
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
The user wants British-English catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subject:** A red brick building, clearly a church.
*   **Signage:** A red sign on the left reads "Akeman Street Baptist Church".
*   **Architecture:**
    *   Red brick facade.
    *   Three arched windows on the upper floor (left, center, right).
    *   Two arched windows on the lower floor (left, right).
    *   A central arched doorway with a maroon double door.
    *   A small cupola or lantern on the roof.
    *   A plaque above the door: "ERECTED 4th May 1808 REBUILT 31st May 1832".
    *   Two stone obelisks/monuments in front of the building.
*   **Setting:**
    *   Surrounded by greenery/bushes/trees.
    *   A brick wall with pillars and gates in the foreground.
    *   Steps leading up to the entrance.
    *   Another brick building visible to the left.
*   **Lighting/Weather:**
    *   Daytime, partly cloudy sky.
    *   Sunlight hitting the roof and upper part of the building.
*   **Context:**
    *   Location: Tr
</think>

Title: Akeman Street Baptist Church, Tring, Hertfordshire

Description: This red brick Baptist church in Tring, Hertfordshire, features a central maroon door, arched windows, and a small cupola, set against a partly cloudy sky. A brick wall with gates and stone monuments frame the entrance to the historic building.

Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, red brick, arched windows, cupola, maroon door, stone monuments, brick wall, gates, partly cloudy sky, historic church, Baptist, England, UK, architecture
```

</details>

<a id="diagnostic-mlx-community-qwen36-27b-mxfp8"></a>

<details>
<summary>mlx-community/Qwen3.6-27B-mxfp8 — unusable — extra text before Title; wrapper tokens visible; title/keyword constraints failed</summary>

### mlx-community/Qwen3.6-27B-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble, configured_wrapper_present,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_5)
- *Unexpected text before Title:* The user wants British-English catalogue
  metadata for the provided image.  **1. Analyze the Image:** * **Subject:** A
  large, red brick building. It looks like a church or chapel. * **Signage:**
  There is a red sign on the left that says "Akeman Street Baptist Church". It
  also lists service times: "Sunday Services 10:30am &amp; 6:00pm". *
  **Architecture:** * Red brick construction. * Three large arched windows on
  the upper level. * Two smaller rectangular windows on the lower level
  (flanking the door). * A large arched double door (dark red/brown wood). * A
  slate roof with a small cupola/spire on top. * A stone plaque above the door
  (looks like a date, possibly 1808 or similar, though hard to read clearly,
  the sign confirms the name). * Stone obelisks on either side of the entrance
  path. * **Setting:** * Surrounded by greenery (bushes, trees). * A brick
  wall and gate in the foreground. * A paved path leading up to the church. *
  Another brick building visible to the left (residential?). * Sky is blue
  with scattered clouds. *
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Title word count:* 7
- *Requested title word range:* [5, 10]
- *Keyword count:* 20
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009585802
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16557
- *Generation tokens:* 450
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
The user wants British-English catalogue metadata for the provided image.

**1. Analyze the Image:**
*   **Subject:** A large, red brick building. It looks like a church or chapel.
*   **Signage:** There is a red sign on the left that says "Akeman Street Baptist Church". It also lists service times: "Sunday Services 10:30am & 6:00pm".
*   **Architecture:**
    *   Red brick construction.
    *   Three large arched windows on the upper level.
    *   Two smaller rectangular windows on the lower level (flanking the door).
    *   A large arched double door (dark red/brown wood).
    *   A slate roof with a small cupola/spire on top.
    *   A stone plaque above the door (looks like a date, possibly 1808 or similar, though hard to read clearly, the sign confirms the name).
    *   Stone obelisks on either side of the entrance path.
*   **Setting:**
    *   Surrounded by greenery (bushes, trees).
    *   A brick wall and gate in the foreground.
    *   A paved path leading up to the church.
    *   Another brick building visible to the left (residential?).
    *   Sky is blue with scattered clouds.
*  
</think>

Title:
Akeman Street Baptist Church in Tring, Hertfordshire

Description:
This image shows the exterior of the Akeman Street Baptist Church in Tring, Hertfordshire, featuring a red brick facade with arched windows and a slate roof topped by a small cupola. A red sign identifies the building and lists service times, while stone obelisks flank the paved path leading to the main entrance.

Keywords:
Akeman Street Baptist Church, Tring, Hertfordshire, England, red brick, church, chapel, architecture, arched windows, slate roof, cupola, stone obelisk, entrance, sign, sky, clouds, brick wall, garden, religious building, UK
```

</details>

<a id="diagnostic-mlx-community-step-37-flash-oq2e"></a>

<details>
<summary>mlx-community/Step-3.7-Flash-oQ2e — unusable — extra text before Title; wrapper tokens visible</summary>

### mlx-community/Step-3.7-Flash-oQ2e

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_catalog_preamble, configured_wrapper_present
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Unexpected text before Title:* Got it, let's tackle this step by step.
  First, the title needs to be 5-10 words, concrete. The subject is Akeman
  Street Baptist Church in Tring, England, right? Wait, let's make it
  accurate. Oh, "Akeman Street Baptist Church, Tring, Hertfordshire" that's
  good, count words: Akeman (1) Street (2) Baptist (3) Church, (4) Tring, (5)
  Hertfordshire (6) – that's 6 words, fits 5-10.  Next, the description. 1-2
  sentences, factual. Let's include the context: it's the Akeman Street
  Baptist Church in Tring, Hertfordshire, UK, a red brick 19th-century chapel
  with arched windows, a central arched entrance, set against a partly cloudy
  sky, with surrounding residential buildings, a church sign, and landscaped
  grounds with a stone monument and gateposts. Wait, make it flow. Let's see:
  "The Akeman Street Baptist Church in Tring, Hertfordshire, UK, is a historic
  red brick chapel featuring arched windows, a central arched entrance, and a
  steeply pitched roof, set amid residential buildings, landscaped grounds
  with a stone monument, and a partly cloudy summer sky." Wait, check the
  details: the sign is there, the brick wall, the gateposts. Oh right, also
  mention the sign? Wait, let's make it accurate
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Resolved model revision:* 3dacb46f724ac89725bcd922fb779c7ed1499fe7
- *Processor class:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00996267
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3485
- *Generation tokens:* 438
- *Configured EOS token ID:* 128007
- *Configured EOS token:* &lt;|im_end|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title needs to be 5-10 words, concrete. The subject is Akeman Street Baptist Church in Tring, England, right? Wait, let's make it accurate. Oh, "Akeman Street Baptist Church, Tring, Hertfordshire" that's good, count words: Akeman (1) Street (2) Baptist (3) Church, (4) Tring, (5) Hertfordshire (6) – that's 6 words, fits 5-10.

Next, the description. 1-2 sentences, factual. Let's include the context: it's the Akeman Street Baptist Church in Tring, Hertfordshire, UK, a red brick 19th-century chapel with arched windows, a central arched entrance, set against a partly cloudy sky, with surrounding residential buildings, a church sign, and landscaped grounds with a stone monument and gateposts. Wait, make it flow. Let's see: "The Akeman Street Baptist Church in Tring, Hertfordshire, UK, is a historic red brick chapel featuring arched windows, a central arched entrance, and a steeply pitched roof, set amid residential buildings, landscaped grounds with a stone monument, and a partly cloudy summer sky." Wait, check the details: the sign is there, the brick wall, the gateposts. Oh right, also mention the sign? Wait, let's make it accurate
</think>
Title: Akeman Street Baptist Church, Tring, Hertfordshire
Description: The Akeman Street Baptist Church in Tring, Hertfordshire, UK, is a historic red brick chapel with arched windows, a central arched entrance, and a steeply pitched roof, situated amid adjacent residential buildings, a church sign, landscaped grounds with a stone monument, and a partly cloudy summer sky.
Keywords: Akeman Street Baptist Church, Tring, Hertfordshire, England, UK, Europe, red brick chapel, arched windows, church sign, stone monument, brick wall, gateposts, residential buildings, partly cloudy sky, landscaping, Christian place of worship
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — usable_with_caveats — role tokens visible</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Role-boundary tokens in output:* ["&lt;end_of_utterance&gt;"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00268806
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2611
- *Generation tokens:* 143
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *Configured thinking end token:* &lt;/think&gt;

#### Complete output

```text
 Title: Akeman Street Baptist Church in Tring, Hertfordshire, England.
Description: The Akeman Street Baptist Church, a red-brick building with a grey slate roof and multiple arched windows, is situated in Tring, Hertfordshire, England. The church is surrounded by trees and has a brick wall with steps leading up to the entrance. A sign with the church's name is visible near the entrance.
Keywords: akeman street baptist church, tring, hertfordshire, england, red brick, grey slate roof, arched windows, trees, brick wall, steps, entrance, sign, church, building, europe, uk, gbr.<end_of_utterance>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                             | Usability           | Observations                                                                                           |
|---------------------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                   | unusable            | missing required fields                                                                                |
| mlx-community/gemma-3n-E4B-it-bf16                | unusable            | missing required fields                                                                                |
| mlx-community/gemma-4-31b-it-4bit                 | unusable            | missing required fields; echoes instructions; extra text before Title                                  |
| mlx-community/llava-v1.6-mistral-7b-8bit          | unusable            | missing required fields; extra text before Title                                                       |
| mlx-community/MolmoPoint-8B-fp16                  | unusable            | missing required fields                                                                                |
| mlx-community/nanoLLaVA-1.5-4bit                  | unusable            | missing required fields                                                                                |
| mlx-community/Qwen2-VL-2B-Instruct-4bit           | unusable            | missing required fields                                                                                |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX     | unusable            | echoes instructions; extra text before Title; cut off at token limit; title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8     | unusable            | echoes instructions; extra text before Title; title/keyword constraints failed                         |
| mlx-community/GLM-4.6V-Flash-mxfp4                | unusable            | extra text before Title                                                                                |
| mlx-community/gemma-3-27b-it-qat-4bit             | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Molmo-7B-D-0924-8bit                | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/pixtral-12b-8bit                    | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/Qwen3-VL-2B-Instruct-bf16           | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx          | usable_with_caveats | title/keyword constraints failed                                                                       |
| mlx-community/X-Reasoner-7B-8bit                  | usable_with_caveats | title/keyword constraints failed                                                                       |
| Qwen/Qwen3-VL-2B-Instruct                         | usable_with_caveats | title/keyword constraints failed                                                                       |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                        | Performance                                                                               |
|-------------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------|
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed     | 2389 prompt / 89 generated; 21.0 tok/s; 23 GB peak; cleanup 0.000968/0.0 GB active/cache  |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed | 2111 prompt / 89 generated; 29.7 tok/s; 17 GB peak; cleanup 0.003/0.0 GB active/cache     |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | rev 16a710cf8afc; Lfm2VlProcessor; stop completed       | 579 prompt / 124 generated; 158 tok/s; 4.1 GB peak; cleanup 0.00379/0.0 GB active/cache   |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | rev 8451adc50203; MllamaProcessor; stop completed       | 300 prompt / 90 generated; 11.2 tok/s; 15 GB peak; cleanup 0.00431/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed     | 2922 prompt / 157 generated; 48.0 tok/s; 13 GB peak; cleanup 0.00508/0.0 GB active/cache  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed     | 2921 prompt / 128 generated; 160 tok/s; 7.8 GB peak; cleanup 0.0056/0.0 GB active/cache   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed        | 1125 prompt / 168 generated; 55.7 tok/s; 9.4 GB peak; cleanup 0.00711/0.0 GB active/cache |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 60,712,161 bytes
- *Image SHA-256:* 2d3e8ab39253f25bfa3f4a37188a72d369bb79657c8f7011611e1f58fb3afc23

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-08 15:43:55 UTC+01:00
- GPS: 51.815915°N, 0.638706°W

Descriptive hints:
- Title hint: Town centre, Tring, England, UK, GBR, Europe
- Description hint: Akeman Street Baptist Church, Tring, Herts
- Keyword hints: Adobe Stock, Akeman Street Baptist Church, Any Vision, Buckinghamshire, Bushes, Chapel, Chimney, Christian, Church, Clouds, England, Entrance, Europe, Hertfordshire, Locations, Objects, Red brick, Roof, Sign, Sky

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

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/Inkling-Small-mlx-4bit             | f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612 |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                 | 6c33f49ebc0b50b75385f49ad3beddcb720d0c75 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/gemma-4-26b-a4b-it-4bit            | 0d77464eeb233a2da68ebf9d7dc4edaac7db956d |
| mlx-community/MiniCPM-V-4.6-8bit                 | 03721395f6b82cd000cc74cde28fcff8abd9a04c |
| mlx-community/GLM-4.1V-9B-Thinking-8bit          | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Ornith-1.0-35B-bf16                | 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b |
| mlx-community/Qwen3-VL-2B-Thinking-bf16          | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/Qwen3.5-35B-A3B-4bit               | 1e20fd8d42056f870933bf98ca6211024744f7ec |
| mlx-community/Qwen3.5-9B-MLX-4bit                | 938d8919941c6e7efd3c7150eff7fe9d12afa631 |
| mlx-community/Qwen3.6-27B-mxfp8                  | 5db9fd9c38ce083048ee12217eceed7482ca7ad4 |
| mlx-community/Step-3.7-Flash-oQ2e                | 3dacb46f724ac89725bcd922fb779c7ed1499fe7 |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.11                                                                                                                                          |
| mlx                        | 0.32.1.dev20260808+8d6662986                                                                                                                    |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.7                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.27.0                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (165,577,880 bytes, sha256=0d3bca561dcf0924959abae6246546e2ad15baedff1ff9e418c7fd25d07b187e) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,720,768 bytes, sha256=67adf9bea8131724a4b573c7756226f950a068f84a5a61414988d8dac1e8c7de)  |
| RAM                        | 128.0 GB                                                                                                                                        |

<!-- markdownlint-enable MD004 MD037 -->
