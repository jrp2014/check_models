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
| none                           | 30    |
| observation needs reproduction | 11    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 15    |
| usable              | 15    |
| usable with caveats | 11    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 5     |
| Unrecognised model control tokens remain visible                                      | 1     |
| Required fields are missing or empty                                                  | 11    |
| Response repeats the task instructions instead of only returning the requested fields | 3     |
| Extra text appears before the Title field                                             | 6     |
| Response appears cut off at the token limit                                           | 10    |
| Internal reasoning block appears incomplete                                           | 4     |
| Conversation-role control tokens remain visible                                       | 1     |
| Title or keywords do not meet requested constraints                                   | 14    |

## Triage

| Model                                                                                                           | Execution | Usability           | Maintainer status              | Observations                                                                                                             |
|-----------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/Inkling-Small-mlx-4bit](#diagnostic-mlx-community-inkling-small-mlx-4bit)                        | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                     |
| [mlx-community/LFM2.5-VL-1.6B-bf16](#diagnostic-mlx-community-lfm25-vl-16b-bf16)                                | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [mlx-community/llava-v1.6-mistral-7b-8bit](#diagnostic-mlx-community-llava-v16-mistral-7b-8bit)                 | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; extra text before Title; cut off at token limit                                  |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; echoes instructions; cut off at token limit                                      |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                  |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                   |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed | unusable            | observation_needs_reproduction | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed | unusable            | observation_needs_reproduction | missing required fields; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                  | completed | unusable            | observation_needs_reproduction | missing required fields; cut off at token limit; incomplete thinking block                                               |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                   | completed | unusable            | observation_needs_reproduction | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed             |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible; title/keyword constraints failed                                                                    |

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
- *Post-cleanup active memory (GB):* 0.002704444
- *Post-cleanup cache memory (GB):* 0.0

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11958, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11435, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 750, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12388, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11973, in _run_model_generation
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
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 3352.02it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[21:24:49] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-lfm25-vl-16b-bf16"></a>

<details>
<summary>mlx-community/LFM2.5-VL-1.6B-bf16 — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/LFM2.5-VL-1.6B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Repeated fragment:* keyword: "sunlight"
- *Title word count:* 13
- *Requested title word range:* [5, 10]
- *Keyword count:* 149
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["sunlight", "historic", "modern", "parking",
  "stonework", "spires"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003802208
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 615
- *Generation tokens:* 500
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe

Description: A stunning view of Arundel Cathedral of Our Lady and St Philip Howard, showcasing its French-Gothic architecture against a clear blue sky. The cathedral's intricate spires and detailed stonework are highlighted by the warm sunlight, while a red car is parked nearby, adding a touch of modernity to the historic scene.

Keywords: Arundel Cathedral, French-Gothic architecture, Arundel, England, UK, GBR, Europe, Cathedral, Church, Red car, Blue sky, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Warm, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stonework, Spires, Sunlight, Historic, Modern, Parking, Stone
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Repeated fragment:* phrase: "st. philip's church of..."
- *Title word count:* 13
- *Requested title word range:* [5, 10]
- *Keyword count:* 59
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["st philip s church of our lady and st philip s
  church of our lady and st philip s"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.004326508
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 333
- *Generation tokens:* 500
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;

#### Complete output

```text
Title: Arundel Cathedral of Our Lady and St. Philip Howard in the Afternoon Sun

Description: The Arundel Cathedral of Our Lady and St. Philip Howard is captured in the late afternoon sun, showcasing its intricate architecture and surrounding neighborhood. The image features a red car parked in a garage, surrounded by a mix of old and new buildings, with the cathedral's spires and towers rising above the rooftops.

Keywords: Arundel, Cathedral, Church, Church Architecture, Church Building, Church Exterior, Church Interior, Church Landscaping, Church Lighting, Church Location, Church Neighborhood, Church Parking, Church Roof, Church Spires, Church Towers, Church Windows, English Architecture, European Architecture, Gothic Architecture, Historic Architecture, Historic Buildings, Historic Landmarks, Historic Landscapes, Historic Neighborhoods, Historic Places, Historic Sites, Historic Towns, Historic Villages, Old Buildings, Old Churches, Old Landmarks, Old Neighborhoods, Old Places, Old Towns, Old Villages, Religious Architecture, Religious Buildings, Religious Landmarks, Religious Landscapes, Religious Neighborhoods, Religious Places, Religious Sites, Religious Towns, Religious Villages, St. Philip Howard, St. Philip's Church, St. Philip's Cathedral, St. Philip's Church of Our Lady, St. Philip's Church of Our Lady and St. Philip, St. Philip's Church of Our Lady and St. Philip's, St. Philip's Church of Our Lady and St. Philip's Cathedral, St. Philip's Church of Our Lady and St. Philip's Church, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church of Our Lady, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's Church, St. Philip's Church of Our Lady and St. Philip's Church of Our Lady and St. Philip's
```

</details>

<a id="diagnostic-mlx-community-llava-v16-mistral-7b-8bit"></a>

<details>
<summary>mlx-community/llava-v1.6-mistral-7b-8bit — unusable — repeated text; missing required fields; extra text before Title; cut off at token limit</summary>

### mlx-community/llava-v1.6-mistral-7b-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, unexpected_catalog_preamble
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Missing sections:* ["title", "description"]
- *Repeated fragment:* phrase: "arundel cathedral of our..."
- *Unexpected text before Title:* Arundel Cathedral of Our Lady &amp; St.
  Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of
  Our Lady and St Philip Howard
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail"]
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Processor class:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.012502264
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2552
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/] Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard

Title:
Description:
Keywords: [/car, church, cathedral, arundel, uk, europe, sky, roof, building, architecture, gothic, windows,
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
- *Repeated fragment:* phrase: "- 10-18 unique, comma-separate..."
- *Echoed instruction fragments:* ["title hint:", "description hint:", "10 18
  unique comma separated keywords covering relevant"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.014157068
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4444
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Title hint:
- Description hint:
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18 unique, comma-separated keywords covering relevant context and visible details.
- 10-18
```

</details>

<a id="diagnostic-mlx-community-x-reasoner-7b-8bit"></a>

<details>
<summary>mlx-community/X-Reasoner-7B-8bit — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Repeated fragment:* keyword: "tranquil"
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 184
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["stone wall", "dusk", "peaceful", "historic",
  "traditional", "residential", "urban", "architectural", "landmark",
  "cultural", "scenic", "picturesque", "tranquil", "serene", "calm", "quiet"]
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.010306734
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16592
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Arundel Cathedral and Cottage at Dusk

Description: A serene view of Arundel Cathedral of Our Lady & St. Philip Howard, framed by a quaint cottage and vibrant red car, under a clear blue sky at dusk.

Keywords: Arundel, Arundel Cathedral, England, UK, Europe, Gothic architecture, French-Gothic, blue sky, cottage, red car, parking, roof, flower, bush, neighborhood, streetlamp, stone wall, window, cross, steeple, garage, 20 mph sign, brick building, white house, stone pathway, lamp post, 2026-08-06, 18:26:35 UTC+01:00, 50.806659°N, 0.551382°W, flower box, potted flowers, stone fence, white garage door, satellite dish, chimney, streetlight, lamp, window box, flower pot, stone wall, brick, white, red, blue, green, yellow, dusk, sunlight, shadow, clear sky, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, serene, calm, quiet, evening, sunset, dusk, twilight, peaceful, calm, quiet, serene, tranquil, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet, serene, peaceful, historic, traditional, residential, urban, architectural, landmark, cultural, scenic, picturesque, tranquil, calm, quiet,
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
- *Post-cleanup active memory (GB):* 0.0024423
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6483
- *Generation tokens:* 100
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: Arundel Cathedral and Red Car in English Village
Description: A red car is parked in a cobblestone driveway beside a white cottage, with Arundel Cathedral’s Gothic spire and rose window visible under a clear blue sky in Arundel, England.
Keywords: Arundel Cathedral, Arundel, England, UK, Cathedral, Church, Red Car, Cottage, Blue Sky, Flower, Roof, Neighborhood, Gothic, Parking, Bush, Stone Wall<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — missing required fields; cut off at token limit; incomplete thinking block</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Missing sections:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001197102
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1665
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are a bit of a mess, but the image is clear. It's a view of Arundel Cathedral, a Gothic church, with a red car and a stone wall. I'll go with "Arundel Cathedral and Red Car at Sunset." That's descriptive and accurate.

Now for the description. I need to weave in the context – the date, time, and location – with the visual details. The hints are a bit off, but the image is what matters. I'll start with the location and time, then describe the scene: the cathedral, the red car, the stone wall, the sunset. I'll make sure to mention the Gothic architecture and the red car. I'll also note the stone wall and the red car. The final description will be: "A panoramic view of Arundel Cathedral in England, captured at sunset on August 6, 2026. The Gothic church is silhouetted against a bright blue sky, with a red car parked near a stone wall and a stone archway in the foreground."

Finally, the keywords. I'll pull from the hints, but I'll prioritize what's actually in the image. I'll include the location, the cathedral, the car, the architecture, the sky, and the time of day. I'll also add some general terms like "architecture" and "road." I'll make sure to include the date and time. I'll also add "stone wall" and "red car" to the list. I'll make sure to include "sunset" and "blue sky". I'll also add "Gothic" and "England". I'll make sure to include "road" and "stone archway". I'll also add "sunset" and "blue sky". I'll also add "Gothic" and "England". I'll also add "road" and "stone archway".

Here's the final list: Arundel, Arundel Cathedral, England, UK, Gothic architecture, red car, stone wall, blue sky, sunset, panoramic view, stone
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — missing required fields; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo, unexpected_catalog_preamble,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Missing sections:* ["title", "description"]
- *Echoed instruction fragments:* ["retain details supported by the image
  correct conflicts"]
- *Unexpected text before Title:* ◁think▷Okay, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. The user wants a title, a descriptive paragraph, and
  10-18 keywords.  Starting with the title: The main subject is Arundel
  Cathedral, so the title should be concise. Maybe "Arundel Cathedral of Our
  Lady &amp; St Philip Howard, Arundel, UK" but needs to be 5-10 words. Let's
  check the hint: "Arundel Cathedral of Our Lady &amp; St Philip Howard,
  Arundel, England, UK, GBR, Europe" is a bit long. Maybe shorten to "Arundel
  Cathedral of Our Lady &amp; St Philip Howard, Arundel, UK" (9 words).  Next,
  the description. The context says October 31, 2021, but the image is from
  2026? Wait, the capture date is 2026, but the description hint is from 2021.
  Hmm, need to use the image's date? Wait, the user says to treat capture
  date/time as authoritative but not claim it's visible. So maybe use the
  image's date? Wait, the image's context says capture date is 2026-08-06, but
  the description hint is October 31, 2021. There's a conflict. The user says
  to retain details supported by the image, correct conflicts, and add
  important visible details. The image shows a clear view of the cathedral,
  houses, a red car, blue sky, etc. So maybe the description should be based
  on the image, not the hint's date. So the description should mention the
  cathedral, the red car, the houses, the blue sky, maybe the architectural
  details. So something like "View of Arundel Cathedral of Our Lady and St
  Philip Howard in Arundel, UK, with residential buildings, a red car, and a
  clear blue sky in the background." That's a 1-2 sentence.
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003671136
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1355
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a title, a descriptive paragraph, and 10-18 keywords. 

Starting with the title: The main subject is Arundel Cathedral, so the title should be concise. Maybe "Arundel Cathedral of Our Lady & St Philip Howard, Arundel, UK" but needs to be 5-10 words. Let's check the hint: "Arundel Cathedral of Our Lady & St Philip Howard, Arundel, England, UK, GBR, Europe" is a bit long. Maybe shorten to "Arundel Cathedral of Our Lady & St Philip Howard, Arundel, UK" (9 words).

Next, the description. The context says October 31, 2021, but the image is from 2026? Wait, the capture date is 2026, but the description hint is from 2021. Hmm, need to use the image's date? Wait, the user says to treat capture date/time as authoritative but not claim it's visible. So maybe use the image's date? Wait, the image's context says capture date is 2026-08-06, but the description hint is October 31, 2021. There's a conflict. The user says to retain details supported by the image, correct conflicts, and add important visible details. The image shows a clear view of the cathedral, houses, a red car, blue sky, etc. So maybe the description should be based on the image, not the hint's date. So the description should mention the cathedral, the red car, the houses, the blue sky, maybe the architectural details. So something like "View of Arundel Cathedral of Our Lady and St Philip Howard in Arundel, UK, with residential buildings, a red car, and a clear blue sky in the background." That's a 1-2 sentence.

Keywords: Need 10-18 unique keywords. From the image: Arundel Cathedral, Our Lady & St Philip Howard, Arundel, UK, England, Europe, Red car, Blue sky, Brick buildings, Stone walls, Dormer windows, Church spires, Residential area, Flowers, Garage door, Street, Streetlamp, House, etc. Let's pick the most distinctive ones: Arundel Cathedral, Our Lady
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Thinking-bf16 — unusable — missing required fields; cut off at token limit; incomplete thinking block</summary>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Missing sections:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.008078472
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16583
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title. The context says it's Arundel Cathedral of Our Lady & St. Philip Howard, so the title should be concise. The hint says "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe" but we need a 5-10 word title. Let's check the image: it's a view of the cathedral, so maybe "Arundel Cathedral, England, UK, Gothic Architecture, Blue Sky, Residential Neighborhood" but need to be precise. Wait, the title hint is the main subject, so "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England" but maybe shorter. Let's go with "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Area" but need to fit 5-10 words. Maybe "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK" but that's 10 words. Wait, the title should be a concrete title. Let's see the image: the main subject is the cathedral, so "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Neighborhood" but maybe "Arundel Cathedral, England, UK, Gothic Architecture, Blue Sky, Residential Area" – but let's check the context. The context says "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe" so the title should include that. Wait, the title hint is "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe" but we need a title. Let's make it "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK" – that's 10 words. Wait, maybe "Arundel Cathedral, England, UK, Gothic, Blue Sky, Residential Neighborhood" – but let's check the image. The image shows the cathedral with a residential area in front, so the title should capture that. Let's go with "Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK" – that's 10 words. Wait, maybe "Arundel Cathedral, England, UK, Gothic
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
  First, the title needs to be 5-10 words. The main subject is Arundel
  Cathedral with surrounding buildings. So maybe "Arundel Cathedral and
  Neighborhood View".  Then the description: need to combine context
  (date/time, location) with visible elements. The image shows the cathedral,
  houses, a red car, blue sky, etc. So something like "Arundel Cathedral of
  Our Lady &amp; St. Philip Howard in Arundel, England, viewed from a
  residential neighborhood with a red car parked near the cathedral under a
  clear blue sky, captured on 2026-08-06."  Wait, but the description hint had
  October 31, 2021, but the authoritative context says 2026-08-06. So need to
  use the authoritative date. Also, the description should be factual. Let's
  make it concise.  Keywords need to be 10-18 unique. Let's list: Arundel
  Cathedral, England, UK, red car, blue sky, neighborhood, church, cottage,
  stone wall, flower, Gothic architecture, parking, roof, sky, brick building,
  residential area.  Now, let's structure each part.
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 39
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["england", "red car", "blue sky", "neighborhood",
  "church", "cottage", "stone wall", "flower", "parking", "car", "cathedral"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001819696
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6479
- *Generation tokens:* 500
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. The main subject is Arundel Cathedral with surrounding buildings. So maybe "Arundel Cathedral and Neighborhood View". 

Then the description: need to combine context (date/time, location) with visible elements. The image shows the cathedral, houses, a red car, blue sky, etc. So something like "Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England, viewed from a residential neighborhood with a red car parked near the cathedral under a clear blue sky, captured on 2026-08-06."

Wait, but the description hint had October 31, 2021, but the authoritative context says 2026-08-06. So need to use the authoritative date. Also, the description should be factual. Let's make it concise.

Keywords need to be 10-18 unique. Let's list: Arundel Cathedral, England, UK, red car, blue sky, neighborhood, church, cottage, stone wall, flower, Gothic architecture, parking, roof, sky, brick building, residential area.

Now, let's structure each part.

Title: "Arundel Cathedral and Residential View"

Description: "Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England, viewed from a residential neighborhood with a red car parked near the cathedral under a clear blue sky, captured on 2026-08-06."

Wait, need to make sure it's 1-2 sentences. Let's adjust. "Arundel Cathedral of Our Lady & St. Philip Howard in Arundel, England, viewed from a residential area with a red car near the cathedral, under a clear blue sky, captured on 2026-08-06 UTC+01:00."

Keywords: Arundel Cathedral, England, UK, red car, blue sky, neighborhood, church, cottage, stone wall, flower, Gothic architecture, parking, roof, sky, brick building, residential area, stone wall, car, cathedral, England.

Wait, let's check the hints. The keyword hints include: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — usable_with_caveats — role tokens visible; title/keyword constraints failed</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Role-boundary tokens in output:* ["&lt;end_of_utterance&gt;"]
- *Title word count:* 13
- *Requested title word range:* [5, 10]
- *Keyword count:* 18
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002704444
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2644
- *Generation tokens:* 141
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 Title: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe.
Description: The Arundel Cathedral of Our Lady and St. Philip Howard is a French-Gothic style cathedral located in Arundel, England, UK. The cathedral is surrounded by a neighborhood of brick cottages and has a red car parked in front of it.
Keywords: Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky.<end_of_utterance>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                         | Usability           | Observations                                                                                  |
|-----------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------|
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | unusable            | missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16               | unusable            | missing required fields; extra text before Title                                              |
| mlx-community/gemma-3n-E4B-it-bf16            | unusable            | missing required fields                                                                       |
| mlx-community/MiniCPM-V-4.6-8bit              | unusable            | missing required fields; extra text before Title                                              |
| mlx-community/MolmoPoint-8B-fp16              | unusable            | missing required fields                                                                       |
| mlx-community/nanoLLaVA-1.5-4bit              | unusable            | missing required fields                                                                       |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16              | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/diffusiongemma-26B-A4B-it-8bit  | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/GLM-4.6V-Flash-mxfp4            | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Molmo-7B-D-0924-8bit            | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/pixtral-12b-8bit                | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Qwen2-VL-2B-Instruct-4bit       | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Qwen3-VL-2B-Instruct-bf16       | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx      | usable_with_caveats | title/keyword constraints failed                                                              |
| Qwen/Qwen3-VL-2B-Instruct                     | usable_with_caveats | title/keyword constraints failed                                                              |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                           | Performance                                                                                |
|-------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed        | 2426 prompt / 104 generated; 29.5 tok/s; 23 GB peak; cleanup 0.000968/0.0 GB active/cache  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | rev ded389e478f8; DiffusionGemma4Processor; stop completed | 619 prompt / 83 generated; 54.0 tok/s; 28 GB peak; cleanup 0.0103/0.0 GB active/cache      |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed          | 618 prompt / 126 generated; 31.5 tok/s; 18 GB peak; cleanup 0.0108/0.0 GB active/cache     |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed          | 623 prompt / 101 generated; 129 tok/s; 16 GB peak; cleanup 0.0119/0.0 GB active/cache      |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed          | 623 prompt / 85 generated; 26.3 tok/s; 20 GB peak; cleanup 0.0124/0.0 GB active/cache      |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed    | 2147 prompt / 101 generated; 35.1 tok/s; 17 GB peak; cleanup 0.00302/0.0 GB active/cache   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed        | 2959 prompt / 198 generated; 66.4 tok/s; 13 GB peak; cleanup 0.00511/0.0 GB active/cache   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed        | 2959 prompt / 188 generated; 63.5 tok/s; 13 GB peak; cleanup 0.00538/0.0 GB active/cache   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed        | 2958 prompt / 161 generated; 187 tok/s; 7.8 GB peak; cleanup 0.00564/0.0 GB active/cache   |
| mlx-community/Ornith-1.0-35B-bf16                     | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed         | 16596 prompt / 102 generated; 63.3 tok/s; 74 GB peak; cleanup 0.00708/0.0 GB active/cache  |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed           | 1164 prompt / 119 generated; 55.6 tok/s; 9.3 GB peak; cleanup 0.00714/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed         | 16596 prompt / 102 generated; 109 tok/s; 24 GB peak; cleanup 0.00859/0.0 GB active/cache   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed         | 16596 prompt / 88 generated; 90.7 tok/s; 10.0 GB peak; cleanup 0.00911/0.0 GB active/cache |
| mlx-community/Qwen3.6-27B-mxfp8                       | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed         | 16596 prompt / 103 generated; 17.9 tok/s; 35 GB peak; cleanup 0.00962/0.0 GB active/cache  |
| mlx-community/Step-3.7-Flash-oQ2e                     | rev 3dacb46f724a; Step3VLProcessor; stop completed         | 3524 prompt / 111 generated; 45.9 tok/s; 70 GB peak; cleanup 0.01/0.0 GB active/cache      |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 45,221,684 bytes
- *Image SHA-256:* c285c098dd74e95801ffa1682a82ee2233a8cbb0baed91714ede8a361b9c6438

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-06 18:26:35 UTC+01:00
- GPS: 50.806659°N, 0.551382°W

Descriptive hints:
- Title hint: Arundel Cathedral of Our Lady & St. Philip Howard, Arundel, England, UK, GBR, Europe
- Description hint: Arundel, UK - October 31, 2021: View of Arundel Cathedral of Our Lady and St Philip Howard
- Keyword hints: Adobe Stock, Any Vision, Arundel, Arundel Cathedral of Our Lady & St. Philip Howard, Blue sky, Bush, Car, Cathedral, Church, Cottage, England, Europe, Flower, French-Gothic, Neighborhood, Objects, Parking, Red Car, Roof, Sky

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
| mlx-community/LFM2.5-VL-1.6B-bf16                | 16a710cf8afca206ff16a95a4ad6fe657f876ce1 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/llava-v1.6-mistral-7b-8bit         | b8df5f329d95a7abe6429ed46093f9b84e8e6396 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/Qwen3-VL-2B-Thinking-bf16          | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/GLM-4.1V-9B-Thinking-8bit          | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.11                                                                                                                                          |
| mlx                        | 0.32.1.dev20260809+8c28c385f                                                                                                                    |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.7                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.27.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.6.0                                                                                                                                   |
| macOS Version              | 26.6.1                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (165,677,336 bytes, sha256=26e8fdf2551230764eb86a576256ee63d77ea1d7bde4fc72faa3ecc6eafdd95f) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,740,096 bytes, sha256=511ad05d90fe6e27ae31250a8ca8906f0416b0da931c490cff61b02b519b4303)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
