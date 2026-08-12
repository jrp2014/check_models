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
| none                           | 27    |
| observation needs reproduction | 14    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 16    |
| usable              | 14    |
| usable with caveats | 11    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| No response text was returned                                                         | 1     |
| Response repeats the same text                                                        | 4     |
| Unrecognised model control tokens remain visible                                      | 3     |
| Required fields are missing or empty                                                  | 12    |
| Response repeats the task instructions instead of only returning the requested fields | 2     |
| Extra text appears before the Title field                                             | 4     |
| Response appears cut off at the token limit                                           | 10    |
| Internal reasoning block appears incomplete                                           | 5     |
| Conversation-role control tokens remain visible                                       | 1     |
| Title or keywords do not meet requested constraints                                   | 12    |

## Triage

| Model                                                                                                           | Execution | Usability           | Maintainer status              | Observations                                                                                                                      |
|-----------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/Inkling-Small-mlx-4bit](#diagnostic-mlx-community-inkling-small-mlx-4bit)                        | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                              |
| [mlx-community/gemma-3n-E4B-it-bf16](#diagnostic-mlx-community-gemma-3n-e4b-it-bf16)                            | completed | unusable            | observation_needs_reproduction | empty response; missing required fields                                                                                           |
| [mlx-community/GLM-4.6V-Flash-mxfp4](#diagnostic-mlx-community-glm-46v-flash-mxfp4)                             | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; cut off at token limit                                                                    |
| [mlx-community/SmolVLM2-2.2B-Instruct-mlx](#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx)                 | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                           |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                            |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                            |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                            |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block                               |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                   | completed | unusable            | observation_needs_reproduction | missing required fields; cut off at token limit; incomplete thinking block                                                        |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block                               |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                  | completed | unusable            | observation_needs_reproduction | missing required fields; cut off at token limit; incomplete thinking block                                                        |
| [mlx-community/MiniCPM-V-4.6-8bit](#diagnostic-mlx-community-minicpm-v-46-8bit)                                 | completed | unusable            | observation_needs_reproduction | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible; title/keyword constraints failed                                                                             |

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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12108, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11587, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 786, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 966, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 842, in load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12536, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12123, in _run_model_generation
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
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 2939.06it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[22:27:38] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
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
- *Post-cleanup active memory (GB):* 0.011388122
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 602
- *Generation tokens:* 1
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
(empty)
```

</details>

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
- *Repeated fragment:* keyword: "cityscape"
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 156
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["bristol marina", "bristol uk", "boats docked",
  "seagulls", "historic architecture", "cityscape", "england", "europe",
  "marina", "dock", "boat", "water", "birds", "city centre", "bristol
  cathedral", "canons marsh", "coyote ugly", "building", "architecture",
  "great britain", "gbr", "uk", "bristol", "marina view", "docked boats",
  "seagulls in water", "historic buildings"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002114608
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6367
- *Generation tokens:* 500
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
Title: Bristol Marina View
Description: An image of boats docked at a marina in Bristol, England, with seagulls floating in the foreground and historic architecture visible in the background.
Keywords: Bristol Marina, Bristol UK, Boats Docked, Seagulls, Historic Architecture, Cityscape, England, Europe, Marina, Dock, Boat, Water, Birds, City Centre, Bristol Cathedral, Canons Marsh, Coyote Ugly, Building, Architecture, Great Britain, GBR, Europe, UK, England, Bristol, Marina View, Docked Boats, Seagulls in Water, Historic Buildings, Cityscape, Bristol Marina, Bristol UK, Boats Docked, Seagulls, Historic Architecture, Cityscape, England, Europe, Marina, Dock, Boat, Water, Birds, City Centre, Bristol Cathedral, Canons Marsh, Coyote Ugly, Building, Architecture, Great Britain, GBR, Europe, UK, England, Bristol, Marina View, Docked Boats, Seagulls in Water, Historic Buildings, Cityscape, Bristol Marina, Bristol UK, Boats Docked, Seagulls, Historic Architecture, Cityscape, England, Europe, Marina, Dock, Boat, Water, Birds, City Centre, Bristol Cathedral, Canons Marsh, Coyote Ugly, Building, Architecture, Great Britain, GBR, Europe, UK, England, Bristol, Marina View, Docked Boats, Seagulls in Water, Historic Buildings, Cityscape, Bristol Marina, Bristol UK, Boats Docked, Seagulls, Historic Architecture, Cityscape, England, Europe, Marina, Dock, Boat, Water, Birds, City Centre, Bristol Cathedral, Canons Marsh, Coyote Ugly, Building, Architecture, Great Britain, GBR, Europe, UK, England, Bristol, Marina View, Docked Boats, Seagulls in Water, Historic Buildings, Cityscape, Bristol Marina, Bristol UK, Boats Docked, Seagulls, Historic Architecture, Cityscape, England, Europe, Marina, Dock, Boat, Water, Birds, City Centre, Bristol Cathedral, Canons Marsh, Coyote Ugly, Building, Architecture, Great Britain, GBR, Europe, UK, England, Bristol, Marina View, Docked Boats, Seagulls in Water, Historic Buildings, Cityscape, Bristol Marina, Bristol UK, Boats Docked, Seagulls, Historic Architecture, City
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated text; missing required fields; cut off at token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* Coastline,
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.014157064
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4431
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Architecture, Building, City Centre, Cityscape, Coast, Coastline, Coastal, Coastal, Coastal waters, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline, Coastline,
```

</details>

<a id="diagnostic-mlx-community-smolvlm2-22b-instruct-mlx"></a>

<details>
<summary>mlx-community/SmolVLM2-2.2B-Instruct-mlx — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Repeated fragment:* keyword: "marina"
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 174
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["england", "europe", "marina", "church", "city",
  "cityscape", "coyote ugly", "dock", "great britain", "gull", "harbor",
  "architecture", "building", "canons marsh"]
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Processor class:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.009733256
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 441
- *Generation tokens:* 500
- *Configured EOS token ID:* 49279
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 Title: Bristol, England, UK, GBR, Europe
Description: A marina with boats docked, seagulls flying and floating in the water, and a church in the background.
Keywords: Bristol, England, UK, GBR, Europe, Marina, Boat, Boats, Church, City, City Centre, Cityscape, Coyote Ugly, Dock, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building, Canons Marsh, Church, City, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Architecture, Building,
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
- *Repeated fragment:* keyword: "dock"
- *Title word count:* 7
- *Requested title word range:* [5, 10]
- *Keyword count:* 172
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["bristol", "england", "uk", "europe", "great
  britain", "marina", "ferris wheel", "seagulls", "waterfront", "cityscape",
  "architecture", "church", "dock", "harbor", "gull"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.010306732
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16572
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Bristol Marina with Ferris Wheel and Cathedral

Description:
A vibrant scene of boats docked at a marina in Bristol, with the iconic Ferris wheel and Bristol Cathedral rising in the background, while seagulls float and fly in the foreground under a bright, partly cloudy sky.

Keywords: Bristol, England, UK, Europe, Great Britain, Marina, Boats, Ferris Wheel, Cathedral, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Harbor, Urban, Tourism, Landmark, Gull, City Centre, Bristol Cathedral, Canons Marsh, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock, Ferris Wheel, Bristol, England, UK, Europe, Great Britain, Gull, Harbor, Marina, Seagulls, Waterfront, Cityscape, Architecture, Church, Dock
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
- *Post-cleanup active memory (GB):* 0.010306752
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 604
- *Generation tokens:* 87
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Boats at Canons Marsh Marina in Bristol City Centre
Description: Numerous boats are docked at the marina in Bristol, featuring a flock of seagulls floating in the water with Bristol Cathedral and a Ferris wheel in the background.
Keywords: Bristol, England, Canons Marsh, Marina, Boats, Cathedral, Ferris wheel, Seagull, Cityscape, Architecture, Harbour, Dock, Waterfront, UK, Europe
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
- *Post-cleanup active memory (GB):* 0.010306752
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 604
- *Generation tokens:* 81
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Boats at Canons Marsh Marina near Bristol Cathedral
Description: Various motorboats are docked at the marina in Bristol, with a flock of seagulls floating and flying over the water in the foreground.
Keywords: Bristol, Canons Marsh, Cathedral, Marina, Boats, Seagulls, Harbour, Architecture, Cityscape, England, Europe, Water, Dock, Birds, Waterfront
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
- *Post-cleanup active memory (GB):* 0.002425916
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6367
- *Generation tokens:* 97
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: Bristol Marina with Seagulls and Cathedral
Description: A marina in Bristol, England, with boats docked at the pier, seagulls floating in the water, and Bristol Cathedral and a Ferris wheel visible in the background under a partly cloudy sky.
Keywords: Bristol, Marina, Boats, Seagulls, Bristol Cathedral, Ferris wheel, City Centre, England, UK, Europe, Harbor, Dock, Architecture, Building, Cityscape<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — missing required fields; extra text before Title; cut off at token limit; incomplete thinking block</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* Let's tackle this step by step. First, the
  title needs to be concrete, 5-10 words. The context says Bristol, UK, with
  landmarks like Bristol Cathedral, Coyote Ugly Bar, marina, seagulls. So
  maybe "Bristol Cathedral and Coyote Ugly Marina at Sunset" but wait, the
  capture time is 16:45 UTC+01:00, which is late afternoon, not sunset. Wait,
  the hint says "City Centre, Bristol, England, UK, GBR, Europe" so need to
  include key landmarks. Let's check the keywords: Bristol, Bristol Cathedral,
  Coyote Ugly, Marina, Seagulls, Harbour, Cityscape, England, UK, Europe. So
  title: "Bristol Cathedral and Coyote Ugly Marina with Seagulls at Dusk"
  (wait, 16:45 is late afternoon, dusk is around that time). Wait, but the
  hint says "City Centre, Bristol, England, UK, GBR, Europe" so maybe "Bristol
  City Centre Marina with Bristol Cathedral and Coyote Ugly Bar" but need to
  be concise. Wait, the user said "concrete 5-10-word title". Let's go with
  "Bristol Cathedral and Coyote Ugly Marina at Late Afternoon" (that's 8
  words).  Next, the description: needs to combine context (capture date/time:
  2026-08-12 16:45:16 UTC+01:00, GPS 51.451700°N, 2.600800°W which is Bristol)
  with main subject (boats at marina), setting (Bristol city centre, near
  Bristol Cathedral, Coyote Ugly Bar), action (seagulls flying/floating),
  lighting (late afternoon, since 16:45 is late afternoon). So: "A late
  afternoon view of boats docked at Bristol's marina near Bristol Cathedral
  and the Coyote Ugly Bar, with seagulls floating on the water in the city
  centre."
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001180718
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1643
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Let's tackle this step by step. First, the title needs to be concrete, 5-10 words. The context says Bristol, UK, with landmarks like Bristol Cathedral, Coyote Ugly Bar, marina, seagulls. So maybe "Bristol Cathedral and Coyote Ugly Marina at Sunset" but wait, the capture time is 16:45 UTC+01:00, which is late afternoon, not sunset. Wait, the hint says "City Centre, Bristol, England, UK, GBR, Europe" so need to include key landmarks. Let's check the keywords: Bristol, Bristol Cathedral, Coyote Ugly, Marina, Seagulls, Harbour, Cityscape, England, UK, Europe. So title: "Bristol Cathedral and Coyote Ugly Marina with Seagulls at Dusk" (wait, 16:45 is late afternoon, dusk is around that time). Wait, but the hint says "City Centre, Bristol, England, UK, GBR, Europe" so maybe "Bristol City Centre Marina with Bristol Cathedral and Coyote Ugly Bar" but need to be concise. Wait, the user said "concrete 5-10-word title". Let's go with "Bristol Cathedral and Coyote Ugly Marina at Late Afternoon" (that's 8 words). 

Next, the description: needs to combine context (capture date/time: 2026-08-12 16:45:16 UTC+01:00, GPS 51.451700°N, 2.600800°W which is Bristol) with main subject (boats at marina), setting (Bristol city centre, near Bristol Cathedral, Coyote Ugly Bar), action (seagulls flying/floating), lighting (late afternoon, since 16:45 is late afternoon). So: "A late afternoon view of boats docked at Bristol's marina near Bristol Cathedral and the Coyote Ugly Bar, with seagulls floating on the water in the city centre." 

Keywords: Need 10-18 unique, comma-separated. From context: Bristol, Bristol Cathedral, Coyote Ugly, Marina, Se
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — missing required fields; cut off at token limit; incomplete thinking block</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Missing sections:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001803312
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6363
- *Generation tokens:* 500
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image, it's Bristol's harbor with boats, cathedral, etc. So maybe "Bristol Harbour with Cathedral and Ferris Wheel". Wait, need to keep it concise. Let's check the hints: City Centre, Bristol, England... so maybe "Bristol Marina with Cathedral View". Wait, let's make it 5-10 words. How about "Bristol Harbour Scene with Cathedral and Boats". Hmm, maybe "Bristol City Centre Marina with Cathedral". Wait, let's go with something that's clear. Let's see the image has boats, cathedral, ferris wheel, water with birds. So title could be "Bristol Harbour with Cathedral and Boats". That's about 6 words.

Next, description: need 1-2 sentences. Combine context (capture date/time, GPS) with main visible subject (boats, marina), setting (city centre, harbor), action (seagulls flying/float), lighting (sunny, clear sky), distinctive details (cathedral, ferris wheel, Coyote Ugly building). So: "Captured on 2026-08-12 at 16:45 UTC+01:00, this image shows a marina in Bristol's city centre with boats docked, seagulls in the water, and the Bristol Cathedral and a Ferris wheel in the background under clear daylight." Wait, need to make it 1-2 sentences. Let's adjust: "This image, captured on 2026-08-12 at 16:45 UTC+01:00, depicts a marina in Bristol's city centre with boats, seagulls in the water, and the Bristol Cathedral and Ferris wheel visible in the background under bright daylight." That's two sentences? Wait, no, 1-2. Let's make it one: "Captured on 2026-08-12 at 16:45 UTC+01:00, the image shows a marina in Bristol's city centre with boats, seagulls in the water, and the Bristol Cathedral and Ferris wheel in the background under clear daylight." That's one sentence.

Then keywords: need 10-18 unique, comma-separated. From hints: Architecture, Bird, Boat, Boats, Bristol, Bristol Cathedral, Building,
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — missing required fields; extra text before Title; cut off at token limit; incomplete thinking block</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Missing sections:* ["title"]
- *Unexpected text before Title:* ◁think▷Got it, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. Let's start by analyzing the image and the hints.
  First, the context says the capture date/time is 2026-08-12 16:45:16
  UTC+01:00, and GPS is 51.451700°N, 2.600800°W. The location is City Centre,
  Bristol, England, UK, GBR, Europe. The description hint mentions boats
  docked at a marina, Bristol UK, seagulls in the foreground. Keywords include
  Architecture, Bird, Boat, Boats, Bristol, Bristol Cathedral, Building,
  Canons Marsh, Church, City, City Centre, Coyote Ugly, Dock, England, Europe,
  Great Britain, Gull, Harbor, Marina.  Now, the image shows a marina in
  Bristol with boats docked, seagulls in the water, buildings including what
  looks like Bristol Cathedral in the background, a Ferris wheel (maybe the
  "Coyote Ugly" place), and the cityscape. The sky is partly cloudy.  For the
  title, need a 5-10 word British-English title. Maybe "Bristol Marina with
  Boats and Seagulls in City Centre". Wait, check the keywords: "Bristol
  Cathedral" is a key element, so maybe include that. Also, "Coyote Ugly" is a
  known bar in Bristol, so that's a distinctive detail. So title could be
  "Bristol City Centre Marina with Boats, Seagulls, and Coyote Ugly". That's
  10 words.
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003671136
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1338
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

First, the context says the capture date/time is 2026-08-12 16:45:16 UTC+01:00, and GPS is 51.451700°N, 2.600800°W. The location is City Centre, Bristol, England, UK, GBR, Europe. The description hint mentions boats docked at a marina, Bristol UK, seagulls in the foreground. Keywords include Architecture, Bird, Boat, Boats, Bristol, Bristol Cathedral, Building, Canons Marsh, Church, City, City Centre, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina.

Now, the image shows a marina in Bristol with boats docked, seagulls in the water, buildings including what looks like Bristol Cathedral in the background, a Ferris wheel (maybe the "Coyote Ugly" place), and the cityscape. The sky is partly cloudy. 

For the title, need a 5-10 word British-English title. Maybe "Bristol Marina with Boats and Seagulls in City Centre". Wait, check the keywords: "Bristol Cathedral" is a key element, so maybe include that. Also, "Coyote Ugly" is a known bar in Bristol, so that's a distinctive detail. So title could be "Bristol City Centre Marina with Boats, Seagulls, and Coyote Ugly". That's 10 words.

Description: Need a 1-2 sentence factual description. Mention the marina, boats, seagulls, city elements like Bristol Cathedral, Coyote Ugly, and the setting. So: "A marina in Bristol's city centre featuring docked boats, seagulls in the water, and the iconic Bristol Cathedral in the background, with the 'Coyote Ugly' establishment visible among buildings under a partly cloudy sky."

Keywords: Need 10-18 unique keywords. Let's list them: Bristol, City Centre, Marina, Boats, Seagulls, Bristol Cathedral, Coyote Ugly, Building, England, Europe, Cityscape, Dock. That's 10. Maybe add "Harbor" instead of Building? Wait, the image has a harbor. Let's check the hints
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
- *Post-cleanup active memory (GB):* 0.00807847
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16563
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title. The context says Bristol, UK, and the image has boats, seagulls, marina. The title hint is City Centre, Bristol, England, UK, GBR, Europe. So need a concise title. Maybe "Bristol City Centre Marina with Seagulls" but let's check. The image has a marina with boats and seagulls, so "Bristol Marina, Seagulls, City Centre" but need 5-10 words. Let's see: "Bristol City Centre Marina with Seagulls" is 10 words. Wait, maybe "Bristol City Centre Marina, Seagulls, Boats" but the title should be catchy. Wait, the context says Bristol UK, so "Bristol City Centre Marina" is good.

Next, the description. Needs to combine context with main visible subject, setting, action, lighting, distinctive details. Context: capture date/time is 2026-08-12 16:45:16 UTC+01:00, GPS 51.451700N, 2.600800W. The image shows boats docked at a marina, seagulls in the water, buildings including Bristol Cathedral (the Gothic cathedral in the background), the "Coyote Ugly" building, and the Ferris wheel. The lighting is bright, sunny, clear sky. The main visible subject is the marina with boats and seagulls. So the description should mention Bristol City Centre, marina, boats, seagulls, Bristol Cathedral, sunny day, etc.

Then keywords: need 10-18 unique, comma-separated. From context: Architecture, Bird, Boat, Boats, Bristol, Bristol Cathedral, Building, Canons Marsh, Church, City, City Centre, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina. But need to check which are visible. The image has Bristol Cathedral (yes, the Gothic building in the background), Coyote Ugly building (the green and red building with "Coyote Ugly" sign), marina, boats, seagulls (gulls), harbor, cityscape, Bristol, etc. Let's list visible ones: Architecture (yes, the cathedral
```

</details>

<a id="diagnostic-mlx-community-minicpm-v-46-8bit"></a>

<details>
<summary>mlx-community/MiniCPM-V-4.6-8bit — unusable — echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/MiniCPM-V-4.6-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, prompt_instruction_echo,
  unexpected_catalog_preamble, thinking_trace_incomplete,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Echoed instruction fragments:* ["main visible subject setting action
  lighting and distinctive"]
- *Unexpected text before Title:* &lt;think&gt; We need to create a title,
  description, and keywords based on the image. Let's analyze the image.  The
  image shows a harbor or marina with boats, seagulls, buildings, and a
  cathedral (Bristol Cathedral?). There's a ferris wheel and some modern
  buildings. The context is Bristol, UK.  First, title: need 5-10 words. Let's
  think: "Boat marina with seagulls and Bristol cathedral". That's 6 words.
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 26
- *Requested keyword count range:* [10, 18]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "abrupt_tail"]
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.004850798
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 945
- *Generation tokens:* 500
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
<think>
We need to create a title, description, and keywords based on the image. Let's analyze the image.

The image shows a harbor or marina with boats, seagulls, buildings, and a cathedral (Bristol Cathedral?). There's a ferris wheel and some modern buildings. The context is Bristol, UK.

First, title: need 5-10 words. Let's think: "Boat marina with seagulls and Bristol cathedral". That's 6 words.

Description: Combine relevant context and main visible subject, setting, action, lighting, and distinctive details. The main subject is boats docked at a marina, with seagulls in the water, and background showing Bristol Cathedral and other buildings. Lighting is sunny with clouds.

Keywords: need 10-18 unique, comma-separated. Cover relevant context and visible details. Let's list: boat, marina, seagulls, Bristol, cathedral, buildings, water, dock, ferry, trees, sky, Europe, UK, England, Gull, harbor, city, architecture, modern, red building, white boats, etc.

Make sure to be unique and cover the main elements.

Now, structure:

Title: [concrete 5-10 words]

Description: [1-2 sentences]

Keywords: [10-18 unique, comma-separated]

Let's draft.

Title: Boat marina with seagulls and Bristol cathedral.

Description: Shows boats docked at a marina with seagulls in the water, featuring Bristol Cathedral and urban buildings under sunny skies.

Keywords: boat, marina, seagulls, Bristol, cathedral, buildings, water, dock, harbor, city, architecture, England, UK, Europe, Gull, trees, sky, modern, red building, white boats.

Check count: title 5-10 words: "Boat marina with seagulls and Bristol cathedral" is 5 words. Good.

Description: two sentences, combine context and main visible subject, setting, action, lighting, distinctive details. Looks okay.

Keywords: list 10-18 unique terms. Let's ensure they are unique and cover the relevant context and visible details.

List: boat, marina, seagulls, Bristol, cathedral, buildings, water, dock, harbor, city, architecture, England, UK, Europe, Gull, trees
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
- *Title word count:* 11
- *Requested title word range:* [5, 10]
- *Keyword count:* 15
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00268806
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2627
- *Generation tokens:* 208
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 Title: Boats Docked at Bristol Marina with City Centre in the Background.
Description: This image depicts a marina in Bristol, England, featuring a variety of boats docked along the waterfront. The boats vary in size and design, with some having multiple decks and others appearing more compact. The marina is surrounded by a mix of greenery, including trees and shrubs, which provide a natural backdrop. In the background, the cityscape of Bristol is visible, with notable structures such as the Bristol Cathedral and other buildings. The sky is partly cloudy, with patches of sunlight illuminating the scene. The water in the marina is calm, and a flock of seagulls can be seen flying and floating on the surface. The overall atmosphere is serene and picturesque, capturing the essence of a tranquil day by the water.
Keywords: Bristol, marina, boats, dock, cityscape, Bristol Cathedral, seagulls, water, trees, shrubs, calm, partly cloudy, sunlight, serene, picturesque.<end_of_utterance>
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
| mlx-community/FastVLM-0.5B-bf16               | unusable            | missing required fields                                                                       |
| mlx-community/llava-v1.6-mistral-7b-8bit      | unusable            | missing required fields                                                                       |
| mlx-community/MolmoPoint-8B-fp16              | unusable            | missing required fields                                                                       |
| mlx-community/nanoLLaVA-1.5-4bit              | unusable            | missing required fields                                                                       |
| mlx-community/Qwen2-VL-2B-Instruct-4bit       | unusable            | missing required fields                                                                       |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16              | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/gemma-4-26b-a4b-it-4bit         | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/InternVL3-8B-bf16               | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/LFM2.5-VL-1.6B-bf16             | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Molmo-7B-D-0924-8bit            | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Qwen3-VL-2B-Instruct-bf16       | usable_with_caveats | title/keyword constraints failed                                                              |
| Qwen/Qwen3-VL-2B-Instruct                     | usable_with_caveats | title/keyword constraints failed                                                              |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                    | Performance                                                                                 |
|-------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------|
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed | 2402 prompt / 106 generated; 24.8 tok/s; 23 GB peak; cleanup 0.000968/0.0 GB active/cache   |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed   | 603 prompt / 131 generated; 29.7 tok/s; 18 GB peak; cleanup 0.0108/0.0 GB active/cache      |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed   | 608 prompt / 102 generated; 27.6 tok/s; 20 GB peak; cleanup 0.0124/0.0 GB active/cache      |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | rev 8451adc50203; MllamaProcessor; stop completed   | 316 prompt / 89 generated; 12.5 tok/s; 15 GB peak; cleanup 0.00433/0.0 GB active/cache      |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed | 2935 prompt / 104 generated; 55.8 tok/s; 13 GB peak; cleanup 0.00511/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed | 2935 prompt / 170 generated; 45.8 tok/s; 13 GB peak; cleanup 0.00538/0.0 GB active/cache    |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed | 2934 prompt / 105 generated; 147 tok/s; 7.8 GB peak; cleanup 0.00564/0.0 GB active/cache    |
| mlx-community/Ornith-1.0-35B-bf16                     | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed  | 16576 prompt / 124 generated; 59.1 tok/s; 74 GB peak; cleanup 0.00708/0.0 GB active/cache   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed    | 1144 prompt / 98 generated; 53.7 tok/s; 9.4 GB peak; cleanup 0.00714/0.0 GB active/cache    |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed  | 3125 prompt / 113 generated; 39.6 tok/s; 16 GB peak; cleanup 0.0144/0.0 GB active/cache     |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed  | 16576 prompt / 111 generated; 102 tok/s; 24 GB peak; cleanup 0.00859/0.0 GB active/cache    |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed  | 16576 prompt / 117 generated; 68.0 tok/s; 10.0 GB peak; cleanup 0.00911/0.0 GB active/cache |
| mlx-community/Qwen3.6-27B-mxfp8                       | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed  | 16576 prompt / 125 generated; 15.0 tok/s; 35 GB peak; cleanup 0.00962/0.0 GB active/cache   |
| mlx-community/Step-3.7-Flash-oQ2e                     | rev 3dacb46f724a; Step3VLProcessor; stop completed  | 3503 prompt / 116 generated; 42.7 tok/s; 70 GB peak; cleanup 0.01/0.0 GB active/cache       |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,049 x 6,032 pixels
- *Image size:* 54,967,584 bytes
- *Image SHA-256:* 49cac3fb93699eb78136c721288d36296a56b950911c604e4b91a246d8132e9a

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-12 16:45:16 UTC+01:00
- GPS: 51.451700°N, 2.600800°W

Descriptive hints:
- Title hint: City Centre, Bristol, England, UK, GBR, Europe
- Description hint: This is an image of boats docked at a marina. Bristol UK. A flock of seagulls flies and floats in the water in the foreground.
- Keyword hints: Architecture, Bird, Boat, Boats, Bristol, Bristol Cathedral, Building, Canons Marsh, Church, City, City Centre, Cityscape, Coyote Ugly, Dock, England, Europe, Great Britain, Gull, Harbor, Marina

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
| mlx-community/gemma-3n-E4B-it-bf16               | d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a |
| mlx-community/GLM-4.6V-Flash-mxfp4               | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx         | 844516024a1c4400d34489b89ee067d794e432ed |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/GLM-4.1V-9B-Thinking-8bit          | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/Qwen3-VL-2B-Thinking-bf16          | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/MiniCPM-V-4.6-8bit                 | 03721395f6b82cd000cc74cde28fcff8abd9a04c |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.13                                                                                                                                          |
| mlx                        | 0.32.1.dev20260812+52960f80f                                                                                                                    |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.8                                                                                                                                           |
| transformers               | 5.15.0                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (174,800,976 bytes, sha256=7841f37301ab31cebba1227b6cdc3bb59d91dd35ce1586512266a8997e33b9d6) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,961,152 bytes, sha256=a0d96669c1c798f10b8e9c1be789b0eb6aadba46d0818ed94d2af055e70c36b6)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
