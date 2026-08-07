# Diagnostics

## Run Summary

Outcome counts

| Outcome             |   Count |
|---------------------|---------|
| Attempted           |      64 |
| Conclusive outcomes |      64 |
| Completed           |      63 |
| Crashed             |       1 |
| Indeterminate       |       0 |

Maintainer status counts

| Maintainer status              |   Count |
|--------------------------------|---------|
| actionable failure             |       1 |
| none                           |      53 |
| observation needs reproduction |      10 |

Usability counts

| Usability           |   Count |
|---------------------|---------|
| not evaluated       |       1 |
| unusable            |       4 |
| usable              |      53 |
| usable with caveats |       6 |

Observation counts

| Observation                                      |   Count |
|--------------------------------------------------|---------|
| Conversation-role control tokens remain visible  |       3 |
| Internal reasoning block appears incomplete      |       1 |
| Response appears cut off at the token limit      |       4 |
| Response is unusually short                      |       1 |
| Response repeats the same text                   |       1 |
| Unrecognised model control tokens remain visible |       2 |

## Triage

| Model                                                                                                           | Execution   | Usability           | Maintainer status              | Observations                                      |
|-----------------------------------------------------------------------------------------------------------------|-------------|---------------------|--------------------------------|---------------------------------------------------|
| [mlx-community/Inkling-Small-mlx-4bit](#diagnostic-mlx-community-inkling-small-mlx-4bit)                        | crashed     | not_evaluated       | actionable_failure             | none                                              |
| [mlx-community/gemma-3n-E2B-4bit](#diagnostic-mlx-community-gemma-3n-e2b-4bit)                                  | completed   | unusable            | observation_needs_reproduction | repeated text; cut off at token limit             |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed   | usable_with_caveats | observation_needs_reproduction | control tokens visible                            |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed   | usable_with_caveats | observation_needs_reproduction | control tokens visible                            |
| [mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX](#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx)       | completed   | unusable            | observation_needs_reproduction | cut off at token limit                            |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed   | unusable            | observation_needs_reproduction | cut off at token limit                            |
| [mlx-community/Kimi-VL-A3B-Thinking-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit)                  | completed   | unusable            | observation_needs_reproduction | cut off at token limit; incomplete thinking block |
| [jqlive/Kimi-VL-A3B-Thinking-2506-6bit](#diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit)                      | completed   | usable_with_caveats | observation_needs_reproduction | role tokens visible                               |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed   | usable_with_caveats | observation_needs_reproduction | role tokens visible                               |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed   | usable_with_caveats | observation_needs_reproduction | role tokens visible                               |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed   | usable_with_caveats | observation_needs_reproduction | very short response                               |

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
- *Phase:* model_load
- *Stage:* Model Error
- *Package:* mlx
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
- *Post-cleanup active memory (GB):* 0.00442481
- *Post-cleanup cache memory (GB):* 0.0

#### Complete traceback

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11977, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11313, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 712, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12230, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11992, in _run_model_generation
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

#### Captured stdout/stderr

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 54 files:   0%|          | 0/54 [00:00<?, ?it/s]
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 3343.46it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[18:42:53] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-gemma-3n-e2b-4bit"></a>

<details>
<summary>mlx-community/gemma-3n-E2B-4bit — unusable — repeated text; cut off at token limit</summary>

### mlx-community/gemma-3n-E2B-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Repeated fragment:* phrase: "this image this image..."
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* ec68dc186276e20e4bed30b96a2b5c667e0a81e3
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.017188058
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 264
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
 this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this image this
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
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.015566016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 280
- *Generation tokens:* 88
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>A high-angle, eye-level shot shows two tabby cats sleeping on a pink blanket on a red couch. The cat on the left is a dark brown tabby, lying on its side with its head tucked in. The cat on the right is a lighter brown and tan tabby, lying on its side with its head turned down. There are two remote controls between the cats. The lighting is bright and even.
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
- *Unexpected special tokens:* ["&lt;|channel&gt;", "&lt;channel|&gt;"]
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.015566016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 280
- *Generation tokens:* 85
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>A high-angle medium-shot shows two tabby cats sleeping on a pink blanket on a red couch. The cat on the left is lying on its side with its head curled up and its paws tucked in. The cat on the right is lying on its side with its head down and its legs stretched out. Two remote controls are on the couch between the cats. The lighting is bright and even.
```

</details>

<a id="diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

<details>
<summary>mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX — unusable — cut off at token limit</summary>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation
- *Token-cap degradation evidence:* ["abrupt_tail"]
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Processor class:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002065504
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1329
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Here are my reasoning steps:
The user asks: "Describe this image". The image is of two cats sleeping on a pink couch, with a remote control and a DVD case (or something) on the couch. The cats are both tabby, with similar patterns. They are lying on their sides, curled up, with paws extended. The remote is on the left side, the DVD case on the right side. The couch is pink. The cats appear to be sleeping. The cat on the left is lying on its side with its head turned slightly upward, its front paws stretched out, and its tail curled around its body. The cat on the right is lying on its side, with its head turned to the left, its front paws tucked under its chest, and its tail draped over its back. The remote is a typical TV remote with colored buttons. The DVD case is partially visible, showing a blue spine with some text. The couch is plush. The lighting is soft. The cats have a mix of brown, black, and tan stripes. The overall scene is cozy and relaxed.

We need to produce a description. The user didn't ask for anything else. So we can give a detailed description. The user is presumably just wanting a description. There's no disallowed content. So we can comply.

We should be mindful of the policy: no disallowed content. It's a normal image. So we can comply.

We can also mention the cat's positions relative to each other and the objects. The cat on the left is near the remote, the cat on the right near the DVD case. The couch is pink. The cats are tabby. The remote is white with black and colored buttons. The DVD case is blue with white text. The cats are sleeping. The scene is cozy.

We can also mention the cat's fur pattern: stripes, spots, etc. The cat's eyes are closed. The cat's ears are perked. The cat's paws are visible. The cat's tail is visible. The cat's whiskers are visible. The cat's nose is visible. The cat's ears are pointed. The cat's fur is fluffy.

We can also mention the couch's texture: plush, soft. The lighting is warm. The scene is likely in a living room.

We can also mention the remote is a typical TV remote with a power button, volume, channel, etc. The
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — cut off at token limit</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation
- *Thinking trace markers:* ["&lt;think&gt;", "&lt;/think&gt;"]
- *Token-cap degradation evidence:* ["dangling_markdown"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002557024
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 437
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Alright, let's get to work. The user wants a detailed description of this image, and I'm the one to provide it. First, I need to break down the request. The user has given me a complex prompt, a jumble of words that I need to decipher. It's a classic case of "gibberish," but I can handle it.

My initial step is to analyze the user's input. I see a jumble of words, a mix of cat breeds, colors, and objects. It's a mess, but I can see the core elements: cats, a couch, and a remote. The user wants a description, so I need to extract the meaning from this chaos.

Next, I'll sift through the text, pulling out the key elements. I'll identify the cats, the couch, and the remote. I'll also note the colors and positions. I'll ignore the gibberish, the nonsensical phrases.

Now, I'll synthesize the information. I'll start with the main subject: two cats on a couch. I'll describe their appearance, focusing on the colors and breeds. Then, I'll describe the setting: the couch, the remote, and the overall atmosphere. I'll make sure to mention the relaxed, cozy vibe.

Finally, I'll structure the description. I'll start with a general overview, then go into detail about the cats, the couch, and the remote. I'll use descriptive language to paint a picture. I'll make sure the description is clear, concise, and engaging. I'll also make sure to mention the "gibberish" in the prompt, but I won't let it dominate the description.

I'll review the final description to ensure it's accurate, comprehensive, and captures the essence of the image. I'll make sure it's well-written and easy to understand. I'll also make sure to mention the "gibberish" in the prompt, but I won't let it dominate the description.
</think>

Based on the image provided, here is a description of the scene:

This is a heartwarming and cozy image of two cats relaxing together on a plush, red couch.

*   **
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-8bit — unusable — cut off at token limit; incomplete thinking block</summary>

### mlx-community/Kimi-VL-A3B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, thinking_trace_incomplete
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace"]
- *Resolved model revision:* 85daf3dc2490c0f824143338f08ba45f475c9ce4
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00635813
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 397
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Okay, let me try to figure out how to describe this image. First, I need to look at the image carefully. There are two cats on a pink surface, maybe a blanket or a bed. Both cats seem to be lying down. The one on the left is a tabby with gray and black stripes, and the one on the right is a darker tabby, maybe a bit bigger. They both look like they're sleeping or resting. There's a remote control near the left cat and another one near the right cat. The background is pink, which might be a couch or a bed. The image is a bit blurry, especially around the edges, which makes it look like a low-resolution or heavily compressed image. The colors are a bit oversaturated, especially the pink background. The cats' fur has some highlights, maybe from light reflecting off them. The cats seem to be in a relaxed position, maybe after playing or moving around since one is stretched out. The remotes are standard TV remotes, which adds a bit of context that this might be a cozy indoor setting. The image might have been taken in a living room. The lighting seems artificial, maybe from a lamp or overhead light. The cats' eyes aren't clearly visible because they're closed. The overall mood is calm and relaxed. I should mention the colors, the subjects (cats and remotes), the setting, and the image quality issues like blurriness and color saturation. Also, the composition with the remotes adds a bit of a humorous touch, like the cats are ignoring the remotes. Maybe mention the possible breeds of the cats, but I'm not sure. The image might have been edited or filtered, given the color enhancements. The pink background contrasts with the cats' fur colors. The cats' positions suggest they might have been moving before settling down. The remotes are standard, not specialized, so maybe they're in a typical living room. The image could be a candid shot capturing a quiet moment. The blurriness might be due to motion or a low-quality camera. The lighting is even, not harsh shadows, so maybe it's evening or night time. The cats seem comfortable, indicating they feel safe. The image might be meant to be cute or humorous because of the remotes next to them. The pink background makes the cats stand out. The image could be shared on social media or as a pet owner's photo. The cats' poses are natural, not posed.
```

</details>

<a id="diagnostic-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

<details>
<summary>jqlive/Kimi-VL-A3B-Thinking-2506-6bit — usable_with_caveats — role tokens visible</summary>

### jqlive/Kimi-VL-A3B-Thinking-2506-6bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* 4b3b11ce0874c36a99e13e17e355049042f8620a
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.001213516
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 397
- *Generation tokens:* 221
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷So, let's analyze the image. The image shows two tabby cats lying on a pink fabric surface. There are two remote controls, one near each cat. The cats have a mix of brown, black, and white fur. The lighting is high-contrast, giving a stylized, almost posterized look. The scene is vibrant with the pink background. The cats appear relaxed, possibly sleeping or resting. The remotes are light blue with buttons, typical of TV or other electronic remotes. The overall style is artistic with strong color saturation.◁/think▷The image depicts two tabby cats resting on a vibrant pink fabric surface. One cat is positioned on the left, with a light blue remote control nearby, while the other cat is on the right, with a similar remote. Both cats have a mix of brown, black, and white fur patterns. The image has a stylized, high-contrast, posterized aesthetic, emphasizing vivid colors and sharp details. The cats appear relaxed, likely sleeping or resting, against the bright pink background.<|im_assistant|>
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
- *Role-boundary tokens in output:* ["&lt;end_of_utterance&gt;"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00442481
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2325
- *Generation tokens:* 354
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 The image depicts two cats lying on a pink fabric, possibly a blanket or a piece of clothing, which is draped over a piece of furniture, likely a couch or a chair. The cats are positioned side by side, and they appear to be in a relaxed state, possibly sleeping or resting. The fabric is a solid pink color, and it is smooth and soft-looking, suggesting it is made of a comfortable material like cotton or fleece.

The cats are both of a similar size and have a similar color pattern. They have a mix of brown and black stripes, with the brown stripes being more prominent. Their fur appears to be short and well-groomed, indicating that they are well-cared for. The cats' eyes are closed, and their ears are slightly tilted back, which is a common posture for cats when they are relaxed or sleeping.

In the background, there are two remote controls, one on each side of the cats. The remotes are white with black buttons, and they are placed on the pink fabric. The remotes are not in use, and they are simply resting on the fabric.

The overall scene is calm and serene, with the cats and remotes creating a cozy and comfortable setting. The pink fabric adds a soft and warm touch to the image, making it feel inviting and relaxing. The cats' presence adds a sense of companionship and warmth to the scene, making it a pleasant and homely image.

In summary, the image features two cats lying on a pink fabric, with two remotes in the background, creating a cozy and relaxed atmosphere. The cats are well-groomed and appear to be sleeping or resting, while the pink fabric and remotes add to the comfort and homeliness of the scene.<end_of_utterance>
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
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.005702762
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 397
- *Generation tokens:* 241
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷So, let's analyze the image. The image shows two tabby cats lying on a pink fabric surface. There are two remote controls, one near each cat. The cats have a mix of brown, black, and white fur. The lighting is high-contrast, making the colors very vivid. The scene is likely a cozy indoor setting, maybe a couch or bed. The cats appear relaxed, possibly sleeping or resting. The remotes are light blue with buttons, typical of TV or other electronic remotes. The overall style is a bit stylized with high saturation, giving it a vibrant look.◁/think▷The image depicts two tabby cats resting on a pink fabric surface. One cat is positioned on the left, with a light blue remote control near its head, while the other cat is on the right, with a similar remote control closer to its body. Both cats have a mix of brown, black, and white fur patterns. The scene has a high-contrast, stylized appearance with vivid colors, suggesting a cozy indoor setting, likely a couch or bed. The lighting emphasizes the cats' fur details and the texture of the fabric.<|im_assistant|>
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — usable_with_caveats — very short response</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* minimal_output
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.02413492
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4101
- *Generation tokens:* 2
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
Cat
```

</details>

## Indeterminate Attempts

None.

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                        | Performance                                                                                    |
|-------------------------------------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------------------|
| HuggingFaceTB/SmolVLM-Instruct                        | rev 81cd9a775a4d; Idefics3Processor; stop completed     | 1194 prompt / 63 generated; 128 tok/s; 5.5 GB peak; cleanup 0.000116/0.0 GB active/cache       |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | rev 6c33f49ebc0b; Lfm2VlProcessor; stop completed       | 78 prompt / 169 generated; 523 tok/s; 1.0 GB peak; cleanup 0.000247/0.0 GB active/cache        |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | rev 9eb2daaa8597; MllamaProcessor; stop completed       | 15 prompt / 130 generated; 4.96 tok/s; 25 GB peak; cleanup 0.00174/0.0 GB active/cache         |
| microsoft/Phi-3.5-vision-instruct                     | rev 12b77fb40b63; Phi3VProcessor; stop completed        | 768 prompt / 20 generated; 51.9 tok/s; 9.3 GB peak; cleanup 0.0018/0.0 GB active/cache         |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed     | 439 prompt / 107 generated; 25.1 tok/s; 20 GB peak; cleanup 0.00233/0.0 GB active/cache        |
| mlx-community/FastVLM-0.5B-bf16                       | rev 81ffe9290466; FastVLMProcessor; stop completed      | 24 prompt / 428 generated; 351 tok/s; 2.0 GB peak; cleanup 0.00287/0.0 GB active/cache         |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed       | 273 prompt / 303 generated; 30.5 tok/s; 18 GB peak; cleanup 0.0161/0.0 GB active/cache         |
| mlx-community/gemma-3-27b-it-qat-8bit                 | rev c408904bc9a0; Gemma3Processor; stop completed       | 273 prompt / 288 generated; 17.0 tok/s; 32 GB peak; cleanup 0.0166/0.0 GB active/cache         |
| mlx-community/gemma-3n-E4B-it-bf16                    | rev d9c02d0b2fa8; Gemma3nProcessor; stop completed      | 272 prompt / 106 generated; 48.2 tok/s; 17 GB peak; cleanup 0.0177/0.0 GB active/cache         |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed       | 284 prompt / 148 generated; 131 tok/s; 16 GB peak; cleanup 0.0183/0.0 GB active/cache          |
| mlx-community/gemma-4-31b-bf16                        | rev 19f0f1af698c; Gemma4Processor; stop completed       | 272 prompt / 15 generated; insufficient sample; 63 GB peak; cleanup 0.0188/0.0 GB active/cache |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed       | 284 prompt / 98 generated; 26.9 tok/s; 19 GB peak; cleanup 0.0193/0.0 GB active/cache          |
| mlx-community/GLM-4.1V-9B-Thinking-8bit               | rev 9677807f1065; Glm46VProcessor; stop completed       | 402 prompt / 301 generated; 48.1 tok/s; 13 GB peak; cleanup 0.00318/0.0 GB active/cache        |
| mlx-community/GLM-4.6V-Flash-6bit                     | rev df9464782d34; Glm46VProcessor; stop completed       | 406 prompt / 197 generated; 63.6 tok/s; 10 GB peak; cleanup 0.00351/0.0 GB active/cache        |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | rev 773591fa7388; Glm46VProcessor; stop completed       | 406 prompt / 205 generated; 91.7 tok/s; 7.7 GB peak; cleanup 0.00383/0.0 GB active/cache       |
| mlx-community/GLM-4.6V-nvfp4                          | rev 2da6855d4e28; Glm46VMoEProcessor; stop completed    | 406 prompt / 172 generated; 51.3 tok/s; 63 GB peak; cleanup 0.00416/0.0 GB active/cache        |
| mlx-community/InternVL3-14B-8bit                      | rev 50efc568c7df; InternVLChatProcessor; stop completed | 3340 prompt / 77 generated; 28.2 tok/s; 19 GB peak; cleanup 0.00474/0.0 GB active/cache        |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed | 3340 prompt / 99 generated; 32.3 tok/s; 18 GB peak; cleanup 0.00505/0.0 GB active/cache        |
| mlx-community/LFM2-VL-1.6B-8bit                       | rev 294b90e5ae23; Lfm2VlProcessor; stop completed       | 267 prompt / 103 generated; 315 tok/s; 3.0 GB peak; cleanup 0.00649/0.0 GB active/cache        |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | rev 16a710cf8afc; Lfm2VlProcessor; stop completed       | 267 prompt / 108 generated; 185 tok/s; 4.1 GB peak; cleanup 0.00662/0.0 GB active/cache        |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | rev 8451adc50203; MllamaProcessor; stop completed       | 14 prompt / 155 generated; 16.5 tok/s; 15 GB peak; cleanup 0.00714/0.0 GB active/cache         |
| mlx-community/llava-v1.6-mistral-7b-8bit              | rev b8df5f329d95; LlavaNextProcessor; stop completed    | 2354 prompt / 82 generated; 62.0 tok/s; 9.7 GB peak; cleanup 0.0194/0.0 GB active/cache        |
| mlx-community/MiniCPM-V-4.6-8bit                      | rev 03721395f6b8; MiniCPMVProcessor; stop completed     | 226 prompt / 112 generated; 252 tok/s; 3.0 GB peak; cleanup 0.00767/0.0 GB active/cache        |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed     | 972 prompt / 155 generated; 65.1 tok/s; 9.8 GB peak; cleanup 0.00793/0.0 GB active/cache       |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed     | 972 prompt / 156 generated; 60.0 tok/s; 10 GB peak; cleanup 0.00819/0.0 GB active/cache        |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed     | 971 prompt / 178 generated; 188 tok/s; 4.5 GB peak; cleanup 0.00846/0.0 GB active/cache        |
| mlx-community/Molmo-7B-D-0924-8bit                    | rev 90a14ed7a230; MolmoProcessor; stop completed        | 747 prompt / 174 generated; 50.2 tok/s; 11 GB peak; cleanup 0.00877/0.0 GB active/cache        |
| mlx-community/Molmo-7B-D-0924-bf16                    | rev d871cbdb87a4; MolmoProcessor; stop completed        | 747 prompt / 199 generated; 29.5 tok/s; 18 GB peak; cleanup 0.00908/0.0 GB active/cache        |
| mlx-community/MolmoPoint-8B-fp16                      | rev 0a60033b4e48; MolmoPointProcessor; stop completed   | 764 prompt / 158 generated; 5.95 tok/s; 23 GB peak; cleanup 0.0097/0.0 GB active/cache         |
| mlx-community/nanoLLaVA-1.5-4bit                      | rev 524020474496; Qwen2Tokenizer; stop completed        | 20 prompt / 123 generated; 365 tok/s; 1.8 GB peak; cleanup 0.02/0.0 GB active/cache            |
| mlx-community/Ornith-1.0-35B-bf16                     | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed      | 317 prompt / 305 generated; 59.7 tok/s; 71 GB peak; cleanup 0.0102/0.0 GB active/cache         |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | rev 1485fa9b3c7a; PaliGemmaProcessor; stop completed    | 1029 prompt / 180 generated; 33.3 tok/s; 11 GB peak; cleanup 0.021/0.0 GB active/cache         |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | rev 7c412694b919; PaliGemmaProcessor; stop completed    | 1029 prompt / 159 generated; 5.34 tok/s; 26 GB peak; cleanup 0.0221/0.0 GB active/cache        |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | rev f66333527ce7; PaliGemmaProcessor; stop completed    | 1029 prompt / 126 generated; 18.8 tok/s; 10 GB peak; cleanup 0.0231/0.0 GB active/cache        |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed        | 768 prompt / 20 generated; 59.0 tok/s; 9.3 GB peak; cleanup 0.0103/0.0 GB active/cache         |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed      | 1237 prompt / 199 generated; 34.2 tok/s; 15 GB peak; cleanup 0.0244/0.0 GB active/cache        |
| mlx-community/pixtral-12b-bf16                        | rev 378cf65efd79; PixtralProcessor; stop completed      | 1237 prompt / 204 generated; 18.3 tok/s; 27 GB peak; cleanup 0.0247/0.0 GB active/cache        |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | rev 01af461cdb95; Qwen2VLProcessor; stop completed      | 415 prompt / 69 generated; 317 tok/s; 2.5 GB peak; cleanup 0.0106/0.0 GB active/cache          |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | rev c8a67a843274; Qwen3VLProcessor; stop completed      | 313 prompt / 281 generated; 121 tok/s; 5.2 GB peak; cleanup 0.0109/0.0 GB active/cache         |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | rev c325e5ea14c2; Qwen3VLProcessor; stop completed      | 315 prompt / 401 generated; 123 tok/s; 5.3 GB peak; cleanup 0.0112/0.0 GB active/cache         |
| mlx-community/Qwen3.5-27B-4bit                        | rev 45797d2985a1; Qwen3VLProcessor; stop completed      | 317 prompt / 315 generated; 29.6 tok/s; 19 GB peak; cleanup 0.0117/0.0 GB active/cache         |
| mlx-community/Qwen3.5-27B-mxfp8                       | rev 2d6caf2325c2; Qwen3VLProcessor; stop completed      | 317 prompt / 305 generated; 17.2 tok/s; 30 GB peak; cleanup 0.0122/0.0 GB active/cache         |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed      | 317 prompt / 326 generated; 103 tok/s; 21 GB peak; cleanup 0.0127/0.0 GB active/cache          |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | rev b729d115bb2c; Qwen3VLProcessor; stop completed      | 317 prompt / 299 generated; 90.0 tok/s; 30 GB peak; cleanup 0.0132/0.0 GB active/cache         |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | rev 731d09ba3597; Qwen3VLProcessor; stop completed      | 317 prompt / 299 generated; 70.0 tok/s; 71 GB peak; cleanup 0.0137/0.0 GB active/cache         |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed      | 317 prompt / 265 generated; 98.9 tok/s; 7.0 GB peak; cleanup 0.0143/0.0 GB active/cache        |
| mlx-community/Qwen3.6-27B-mxfp8                       | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed      | 317 prompt / 283 generated; 19.3 tok/s; 30 GB peak; cleanup 0.0148/0.0 GB active/cache         |
| mlx-community/SmolVLM-Instruct-bf16                   | rev cae61cdedd06; Idefics3Processor; stop completed     | 1194 prompt / 63 generated; 125 tok/s; 5.5 GB peak; cleanup 0.0149/0.0 GB active/cache         |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | rev 844516024a1c; SmolVLMProcessor; stop completed      | 95 prompt / 143 generated; 132 tok/s; 5.5 GB peak; cleanup 0.015/0.0 GB active/cache           |
| mlx-community/Step-3.7-Flash-oQ2e                     | rev 3dacb46f724a; Step3VLProcessor; stop completed      | 188 prompt / 113 generated; 47.2 tok/s; 65 GB peak; cleanup 0.0153/0.0 GB active/cache         |
| mlx-community/X-Reasoner-7B-8bit                      | rev 21732e74613b; Qwen2_5_VLProcessor; stop completed   | 415 prompt / 104 generated; 66.1 tok/s; 10 GB peak; cleanup 0.0156/0.0 GB active/cache         |
| qnguyen3/nanoLLaVA                                    | rev 13d60cec183a; Qwen2Tokenizer; stop completed        | 20 prompt / 44 generated; 91.0 tok/s; 4.0 GB peak; cleanup 0.0253/0.0 GB active/cache          |
| Qwen/Qwen3-VL-2B-Instruct                             | rev 89644892e4d8; Qwen3VLProcessor; stop completed      | 313 prompt / 281 generated; 136 tok/s; 5.2 GB peak; cleanup 0.000558/0.0 GB active/cache       |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 640 x 480 pixels
- *Image size:* 173,131 bytes
- *Image SHA-256:* dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e

<details>
<summary>Exact prompt</summary>

```text
Describe this image
```

</details>

The original local input is not published, so this report does not claim a
complete reproduction command. Use a shareable equivalent image or add the
original image before filing.

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/Inkling-Small-mlx-4bit             | f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612 |
| mlx-community/gemma-3n-E2B-4bit                  | ec68dc186276e20e4bed30b96a2b5c667e0a81e3 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX    | 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2 |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/Kimi-VL-A3B-Thinking-8bit          | 85daf3dc2490c0f824143338f08ba45f475c9ce4 |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit            | 4b3b11ce0874c36a99e13e17e355049042f8620a |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.11                                                                                                                                          |
| mlx                        | 0.32.1.dev20260807+8056817bd                                                                                                                    |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (164,591,992 bytes, sha256=521bd22d775f929912876487aa2def0a121b9b574141b0a47583b5f3c80e01ed) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,678,352 bytes, sha256=8981b1ac0f9a8fe29383470e279aa887ca562237e73a3fe9e8835f623d791776)  |
| RAM                        | 128.0 GB                                                                                                                                        |
