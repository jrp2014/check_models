# Diagnostics

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
| none                           | 12    |
| observation needs reproduction | 29    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 16    |
| usable              | 12    |
| usable with caveats | 13    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Conversation-role control tokens remain visible                                       | 1     |
| Extra text appears before the Title field                                             | 6     |
| Internal reasoning block appears incomplete                                           | 3     |
| Required fields are missing or empty                                                  | 15    |
| Response appears cut off at the token limit                                           | 8     |
| Response repeats the same text                                                        | 3     |
| Response repeats the task instructions instead of only returning the requested fields | 2     |
| Title or keywords do not meet requested constraints                                   | 13    |
| Title, Description and Keywords copy all supplied hints unchanged                     | 1     |
| Unrecognised model control tokens remain visible                                      | 1     |

## Triage

| Model                                                                                                                      | Execution | Usability           | Maintainer status              | Observations                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-----------------------------------------------------------------------------------------------------|
| [mlx-community/Inkling-Small-mlx-4bit](#diagnostic-mlx-community-inkling-small-mlx-4bit)                                   | crashed   | not_evaluated       | actionable_failure             | none                                                                                                |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16)            | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; cut off at token limit                                      |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                             | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; cut off at token limit                                      |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                           | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                                    | completed | unusable            | observation_needs_reproduction | control tokens visible; missing required fields; extra text before Title                            |
| [mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX](#diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx)                  | completed | unusable            | observation_needs_reproduction | missing required fields; echoes instructions; cut off at token limit                                |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)                   | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title                                                    |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)                 | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title                                                    |
| [mlx-community/FastVLM-0.5B-bf16](#diagnostic-mlx-community-fastvlm-05b-bf16)                                              | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title                                                    |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                              | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)                   | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [mlx-community/LFM2.5-VL-1.6B-bf16](#diagnostic-mlx-community-lfm25-vl-16b-bf16)                                           | completed | unusable            | observation_needs_reproduction | missing required fields                                                                             |
| [mlx-community/llava-v1.6-mistral-7b-8bit](#diagnostic-mlx-community-llava-v16-mistral-7b-8bit)                            | completed | unusable            | observation_needs_reproduction | missing required fields                                                                             |
| [mlx-community/MiniCPM-V-4.6-8bit](#diagnostic-mlx-community-minicpm-v-46-8bit)                                            | completed | unusable            | observation_needs_reproduction | missing required fields; cut off at token limit; incomplete thinking block                          |
| [mlx-community/MolmoPoint-8B-fp16](#diagnostic-mlx-community-molmopoint-8b-fp16)                                           | completed | unusable            | observation_needs_reproduction | missing required fields                                                                             |
| [mlx-community/nanoLLaVA-1.5-4bit](#diagnostic-mlx-community-nanollava-15-4bit)                                            | completed | unusable            | observation_needs_reproduction | missing required fields                                                                             |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                             | completed | unusable            | observation_needs_reproduction | missing required fields; echoes instructions; cut off at token limit                                |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                                 | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                                 |
| [LiquidAI/LFM2.5-VL-450M-MLX-bf16](#diagnostic-liquidai-lfm25-vl-450m-mlx-bf16)                                            | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit](#diagnostic-mlx-community-devstral-small-2-24b-instruct-2512-5bit) | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/gemma-3-27b-it-qat-4bit](#diagnostic-mlx-community-gemma-3-27b-it-qat-4bit)                                 | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/gemma-3n-E4B-it-bf16](#diagnostic-mlx-community-gemma-3n-e4b-it-bf16)                                       | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/GLM-4.6V-Flash-mxfp4](#diagnostic-mlx-community-glm-46v-flash-mxfp4)                                        | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/Molmo-7B-D-0924-8bit](#diagnostic-mlx-community-molmo-7b-d-0924-8bit)                                       | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/Ornith-1.0-35B-bf16](#diagnostic-mlx-community-ornith-10-35b-bf16)                                          | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/Qwen2-VL-2B-Instruct-4bit](#diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit)                             | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed; draft hints copied unchanged                                      |
| [mlx-community/Qwen3-VL-2B-Instruct-bf16](#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16)                             | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/Qwen3.6-27B-mxfp8](#diagnostic-mlx-community-qwen36-27b-mxfp8)                                              | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [mlx-community/SmolVLM2-2.2B-Instruct-mlx](#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx)                            | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |
| [Qwen/Qwen3-VL-2B-Instruct](#diagnostic-qwen-qwen3-vl-2b-instruct)                                                         | completed | usable_with_caveats | observation_needs_reproduction | title/keyword constraints failed                                                                    |

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
- *Post-cleanup active memory (GB):* 0.00268806
- *Post-cleanup cache memory (GB):* 0.0

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11841, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11318, in _load_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12271, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11856, in _run_model_generation
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
Fetching 54 files: 100%|##########| 54/54 [00:00<00:00, 3965.34it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[20:30:31] DEBUG    HF Cache Info for mlx-community/Inkling-Small-mlx-4bit: size=146358.0 MB, files=58
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — repeated text; missing required fields; cut off at token limit</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "boat trip, boat ride,..."
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001180718
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1590
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Alright, let's get to work. I've got the image and the hints, and the task is to create some metadata. First, the title. "Two inflatable boats speeding across the sea" is a good start, but I need something more evocative. "Boats on the Seafront at Seaford" is a bit bland. "Speedboats on the Seafront at Seaford" is better, but I think I'll go with "Two inflatable boats speeding across the seafront at Seaford." That's descriptive and accurate.

Now, the description. I need to weave in the context and the hints. The date and time are important, so I'll start with that. "Captured on August 7th, 2026, at 17:17 UTC+01:00, this image shows two inflatable motorboats speeding across the seafront at Seaford, England." Then, I'll add the details from the hints: "Against a clear blue sky, the boats leave white wakes as they race across the water, with people visible on board." That's a solid, factual description.

Finally, the keywords. I need to cover everything. I'll start with the obvious: "boats, water, sea, seafront, Seaford, England, UK, Europe, inflatable boat, motorboat, speeding, wake, white wake, blue sky, horizon, people, action, summer, holiday, outboard motor, racing, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat ride, boat trip, boat
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
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "- authoritative context: -..."
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.014140684
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4404
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Author
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
- *Repeated fragment:* keyword: "england"
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 166
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["seaford", "england", "uk", "europe", "inflatable
  boat", "motorboat", "people", "horizon", "blue sky", "clear", "water",
  "action", "holiday", "white wake", "driver", "yamaha", "rigid hull", "lady
  maverick", "yamaha engine", "clear day", "open sea", "summer", "adventure",
  "travel", "tourism", "seaside", "gbr", "blue", "horizon line", "inflatable",
  "outboard motor", "speed boat"]
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.01029035
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16639
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Inflatable Boats Speeding Across the Ocean

Description:
Two inflatable motorboats, one labeled "Rapid," cruise across the open sea, leaving white wakes as they move, under a clear blue sky with a distinct horizon line, capturing a vibrant holiday scene.

Keywords: Seaford, England, UK, Europe, Inflatable boat, Motorboat, People, Horizon, Blue sky, Clear, Ocean, Water, Action, Holiday, Vehicles, Speeding, White wake, Driver, Yamaha, Rigid Hull, Lady Maverick, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull, Lady Maverick, White wake, Driver, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull, Lady Maverick, White wake, Driver, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull, Lady Maverick, White wake, Driver, Yamaha engine, Clear day, Open sea, Summer, Adventure, Travel, Tourism, Seaside, England, UK, Europe, GBR, Blue, Horizon line, Inflatable, Outboard motor, Speed boat, Seaford, England, UK, Europe, GBR, Blue sky, Clear, Open sea, Water, Action, Holiday, People, Horizon, Inflatable boat, Motorboat, Yamaha, Rigid Hull,
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — unusable — control tokens visible; missing required fields; extra text before Title</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble,
  unexpected_special_token
- *Missing sections:* ["title"]
- *Unexpected text before Title:* &lt;|begin_of_box|&gt;Title: Two inflatable
  boats speeding across the ocean
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002425916
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6330
- *Generation tokens:* 92
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: Two inflatable boats speeding across the ocean
Description: Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line, with people aboard each vessel.
Keywords: inflatable boat, motorboat, ocean, speeding, white wakes, clear blue sky, horizon line, people, boats, water, action, sea, seaford, england, uk, europe<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

<details>
<summary>mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX — unusable — missing required fields; echoes instructions; cut off at token limit</summary>

### mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Echoed instruction fragments:* ["title hint:", "description hint:",
  "keyword hints:", "british english catalogue metadata from the image and"]
- *Token-cap degradation evidence:* ["missing_sections", "dangling_markdown"]
- *Resolved model revision:* 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2
- *Processor class:* mlx_vlm.models.pixtral.processing_pixtral.PixtralProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.000705582
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2740
- *Generation tokens:* 500
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Here are my reasoning steps:
We need to produce British-English catalogue metadata from the image and supplied context. The context includes capture date/time: 2026-08-07 17:17:16 UTC+01:00. The descriptive hints: Title hint: Seafront, Seaford, England, UK, GBR, Europe. Description hint: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. Keyword hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.

We need to produce:

- a concrete 5-10-word title (so 5-10 words). Should be British English. Should reflect the main subject. Possibly "Seaford Inflatable Boats Speeding Across the Sea". That's 7 words? Let's count: Seaford (1) Inflatable (2) Boats (3) Speeding (4) Across (5) the (6) Sea (7). That's 7 words. Good.

- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details. Should be 1-2 sentences. Should mention capture date/time? The instruction: "Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible." So we can incorporate the date/time in description, but not claim it's visible. So we can say "Captured on 7 August 2026 at 17:17 UTC+01, the image shows two inflatable motorboats racing across the open sea off Seaford, leaving white wakes behind them under a clear blue sky with a distinct horizon." That's one sentence. Or two sentences: "Captured on 7 August 2026 at 17:17 UTC+01, the image shows two inflatable motorboats racing across the open sea off Seaford. They leave white wakes behind them under a clear blue sky with a distinct horizon." That's two sentences. That's fine.

-
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-8bit — unusable — missing required fields; extra text before Title</summary>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble
- *Missing sections:* ["title"]
- *Unexpected text before Title:* Title: Two Inflatable Boats Speeding Across
  the Seaford Coast
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.010290372
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 584
- *Generation tokens:* 83
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Two Inflatable Boats Speeding Across the Seaford Coast
Description: Two inflatable motorboats with outboard engines speed across the blue ocean water, leaving white wakes under a clear sky. 
Keywords: Seaford, England, UK, inflatable boat, motorboat, water, ocean, action, sky, horizon, people, vehicles, travel, holiday, transport, transport
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-mxfp8 — unusable — missing required fields; extra text before Title</summary>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble
- *Missing sections:* ["title"]
- *Unexpected text before Title:* Title: Two Motorboats Speeding Across the
  Sea at Seaford
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.010290372
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 584
- *Generation tokens:* 85
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Two Motorboats Speeding Across the Sea at Seaford
Description: Two inflatable boats with outboard motors speed across the blue ocean water, leaving white wakes behind them under a clear sky. 
Keywords: Seaford, England, UK, inflatable boat, motorboat, people, action, water, ocean, blue sky, horizon, holiday, vehicles, riding, sailing, Europe
```

</details>

<a id="diagnostic-mlx-community-fastvlm-05b-bf16"></a>

<details>
<summary>mlx-community/FastVLM-0.5B-bf16 — unusable — missing required fields; extra text before Title</summary>

### mlx-community/FastVLM-0.5B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_catalog_preamble
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* A concrete 5-10-word title: "Two inflatable
  boats speeding across the ocean, leaving white wakes behind them, against a
  clear blue sky and a distinct horizon line." A 1-2-sentence factual
  description: "Two inflatable boats with outboard motors are speeding across
  the ocean, leaving white wakes behind them, against a clear blue sky and a
  distinct horizon line."
- *Resolved model revision:* 81ffe929046666c43de53691147b1669ba0f3a4c
- *Processor class:* mlx_vlm.models.fastvlm.processing.FastVLMProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.001492016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 319
- *Generation tokens:* 130
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
A concrete 5-10-word title: "Two inflatable boats speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line."
A 1-2-sentence factual description: "Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line."
Keywords: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — missing required fields; extra text before Title; cut off at token limit; incomplete thinking block</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_incomplete
- *Missing sections:* ["description", "keywords"]
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this task
  step by step. First, the title needs to be 5-10 words, concrete. Looking at
  the image: two inflatable boats, sea, action. So maybe "Inflatable Boats
  Speeding Over Ocean".  Next, the description: need to combine context
  (date/time, location) with main subject, setting, action, lighting,
  distinctive details. The image has two inflatable boats, ocean, clear blue
  sky, horizon, people. So something like "Two inflatable boats with outboard
  motors speed across the ocean on 2026-08-07, leaving white wakes under a
  clear blue sky with a distinct horizon." Wait, need to make sure it's 1-2
  sentences. Let's check the hints: description hint says "Two inflatable
  boats with outboard motors are speeding across the ocean, leaving white
  wakes behind them, against a clear blue sky and a distinct horizon line." So
  combine with context. So "On 2026-08-07, two inflatable boats with outboard
  motors speed across the ocean, leaving white wakes behind them under a clear
  blue sky with a distinct horizon."  Then keywords: need 10-18 unique,
  comma-separated. From hints: Adobe Stock, Any Vision (maybe not, since it's
  not visible), Blue sky, Driver, England, Europe, Holiday, Horizon,
  Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK,
  Vehicles, Water, action, beautiful. But need to pick unique ones. Let's
  list: Inflatable boat, Motorboat, Ocean, Seaford, England, UK, Europe,
  People, Blue sky, Horizon, Water, Action, Vehicles, Holiday, Sky, Driver,
  Wakes. Wait, need to make sure they are unique. Let's check the hints and
  image. The image has inflatable boats, motorboats, ocean, seaford, england,
  uk, europe, people, blue sky, horizon, water, action, vehicles, holiday,
  sky, driver. So keywords: Inflatable boat, Motorboat, Ocean, Seaford,
  England, UK, Europe, People, Blue sky, Horizon, Water, Action, Vehicles,
  Holiday, Sky, Driver, Wakes. That's around 16, which is within 10-18.  Now,
  let's structure each part:
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001803312
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6326
- *Generation tokens:* 500
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: two inflatable boats, sea, action. So maybe "Inflatable Boats Speeding Over Ocean". 

Next, the description: need to combine context (date/time, location) with main subject, setting, action, lighting, distinctive details. The image has two inflatable boats, ocean, clear blue sky, horizon, people. So something like "Two inflatable boats with outboard motors speed across the ocean on 2026-08-07, leaving white wakes under a clear blue sky with a distinct horizon." Wait, need to make sure it's 1-2 sentences. Let's check the hints: description hint says "Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line." So combine with context. So "On 2026-08-07, two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them under a clear blue sky with a distinct horizon." 

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (maybe not, since it's not visible), Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful. But need to pick unique ones. Let's list: Inflatable boat, Motorboat, Ocean, Seaford, England, UK, Europe, People, Blue sky, Horizon, Water, Action, Vehicles, Holiday, Sky, Driver, Wakes. Wait, need to make sure they are unique. Let's check the hints and image. The image has inflatable boats, motorboats, ocean, seaford, england, uk, europe, people, blue sky, horizon, water, action, vehicles, holiday, sky, driver. So keywords: Inflatable boat, Motorboat, Ocean, Seaford, England, UK, Europe, People, Blue sky, Horizon, Water, Action, Vehicles, Holiday, Sky, Driver, Wakes. That's around 16, which is within 10-18. 

Now, let's structure each part:

Title: "Inflatable Boats Speeding
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
- *Missing sections:* ["title", "description"]
- *Unexpected text before Title:* ◁think▷Got it, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. Let's start by analyzing the image and the hints.
  First, the context says the capture date/time is 2026-08-07 17:17:16
  UTC+01:00, but we don't need to claim it's visible. The title hint is
  "Seafront, Seaford, England, UK, GBR, Europe". The description hint mentions
  two inflatable boats with outboard motors, white wakes, clear blue sky,
  distinct horizon. Keywords include Adobe Stock, Any Vision, Blue sky,
  Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat,
  People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action,
  beautiful.  Now, the image shows two motorboats (inflatable boats with
  outboard motors) on the sea, people on board, creating white wakes. The
  setting is Seaford, England, UK, Europe. The sky is clear blue, horizon is
  distinct. So the title should be concise, 5-10 words. Let's think: "Two
  inflatable motorboats speed across Seaford's sea with white wakes under a
  clear blue sky."  Description needs to be 1-2 sentences, combining context
  and main elements: "Two inflatable motorboats with outboard motors create
  white wakes as they speed across the sea at Seaford, England, under a clear
  blue sky with a distinct horizon."
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "unfinished_section", "abrupt_tail"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003654752
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1313
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

First, the context says the capture date/time is 2026-08-07 17:17:16 UTC+01:00, but we don't need to claim it's visible. The title hint is "Seafront, Seaford, England, UK, GBR, Europe". The description hint mentions two inflatable boats with outboard motors, white wakes, clear blue sky, distinct horizon. Keywords include Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful.

Now, the image shows two motorboats (inflatable boats with outboard motors) on the sea, people on board, creating white wakes. The setting is Seaford, England, UK, Europe. The sky is clear blue, horizon is distinct. So the title should be concise, 5-10 words. Let's think: "Two inflatable motorboats speed across Seaford's sea with white wakes under a clear blue sky."

Description needs to be 1-2 sentences, combining context and main elements: "Two inflatable motorboats with outboard motors create white wakes as they speed across the sea at Seaford, England, under a clear blue sky with a distinct horizon."

Keywords: need 10-18 unique, comma-separated. Let's list them: Seaford, England, UK, Europe, Two inflatable motorboats, outboard motors, white wakes, clear blue sky, distinct horizon, people, motorboats, sea, action, holiday, vehicles, water, blue sky, horizon line, speed, ocean, boats, passengers, water splashes.

Wait, check the hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful. But the image might not show "Driver" or "Sailing" as the boats are moving, maybe "Riding" is better. Also, "beautiful" is a hint but maybe not visible. Let's focus on what's visible. So keywords
```

</details>

<a id="diagnostic-mlx-community-lfm25-vl-16b-bf16"></a>

<details>
<summary>mlx-community/LFM2.5-VL-1.6B-bf16 — unusable — missing required fields</summary>

### mlx-community/LFM2.5-VL-1.6B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003785824
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 571
- *Generation tokens:* 93
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Seafoam Adventure, Two Inflatable Boats, Clear Sky, Horizon, People, Motorboat, Adventure, Seaford, England, Europe, Action, Water, Adventure, Inflatable Boat, Motorboat, People, Riding, Sailing, Seafoam, Blue Sky, UK, Vehicles, Water, Action, Beautiful, 2026-08-07 17:17:16 UTC+01:00
```

</details>

<a id="diagnostic-mlx-community-llava-v16-mistral-7b-8bit"></a>

<details>
<summary>mlx-community/llava-v1.6-mistral-7b-8bit — unusable — missing required fields</summary>

### mlx-community/llava-v1.6-mistral-7b-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Processor class:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.01248588
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2305
- *Generation tokens:* 52
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Seafront, Seaford, England, UK, GBR, Europe, Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. 
```

</details>

<a id="diagnostic-mlx-community-minicpm-v-46-8bit"></a>

<details>
<summary>mlx-community/MiniCPM-V-4.6-8bit — unusable — missing required fields; cut off at token limit; incomplete thinking block</summary>

### mlx-community/MiniCPM-V-4.6-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_incomplete
- *Missing sections:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* 03721395f6b82cd000cc74cde28fcff8abd9a04c
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.004834414
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 920
- *Generation tokens:* 500
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
<think>
We need to create the catalogue metadata. First, the title should be concrete and 5-10 words. Then the description should combine relevant context with main visible subjects, setting, action, lighting, and distinctive details. Then the keywords should be 10-18 unique, covering relevant context and visible details.

Let's extract info:

- The image shows two inflatable boats (inflatable boats) with people, moving across the ocean, leaving wakes. The setting is seafront, Seaford, England, UK, Europe. The time is 2026-08-07 17:17:16 UTC+01:00.

We need to describe: two inflatable boats with people, speeding across the ocean, clear sky, horizon, action (speeding, leaving wakes). Lighting seems sunny.

Now, title: need 5-10 words. Something like "Two Speed Inflatable Boats Across Ocean". But let's count: two, speed, inflatable, boats, across, ocean. That's 6 words. Maybe "Action Scenes of Inflatable Boats on Ocean". But better to be concrete.

Let's think: "Two Speed Inflatable Boats Across Blue Ocean". That's 7 words.

Now description: combine context and visible details. The context says capture date/time is 2026-08-07 17:17:16 UTC+01:00. So we can mention the time. The setting is Seaford, England, UK, Europe. The action is the boats speeding, leaving white wakes, under clear blue sky. So description: Two inflatable boats are speeding across the ocean, creating wakes, under a clear sky, in Seaford, England.

Now keywords: need 10-18 unique, covering relevant context and visible details. Let's list:

1. inflatable boats
2. speed, speeding
3. ocean, sea, water
4. blue sky, clear, sunny
5. horizon, Seaford, England, UK, Europe
6. people, passengers, riders
7. outboard motors, motorboats
8. white wakes, wakes, splashes
9. action, action scenes
10. holiday, leisure, travel
11. water, aquatic, marine
12. boats, vessels
```

</details>

<a id="diagnostic-mlx-community-molmopoint-8b-fp16"></a>

<details>
<summary>mlx-community/MolmoPoint-8B-fp16 — unusable — missing required fields</summary>

### mlx-community/MolmoPoint-8B-fp16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["title", "description", "keywords"]
- *Resolved model revision:* 0a60033b4e4813fb53df4c7523857d2ec972c7d9
- *Processor class:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.006554746
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3111
- *Generation tokens:* 156
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Inflatable Boats Speeding Across Open Sea Near Seaford

Two inflatable motorboats with outboard engines race across a vast, dark blue ocean, leaving white wakes behind them. The boats carry multiple passengers and feature distinctive blue and white designs, with one displaying a blue and white flag. The scene is set against a clear blue sky with a distinct horizon line, captured on August 7, 2026, at 17:17:16 UTC+01:00.

Inflatable boat, motorboat, Seaford, England, UK, Europe, open sea, horizon, blue sky, white wake, passengers, outboard motor, action, vehicles, water, sailing, holiday, Any Vision, Adobe Stock
```

</details>

<a id="diagnostic-mlx-community-nanollava-15-4bit"></a>

<details>
<summary>mlx-community/nanoLLaVA-1.5-4bit — unusable — missing required fields</summary>

### mlx-community/nanoLLaVA-1.5-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections
- *Missing sections:* ["keywords"]
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Processor class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.013108472
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 315
- *Generation tokens:* 71
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: "Seaford in Action"
Description: A serene scene of two inflatable boats, one driving away from the other, captured in motion on a clear blue sky. The boats are driving on a calm sea, with a distinct horizon line in the background. The image is a testament to the beauty of nature and the joy of outdoor activities.
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Thinking-bf16 — unusable — missing required fields; echoes instructions; cut off at token limit</summary>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  prompt_instruction_echo
- *Missing sections:* ["title", "description", "keywords"]
- *Echoed instruction fragments:* ["title hint:"]
- *Token-cap degradation evidence:* ["missing_sections", "abrupt_tail"]
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.008062088
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16630
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title needs to be 5-10 words. The image shows two inflatable boats (motorboats) on the ocean, so maybe "Two motorboats speeding across the sea" but need to be concise. Let's see: "Two inflatable motorboats racing on the ocean" – that's 8 words, good.

Next, the description. It has to combine context with visible subject, setting, action, lighting, details. Context is capture date/time, but we don't claim that. The main subject is two inflatable boats with outboard motors, speeding, leaving wakes, clear blue sky, horizon line. So: "Two inflatable motorboats with outboard motors speed across the ocean, leaving white wakes, under a clear blue sky with distinct horizon line, on a calm sea."

Wait, the context says capture date/time is 2026-08-07 17:17:16 UTC+01:00, but we don't include that in the description. The descriptive hints mention "clear blue sky" which is visible, "distinct horizon line" – the image has a horizon line, so that's there. The boats are speeding, so "speeding" is key. Let's check the hints: "Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line." So the description should include that. Let's make it: "Two inflatable motorboats with outboard motors speed across the ocean, leaving white wakes, under a clear blue sky with a distinct horizon line, on a calm sea."

Now keywords. Need 10-18 unique, comma-separated. From context: UK, Europe, England, Seafront, Seaford (but the title hint says Seaford, England, UK, GBR, Europe). Wait, the title hint is "Seafront, Seaford, England, UK, GBR, Europe". But the image doesn't show a seafront, just the ocean. Wait, the descriptive hints say "Title hint: Seafront, Seaford, England, UK, GBR, Europe". But the image is of two boats on the ocean, so maybe Seaford is a location, but the image doesn't show it. Wait, the problem says: "Descriptive hints: Title hint
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
- *Post-cleanup active memory (GB):* 0.00268806
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2608
- *Generation tokens:* 130
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 Title: Inflatable Boats Speed Across the Ocean on a Clear Day.

Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are carrying several people, and the water is calm with no visible waves. The sky is clear with no clouds, and the sun is not visible, suggesting it might be early morning or late afternoon.

Keywords: inflatable boats, ocean, outboard motors, white wakes, clear blue sky, horizon line, people, calm water, clear sky, no clouds, early morning, late afternoon.<end_of_utterance>
```

</details>

<a id="diagnostic-liquidai-lfm25-vl-450m-mlx-bf16"></a>

<details>
<summary>LiquidAI/LFM2.5-VL-450M-MLX-bf16 — usable_with_caveats — title/keyword constraints failed</summary>

### LiquidAI/LFM2.5-VL-450M-MLX-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 18
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["seaford", "uk"]
- *Resolved model revision:* 6c33f49ebc0b50b75385f49ad3beddcb720d0c75
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00013212
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 385
- *Generation tokens:* 107
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Seafront, Seaford, England, UK, GBR, Europe

Description:
Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.

Keywords:
Seafront, Seaford, England, UK, GBR, Europe, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, Action, Beautiful
```

</details>

<a id="diagnostic-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

<details>
<summary>mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 7
- *Requested title word range:* [5, 10]
- *Keyword count:* 24
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 0a970d20ad7da57b0e2accc35c5b8628f5d02063
- *Processor class:* mlx_vlm.models.mistral3.processing_mistral3.Mistral3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.000967726
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2045
- *Generation tokens:* 110
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Title:
Two Inflatable Boats Racing Across the Ocean

Description:
Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, under a clear blue sky with a distinct horizon line.

Keywords:
Adobe Stock, inflatable boats, outboard motors, ocean, blue sky, horizon, speeding, white wakes, clear sky, racing, water, vehicles, action, holiday, sailing, people, riding, driver, beautiful, seafront, Seaford, England, UK, Europe
```

</details>

<a id="diagnostic-mlx-community-gemma-3-27b-it-qat-4bit"></a>

<details>
<summary>mlx-community/gemma-3-27b-it-qat-4bit — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/gemma-3-27b-it-qat-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 8
- *Requested title word range:* [5, 10]
- *Keyword count:* 24
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* fc4e000f32af1b7b6779294e490a7d2a80bac611
- *Processor class:* mlx_vlm.models.gemma3.processing_gemma3.Gemma3Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.010831044
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 579
- *Generation tokens:* 155
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
Title: Seaford Ribs Speeding Across the Channel, August 2026

Description: Two black rigid inflatable boats (RIBs) with passengers are travelling at speed on the sea off Seaford, England, captured on 7th August 2026 at 17:17 UTC+01:00 under clear, bright conditions. Distinct white wakes trail behind the vessels against a flat, calm horizon.

Keywords: Seaford, England, UK, Channel, Rib, Inflatable boat, Motorboat, Speedboat, Water, Sea, Ocean, August, 2026, Passengers, Vehicles, Action, Horizon, Blue sky, Daytime, Summer, Transport, Recreation, Coastal, Rigid Hull
```

</details>

<a id="diagnostic-mlx-community-gemma-3n-e4b-it-bf16"></a>

<details>
<summary>mlx-community/gemma-3n-E4B-it-bf16 — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/gemma-3n-E4B-it-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 7
- *Requested title word range:* [5, 10]
- *Keyword count:* 19
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a
- *Processor class:* mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.011371742
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 578
- *Generation tokens:* 130
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
**Title:** Two Inflatable Boats Speed Across the Ocean

**Description:** Two blue inflatable boats, each with an outboard motor, are speeding across a vast, choppy blue ocean. White wakes trail behind them, contrasting against the deep blue water. The sky is a clear, pale blue, meeting the horizon line in the distance. Several people are visible in each boat, appearing to be enjoying a fast ride.

**Keywords:** Inflatable boat, motorboat, ocean, water, speed, blue, sky, people, riding, action, waves, sea, travel, adventure, summer, coastal, UK, England, Seaford.
```

</details>

<a id="diagnostic-mlx-community-glm-46v-flash-mxfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-mxfp4 — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/GLM-4.6V-Flash-mxfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 21
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002114608
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6330
- *Generation tokens:* 104
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
Title: Two Inflatable Boats
Description: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
Keywords: Two Inflatable Boats, outboard motors, speeding across ocean, white wakes, clear blue sky, distinct horizon line, inflatable boat, motorboat, people, riding, Sailing, Seaford, England, UK, Europe, Holiday, Horizon, Sky, Water, action, beautiful
```

</details>

<a id="diagnostic-mlx-community-molmo-7b-d-0924-8bit"></a>

<details>
<summary>mlx-community/Molmo-7B-D-0924-8bit — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/Molmo-7B-D-0924-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 19
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 90a14ed7a230088904c7556fbe6d67b295c33f5f
- *Processor class:* mlx_vlm.models.molmo.processing_molmo.MolmoProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.005932144
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1498
- *Generation tokens:* 105
- *Configured EOS token ID:* 151643
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
 Title: Speedboats Raging on Seaford's Shores

Description: Two inflatable motorboats race across the ocean, leaving white wakes behind them against a clear blue sky and distinct horizon, with passengers enjoying the thrilling ride on a sunny day in Seaford, England.

Keywords: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Seaford, Sky, UK, Vehicles, Water, action, beautiful
```

</details>

<a id="diagnostic-mlx-community-ornith-10-35b-bf16"></a>

<details>
<summary>mlx-community/Ornith-1.0-35B-bf16 — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/Ornith-1.0-35B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 8
- *Requested title word range:* [5, 10]
- *Keyword count:* 19
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.007062654
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16643
- *Generation tokens:* 107
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Two inflatable motorboats speeding across calm blue sea

Description: Two rigid inflatable boats carrying passengers cruise across the open sea under a clear sky, leaving white wakes behind them. The scene captures a bright, sunny day on the water with a distinct horizon line separating the deep blue sea from the pale sky.

Keywords: inflatable boat, motorboat, sea, ocean, people, passengers, wake, speed, blue sky, horizon, sunny, vessels, water, leisure, boating, England, UK, Europe, adventure
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-Instruct-4bit — usable_with_caveats — title/keyword constraints failed; draft hints copied unchanged</summary>

### mlx-community/Qwen2-VL-2B-Instruct-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation, draft_returned_unchanged
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 20
- *Requested keyword count range:* [10, 18]
- *Draft fields returned unchanged:* ["title", "description", "keywords"]
- *Resolved model revision:* 01af461cdb9574acc09084a0ef94e216e142b085
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.007439496
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16639
- *Generation tokens:* 102
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Seafront, Seaford, England, UK, GBR, Europe

Description:
Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.

Keywords:
Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Instruct-bf16 — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 20
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["water"]
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.007750792
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16628
- *Generation tokens:* 90
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Two boats on the sea

Description: Two inflatable motorboats speed across the blue sea, leaving white wakes, under a clear sky with a distinct horizon, with people on board enjoying the water.

Keywords: motorboat, inflatable boat, sea, people, sky, horizon, water, action, blue sky, England, UK, Europe, holiday, vehicles, seafront, Seaford, Sailing, Riding, Water, beautiful
```

</details>

<a id="diagnostic-mlx-community-qwen36-27b-mxfp8"></a>

<details>
<summary>mlx-community/Qwen3.6-27B-mxfp8 — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/Qwen3.6-27B-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 19
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 5db9fd9c38ce083048ee12217eceed7482ca7ad4
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009602186
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16643
- *Generation tokens:* 87
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Two Speedboats Racing Across the Sea

Description: Two motorised inflatable boats carrying passengers speed across the blue ocean, leaving white wakes behind them under a clear sky.

Keywords: Sea, Ocean, Boats, Motorboats, Inflatable boats, People, Passengers, Speeding, Wakes, Blue water, Clear sky, Horizon, Summer, Daytime, England, UK, Europe, Action, Water sports
```

</details>

<a id="diagnostic-mlx-community-smolvlm2-22b-instruct-mlx"></a>

<details>
<summary>mlx-community/SmolVLM2-2.2B-Instruct-mlx — usable_with_caveats — title/keyword constraints failed</summary>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 2
- *Requested title word range:* [5, 10]
- *Keyword count:* 20
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Processor class:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.009716874
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 411
- *Generation tokens:* 125
- *Configured EOS token ID:* 49279
- *Configured EOS token:* &lt;end_of_utterance&gt;

#### Complete output

```text
 Title: "Seafront Adventure"
Description: "Two inflatable boats with outboard motors speed across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line. The boats are being ridden by people enjoying a holiday in England, Europe, with a beautiful view of the horizon and clear blue sky."
Keywords: "Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful"
```

</details>

<a id="diagnostic-qwen-qwen3-vl-2b-instruct"></a>

<details>
<summary>Qwen/Qwen3-VL-2B-Instruct — usable_with_caveats — title/keyword constraints failed</summary>

### Qwen/Qwen3-VL-2B-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* catalog_constraint_violation
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 20
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["water"]
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.000443432
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16628
- *Generation tokens:* 90
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Two boats on the sea

Description: Two inflatable motorboats speed across the blue sea, leaving white wakes, under a clear sky with a distinct horizon, with people on board enjoying the water.

Keywords: motorboat, inflatable boat, sea, people, sky, horizon, water, action, blue sky, England, UK, Europe, holiday, vehicles, seafront, Seaford, Sailing, Riding, Water, beautiful
```

</details>

## Indeterminate Attempts

None.

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                             | Runtime identity                                        | Performance                                                                               |
|---------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------------------------------------|
| mlx-community/gemma-4-26b-a4b-it-4bit             | rev 0d77464eeb23; Gemma4Processor; stop completed       | 588 prompt / 96 generated; 114 tok/s; 16 GB peak; cleanup 0.0119/0.0 GB active/cache      |
| mlx-community/gemma-4-31b-it-4bit                 | rev 696d436c4047; Gemma4Processor; stop completed       | 588 prompt / 83 generated; 18.5 tok/s; 20 GB peak; cleanup 0.0124/0.0 GB active/cache     |
| mlx-community/InternVL3-8B-bf16                   | rev e0df3dd79263; InternVLChatProcessor; stop completed | 2610 prompt / 103 generated; 23.6 tok/s; 17 GB peak; cleanup 0.003/0.0 GB active/cache    |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit  | rev 8451adc50203; MllamaProcessor; stop completed       | 297 prompt / 89 generated; 13.7 tok/s; 15 GB peak; cleanup 0.00431/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | rev 7c992876448f; Mistral3Processor; stop completed     | 2578 prompt / 121 generated; 59.2 tok/s; 12 GB peak; cleanup 0.0051/0.0 GB active/cache   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | rev 28777b889d84; Mistral3Processor; stop completed     | 2578 prompt / 171 generated; 40.3 tok/s; 12 GB peak; cleanup 0.00536/0.0 GB active/cache  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit   | rev a962dcb09eee; Mistral3Processor; stop completed     | 2577 prompt / 117 generated; 144 tok/s; 6.4 GB peak; cleanup 0.00562/0.0 GB active/cache  |
| mlx-community/Phi-3.5-vision-instruct-bf16        | rev d8da684308c2; Phi3VProcessor; stop completed        | 1112 prompt / 100 generated; 55.8 tok/s; 9.4 GB peak; cleanup 0.00713/0.0 GB active/cache |
| mlx-community/pixtral-12b-8bit                    | rev 79e24b66302d; PixtralProcessor; stop completed      | 2649 prompt / 82 generated; 39.7 tok/s; 15 GB peak; cleanup 0.0144/0.0 GB active/cache    |
| mlx-community/Qwen3.5-35B-A3B-4bit                | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed      | 16643 prompt / 79 generated; 94.0 tok/s; 24 GB peak; cleanup 0.00857/0.0 GB active/cache  |
| mlx-community/Qwen3.5-9B-MLX-4bit                 | rev 938d8919941c; Qwen3VLProcessor; stop completed      | 16643 prompt / 106 generated; 90.7 tok/s; 10 GB peak; cleanup 0.00909/0.0 GB active/cache |
| mlx-community/Step-3.7-Flash-oQ2e                 | rev 3dacb46f724a; Step3VLProcessor; stop completed      | 3478 prompt / 118 generated; 40.4 tok/s; 70 GB peak; cleanup 0.00998/0.0 GB active/cache  |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,964 x 5,605 pixels
- *Image size:* 39,212,214 bytes
- *Image SHA-256:* f5cc97b21d6d751921d8c5b18cbc80b9b8bca1839b8ff95e1a75d7427992e488

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-07 17:17:16 UTC+01:00

Descriptive hints:
- Title hint: Seafront, Seaford, England, UK, GBR, Europe
- Description hint: Two inflatable boats with outboard motors are speeding across the ocean, leaving white wakes behind them, against a clear blue sky and a distinct horizon line.
- Keyword hints: Adobe Stock, Any Vision, Blue sky, Driver, England, Europe, Holiday, Horizon, Inflatable boat, Motorboat, People, Riding, Sailing, Seaford, Sky, UK, Vehicles, Water, action, beautiful

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

| Model                                                 | Resolved revision                        |
|-------------------------------------------------------|------------------------------------------|
| mlx-community/Inkling-Small-mlx-4bit                  | f0cafad5b1a3e54be06ba03fe07b4cd4e8bcc612 |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/paligemma2-3b-pt-896-4bit               | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/X-Reasoner-7B-8bit                      | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/GLM-4.6V-nvfp4                          | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/FastVLM-0.5B-bf16                       | 81ffe929046666c43de53691147b1669ba0f3a4c |
| mlx-community/GLM-4.1V-9B-Thinking-8bit               | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | 16a710cf8afca206ff16a95a4ad6fe657f876ce1 |
| mlx-community/llava-v1.6-mistral-7b-8bit              | b8df5f329d95a7abe6429ed46093f9b84e8e6396 |
| mlx-community/MiniCPM-V-4.6-8bit                      | 03721395f6b82cd000cc74cde28fcff8abd9a04c |
| mlx-community/MolmoPoint-8B-fp16                      | 0a60033b4e4813fb53df4c7523857d2ec972c7d9 |
| mlx-community/nanoLLaVA-1.5-4bit                      | 5240204744963d72823e5de933c528c4aa82dfca |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/Idefics3-8B-Llama3-bf16                 | 8c2a30c48864f3251701b7bde40f601d25535098 |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | 6c33f49ebc0b50b75385f49ad3beddcb720d0c75 |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | 0a970d20ad7da57b0e2accc35c5b8628f5d02063 |
| mlx-community/gemma-3-27b-it-qat-4bit                 | fc4e000f32af1b7b6779294e490a7d2a80bac611 |
| mlx-community/gemma-3n-E4B-it-bf16                    | d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b |
| mlx-community/Molmo-7B-D-0924-8bit                    | 90a14ed7a230088904c7556fbe6d67b295c33f5f |
| mlx-community/Ornith-1.0-35B-bf16                     | 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | 01af461cdb9574acc09084a0ef94e216e142b085 |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | c8a67a84327484ba87f5ec4f8fb927cdafd791aa |
| mlx-community/Qwen3.6-27B-mxfp8                       | 5db9fd9c38ce083048ee12217eceed7482ca7ad4 |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | 844516024a1c4400d34489b89ee067d794e432ed |
| Qwen/Qwen3-VL-2B-Instruct                             | 89644892e4d85e24eaac8bacfd4f463576704203 |

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
