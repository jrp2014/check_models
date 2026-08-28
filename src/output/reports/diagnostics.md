# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

## Run Summary

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 41    |
| Conclusive outcomes | 41    |
| Completed           | 39    |
| Crashed             | 2     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 2     |
| none                           | 30    |
| observation needs reproduction | 9     |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 2     |
| unusable            | 9     |
| usable              | 12    |
| usable with caveats | 18    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 5     |
| Unrecognised model control tokens remain visible                                      | 3     |
| Required fields are missing or empty                                                  | 4     |
| Response repeats the task instructions instead of only returning the requested fields | 1     |
| Extra text appears before the Title field                                             | 2     |
| Response appears cut off at the token limit                                           | 6     |
| Internal reasoning block appears incomplete                                           | 1     |
| Conversation-role control tokens remain visible                                       | 1     |
| Title or keywords do not meet requested constraints                                   | 19    |
| Title, Description and Keywords copy all supplied hints unchanged                     | 3     |

## Triage

| Model                                                                                                           | Execution | Usability           | Maintainer status              | Observations                                                                                                                |
|-----------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/LFM2.5-VL-3B-OptiQ-4bit](#diagnostic-mlx-community-lfm25-vl-3b-optiq-4bit)                       | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                        |
| [tencent/Youtu-VL-4B-Instruct](#diagnostic-tencent-youtu-vl-4b-instruct)                                        | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                        |
| [jinaai/jina-vlm-mlx](#diagnostic-jinaai-jina-vlm-mlx)                                                          | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                   | completed | unusable            | observation_needs_reproduction | repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [mlx-community/LFM2.5-VL-1.6B-bf16](#diagnostic-mlx-community-lfm25-vl-16b-bf16)                                | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                     |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                      |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                      |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                      |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                                                         |

## Crashes requiring action

<a id="diagnostic-mlx-community-lfm25-vl-3b-optiq-4bit"></a>

### mlx-community/LFM2.5-VL-3B-OptiQ-4bit

#### Root exception and chain

```text
builtins.ValueError: Received 600 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
builtins.ValueError: Model loading failed: Received 600 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Phase:* model_load
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model loading failed: Received 600 parameters not in model;
  families: model; representative parameters: model.embed_tokens.biases,
  model.embed_tokens.scales, model.embed_tokens.weight.
- *Root error type:* ValueError
- *Root error message:* Received 600 parameters not in model; families: model;
  representative parameters: model.embed_tokens.biases,
  model.embed_tokens.scales, model.embed_tokens.weight.
- *Resolved model revision:* 12c5ae49304158b0a133fcea9ba4486a6d6c8cad
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.00409712
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 2.81
- *Parameter count:* 3.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 128,000 (text_config.max_position_embeddings)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13505, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12735, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 822, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1202, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1067, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 600 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.embedding_norm.weight,
model.layers.0.conv.conv.weight,
model.layers.0.conv.in_proj.biases,
model.layers.0.conv.in_proj.scales,
model.layers.0.conv.in_proj.weight,
model.layers.0.conv.out_proj.biases,
model.layers.0.conv.out_proj.scales,
model.layers.0.conv.out_proj.weight,
model.layers.0.feed_forward.w1.biases,
model.layers.0.feed_forward.w1.scales,
model.layers.0.feed_forward.w1.weight,
model.layers.0.feed_forward.w2.biases,
model.layers.0.feed_forward.w2.scales,
model.layers.0.feed_forward.w2.weight,
model.layers.0.feed_forward.w3.biases,
model.layers.0.feed_forward.w3.scales,
model.layers.0.feed_forward.w3.weight,
model.layers.0.ffn_norm.weight,
model.layers.0.operator_norm.weight,
model.layers.1.conv.conv.weight,
model.layers.1.conv.in_proj.biases,
model.layers.1.conv.in_proj.scales,
model.layers.1.conv.in_proj.weight,
model.layers.1.conv.out_proj.biases,
model.layers.1.conv.out_proj.scales,
model.layers.1.conv.out_proj.weight,
model.layers.1.feed_forward.w1.biases,
model.layers.1.feed_forward.w1.scales,
model.layers.1.feed_forward.w1.weight,
model.layers.1.feed_forward.w2.biases,
model.layers.1.feed_forward.w2.scales,
model.layers.1.feed_forward.w2.weight,
model.layers.1.feed_forward.w3.biases,
model.layers.1.feed_forward.w3.scales,
model.layers.1.feed_forward.w3.weight,
model.layers.1.ffn_norm.weight,
model.layers.1.operator_norm.weight,
model.layers.10.conv.conv.weight,
model.layers.10.conv.in_proj.biases,
model.layers.10.conv.in_proj.scales,
model.layers.10.conv.in_proj.weight,
model.layers.10.conv.out_proj.biases,
model.layers.10.conv.out_proj.scales,
model.layers.10.conv.out_proj.weight,
model.layers.10.feed_forward.w1.biases,
model.layers.10.feed_forward.w1.scales,
model.layers.10.feed_forward.w1.weight,
model.layers.10.feed_forward.w2.biases,
model.layers.10.feed_forward.w2.scales,
model.layers.10.feed_forward.w2.weight,
model.layers.10.feed_forward.w3.biases,
model.layers.10.feed_forward.w3.scales,
model.layers.10.feed_forward.w3.weight,
model.layers.10.ffn_norm.weight,
model.layers.10.operator_norm.weight,
model.layers.11.conv.conv.weight,
model.layers.11.conv.in_proj.biases,
model.layers.11.conv.in_proj.scales,
model.layers.11.conv.in_proj.weight,
model.layers.11.conv.out_proj.biases,
model.layers.11.conv.out_proj.scales,
model.layers.11.conv.out_proj.weight,
model.layers.11.feed_forward.w1.biases,
model.layers.11.feed_forward.w1.scales,
model.layers.11.feed_forward.w1.weight,
model.layers.11.feed_forward.w2.biases,
model.layers.11.feed_forward.w2.scales,
model.layers.11.feed_forward.w2.weight,
model.layers.11.feed_forward.w3.biases,
model.layers.11.feed_forward.w3.scales,
model.layers.11.feed_forward.w3.weight,
model.layers.11.ffn_norm.weight,
model.layers.11.operator_norm.weight,
model.layers.12.conv.conv.weight,
model.layers.12.conv.in_proj.biases,
model.layers.12.conv.in_proj.scales,
model.layers.12.conv.in_proj.weight,
model.layers.12.conv.out_proj.biases,
model.layers.12.conv.out_proj.scales,
model.layers.12.conv.out_proj.weight,
model.layers.12.feed_forward.w1.biases,
model.layers.12.feed_forward.w1.scales,
model.layers.12.feed_forward.w1.weight,
model.layers.12.feed_forward.w2.biases,
model.layers.12.feed_forward.w2.scales,
model.layers.12.feed_forward.w2.weight,
model.layers.12.feed_forward.w3.biases,
model.layers.12.feed_forward.w3.scales,
model.layers.12.feed_forward.w3.weight,
model.layers.12.ffn_norm.weight,
model.layers.12.operator_norm.weight,
model.layers.13.feed_forward.w1.biases,
model.layers.13.feed_forward.w1.scales,
model.layers.13.feed_forward.w1.weight,
model.layers.13.feed_forward.w2.biases,
model.layers.13.feed_forward.w2.scales,
model.layers.13.feed_forward.w2.weight,
model.layers.13.feed_forward.w3.biases,
model.layers.13.feed_forward.w3.scales,
model.layers.13.feed_forward.w3.weight,
model.layers.13.ffn_norm.weight,
model.layers.13.operator_norm.weight,
model.layers.13.self_attn.k_layernorm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.out_proj.biases,
model.layers.13.self_attn.out_proj.scales,
model.layers.13.self_attn.out_proj.weight,
model.layers.13.self_attn.q_layernorm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.conv.conv.weight,
model.layers.14.conv.in_proj.biases,
model.layers.14.conv.in_proj.scales,
model.layers.14.conv.in_proj.weight,
model.layers.14.conv.out_proj.biases,
model.layers.14.conv.out_proj.scales,
model.layers.14.conv.out_proj.weight,
model.layers.14.feed_forward.w1.biases,
model.layers.14.feed_forward.w1.scales,
model.layers.14.feed_forward.w1.weight,
model.layers.14.feed_forward.w2.biases,
model.layers.14.feed_forward.w2.scales,
model.layers.14.feed_forward.w2.weight,
model.layers.14.feed_forward.w3.biases,
model.layers.14.feed_forward.w3.scales,
model.layers.14.feed_forward.w3.weight,
model.layers.14.ffn_norm.weight,
model.layers.14.operator_norm.weight,
model.layers.15.conv.conv.weight,
model.layers.15.conv.in_proj.biases,
model.layers.15.conv.in_proj.scales,
model.layers.15.conv.in_proj.weight,
model.layers.15.conv.out_proj.biases,
model.layers.15.conv.out_proj.scales,
model.layers.15.conv.out_proj.weight,
model.layers.15.feed_forward.w1.biases,
model.layers.15.feed_forward.w1.scales,
model.layers.15.feed_forward.w1.weight,
model.layers.15.feed_forward.w2.biases,
model.layers.15.feed_forward.w2.scales,
model.layers.15.feed_forward.w2.weight,
model.layers.15.feed_forward.w3.biases,
model.layers.15.feed_forward.w3.scales,
model.layers.15.feed_forward.w3.weight,
model.layers.15.ffn_norm.weight,
model.layers.15.operator_norm.weight,
model.layers.16.conv.conv.weight,
model.layers.16.conv.in_proj.biases,
model.layers.16.conv.in_proj.scales,
model.layers.16.conv.in_proj.weight,
model.layers.16.conv.out_proj.biases,
model.layers.16.conv.out_proj.scales,
model.layers.16.conv.out_proj.weight,
model.layers.16.feed_forward.w1.biases,
model.layers.16.feed_forward.w1.scales,
model.layers.16.feed_forward.w1.weight,
model.layers.16.feed_forward.w2.biases,
model.layers.16.feed_forward.w2.scales,
model.layers.16.feed_forward.w2.weight,
model.layers.16.feed_forward.w3.biases,
model.layers.16.feed_forward.w3.scales,
model.layers.16.feed_forward.w3.weight,
model.layers.16.ffn_norm.weight,
model.layers.16.operator_norm.weight,
model.layers.17.feed_forward.w1.biases,
model.layers.17.feed_forward.w1.scales,
model.layers.17.feed_forward.w1.weight,
model.layers.17.feed_forward.w2.biases,
model.layers.17.feed_forward.w2.scales,
model.layers.17.feed_forward.w2.weight,
model.layers.17.feed_forward.w3.biases,
model.layers.17.feed_forward.w3.scales,
model.layers.17.feed_forward.w3.weight,
model.layers.17.ffn_norm.weight,
model.layers.17.operator_norm.weight,
model.layers.17.self_attn.k_layernorm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.out_proj.biases,
model.layers.17.self_attn.out_proj.scales,
model.layers.17.self_attn.out_proj.weight,
model.layers.17.self_attn.q_layernorm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.conv.conv.weight,
model.layers.18.conv.in_proj.biases,
model.layers.18.conv.in_proj.scales,
model.layers.18.conv.in_proj.weight,
model.layers.18.conv.out_proj.biases,
model.layers.18.conv.out_proj.scales,
model.layers.18.conv.out_proj.weight,
model.layers.18.feed_forward.w1.biases,
model.layers.18.feed_forward.w1.scales,
model.layers.18.feed_forward.w1.weight,
model.layers.18.feed_forward.w2.biases,
model.layers.18.feed_forward.w2.scales,
model.layers.18.feed_forward.w2.weight,
model.layers.18.feed_forward.w3.biases,
model.layers.18.feed_forward.w3.scales,
model.layers.18.feed_forward.w3.weight,
model.layers.18.ffn_norm.weight,
model.layers.18.operator_norm.weight,
model.layers.19.conv.conv.weight,
model.layers.19.conv.in_proj.biases,
model.layers.19.conv.in_proj.scales,
model.layers.19.conv.in_proj.weight,
model.layers.19.conv.out_proj.biases,
model.layers.19.conv.out_proj.scales,
model.layers.19.conv.out_proj.weight,
model.layers.19.feed_forward.w1.biases,
model.layers.19.feed_forward.w1.scales,
model.layers.19.feed_forward.w1.weight,
model.layers.19.feed_forward.w2.biases,
model.layers.19.feed_forward.w2.scales,
model.layers.19.feed_forward.w2.weight,
model.layers.19.feed_forward.w3.biases,
model.layers.19.feed_forward.w3.scales,
model.layers.19.feed_forward.w3.weight,
model.layers.19.ffn_norm.weight,
model.layers.19.operator_norm.weight,
model.layers.2.feed_forward.w1.biases,
model.layers.2.feed_forward.w1.scales,
model.layers.2.feed_forward.w1.weight,
model.layers.2.feed_forward.w2.biases,
model.layers.2.feed_forward.w2.scales,
model.layers.2.feed_forward.w2.weight,
model.layers.2.feed_forward.w3.biases,
model.layers.2.feed_forward.w3.scales,
model.layers.2.feed_forward.w3.weight,
model.layers.2.ffn_norm.weight,
model.layers.2.operator_norm.weight,
model.layers.2.self_attn.k_layernorm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.out_proj.biases,
model.layers.2.self_attn.out_proj.scales,
model.layers.2.self_attn.out_proj.weight,
model.layers.2.self_attn.q_layernorm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.conv.conv.weight,
model.layers.20.conv.in_proj.biases,
model.layers.20.conv.in_proj.scales,
model.layers.20.conv.in_proj.weight,
model.layers.20.conv.out_proj.biases,
model.layers.20.conv.out_proj.scales,
model.layers.20.conv.out_proj.weight,
model.layers.20.feed_forward.w1.biases,
model.layers.20.feed_forward.w1.scales,
model.layers.20.feed_forward.w1.weight,
model.layers.20.feed_forward.w2.biases,
model.layers.20.feed_forward.w2.scales,
model.layers.20.feed_forward.w2.weight,
model.layers.20.feed_forward.w3.biases,
model.layers.20.feed_forward.w3.scales,
model.layers.20.feed_forward.w3.weight,
model.layers.20.ffn_norm.weight,
model.layers.20.operator_norm.weight,
model.layers.21.feed_forward.w1.biases,
model.layers.21.feed_forward.w1.scales,
model.layers.21.feed_forward.w1.weight,
model.layers.21.feed_forward.w2.biases,
model.layers.21.feed_forward.w2.scales,
model.layers.21.feed_forward.w2.weight,
model.layers.21.feed_forward.w3.biases,
model.layers.21.feed_forward.w3.scales,
model.layers.21.feed_forward.w3.weight,
model.layers.21.ffn_norm.weight,
model.layers.21.operator_norm.weight,
model.layers.21.self_attn.k_layernorm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.out_proj.biases,
model.layers.21.self_attn.out_proj.scales,
model.layers.21.self_attn.out_proj.weight,
model.layers.21.self_attn.q_layernorm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.conv.conv.weight,
model.layers.22.conv.in_proj.biases,
model.layers.22.conv.in_proj.scales,
model.layers.22.conv.in_proj.weight,
model.layers.22.conv.out_proj.biases,
model.layers.22.conv.out_proj.scales,
model.layers.22.conv.out_proj.weight,
model.layers.22.feed_forward.w1.biases,
model.layers.22.feed_forward.w1.scales,
model.layers.22.feed_forward.w1.weight,
model.layers.22.feed_forward.w2.biases,
model.layers.22.feed_forward.w2.scales,
model.layers.22.feed_forward.w2.weight,
model.layers.22.feed_forward.w3.biases,
model.layers.22.feed_forward.w3.scales,
model.layers.22.feed_forward.w3.weight,
model.layers.22.ffn_norm.weight,
model.layers.22.operator_norm.weight,
model.layers.23.conv.conv.weight,
model.layers.23.conv.in_proj.biases,
model.layers.23.conv.in_proj.scales,
model.layers.23.conv.in_proj.weight,
model.layers.23.conv.out_proj.biases,
model.layers.23.conv.out_proj.scales,
model.layers.23.conv.out_proj.weight,
model.layers.23.feed_forward.w1.biases,
model.layers.23.feed_forward.w1.scales,
model.layers.23.feed_forward.w1.weight,
model.layers.23.feed_forward.w2.biases,
model.layers.23.feed_forward.w2.scales,
model.layers.23.feed_forward.w2.weight,
model.layers.23.feed_forward.w3.biases,
model.layers.23.feed_forward.w3.scales,
model.layers.23.feed_forward.w3.weight,
model.layers.23.ffn_norm.weight,
model.layers.23.operator_norm.weight,
model.layers.24.feed_forward.w1.biases,
model.layers.24.feed_forward.w1.scales,
model.layers.24.feed_forward.w1.weight,
model.layers.24.feed_forward.w2.biases,
model.layers.24.feed_forward.w2.scales,
model.layers.24.feed_forward.w2.weight,
model.layers.24.feed_forward.w3.biases,
model.layers.24.feed_forward.w3.scales,
model.layers.24.feed_forward.w3.weight,
model.layers.24.ffn_norm.weight,
model.layers.24.operator_norm.weight,
model.layers.24.self_attn.k_layernorm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.out_proj.biases,
model.layers.24.self_attn.out_proj.scales,
model.layers.24.self_attn.out_proj.weight,
model.layers.24.self_attn.q_layernorm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.conv.conv.weight,
model.layers.25.conv.in_proj.biases,
model.layers.25.conv.in_proj.scales,
model.layers.25.conv.in_proj.weight,
model.layers.25.conv.out_proj.biases,
model.layers.25.conv.out_proj.scales,
model.layers.25.conv.out_proj.weight,
model.layers.25.feed_forward.w1.biases,
model.layers.25.feed_forward.w1.scales,
model.layers.25.feed_forward.w1.weight,
model.layers.25.feed_forward.w2.biases,
model.layers.25.feed_forward.w2.scales,
model.layers.25.feed_forward.w2.weight,
model.layers.25.feed_forward.w3.biases,
model.layers.25.feed_forward.w3.scales,
model.layers.25.feed_forward.w3.weight,
model.layers.25.ffn_norm.weight,
model.layers.25.operator_norm.weight,
model.layers.26.conv.conv.weight,
model.layers.26.conv.in_proj.biases,
model.layers.26.conv.in_proj.scales,
model.layers.26.conv.in_proj.weight,
model.layers.26.conv.out_proj.biases,
model.layers.26.conv.out_proj.scales,
model.layers.26.conv.out_proj.weight,
model.layers.26.feed_forward.w1.biases,
model.layers.26.feed_forward.w1.scales,
model.layers.26.feed_forward.w1.weight,
model.layers.26.feed_forward.w2.biases,
model.layers.26.feed_forward.w2.scales,
model.layers.26.feed_forward.w2.weight,
model.layers.26.feed_forward.w3.biases,
model.layers.26.feed_forward.w3.scales,
model.layers.26.feed_forward.w3.weight,
model.layers.26.ffn_norm.weight,
model.layers.26.operator_norm.weight,
model.layers.27.feed_forward.w1.biases,
model.layers.27.feed_forward.w1.scales,
model.layers.27.feed_forward.w1.weight,
model.layers.27.feed_forward.w2.biases,
model.layers.27.feed_forward.w2.scales,
model.layers.27.feed_forward.w2.weight,
model.layers.27.feed_forward.w3.biases,
model.layers.27.feed_forward.w3.scales,
model.layers.27.feed_forward.w3.weight,
model.layers.27.ffn_norm.weight,
model.layers.27.operator_norm.weight,
model.layers.27.self_attn.k_layernorm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.out_proj.biases,
model.layers.27.self_attn.out_proj.scales,
model.layers.27.self_attn.out_proj.weight,
model.layers.27.self_attn.q_layernorm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.conv.conv.weight,
model.layers.28.conv.in_proj.biases,
model.layers.28.conv.in_proj.scales,
model.layers.28.conv.in_proj.weight,
model.layers.28.conv.out_proj.biases,
model.layers.28.conv.out_proj.scales,
model.layers.28.conv.out_proj.weight,
model.layers.28.feed_forward.w1.biases,
model.layers.28.feed_forward.w1.scales,
model.layers.28.feed_forward.w1.weight,
model.layers.28.feed_forward.w2.biases,
model.layers.28.feed_forward.w2.scales,
model.layers.28.feed_forward.w2.weight,
model.layers.28.feed_forward.w3.biases,
model.layers.28.feed_forward.w3.scales,
model.layers.28.feed_forward.w3.weight,
model.layers.28.ffn_norm.weight,
model.layers.28.operator_norm.weight,
model.layers.29.conv.conv.weight,
model.layers.29.conv.in_proj.biases,
model.layers.29.conv.in_proj.scales,
model.layers.29.conv.in_proj.weight,
model.layers.29.conv.out_proj.biases,
model.layers.29.conv.out_proj.scales,
model.layers.29.conv.out_proj.weight,
model.layers.29.feed_forward.w1.biases,
model.layers.29.feed_forward.w1.scales,
model.layers.29.feed_forward.w1.weight,
model.layers.29.feed_forward.w2.biases,
model.layers.29.feed_forward.w2.scales,
model.layers.29.feed_forward.w2.weight,
model.layers.29.feed_forward.w3.biases,
model.layers.29.feed_forward.w3.scales,
model.layers.29.feed_forward.w3.weight,
model.layers.29.ffn_norm.weight,
model.layers.29.operator_norm.weight,
model.layers.3.conv.conv.weight,
model.layers.3.conv.in_proj.biases,
model.layers.3.conv.in_proj.scales,
model.layers.3.conv.in_proj.weight,
model.layers.3.conv.out_proj.biases,
model.layers.3.conv.out_proj.scales,
model.layers.3.conv.out_proj.weight,
model.layers.3.feed_forward.w1.biases,
model.layers.3.feed_forward.w1.scales,
model.layers.3.feed_forward.w1.weight,
model.layers.3.feed_forward.w2.biases,
model.layers.3.feed_forward.w2.scales,
model.layers.3.feed_forward.w2.weight,
model.layers.3.feed_forward.w3.biases,
model.layers.3.feed_forward.w3.scales,
model.layers.3.feed_forward.w3.weight,
model.layers.3.ffn_norm.weight,
model.layers.3.operator_norm.weight,
model.layers.4.conv.conv.weight,
model.layers.4.conv.in_proj.biases,
model.layers.4.conv.in_proj.scales,
model.layers.4.conv.in_proj.weight,
model.layers.4.conv.out_proj.biases,
model.layers.4.conv.out_proj.scales,
model.layers.4.conv.out_proj.weight,
model.layers.4.feed_forward.w1.biases,
model.layers.4.feed_forward.w1.scales,
model.layers.4.feed_forward.w1.weight,
model.layers.4.feed_forward.w2.biases,
model.layers.4.feed_forward.w2.scales,
model.layers.4.feed_forward.w2.weight,
model.layers.4.feed_forward.w3.biases,
model.layers.4.feed_forward.w3.scales,
model.layers.4.feed_forward.w3.weight,
model.layers.4.ffn_norm.weight,
model.layers.4.operator_norm.weight,
model.layers.5.feed_forward.w1.biases,
model.layers.5.feed_forward.w1.scales,
model.layers.5.feed_forward.w1.weight,
model.layers.5.feed_forward.w2.biases,
model.layers.5.feed_forward.w2.scales,
model.layers.5.feed_forward.w2.weight,
model.layers.5.feed_forward.w3.biases,
model.layers.5.feed_forward.w3.scales,
model.layers.5.feed_forward.w3.weight,
model.layers.5.ffn_norm.weight,
model.layers.5.operator_norm.weight,
model.layers.5.self_attn.k_layernorm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.out_proj.biases,
model.layers.5.self_attn.out_proj.scales,
model.layers.5.self_attn.out_proj.weight,
model.layers.5.self_attn.q_layernorm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.conv.conv.weight,
model.layers.6.conv.in_proj.biases,
model.layers.6.conv.in_proj.scales,
model.layers.6.conv.in_proj.weight,
model.layers.6.conv.out_proj.biases,
model.layers.6.conv.out_proj.scales,
model.layers.6.conv.out_proj.weight,
model.layers.6.feed_forward.w1.biases,
model.layers.6.feed_forward.w1.scales,
model.layers.6.feed_forward.w1.weight,
model.layers.6.feed_forward.w2.biases,
model.layers.6.feed_forward.w2.scales,
model.layers.6.feed_forward.w2.weight,
model.layers.6.feed_forward.w3.biases,
model.layers.6.feed_forward.w3.scales,
model.layers.6.feed_forward.w3.weight,
model.layers.6.ffn_norm.weight,
model.layers.6.operator_norm.weight,
model.layers.7.conv.conv.weight,
model.layers.7.conv.in_proj.biases,
model.layers.7.conv.in_proj.scales,
model.layers.7.conv.in_proj.weight,
model.layers.7.conv.out_proj.biases,
model.layers.7.conv.out_proj.scales,
model.layers.7.conv.out_proj.weight,
model.layers.7.feed_forward.w1.biases,
model.layers.7.feed_forward.w1.scales,
model.layers.7.feed_forward.w1.weight,
model.layers.7.feed_forward.w2.biases,
model.layers.7.feed_forward.w2.scales,
model.layers.7.feed_forward.w2.weight,
model.layers.7.feed_forward.w3.biases,
model.layers.7.feed_forward.w3.scales,
model.layers.7.feed_forward.w3.weight,
model.layers.7.ffn_norm.weight,
model.layers.7.operator_norm.weight,
model.layers.8.conv.conv.weight,
model.layers.8.conv.in_proj.biases,
model.layers.8.conv.in_proj.scales,
model.layers.8.conv.in_proj.weight,
model.layers.8.conv.out_proj.biases,
model.layers.8.conv.out_proj.scales,
model.layers.8.conv.out_proj.weight,
model.layers.8.feed_forward.w1.biases,
model.layers.8.feed_forward.w1.scales,
model.layers.8.feed_forward.w1.weight,
model.layers.8.feed_forward.w2.biases,
model.layers.8.feed_forward.w2.scales,
model.layers.8.feed_forward.w2.weight,
model.layers.8.feed_forward.w3.biases,
model.layers.8.feed_forward.w3.scales,
model.layers.8.feed_forward.w3.weight,
model.layers.8.ffn_norm.weight,
model.layers.8.operator_norm.weight,
model.layers.9.feed_forward.w1.biases,
model.layers.9.feed_forward.w1.scales,
model.layers.9.feed_forward.w1.weight,
model.layers.9.feed_forward.w2.biases,
model.layers.9.feed_forward.w2.scales,
model.layers.9.feed_forward.w2.weight,
model.layers.9.feed_forward.w3.biases,
model.layers.9.feed_forward.w3.scales,
model.layers.9.feed_forward.w3.weight,
model.layers.9.ffn_norm.weight,
model.layers.9.operator_norm.weight,
model.layers.9.self_attn.k_layernorm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.out_proj.biases,
model.layers.9.self_attn.out_proj.scales,
model.layers.9.self_attn.out_proj.weight,
model.layers.9.self_attn.q_layernorm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14520, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13520, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 600 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.embedding_norm.weight,
model.layers.0.conv.conv.weight,
model.layers.0.conv.in_proj.biases,
model.layers.0.conv.in_proj.scales,
model.layers.0.conv.in_proj.weight,
model.layers.0.conv.out_proj.biases,
model.layers.0.conv.out_proj.scales,
model.layers.0.conv.out_proj.weight,
model.layers.0.feed_forward.w1.biases,
model.layers.0.feed_forward.w1.scales,
model.layers.0.feed_forward.w1.weight,
model.layers.0.feed_forward.w2.biases,
model.layers.0.feed_forward.w2.scales,
model.layers.0.feed_forward.w2.weight,
model.layers.0.feed_forward.w3.biases,
model.layers.0.feed_forward.w3.scales,
model.layers.0.feed_forward.w3.weight,
model.layers.0.ffn_norm.weight,
model.layers.0.operator_norm.weight,
model.layers.1.conv.conv.weight,
model.layers.1.conv.in_proj.biases,
model.layers.1.conv.in_proj.scales,
model.layers.1.conv.in_proj.weight,
model.layers.1.conv.out_proj.biases,
model.layers.1.conv.out_proj.scales,
model.layers.1.conv.out_proj.weight,
model.layers.1.feed_forward.w1.biases,
model.layers.1.feed_forward.w1.scales,
model.layers.1.feed_forward.w1.weight,
model.layers.1.feed_forward.w2.biases,
model.layers.1.feed_forward.w2.scales,
model.layers.1.feed_forward.w2.weight,
model.layers.1.feed_forward.w3.biases,
model.layers.1.feed_forward.w3.scales,
model.layers.1.feed_forward.w3.weight,
model.layers.1.ffn_norm.weight,
model.layers.1.operator_norm.weight,
model.layers.10.conv.conv.weight,
model.layers.10.conv.in_proj.biases,
model.layers.10.conv.in_proj.scales,
model.layers.10.conv.in_proj.weight,
model.layers.10.conv.out_proj.biases,
model.layers.10.conv.out_proj.scales,
model.layers.10.conv.out_proj.weight,
model.layers.10.feed_forward.w1.biases,
model.layers.10.feed_forward.w1.scales,
model.layers.10.feed_forward.w1.weight,
model.layers.10.feed_forward.w2.biases,
model.layers.10.feed_forward.w2.scales,
model.layers.10.feed_forward.w2.weight,
model.layers.10.feed_forward.w3.biases,
model.layers.10.feed_forward.w3.scales,
model.layers.10.feed_forward.w3.weight,
model.layers.10.ffn_norm.weight,
model.layers.10.operator_norm.weight,
model.layers.11.conv.conv.weight,
model.layers.11.conv.in_proj.biases,
model.layers.11.conv.in_proj.scales,
model.layers.11.conv.in_proj.weight,
model.layers.11.conv.out_proj.biases,
model.layers.11.conv.out_proj.scales,
model.layers.11.conv.out_proj.weight,
model.layers.11.feed_forward.w1.biases,
model.layers.11.feed_forward.w1.scales,
model.layers.11.feed_forward.w1.weight,
model.layers.11.feed_forward.w2.biases,
model.layers.11.feed_forward.w2.scales,
model.layers.11.feed_forward.w2.weight,
model.layers.11.feed_forward.w3.biases,
model.layers.11.feed_forward.w3.scales,
model.layers.11.feed_forward.w3.weight,
model.layers.11.ffn_norm.weight,
model.layers.11.operator_norm.weight,
model.layers.12.conv.conv.weight,
model.layers.12.conv.in_proj.biases,
model.layers.12.conv.in_proj.scales,
model.layers.12.conv.in_proj.weight,
model.layers.12.conv.out_proj.biases,
model.layers.12.conv.out_proj.scales,
model.layers.12.conv.out_proj.weight,
model.layers.12.feed_forward.w1.biases,
model.layers.12.feed_forward.w1.scales,
model.layers.12.feed_forward.w1.weight,
model.layers.12.feed_forward.w2.biases,
model.layers.12.feed_forward.w2.scales,
model.layers.12.feed_forward.w2.weight,
model.layers.12.feed_forward.w3.biases,
model.layers.12.feed_forward.w3.scales,
model.layers.12.feed_forward.w3.weight,
model.layers.12.ffn_norm.weight,
model.layers.12.operator_norm.weight,
model.layers.13.feed_forward.w1.biases,
model.layers.13.feed_forward.w1.scales,
model.layers.13.feed_forward.w1.weight,
model.layers.13.feed_forward.w2.biases,
model.layers.13.feed_forward.w2.scales,
model.layers.13.feed_forward.w2.weight,
model.layers.13.feed_forward.w3.biases,
model.layers.13.feed_forward.w3.scales,
model.layers.13.feed_forward.w3.weight,
model.layers.13.ffn_norm.weight,
model.layers.13.operator_norm.weight,
model.layers.13.self_attn.k_layernorm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.out_proj.biases,
model.layers.13.self_attn.out_proj.scales,
model.layers.13.self_attn.out_proj.weight,
model.layers.13.self_attn.q_layernorm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.conv.conv.weight,
model.layers.14.conv.in_proj.biases,
model.layers.14.conv.in_proj.scales,
model.layers.14.conv.in_proj.weight,
model.layers.14.conv.out_proj.biases,
model.layers.14.conv.out_proj.scales,
model.layers.14.conv.out_proj.weight,
model.layers.14.feed_forward.w1.biases,
model.layers.14.feed_forward.w1.scales,
model.layers.14.feed_forward.w1.weight,
model.layers.14.feed_forward.w2.biases,
model.layers.14.feed_forward.w2.scales,
model.layers.14.feed_forward.w2.weight,
model.layers.14.feed_forward.w3.biases,
model.layers.14.feed_forward.w3.scales,
model.layers.14.feed_forward.w3.weight,
model.layers.14.ffn_norm.weight,
model.layers.14.operator_norm.weight,
model.layers.15.conv.conv.weight,
model.layers.15.conv.in_proj.biases,
model.layers.15.conv.in_proj.scales,
model.layers.15.conv.in_proj.weight,
model.layers.15.conv.out_proj.biases,
model.layers.15.conv.out_proj.scales,
model.layers.15.conv.out_proj.weight,
model.layers.15.feed_forward.w1.biases,
model.layers.15.feed_forward.w1.scales,
model.layers.15.feed_forward.w1.weight,
model.layers.15.feed_forward.w2.biases,
model.layers.15.feed_forward.w2.scales,
model.layers.15.feed_forward.w2.weight,
model.layers.15.feed_forward.w3.biases,
model.layers.15.feed_forward.w3.scales,
model.layers.15.feed_forward.w3.weight,
model.layers.15.ffn_norm.weight,
model.layers.15.operator_norm.weight,
model.layers.16.conv.conv.weight,
model.layers.16.conv.in_proj.biases,
model.layers.16.conv.in_proj.scales,
model.layers.16.conv.in_proj.weight,
model.layers.16.conv.out_proj.biases,
model.layers.16.conv.out_proj.scales,
model.layers.16.conv.out_proj.weight,
model.layers.16.feed_forward.w1.biases,
model.layers.16.feed_forward.w1.scales,
model.layers.16.feed_forward.w1.weight,
model.layers.16.feed_forward.w2.biases,
model.layers.16.feed_forward.w2.scales,
model.layers.16.feed_forward.w2.weight,
model.layers.16.feed_forward.w3.biases,
model.layers.16.feed_forward.w3.scales,
model.layers.16.feed_forward.w3.weight,
model.layers.16.ffn_norm.weight,
model.layers.16.operator_norm.weight,
model.layers.17.feed_forward.w1.biases,
model.layers.17.feed_forward.w1.scales,
model.layers.17.feed_forward.w1.weight,
model.layers.17.feed_forward.w2.biases,
model.layers.17.feed_forward.w2.scales,
model.layers.17.feed_forward.w2.weight,
model.layers.17.feed_forward.w3.biases,
model.layers.17.feed_forward.w3.scales,
model.layers.17.feed_forward.w3.weight,
model.layers.17.ffn_norm.weight,
model.layers.17.operator_norm.weight,
model.layers.17.self_attn.k_layernorm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.out_proj.biases,
model.layers.17.self_attn.out_proj.scales,
model.layers.17.self_attn.out_proj.weight,
model.layers.17.self_attn.q_layernorm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.conv.conv.weight,
model.layers.18.conv.in_proj.biases,
model.layers.18.conv.in_proj.scales,
model.layers.18.conv.in_proj.weight,
model.layers.18.conv.out_proj.biases,
model.layers.18.conv.out_proj.scales,
model.layers.18.conv.out_proj.weight,
model.layers.18.feed_forward.w1.biases,
model.layers.18.feed_forward.w1.scales,
model.layers.18.feed_forward.w1.weight,
model.layers.18.feed_forward.w2.biases,
model.layers.18.feed_forward.w2.scales,
model.layers.18.feed_forward.w2.weight,
model.layers.18.feed_forward.w3.biases,
model.layers.18.feed_forward.w3.scales,
model.layers.18.feed_forward.w3.weight,
model.layers.18.ffn_norm.weight,
model.layers.18.operator_norm.weight,
model.layers.19.conv.conv.weight,
model.layers.19.conv.in_proj.biases,
model.layers.19.conv.in_proj.scales,
model.layers.19.conv.in_proj.weight,
model.layers.19.conv.out_proj.biases,
model.layers.19.conv.out_proj.scales,
model.layers.19.conv.out_proj.weight,
model.layers.19.feed_forward.w1.biases,
model.layers.19.feed_forward.w1.scales,
model.layers.19.feed_forward.w1.weight,
model.layers.19.feed_forward.w2.biases,
model.layers.19.feed_forward.w2.scales,
model.layers.19.feed_forward.w2.weight,
model.layers.19.feed_forward.w3.biases,
model.layers.19.feed_forward.w3.scales,
model.layers.19.feed_forward.w3.weight,
model.layers.19.ffn_norm.weight,
model.layers.19.operator_norm.weight,
model.layers.2.feed_forward.w1.biases,
model.layers.2.feed_forward.w1.scales,
model.layers.2.feed_forward.w1.weight,
model.layers.2.feed_forward.w2.biases,
model.layers.2.feed_forward.w2.scales,
model.layers.2.feed_forward.w2.weight,
model.layers.2.feed_forward.w3.biases,
model.layers.2.feed_forward.w3.scales,
model.layers.2.feed_forward.w3.weight,
model.layers.2.ffn_norm.weight,
model.layers.2.operator_norm.weight,
model.layers.2.self_attn.k_layernorm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.out_proj.biases,
model.layers.2.self_attn.out_proj.scales,
model.layers.2.self_attn.out_proj.weight,
model.layers.2.self_attn.q_layernorm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.conv.conv.weight,
model.layers.20.conv.in_proj.biases,
model.layers.20.conv.in_proj.scales,
model.layers.20.conv.in_proj.weight,
model.layers.20.conv.out_proj.biases,
model.layers.20.conv.out_proj.scales,
model.layers.20.conv.out_proj.weight,
model.layers.20.feed_forward.w1.biases,
model.layers.20.feed_forward.w1.scales,
model.layers.20.feed_forward.w1.weight,
model.layers.20.feed_forward.w2.biases,
model.layers.20.feed_forward.w2.scales,
model.layers.20.feed_forward.w2.weight,
model.layers.20.feed_forward.w3.biases,
model.layers.20.feed_forward.w3.scales,
model.layers.20.feed_forward.w3.weight,
model.layers.20.ffn_norm.weight,
model.layers.20.operator_norm.weight,
model.layers.21.feed_forward.w1.biases,
model.layers.21.feed_forward.w1.scales,
model.layers.21.feed_forward.w1.weight,
model.layers.21.feed_forward.w2.biases,
model.layers.21.feed_forward.w2.scales,
model.layers.21.feed_forward.w2.weight,
model.layers.21.feed_forward.w3.biases,
model.layers.21.feed_forward.w3.scales,
model.layers.21.feed_forward.w3.weight,
model.layers.21.ffn_norm.weight,
model.layers.21.operator_norm.weight,
model.layers.21.self_attn.k_layernorm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.out_proj.biases,
model.layers.21.self_attn.out_proj.scales,
model.layers.21.self_attn.out_proj.weight,
model.layers.21.self_attn.q_layernorm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.conv.conv.weight,
model.layers.22.conv.in_proj.biases,
model.layers.22.conv.in_proj.scales,
model.layers.22.conv.in_proj.weight,
model.layers.22.conv.out_proj.biases,
model.layers.22.conv.out_proj.scales,
model.layers.22.conv.out_proj.weight,
model.layers.22.feed_forward.w1.biases,
model.layers.22.feed_forward.w1.scales,
model.layers.22.feed_forward.w1.weight,
model.layers.22.feed_forward.w2.biases,
model.layers.22.feed_forward.w2.scales,
model.layers.22.feed_forward.w2.weight,
model.layers.22.feed_forward.w3.biases,
model.layers.22.feed_forward.w3.scales,
model.layers.22.feed_forward.w3.weight,
model.layers.22.ffn_norm.weight,
model.layers.22.operator_norm.weight,
model.layers.23.conv.conv.weight,
model.layers.23.conv.in_proj.biases,
model.layers.23.conv.in_proj.scales,
model.layers.23.conv.in_proj.weight,
model.layers.23.conv.out_proj.biases,
model.layers.23.conv.out_proj.scales,
model.layers.23.conv.out_proj.weight,
model.layers.23.feed_forward.w1.biases,
model.layers.23.feed_forward.w1.scales,
model.layers.23.feed_forward.w1.weight,
model.layers.23.feed_forward.w2.biases,
model.layers.23.feed_forward.w2.scales,
model.layers.23.feed_forward.w2.weight,
model.layers.23.feed_forward.w3.biases,
model.layers.23.feed_forward.w3.scales,
model.layers.23.feed_forward.w3.weight,
model.layers.23.ffn_norm.weight,
model.layers.23.operator_norm.weight,
model.layers.24.feed_forward.w1.biases,
model.layers.24.feed_forward.w1.scales,
model.layers.24.feed_forward.w1.weight,
model.layers.24.feed_forward.w2.biases,
model.layers.24.feed_forward.w2.scales,
model.layers.24.feed_forward.w2.weight,
model.layers.24.feed_forward.w3.biases,
model.layers.24.feed_forward.w3.scales,
model.layers.24.feed_forward.w3.weight,
model.layers.24.ffn_norm.weight,
model.layers.24.operator_norm.weight,
model.layers.24.self_attn.k_layernorm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.out_proj.biases,
model.layers.24.self_attn.out_proj.scales,
model.layers.24.self_attn.out_proj.weight,
model.layers.24.self_attn.q_layernorm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.conv.conv.weight,
model.layers.25.conv.in_proj.biases,
model.layers.25.conv.in_proj.scales,
model.layers.25.conv.in_proj.weight,
model.layers.25.conv.out_proj.biases,
model.layers.25.conv.out_proj.scales,
model.layers.25.conv.out_proj.weight,
model.layers.25.feed_forward.w1.biases,
model.layers.25.feed_forward.w1.scales,
model.layers.25.feed_forward.w1.weight,
model.layers.25.feed_forward.w2.biases,
model.layers.25.feed_forward.w2.scales,
model.layers.25.feed_forward.w2.weight,
model.layers.25.feed_forward.w3.biases,
model.layers.25.feed_forward.w3.scales,
model.layers.25.feed_forward.w3.weight,
model.layers.25.ffn_norm.weight,
model.layers.25.operator_norm.weight,
model.layers.26.conv.conv.weight,
model.layers.26.conv.in_proj.biases,
model.layers.26.conv.in_proj.scales,
model.layers.26.conv.in_proj.weight,
model.layers.26.conv.out_proj.biases,
model.layers.26.conv.out_proj.scales,
model.layers.26.conv.out_proj.weight,
model.layers.26.feed_forward.w1.biases,
model.layers.26.feed_forward.w1.scales,
model.layers.26.feed_forward.w1.weight,
model.layers.26.feed_forward.w2.biases,
model.layers.26.feed_forward.w2.scales,
model.layers.26.feed_forward.w2.weight,
model.layers.26.feed_forward.w3.biases,
model.layers.26.feed_forward.w3.scales,
model.layers.26.feed_forward.w3.weight,
model.layers.26.ffn_norm.weight,
model.layers.26.operator_norm.weight,
model.layers.27.feed_forward.w1.biases,
model.layers.27.feed_forward.w1.scales,
model.layers.27.feed_forward.w1.weight,
model.layers.27.feed_forward.w2.biases,
model.layers.27.feed_forward.w2.scales,
model.layers.27.feed_forward.w2.weight,
model.layers.27.feed_forward.w3.biases,
model.layers.27.feed_forward.w3.scales,
model.layers.27.feed_forward.w3.weight,
model.layers.27.ffn_norm.weight,
model.layers.27.operator_norm.weight,
model.layers.27.self_attn.k_layernorm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.out_proj.biases,
model.layers.27.self_attn.out_proj.scales,
model.layers.27.self_attn.out_proj.weight,
model.layers.27.self_attn.q_layernorm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.conv.conv.weight,
model.layers.28.conv.in_proj.biases,
model.layers.28.conv.in_proj.scales,
model.layers.28.conv.in_proj.weight,
model.layers.28.conv.out_proj.biases,
model.layers.28.conv.out_proj.scales,
model.layers.28.conv.out_proj.weight,
model.layers.28.feed_forward.w1.biases,
model.layers.28.feed_forward.w1.scales,
model.layers.28.feed_forward.w1.weight,
model.layers.28.feed_forward.w2.biases,
model.layers.28.feed_forward.w2.scales,
model.layers.28.feed_forward.w2.weight,
model.layers.28.feed_forward.w3.biases,
model.layers.28.feed_forward.w3.scales,
model.layers.28.feed_forward.w3.weight,
model.layers.28.ffn_norm.weight,
model.layers.28.operator_norm.weight,
model.layers.29.conv.conv.weight,
model.layers.29.conv.in_proj.biases,
model.layers.29.conv.in_proj.scales,
model.layers.29.conv.in_proj.weight,
model.layers.29.conv.out_proj.biases,
model.layers.29.conv.out_proj.scales,
model.layers.29.conv.out_proj.weight,
model.layers.29.feed_forward.w1.biases,
model.layers.29.feed_forward.w1.scales,
model.layers.29.feed_forward.w1.weight,
model.layers.29.feed_forward.w2.biases,
model.layers.29.feed_forward.w2.scales,
model.layers.29.feed_forward.w2.weight,
model.layers.29.feed_forward.w3.biases,
model.layers.29.feed_forward.w3.scales,
model.layers.29.feed_forward.w3.weight,
model.layers.29.ffn_norm.weight,
model.layers.29.operator_norm.weight,
model.layers.3.conv.conv.weight,
model.layers.3.conv.in_proj.biases,
model.layers.3.conv.in_proj.scales,
model.layers.3.conv.in_proj.weight,
model.layers.3.conv.out_proj.biases,
model.layers.3.conv.out_proj.scales,
model.layers.3.conv.out_proj.weight,
model.layers.3.feed_forward.w1.biases,
model.layers.3.feed_forward.w1.scales,
model.layers.3.feed_forward.w1.weight,
model.layers.3.feed_forward.w2.biases,
model.layers.3.feed_forward.w2.scales,
model.layers.3.feed_forward.w2.weight,
model.layers.3.feed_forward.w3.biases,
model.layers.3.feed_forward.w3.scales,
model.layers.3.feed_forward.w3.weight,
model.layers.3.ffn_norm.weight,
model.layers.3.operator_norm.weight,
model.layers.4.conv.conv.weight,
model.layers.4.conv.in_proj.biases,
model.layers.4.conv.in_proj.scales,
model.layers.4.conv.in_proj.weight,
model.layers.4.conv.out_proj.biases,
model.layers.4.conv.out_proj.scales,
model.layers.4.conv.out_proj.weight,
model.layers.4.feed_forward.w1.biases,
model.layers.4.feed_forward.w1.scales,
model.layers.4.feed_forward.w1.weight,
model.layers.4.feed_forward.w2.biases,
model.layers.4.feed_forward.w2.scales,
model.layers.4.feed_forward.w2.weight,
model.layers.4.feed_forward.w3.biases,
model.layers.4.feed_forward.w3.scales,
model.layers.4.feed_forward.w3.weight,
model.layers.4.ffn_norm.weight,
model.layers.4.operator_norm.weight,
model.layers.5.feed_forward.w1.biases,
model.layers.5.feed_forward.w1.scales,
model.layers.5.feed_forward.w1.weight,
model.layers.5.feed_forward.w2.biases,
model.layers.5.feed_forward.w2.scales,
model.layers.5.feed_forward.w2.weight,
model.layers.5.feed_forward.w3.biases,
model.layers.5.feed_forward.w3.scales,
model.layers.5.feed_forward.w3.weight,
model.layers.5.ffn_norm.weight,
model.layers.5.operator_norm.weight,
model.layers.5.self_attn.k_layernorm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.out_proj.biases,
model.layers.5.self_attn.out_proj.scales,
model.layers.5.self_attn.out_proj.weight,
model.layers.5.self_attn.q_layernorm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.conv.conv.weight,
model.layers.6.conv.in_proj.biases,
model.layers.6.conv.in_proj.scales,
model.layers.6.conv.in_proj.weight,
model.layers.6.conv.out_proj.biases,
model.layers.6.conv.out_proj.scales,
model.layers.6.conv.out_proj.weight,
model.layers.6.feed_forward.w1.biases,
model.layers.6.feed_forward.w1.scales,
model.layers.6.feed_forward.w1.weight,
model.layers.6.feed_forward.w2.biases,
model.layers.6.feed_forward.w2.scales,
model.layers.6.feed_forward.w2.weight,
model.layers.6.feed_forward.w3.biases,
model.layers.6.feed_forward.w3.scales,
model.layers.6.feed_forward.w3.weight,
model.layers.6.ffn_norm.weight,
model.layers.6.operator_norm.weight,
model.layers.7.conv.conv.weight,
model.layers.7.conv.in_proj.biases,
model.layers.7.conv.in_proj.scales,
model.layers.7.conv.in_proj.weight,
model.layers.7.conv.out_proj.biases,
model.layers.7.conv.out_proj.scales,
model.layers.7.conv.out_proj.weight,
model.layers.7.feed_forward.w1.biases,
model.layers.7.feed_forward.w1.scales,
model.layers.7.feed_forward.w1.weight,
model.layers.7.feed_forward.w2.biases,
model.layers.7.feed_forward.w2.scales,
model.layers.7.feed_forward.w2.weight,
model.layers.7.feed_forward.w3.biases,
model.layers.7.feed_forward.w3.scales,
model.layers.7.feed_forward.w3.weight,
model.layers.7.ffn_norm.weight,
model.layers.7.operator_norm.weight,
model.layers.8.conv.conv.weight,
model.layers.8.conv.in_proj.biases,
model.layers.8.conv.in_proj.scales,
model.layers.8.conv.in_proj.weight,
model.layers.8.conv.out_proj.biases,
model.layers.8.conv.out_proj.scales,
model.layers.8.conv.out_proj.weight,
model.layers.8.feed_forward.w1.biases,
model.layers.8.feed_forward.w1.scales,
model.layers.8.feed_forward.w1.weight,
model.layers.8.feed_forward.w2.biases,
model.layers.8.feed_forward.w2.scales,
model.layers.8.feed_forward.w2.weight,
model.layers.8.feed_forward.w3.biases,
model.layers.8.feed_forward.w3.scales,
model.layers.8.feed_forward.w3.weight,
model.layers.8.ffn_norm.weight,
model.layers.8.operator_norm.weight,
model.layers.9.feed_forward.w1.biases,
model.layers.9.feed_forward.w1.scales,
model.layers.9.feed_forward.w1.weight,
model.layers.9.feed_forward.w2.biases,
model.layers.9.feed_forward.w2.scales,
model.layers.9.feed_forward.w2.weight,
model.layers.9.feed_forward.w3.biases,
model.layers.9.feed_forward.w3.scales,
model.layers.9.feed_forward.w3.weight,
model.layers.9.ffn_norm.weight,
model.layers.9.operator_norm.weight,
model.layers.9.self_attn.k_layernorm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.out_proj.biases,
model.layers.9.self_attn.out_proj.scales,
model.layers.9.self_attn.out_proj.weight,
model.layers.9.self_attn.q_layernorm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight.

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|##########| 10/10 [00:00<00:00, 2621.77it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[19:56:34] DEBUG    HF Cache Info for mlx-community/LFM2.5-VL-3B-OptiQ-4bit: size=2698.2 MB, files=12
```

<a id="diagnostic-tencent-youtu-vl-4b-instruct"></a>

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
- *Package:* transformers
- *Error type:* ValueError
- *Error message:* Model loading failed: cannot import name
  'DefaultFastImageProcessorKwargs' from
  'transformers.image_processing_utils_fast' (unknown location)
- *Root error type:* ImportError
- *Root error message:* cannot import name 'DefaultFastImageProcessorKwargs'
  from 'transformers.image_processing_utils_fast' (unknown location)
- *Resolved model revision:* 8d30a0e49662a1d628a472b12df264dbcd768753
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.01279716
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13505, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12735, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 822, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1212, in load
    processor = load_processor(model_path, True, eos_token_ids=eos_token_id, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1357, in load_processor
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14520, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13520, in _run_model_generation
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
Fetching 19 files: 100%|##########| 19/19 [00:00<00:00, 3082.50it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[20:07:00] DEBUG    HF Cache Info for tencent/Youtu-VL-4B-Instruct: size=10229.6 MB, files=27
```

## Completed Runs with Observations

<a id="diagnostic-jinaai-jina-vlm-mlx"></a>

<details>
<summary>jinaai/jina-vlm-mlx — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### jinaai/jina-vlm-mlx

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type jvlm via jina_vlm)
- *Repeated fragment:* keyword: "serene"
- *Title word count:* 4
- *Requested title word range:* [5, 10]
- *Keyword count:* 380
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["boathouse", "pond", "foliage", "tree", "deck",
  "railing", "rustic", "outdoor", "serene", "nature", "birdwatching",
  "pondside"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* a987631a01dc554a787d87a45fb01fb48f8aaca4
- *Processor class:* mlx_vlm.models.jina_vlm.processing_jinavlm.JinaVLMProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.000754728
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2072
- *Prompt composition:* 2,072 = 329 text/template (tokenizer-exact) + 1,743
  non-text (84%, image/audio expansion)
- *Checkpoint weights (GB):* 2.11
- *Quantization:* 4-bit, group 64
- *Declared context length:* 40,960 (text_config.max_sequence_length)
- *Load active memory vs checkpoint:* 1.00x (2.11 GB vs 2.11 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151643
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
A Serene Pondside Hideout
Description:
A rustic wooden boathouse built on stilts overlooks a tranquil pond, surrounded by lush foliage and framed by the overhanging branches of a nearby tree. The wooden deck and railing add to the charm of this peaceful outdoor retreat, perfect for birdwatching or simply enjoying the serenity of nature.
Keywords:
boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond, foliage, tree, deck, railing, rustic, outdoor, serene, nature, birdwatching, pondside, serene, pondside, boathouse, pond,
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Repeated fragment:* keyword: "boardwalk"
- *Declared generation wrappers in output:* ["&lt;/think&gt;"]
- *Thinking trace markers:* ["&lt;think&gt;", "&lt;/think&gt;"]
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 158
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["boardwalk"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001492014
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1641
- *Prompt composition:* 1,641 = 400 text/template (tokenizer-exact) + 1,241
  non-text (76%, image/audio expansion)
- *Checkpoint weights (GB):* 58.80
- *Parameter count:* 28.00B (name-estimate)
- *Declared context length:* 131,072 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (58.80 GB vs 58.80 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;
- *Configured thinking start token:* &lt;think&gt;
- *Configured thinking end token:* &lt;/think&gt;
- *Configured thinking budget:* 800
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Alright, let's get to work. I've got an image and some hints to work with, and my task is to create some metadata for it. First things first, I need to break down the request. It's a straightforward task: title, description, and keywords. The image is of a boathouse, and I have the context to work with.

The image itself is a bit of a mess, but I can see the core elements. It's a wooden boathouse, likely a traditional one, built on stilts over water. The setting is a wetland or marsh, with reeds, willows, and a pond. The lighting is overcast, which is important for the mood.

Now, the hints. The description is a bit repetitive, but I can extract the key elements: weathered wooden boathouse, stilts, pond, tree branches, decking, wetland foliage. The keywords are more useful, but some are redundant. I need to be careful about what to include.

The authoritative context is crucial. The date and time are useless for the image itself, but the GPS coordinates are gold. That's the key to the location. I'll use that to pinpoint the location.

Okay, time to synthesize. The title needs to be concise and descriptive. I'll go with "Wooden Boathouse Over a Wetland Pond." The description needs to be factual and incorporate the context. I'll start with the location, then describe the boathouse, the setting, and the weather. The keywords need to be unique and relevant. I'll use the authoritative GPS coordinates, the key elements from the hints, and the location name.

Let's refine the description. I'll make sure it's accurate and flows well. I'll also make sure to include the location name.

Final check: Title, description, keywords. All good. I'm ready to output the metadata.
</think>

Title: Wooden Boathouse Over a Wetland Pond

Description: A weathered wooden boathouse built on stilts stands over the edge of a serene pond in a wetland, framed by lush overhanging tree branches and surrounding reeds. The scene is set in a tranquil, overcast landscape with a damp, green atmosphere.

Keywords: 51.441113°N, 0.565406°W, boathouse, wetland, pond, reeds, willow, overcast sky, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk, boardwalk,
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — repeated text; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_incomplete,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Repeated fragment:* keyword: "foliage"
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this task
  step by step. First, the title needs to be 5-10 words, concrete. Looking at
  the image: weathered wooden boathouse, stilts, pond, trees, etc. Maybe
  "Weathered Wooden Boathouse by the Pond".  Next, the description: combine
  context (date/time, GPS) with main subject, setting, action, lighting,
  distinctive details. The image has a weathered wooden structure on stilts
  over water, lush trees, cloudy sky, wetland. So something like "A weathered
  wooden boathouse on stilts overlooks a serene pond, surrounded by lush
  foliage and under a cloudy sky, captured on 2026-08-21." Wait, need to make
  sure it's 1-2 sentences. Let's check the hints and image. The boathouse is
  on stilts, over pond, framed by trees, wooden decking. So description: "A
  weathered wooden boathouse on stilts stands over a serene pond, framed by
  lush overhanging trees and wetland vegetation, under a cloudy sky, captured
  on 2026-08-21."  Then keywords: need 10-18 unique, comma-separated. From
  hints and image: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape,
  Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection,
  Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk.
  Let's pick the most relevant ones. Let's list: Cloudy Sky, Foliage, Pond,
  Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves,
  Outdoors, Landscape, Moss, Boardwalk, Architecture, Grass. Wait, need to
  make sure they are unique. Let's check each hint.  Now, let's structure each
  part:
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 10
- *Requested title word range:* [5, 10]
- *Keyword count:* 80
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["foliage", "pond", "wetland", "wooden shed", "trees",
  "water reflection", "marshland", "reeds", "leaves", "outdoors", "landscape",
  "moss", "boardwalk", "architecture", "grass", "bird hide", "birdwatching"]
- *Token-cap degradation evidence:* ["repetitive_tail",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002114608
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6388
- *Prompt composition:* 6,388 = 309 text/template (tokenizer-exact) + 6,079
  non-text (95%, image/audio expansion)
- *Checkpoint weights (GB):* 11.77
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 8-bit, group 64, affine
- *Declared context length:* 65,536 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (11.78 GB vs 11.77 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. Looking at the image: weathered wooden boathouse, stilts, pond, trees, etc. Maybe "Weathered Wooden Boathouse by the Pond". 

Next, the description: combine context (date/time, GPS) with main subject, setting, action, lighting, distinctive details. The image has a weathered wooden structure on stilts over water, lush trees, cloudy sky, wetland. So something like "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21." Wait, need to make sure it's 1-2 sentences. Let's check the hints and image. The boathouse is on stilts, over pond, framed by trees, wooden decking. So description: "A weathered wooden boathouse on stilts stands over a serene pond, framed by lush overhanging trees and wetland vegetation, under a cloudy sky, captured on 2026-08-21." 

Then keywords: need 10-18 unique, comma-separated. From hints and image: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Let's pick the most relevant ones. Let's list: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture, Grass. Wait, need to make sure they are unique. Let's check each hint. 

Now, let's structure each part:

Title: "Weathered Wooden Boathouse by the Pond" (about 8 words, concrete)

Description: Combine context (date/time, GPS) with subject, setting, etc. So: "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21 at 14:34:53 UTC+01:00 near 51.441113°N, 0.565406°W." Wait, but need to make it 1-2 sentences. Maybe shorter: "A weathered wooden boathouse on stilts stands over a serene pond, framed by lush trees and wetland vegetation, under a cloudy sky, captured on 2026-08-21." That's two sentences? Wait, no, 1-2 sentences. Let's make it one: "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21." 

Keywords: Let's pick the most relevant ones from hints and image. Let's list: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture, Grass. Wait, let's check the hints again. The hints include: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. So unique ones: Cloudy Sky, Foliage, Forest (or Trees), Grass, Lake (or Pond), Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, Architecture, Bird hide, Birdwatching, Boardwalk. Let's pick 12-15. Let's do: Cloudy Sky, Foliage, Pond, Wetland, Wooden shed, Trees, Water reflection, Marshland, Reeds, Leaves, Outdoors, Landscape, Moss, Boardwalk, Architecture. That's 15, unique. 

Wait, let's make sure the description is factual, combining context with main subject. So: "A weathered wooden boathouse on stilts stands over a serene pond, surrounded by lush foliage and under a cloudy sky, captured on 2026-08-21 at 14:34:53 UTC+01:00 near 51.441113°N, 0.565406°W." That's two sentences, but maybe can make it one. "A weathered wooden boathouse on stilts overlooks a serene pond, framed by lush trees and wetland vegetation, under a cloudy sky, captured
```

</details>

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
- *Repeated fragment:* keyword: "elevated"
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 395
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["elevated", "wetland", "pond", "outdoor", "railings",
  "reflection", "natural", "environment", "architecture", "birdwatching",
  "peaceful", "serene"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00409712
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2117
- *Prompt composition:* 2,117 = 327 text/template (tokenizer-exact) + 1,790
  non-text (85%, image/audio expansion)
- *Checkpoint weights (GB):* 3.19
- *Parameter count:* 1.60B (name-estimate)
- *Declared context length:* 128,000 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (3.20 GB vs 3.19 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: "Moss-Covered Wooden Boathouse Over Wetland Pond"

Description: A weathered wooden boathouse stands elevated on stilts over a serene pond, surrounded by lush foliage and marshland. The overhanging tree branches frame the structure, while the cloudy sky casts a soft, diffused light over the scene. The wooden decking and railings add to the rustic charm of this tranquil outdoor setting.

Keywords: boathouse, wooden, moss, elevated, wetland, pond, overhanging, foliage, marshland, cloudy, diffused, light, rustic, tranquil, outdoor, setting, structure, railings, decking, railings, trees, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated, wetland, pond, reflection, natural, environment, architecture, birdwatching, outdoor, peaceful, serene, elevated
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
- *Repeated fragment:* keyword: "accurate"
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 436
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["pond", "wetland", "grass", "reeds", "trees",
  "architecture", "moss", "weathered", "landscape", "outdoor", "nature",
  "structure", "environment", "ecosystem", "habitat", "accurate", "precise",
  "detailed", "comprehensive", "thorough"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.01017563
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16563
- *Prompt composition:* 16,563 = 340 text/template (tokenizer-exact) + 16,223
  non-text (98%, image/audio expansion)
- *Checkpoint weights (GB):* 9.45
- *Parameter count:* 7.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 128,000 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (9.46 GB vs 9.45 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Wooden Boathouse Over Serene Pond

Description:
A weathered wooden boathouse, elevated on stilts, stands over a tranquil pond, framed by lush greenery and overhanging tree branches, with a cloudy sky and reflections on the water.

Keywords: 
Wooden boathouse, stilted, pond, serene, cloudy sky, lush foliage, overhanging branches, water reflection, wetland, grass, reeds, trees, architecture, bird hide, birdwatching, boardwalk, moss, weathered, landscape, outdoor, nature, countryside, 2026-08-21, 51.441113°N, 0.565406°W, architecture, boathouse, shed, structure, stillness, calm, natural setting, rural, environment, ecosystem, wildlife, habitat, peaceful, quiet, still, calmness, tranquility, natural beauty, greenery, foliage, leaves, moss, water, reflection, pond, lake, wetland, marshland, forest, grass, reeds, trees, sky, clouds, overcast, weathered, aged, rustic, traditional, historical, cultural, heritage, preservation, conservation, nature reserve, wildlife sanctuary, ecological, biodiversity, ecosystem, habitat, natural, environment, outdoor, outdoors, nature, landscape, scenery, view, perspective, angle, composition, balance, symmetry, harmony, contrast, texture, material, wood, wooden, structure, building, shelter, protection, weather, climate, season, summer, August, 2026, date, time, location, coordinates, GPS, authoritative, factual, descriptive, evidence-based, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed, comprehensive, thorough, accurate, precise, detailed,
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
- *Post-cleanup active memory (GB):* 0.01017565
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 594
- *Prompt composition:* 594 = 333 text/template (tokenizer-exact) + 261
  non-text (44%, image/audio expansion)
- *Checkpoint weights (GB):* 27.97
- *Parameter count:* 26.00B (name-estimate)
- *Quantization:* 8-bit, group 64, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (27.99 GB vs 27.97 GB on disk)
- *Generation tokens:* 88
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: Weathered Wooden Boathouse Over Serene Wetland Pond
Description: A weathered wooden boathouse built on stilts stands over a serene pond, framed by lush green trees and wetland foliage under a cloudy sky.
Keywords: architecture, birdwatching, hide, boardwalk, foliage, forest, landscape, marshland, moss, outdoors, pond, reeds, trees, water reflection, wetland, wooden shed
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
- *Post-cleanup active memory (GB):* 0.01017565
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 594
- *Prompt composition:* 594 = 333 text/template (tokenizer-exact) + 261
  non-text (44%, image/audio expansion)
- *Checkpoint weights (GB):* 27.20
- *Parameter count:* 26.00B (name-estimate)
- *Quantization:* 8-bit, group 32, mxfp8
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (27.21 GB vs 27.20 GB on disk)
- *Generation tokens:* 84
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: Weathered Wooden Boathouse over a Serene Pond
Description: A weathered wooden boathouse built on stilts stands over a pond, framed by lush green foliage under a cloudy sky.
Keywords: Boathouse, Wooden, Pond, Stilts, Wetland, Foliage, Trees, Boardwalk, Re, Marshland, Reflection, Cloudy Sky, Architecture, Nature, Outdoors
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
- *Post-cleanup active memory (GB):* 0.002737212
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6392
- *Prompt composition:* 6,392 = 313 text/template (tokenizer-exact) + 6,079
  non-text (95%, image/audio expansion)
- *Checkpoint weights (GB):* 61.86
- *Quantization:* 4-bit, group 16, nvfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (61.86 GB vs 61.86 GB on disk)
- *Generation tokens:* 95
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Weathered Wooden Boathouse Over Pond
Description: A weathered wooden boathouse on stilts stands over a serene pond, framed by lush tree branches, wooden decking, and surrounding wetland foliage, under a cloudy sky.
Keywords: boathouse, wooden, stilts, pond, cloudy sky, foliage, trees, wetland, marshland, reeds, water reflection, boardwalk, bird hide, architecture, outdoors, landscape<|end_of_box|>
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
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003966048
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1329
- *Prompt composition:* 1,329 = 316 text/template (tokenizer-exact) + 1,013
  non-text (76%, image/audio expansion)
- *Checkpoint weights (GB):* 32.82
- *Parameter count:* 3.00B (name-estimate)
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (32.82 GB vs 32.82 GB on disk)
- *Generation tokens:* 682
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords. 

Starting with the title: The image shows a weathered wooden boathouse on stilts over a pond, surrounded by trees and foliage. The context mentions a serene pond, so maybe something like "Weathered Wooden Boathouse Over Serene Pond" but that's a bit long. Maybe shorten to "Weathered wooden boathouse over serene pond". That's 7 words.

Next, the description: Need to combine context and visible details. The boathouse is weathered, on stilts, over a pond, with lush foliage, trees, wooden decking, wetland. So something like "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees framing the scene under a cloudy sky." That's a bit long, maybe trim to "A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees under a cloudy sky." That's 2 sentences.

Keywords: Need to pick 10-18 unique ones. From the hints: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk. Let's pick the most relevant: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching, Landscape. Wait, maybe check the image: the boathouse is a bird hide, so include that. Also, the decking is visible. So maybe: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching, Landscape. That's 14 keywords. Let me check if they are all visible or inferred. The image has a cloudy sky, foliage, forest, grass, lake, landscape, leaves, marshland, pond, wetland, wooden shed, boardwalk, water reflection, bird hide, birdwatching, and landscape again. Maybe remove one "Landscape" to avoid repetition. So: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching. That's 14.◁/think▷Title: Weathered wooden boathouse over serene pond  
Description: A weathered wooden boathouse on stilts overlooks a serene pond, surrounded by lush foliage and wetland vegetation, with a wooden deck and overhanging trees under a cloudy sky.  
Keywords: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Pond, Wetland, Wooden shed, Boardwalk, Water reflection, Bird hide, Birdwatching<|im_assistant|>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                             | Usability           | Observations                                                         |
|---------------------------------------------------|---------------------|----------------------------------------------------------------------|
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX     | unusable            | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16                   | unusable            | missing required fields                                              |
| mlx-community/gemma-3n-E4B-it-bf16                | unusable            | missing required fields                                              |
| mlx-community/MiniCPM-V-4.6-8bit                  | unusable            | missing required fields; extra text before Title                     |
| mlx-community/GLM-4.6V-Flash-mxfp4                | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/InternVL3-8B-bf16                   | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/North-Micro-Vision-Instruct-4bit    | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Ornith-1.0-35B-bf16                 | usable_with_caveats | title/keyword constraints failed; draft hints copied unchanged       |
| mlx-community/Phi-3.5-vision-instruct-bf16        | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/pixtral-12b-8bit                    | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3-VL-2B-Thinking-bf16           | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3.5-9B-MLX-4bit                 | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3.6-27B-mxfp8                   | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3.8-27B-4bit                    | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx          | usable_with_caveats | title/keyword constraints failed; draft hints copied unchanged       |
| mlx-community/Step-3.7-Flash-oQ2e                 | usable_with_caveats | title/keyword constraints failed; draft hints copied unchanged       |
| Qwen/Qwen3-VL-2B-Instruct                         | usable_with_caveats | title/keyword constraints failed                                     |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                                 | Runtime identity                                          | Performance                                                                               |
|-------------------------------------------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | rev ed71acdae079; Lfm2VlProcessor; stop completed         | 2117 prompt / 118 generated; 481 tok/s; 1.9 GB peak; cleanup 0.000132/0.0 GB active/cache |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed       | 2401 prompt / 121 generated; 29.4 tok/s; 23 GB peak; cleanup 0.00128/0.0 GB active/cache  |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed         | 593 prompt / 153 generated; 31.3 tok/s; 17 GB peak; cleanup 0.0107/0.0 GB active/cache    |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed         | 598 prompt / 100 generated; 129 tok/s; 16 GB peak; cleanup 0.0118/0.0 GB active/cache     |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed         | 598 prompt / 95 generated; 26.4 tok/s; 20 GB peak; cleanup 0.0123/0.0 GB active/cache     |
| mlx-community/granite-4.0-3b-vision-4bit              | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed | 1383 prompt / 85 generated; 171 tok/s; 4.6 GB peak; cleanup 0.0125/0.0 GB active/cache    |
| mlx-community/Idefics3-8B-Llama3-bf16                 | rev 8c2a30c48864; Idefics3Processor; stop completed       | 2618 prompt / 164 generated; 34.0 tok/s; 18 GB peak; cleanup 0.003/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed       | 2934 prompt / 123 generated; 67.6 tok/s; 13 GB peak; cleanup 0.00488/0.0 GB active/cache  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed       | 2933 prompt / 134 generated; 190 tok/s; 7.8 GB peak; cleanup 0.00541/0.0 GB active/cache  |
| mlx-community/Molmo2-8B-4bit                          | rev 4fcbe9265776; Molmo2Processor; stop completed         | 1529 prompt / 133 generated; 72.5 tok/s; 8.8 GB peak; cleanup 0.00572/0.0 GB active/cache |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit          | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed        | 16552 prompt / 129 generated; 87.1 tok/s; 23 GB peak; cleanup 0.00744/0.0 GB active/cache |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed        | 16568 prompt / 111 generated; 110 tok/s; 24 GB peak; cleanup 0.00795/0.0 GB active/cache  |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,409 x 6,273 pixels
- *Image size:* 51,431,731 bytes
- *Image SHA-256:* dadec238f988c92cd592f7ba686543f85856f67b00665ba8d8d2830881d211b5

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-21 14:34:53 UTC+01:00
- GPS: 51.441113°N, 0.565406°W

Descriptive hints:
- Description hint: A weathered wooden boathouse built on stilts stands over the edge of a serene pond, framed by lush overhanging tree branches, wooden decking, and surrounding wetland foliage.
- Keyword hints: Cloudy Sky, Foliage, Forest, Grass, Lake, Landscape, Leaves, Marshland, Moss, Outdoors, Pond, Reeds, Trees, Water reflection, Wetland, Wooden shed, architecture, bird hide, birdwatching, boardwalk

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

The shared command omits per-model automatic thinking flags. When substituting
these models, append the flags recorded in their diagnostics blocks:
`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16` (--enable-thinking
--thinking-budget 800).

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit            | 12c5ae49304158b0a133fcea9ba4486a6d6c8cad |
| tencent/Youtu-VL-4B-Instruct                     | 8d30a0e49662a1d628a472b12df264dbcd768753 |
| jinaai/jina-vlm-mlx                              | a987631a01dc554a787d87a45fb01fb48f8aaca4 |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/GLM-4.1V-9B-Thinking-8bit          | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/LFM2.5-VL-1.6B-bf16                | 16a710cf8afca206ff16a95a4ad6fe657f876ce1 |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.0rc0                                                                                                                                        |
| mlx-vlm source revision    | 24b244ee29f1646e14d1ba935ba2c6bafd3f78f6                                                                                                        |
| mlx                        | 0.32.3.dev20260828+99e45f71d                                                                                                                    |
| mlx source revision        | 99e45f71dcb4318e2c2530e66038045795883ad2                                                                                                        |
| mlx-lm                     | 0.32.0                                                                                                                                          |
| mlx-lm source revision     | 1f9883c91ab726c6a44fc0249adbfea283ca0b33                                                                                                        |
| mlx-audio                  | 0.5.0                                                                                                                                           |
| transformers               | 5.16.1                                                                                                                                          |
| tokenizers                 | 0.23.1                                                                                                                                          |
| huggingface-hub            | 1.29.0                                                                                                                                          |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 25.6.0                                                                                                                                   |
| macOS Version              | 26.6.2                                                                                                                                          |
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
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-314-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (182,433,408 bytes, sha256=612fa3487b6372fdf8e64e4f30f98a8403ca3f1e5ba118be7a94bfbd9fc3335c) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,958,160 bytes, sha256=2dfce4cfa787ed063f5b8a8f0730b9ba0fbb1e0d0b5b550fe64203328746910c)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
