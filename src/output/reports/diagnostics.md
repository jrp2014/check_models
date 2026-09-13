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
| Attempted           | 41    |
| Conclusive outcomes | 41    |
| Completed           | 38    |
| Crashed             | 3     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 3     |
| none                           | 34    |
| observation needs reproduction | 4     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| not assessed         | 3     |
| major concerns       | 6     |
| no concerns detected | 28    |
| concerns detected    | 4     |

Observation counts

| Observation                                                  | Count |
|--------------------------------------------------------------|-------|
| Response repeats the same text                               | 1     |
| Generation was stopped early after sustained repeated output | 2     |
| Unrecognised model control tokens remain visible             | 3     |
| Required labelled fields not detected                        | 4     |
| Response appears cut off at the token limit                  | 2     |
| Conversation-role control tokens remain visible              | 1     |
| Repeated keyword entries                                     | 4     |

## Triage

| Model                                                                                                           | Execution | Mechanical checks | Maintainer status              | Observations                                                                                      |
|-----------------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|---------------------------------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-30B-A3B-4bit](#diagnostic-mlx-community-internvl35-30b-a3b-4bit)                     | crashed   | not assessed      | actionable_failure             | none                                                                                              |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-4bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-4bit) | crashed   | not assessed      | actionable_failure             | none                                                                                              |
| [mlx-community/Mage-VL-OptiQ-4bit](#diagnostic-mlx-community-mage-vl-optiq-4bit)                                | crashed   | not assessed      | actionable_failure             | none                                                                                              |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | completed | major concerns    | observation_needs_reproduction | repeated text; stopped early: repeating; control tokens visible; labelled fields not detected     |
| [mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit](#diagnostic-mlx-community-qwen3-vl-30b-a3b-instruct-4bit)        | completed | major concerns    | observation_needs_reproduction | stopped early: repeating; duplicate keywords                                                      |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit)              | completed | major concerns    | observation_needs_reproduction | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [mlx-community/aya-vision-8b-4bit](#diagnostic-mlx-community-aya-vision-8b-4bit)                                | completed | concerns detected | observation_needs_reproduction | control tokens visible                                                                            |

## Crashes requiring action

<a id="diagnostic-mlx-community-internvl35-30b-a3b-4bit"></a>

### mlx-community/InternVL3_5-30B-A3B-4bit

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
- *Resolved model revision:* ed2ce3381528db1c5b70a2aad78a6390997e9250
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.001819698
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 17.79
- *Parameter count:* 30.00B total, 3.00B active (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 40,960 (text_config.max_position_embeddings)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14168, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13080, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 823, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1306, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 964, in load_model
    model_class, _ = get_model_and_args(config=config, model_path=model_path)
                     ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 784, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15198, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14183, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Fetching 17 files:   0%|          | 0/17 [00:00<?, ?it/s]
Fetching 17 files: 100%|##########| 17/17 [00:00<00:00, 3441.44it/s]
ERROR:root:Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
```

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-4bit"></a>

### mlx-community/Llama-3.2-11B-Vision-Instruct-4bit

#### Root exception and chain

```text
builtins.ValueError: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
builtins.ValueError: Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
```

#### Execution and provenance

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Phase:* generation_before_first_token
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model generation failed for
  mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes
  (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.
- *Root error type:* ValueError
- *Root error message:* [broadcast_shapes] Shapes (1,1,301,6404) and
  (1,32,300,6404) cannot be broadcast.
- *Resolved model revision:* 82f31be9840fa0d4c7e99257fe2e28b59a46df97
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* exception
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.00273725
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 6.01
- *Parameter count:* 11.00B (name-estimate)
- *Quantization:* 4-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13414, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14091, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<5 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13986, in _generate_with_repetition_guard
    for chunk in stream_generate(
                 ~~~~~~~~~~~~~~~^
        model=model, processor=processor, prompt=prompt, image=image, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1081, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 497, in generate_step
    chunk_output = model.language_model(
        inputs=input_ids[:, :n_to_process],
    ...<3 lines>...
        **chunk_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 360, in __call__
    hidden_states = self.model(
        input_ids=inputs,
    ...<5 lines>...
        cache=cache,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 321, in __call__
    layer_outputs = decoder_layer(
        hidden_states,
    ...<3 lines>...
        cache=c,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 251, in __call__
    hidden_states = self.cross_attn(
        hidden_states=hidden_states,
    ...<2 lines>...
        cache=cache,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 80, in __call__
    attn_output = scaled_dot_product_attention(
        query_states,
    ...<4 lines>...
        mask=attention_mask,  # add a dim for batch processing
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 418, in scaled_dot_product_attention
    return mx.fast.scaled_dot_product_attention(
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        queries,
        ^^^^^^^^
    ...<4 lines>...
        sinks=sinks,
        ^^^^^^^^^^^^
    )
    ^
ValueError: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15198, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14195, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14111, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13423, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be broadcast.

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 4797.15it/s]
[02:14:11] Generation error for mlx-community/Llama-3.2-11B-Vision-Instruct-4bit
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13414, in
           _run_generation_guarded
               return generate_once()
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14091, in
           _generate_once
               return _generate_with_repetition_guard(
                   model=prepared.model,
               ...<5 lines>...
                   **prepared.generate_kwargs,
               )
             File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13986, in
           _generate_with_repetition_guard
               for chunk in stream_generate(
                            ~~~~~~~~~~~~~~~^
                   model=model, processor=processor, prompt=prompt, image=image, **kwargs
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               ):
               ^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1081, in
           stream_generate
               for n, (token, logprobs) in enumerate(gen):
                                           ~~~~~~~~~^^^^^
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 497, in
           generate_step
               chunk_output = model.language_model(
                   inputs=input_ids[:, :n_to_process],
               ...<3 lines>...
                   **chunk_kwargs,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 360,
           in __call__
               hidden_states = self.model(
                   input_ids=inputs,
               ...<5 lines>...
                   cache=cache,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 321,
           in __call__
               layer_outputs = decoder_layer(
                   hidden_states,
               ...<3 lines>...
                   cache=c,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 251,
           in __call__
               hidden_states = self.cross_attn(
                   hidden_states=hidden_states,
               ...<2 lines>...
                   cache=cache,
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/mllama/language.py", line 80,
           in __call__
               attn_output = scaled_dot_product_attention(
                   query_states,
               ...<4 lines>...
                   mask=attention_mask,  # add a dim for batch processing
               )
             File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/base.py", line 418, in
           scaled_dot_product_attention
               return mx.fast.scaled_dot_product_attention(
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
                   queries,
                   ^^^^^^^^
               ...<4 lines>...
                   sinks=sinks,
                   ^^^^^^^^^^^^
               )
               ^
           ValueError: [broadcast_shapes] Shapes (1,1,301,6404) and (1,32,300,6404) cannot be
           broadcast.
```

<a id="diagnostic-mlx-community-mage-vl-optiq-4bit"></a>

### mlx-community/Mage-VL-OptiQ-4bit

#### Root exception and chain

```text
builtins.ValueError: Received 904 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
builtins.ValueError: Model loading failed: Received 904 parameters not in model; families: model; representative parameters: model.embed_tokens.biases, model.embed_tokens.scales, model.embed_tokens.weight.
```

#### Execution and provenance

- *Execution:* crashed
- *Mechanical checks:* not assessed
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type mage_vl)
- *Phase:* model_load
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model loading failed: Received 904 parameters not in model;
  families: model; representative parameters: model.embed_tokens.biases,
  model.embed_tokens.scales, model.embed_tokens.weight.
- *Root error type:* ValueError
- *Root error message:* Received 904 parameters not in model; families: model;
  representative parameters: model.embed_tokens.biases,
  model.embed_tokens.scales, model.embed_tokens.weight.
- *Resolved model revision:* bde6c9c7146acff6af09e203245014f19306c5c5
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.00273725
- *Post-cleanup cache memory (GB):* 0.0
- *Checkpoint weights (GB):* 3.92
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14168, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13080, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 823, in _typed_mlx_vlm_load
    loaded: tuple[nn.Module, ProcessorMixin] = _mlx_vlm_load(
                                               ~~~~~~~~~~~~~^
        path_or_hf_repo=path_or_hf_repo,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        **kwargs,
        ^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1306, in load
    model = load_model(model_path, lazy, strict=strict, **kwargs)
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 1171, in load_model
    model.load_weights(list(weights.items()), strict=strict)
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 185, in load_weights
    raise ValueError(
        f"Received {num_extra} parameters not in model: \n{extras}."
    )
ValueError: Received 904 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.layers.0.input_layernorm.weight,
model.layers.0.mlp.down_proj.biases,
model.layers.0.mlp.down_proj.scales,
model.layers.0.mlp.down_proj.weight,
model.layers.0.mlp.gate_proj.biases,
model.layers.0.mlp.gate_proj.scales,
model.layers.0.mlp.gate_proj.weight,
model.layers.0.mlp.up_proj.biases,
model.layers.0.mlp.up_proj.scales,
model.layers.0.mlp.up_proj.weight,
model.layers.0.post_attention_layernorm.weight,
model.layers.0.self_attn.k_norm.weight,
model.layers.0.self_attn.k_proj.biases,
model.layers.0.self_attn.k_proj.scales,
model.layers.0.self_attn.k_proj.weight,
model.layers.0.self_attn.o_proj.biases,
model.layers.0.self_attn.o_proj.scales,
model.layers.0.self_attn.o_proj.weight,
model.layers.0.self_attn.q_norm.weight,
model.layers.0.self_attn.q_proj.biases,
model.layers.0.self_attn.q_proj.scales,
model.layers.0.self_attn.q_proj.weight,
model.layers.0.self_attn.v_proj.biases,
model.layers.0.self_attn.v_proj.scales,
model.layers.0.self_attn.v_proj.weight,
model.layers.1.input_layernorm.weight,
model.layers.1.mlp.down_proj.biases,
model.layers.1.mlp.down_proj.scales,
model.layers.1.mlp.down_proj.weight,
model.layers.1.mlp.gate_proj.biases,
model.layers.1.mlp.gate_proj.scales,
model.layers.1.mlp.gate_proj.weight,
model.layers.1.mlp.up_proj.biases,
model.layers.1.mlp.up_proj.scales,
model.layers.1.mlp.up_proj.weight,
model.layers.1.post_attention_layernorm.weight,
model.layers.1.self_attn.k_norm.weight,
model.layers.1.self_attn.k_proj.biases,
model.layers.1.self_attn.k_proj.scales,
model.layers.1.self_attn.k_proj.weight,
model.layers.1.self_attn.o_proj.biases,
model.layers.1.self_attn.o_proj.scales,
model.layers.1.self_attn.o_proj.weight,
model.layers.1.self_attn.q_norm.weight,
model.layers.1.self_attn.q_proj.biases,
model.layers.1.self_attn.q_proj.scales,
model.layers.1.self_attn.q_proj.weight,
model.layers.1.self_attn.v_proj.biases,
model.layers.1.self_attn.v_proj.scales,
model.layers.1.self_attn.v_proj.weight,
model.layers.10.input_layernorm.weight,
model.layers.10.mlp.down_proj.biases,
model.layers.10.mlp.down_proj.scales,
model.layers.10.mlp.down_proj.weight,
model.layers.10.mlp.gate_proj.biases,
model.layers.10.mlp.gate_proj.scales,
model.layers.10.mlp.gate_proj.weight,
model.layers.10.mlp.up_proj.biases,
model.layers.10.mlp.up_proj.scales,
model.layers.10.mlp.up_proj.weight,
model.layers.10.post_attention_layernorm.weight,
model.layers.10.self_attn.k_norm.weight,
model.layers.10.self_attn.k_proj.biases,
model.layers.10.self_attn.k_proj.scales,
model.layers.10.self_attn.k_proj.weight,
model.layers.10.self_attn.o_proj.biases,
model.layers.10.self_attn.o_proj.scales,
model.layers.10.self_attn.o_proj.weight,
model.layers.10.self_attn.q_norm.weight,
model.layers.10.self_attn.q_proj.biases,
model.layers.10.self_attn.q_proj.scales,
model.layers.10.self_attn.q_proj.weight,
model.layers.10.self_attn.v_proj.biases,
model.layers.10.self_attn.v_proj.scales,
model.layers.10.self_attn.v_proj.weight,
model.layers.11.input_layernorm.weight,
model.layers.11.mlp.down_proj.biases,
model.layers.11.mlp.down_proj.scales,
model.layers.11.mlp.down_proj.weight,
model.layers.11.mlp.gate_proj.biases,
model.layers.11.mlp.gate_proj.scales,
model.layers.11.mlp.gate_proj.weight,
model.layers.11.mlp.up_proj.biases,
model.layers.11.mlp.up_proj.scales,
model.layers.11.mlp.up_proj.weight,
model.layers.11.post_attention_layernorm.weight,
model.layers.11.self_attn.k_norm.weight,
model.layers.11.self_attn.k_proj.biases,
model.layers.11.self_attn.k_proj.scales,
model.layers.11.self_attn.k_proj.weight,
model.layers.11.self_attn.o_proj.biases,
model.layers.11.self_attn.o_proj.scales,
model.layers.11.self_attn.o_proj.weight,
model.layers.11.self_attn.q_norm.weight,
model.layers.11.self_attn.q_proj.biases,
model.layers.11.self_attn.q_proj.scales,
model.layers.11.self_attn.q_proj.weight,
model.layers.11.self_attn.v_proj.biases,
model.layers.11.self_attn.v_proj.scales,
model.layers.11.self_attn.v_proj.weight,
model.layers.12.input_layernorm.weight,
model.layers.12.mlp.down_proj.biases,
model.layers.12.mlp.down_proj.scales,
model.layers.12.mlp.down_proj.weight,
model.layers.12.mlp.gate_proj.biases,
model.layers.12.mlp.gate_proj.scales,
model.layers.12.mlp.gate_proj.weight,
model.layers.12.mlp.up_proj.biases,
model.layers.12.mlp.up_proj.scales,
model.layers.12.mlp.up_proj.weight,
model.layers.12.post_attention_layernorm.weight,
model.layers.12.self_attn.k_norm.weight,
model.layers.12.self_attn.k_proj.biases,
model.layers.12.self_attn.k_proj.scales,
model.layers.12.self_attn.k_proj.weight,
model.layers.12.self_attn.o_proj.biases,
model.layers.12.self_attn.o_proj.scales,
model.layers.12.self_attn.o_proj.weight,
model.layers.12.self_attn.q_norm.weight,
model.layers.12.self_attn.q_proj.biases,
model.layers.12.self_attn.q_proj.scales,
model.layers.12.self_attn.q_proj.weight,
model.layers.12.self_attn.v_proj.biases,
model.layers.12.self_attn.v_proj.scales,
model.layers.12.self_attn.v_proj.weight,
model.layers.13.input_layernorm.weight,
model.layers.13.mlp.down_proj.biases,
model.layers.13.mlp.down_proj.scales,
model.layers.13.mlp.down_proj.weight,
model.layers.13.mlp.gate_proj.biases,
model.layers.13.mlp.gate_proj.scales,
model.layers.13.mlp.gate_proj.weight,
model.layers.13.mlp.up_proj.biases,
model.layers.13.mlp.up_proj.scales,
model.layers.13.mlp.up_proj.weight,
model.layers.13.post_attention_layernorm.weight,
model.layers.13.self_attn.k_norm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.o_proj.biases,
model.layers.13.self_attn.o_proj.scales,
model.layers.13.self_attn.o_proj.weight,
model.layers.13.self_attn.q_norm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.input_layernorm.weight,
model.layers.14.mlp.down_proj.biases,
model.layers.14.mlp.down_proj.scales,
model.layers.14.mlp.down_proj.weight,
model.layers.14.mlp.gate_proj.biases,
model.layers.14.mlp.gate_proj.scales,
model.layers.14.mlp.gate_proj.weight,
model.layers.14.mlp.up_proj.biases,
model.layers.14.mlp.up_proj.scales,
model.layers.14.mlp.up_proj.weight,
model.layers.14.post_attention_layernorm.weight,
model.layers.14.self_attn.k_norm.weight,
model.layers.14.self_attn.k_proj.biases,
model.layers.14.self_attn.k_proj.scales,
model.layers.14.self_attn.k_proj.weight,
model.layers.14.self_attn.o_proj.biases,
model.layers.14.self_attn.o_proj.scales,
model.layers.14.self_attn.o_proj.weight,
model.layers.14.self_attn.q_norm.weight,
model.layers.14.self_attn.q_proj.biases,
model.layers.14.self_attn.q_proj.scales,
model.layers.14.self_attn.q_proj.weight,
model.layers.14.self_attn.v_proj.biases,
model.layers.14.self_attn.v_proj.scales,
model.layers.14.self_attn.v_proj.weight,
model.layers.15.input_layernorm.weight,
model.layers.15.mlp.down_proj.biases,
model.layers.15.mlp.down_proj.scales,
model.layers.15.mlp.down_proj.weight,
model.layers.15.mlp.gate_proj.biases,
model.layers.15.mlp.gate_proj.scales,
model.layers.15.mlp.gate_proj.weight,
model.layers.15.mlp.up_proj.biases,
model.layers.15.mlp.up_proj.scales,
model.layers.15.mlp.up_proj.weight,
model.layers.15.post_attention_layernorm.weight,
model.layers.15.self_attn.k_norm.weight,
model.layers.15.self_attn.k_proj.biases,
model.layers.15.self_attn.k_proj.scales,
model.layers.15.self_attn.k_proj.weight,
model.layers.15.self_attn.o_proj.biases,
model.layers.15.self_attn.o_proj.scales,
model.layers.15.self_attn.o_proj.weight,
model.layers.15.self_attn.q_norm.weight,
model.layers.15.self_attn.q_proj.biases,
model.layers.15.self_attn.q_proj.scales,
model.layers.15.self_attn.q_proj.weight,
model.layers.15.self_attn.v_proj.biases,
model.layers.15.self_attn.v_proj.scales,
model.layers.15.self_attn.v_proj.weight,
model.layers.16.input_layernorm.weight,
model.layers.16.mlp.down_proj.biases,
model.layers.16.mlp.down_proj.scales,
model.layers.16.mlp.down_proj.weight,
model.layers.16.mlp.gate_proj.biases,
model.layers.16.mlp.gate_proj.scales,
model.layers.16.mlp.gate_proj.weight,
model.layers.16.mlp.up_proj.biases,
model.layers.16.mlp.up_proj.scales,
model.layers.16.mlp.up_proj.weight,
model.layers.16.post_attention_layernorm.weight,
model.layers.16.self_attn.k_norm.weight,
model.layers.16.self_attn.k_proj.biases,
model.layers.16.self_attn.k_proj.scales,
model.layers.16.self_attn.k_proj.weight,
model.layers.16.self_attn.o_proj.biases,
model.layers.16.self_attn.o_proj.scales,
model.layers.16.self_attn.o_proj.weight,
model.layers.16.self_attn.q_norm.weight,
model.layers.16.self_attn.q_proj.biases,
model.layers.16.self_attn.q_proj.scales,
model.layers.16.self_attn.q_proj.weight,
model.layers.16.self_attn.v_proj.biases,
model.layers.16.self_attn.v_proj.scales,
model.layers.16.self_attn.v_proj.weight,
model.layers.17.input_layernorm.weight,
model.layers.17.mlp.down_proj.biases,
model.layers.17.mlp.down_proj.scales,
model.layers.17.mlp.down_proj.weight,
model.layers.17.mlp.gate_proj.biases,
model.layers.17.mlp.gate_proj.scales,
model.layers.17.mlp.gate_proj.weight,
model.layers.17.mlp.up_proj.biases,
model.layers.17.mlp.up_proj.scales,
model.layers.17.mlp.up_proj.weight,
model.layers.17.post_attention_layernorm.weight,
model.layers.17.self_attn.k_norm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.o_proj.biases,
model.layers.17.self_attn.o_proj.scales,
model.layers.17.self_attn.o_proj.weight,
model.layers.17.self_attn.q_norm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.input_layernorm.weight,
model.layers.18.mlp.down_proj.biases,
model.layers.18.mlp.down_proj.scales,
model.layers.18.mlp.down_proj.weight,
model.layers.18.mlp.gate_proj.biases,
model.layers.18.mlp.gate_proj.scales,
model.layers.18.mlp.gate_proj.weight,
model.layers.18.mlp.up_proj.biases,
model.layers.18.mlp.up_proj.scales,
model.layers.18.mlp.up_proj.weight,
model.layers.18.post_attention_layernorm.weight,
model.layers.18.self_attn.k_norm.weight,
model.layers.18.self_attn.k_proj.biases,
model.layers.18.self_attn.k_proj.scales,
model.layers.18.self_attn.k_proj.weight,
model.layers.18.self_attn.o_proj.biases,
model.layers.18.self_attn.o_proj.scales,
model.layers.18.self_attn.o_proj.weight,
model.layers.18.self_attn.q_norm.weight,
model.layers.18.self_attn.q_proj.biases,
model.layers.18.self_attn.q_proj.scales,
model.layers.18.self_attn.q_proj.weight,
model.layers.18.self_attn.v_proj.biases,
model.layers.18.self_attn.v_proj.scales,
model.layers.18.self_attn.v_proj.weight,
model.layers.19.input_layernorm.weight,
model.layers.19.mlp.down_proj.biases,
model.layers.19.mlp.down_proj.scales,
model.layers.19.mlp.down_proj.weight,
model.layers.19.mlp.gate_proj.biases,
model.layers.19.mlp.gate_proj.scales,
model.layers.19.mlp.gate_proj.weight,
model.layers.19.mlp.up_proj.biases,
model.layers.19.mlp.up_proj.scales,
model.layers.19.mlp.up_proj.weight,
model.layers.19.post_attention_layernorm.weight,
model.layers.19.self_attn.k_norm.weight,
model.layers.19.self_attn.k_proj.biases,
model.layers.19.self_attn.k_proj.scales,
model.layers.19.self_attn.k_proj.weight,
model.layers.19.self_attn.o_proj.biases,
model.layers.19.self_attn.o_proj.scales,
model.layers.19.self_attn.o_proj.weight,
model.layers.19.self_attn.q_norm.weight,
model.layers.19.self_attn.q_proj.biases,
model.layers.19.self_attn.q_proj.scales,
model.layers.19.self_attn.q_proj.weight,
model.layers.19.self_attn.v_proj.biases,
model.layers.19.self_attn.v_proj.scales,
model.layers.19.self_attn.v_proj.weight,
model.layers.2.input_layernorm.weight,
model.layers.2.mlp.down_proj.biases,
model.layers.2.mlp.down_proj.scales,
model.layers.2.mlp.down_proj.weight,
model.layers.2.mlp.gate_proj.biases,
model.layers.2.mlp.gate_proj.scales,
model.layers.2.mlp.gate_proj.weight,
model.layers.2.mlp.up_proj.biases,
model.layers.2.mlp.up_proj.scales,
model.layers.2.mlp.up_proj.weight,
model.layers.2.post_attention_layernorm.weight,
model.layers.2.self_attn.k_norm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.o_proj.biases,
model.layers.2.self_attn.o_proj.scales,
model.layers.2.self_attn.o_proj.weight,
model.layers.2.self_attn.q_norm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.input_layernorm.weight,
model.layers.20.mlp.down_proj.biases,
model.layers.20.mlp.down_proj.scales,
model.layers.20.mlp.down_proj.weight,
model.layers.20.mlp.gate_proj.biases,
model.layers.20.mlp.gate_proj.scales,
model.layers.20.mlp.gate_proj.weight,
model.layers.20.mlp.up_proj.biases,
model.layers.20.mlp.up_proj.scales,
model.layers.20.mlp.up_proj.weight,
model.layers.20.post_attention_layernorm.weight,
model.layers.20.self_attn.k_norm.weight,
model.layers.20.self_attn.k_proj.biases,
model.layers.20.self_attn.k_proj.scales,
model.layers.20.self_attn.k_proj.weight,
model.layers.20.self_attn.o_proj.biases,
model.layers.20.self_attn.o_proj.scales,
model.layers.20.self_attn.o_proj.weight,
model.layers.20.self_attn.q_norm.weight,
model.layers.20.self_attn.q_proj.biases,
model.layers.20.self_attn.q_proj.scales,
model.layers.20.self_attn.q_proj.weight,
model.layers.20.self_attn.v_proj.biases,
model.layers.20.self_attn.v_proj.scales,
model.layers.20.self_attn.v_proj.weight,
model.layers.21.input_layernorm.weight,
model.layers.21.mlp.down_proj.biases,
model.layers.21.mlp.down_proj.scales,
model.layers.21.mlp.down_proj.weight,
model.layers.21.mlp.gate_proj.biases,
model.layers.21.mlp.gate_proj.scales,
model.layers.21.mlp.gate_proj.weight,
model.layers.21.mlp.up_proj.biases,
model.layers.21.mlp.up_proj.scales,
model.layers.21.mlp.up_proj.weight,
model.layers.21.post_attention_layernorm.weight,
model.layers.21.self_attn.k_norm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.o_proj.biases,
model.layers.21.self_attn.o_proj.scales,
model.layers.21.self_attn.o_proj.weight,
model.layers.21.self_attn.q_norm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.input_layernorm.weight,
model.layers.22.mlp.down_proj.biases,
model.layers.22.mlp.down_proj.scales,
model.layers.22.mlp.down_proj.weight,
model.layers.22.mlp.gate_proj.biases,
model.layers.22.mlp.gate_proj.scales,
model.layers.22.mlp.gate_proj.weight,
model.layers.22.mlp.up_proj.biases,
model.layers.22.mlp.up_proj.scales,
model.layers.22.mlp.up_proj.weight,
model.layers.22.post_attention_layernorm.weight,
model.layers.22.self_attn.k_norm.weight,
model.layers.22.self_attn.k_proj.biases,
model.layers.22.self_attn.k_proj.scales,
model.layers.22.self_attn.k_proj.weight,
model.layers.22.self_attn.o_proj.biases,
model.layers.22.self_attn.o_proj.scales,
model.layers.22.self_attn.o_proj.weight,
model.layers.22.self_attn.q_norm.weight,
model.layers.22.self_attn.q_proj.biases,
model.layers.22.self_attn.q_proj.scales,
model.layers.22.self_attn.q_proj.weight,
model.layers.22.self_attn.v_proj.biases,
model.layers.22.self_attn.v_proj.scales,
model.layers.22.self_attn.v_proj.weight,
model.layers.23.input_layernorm.weight,
model.layers.23.mlp.down_proj.biases,
model.layers.23.mlp.down_proj.scales,
model.layers.23.mlp.down_proj.weight,
model.layers.23.mlp.gate_proj.biases,
model.layers.23.mlp.gate_proj.scales,
model.layers.23.mlp.gate_proj.weight,
model.layers.23.mlp.up_proj.biases,
model.layers.23.mlp.up_proj.scales,
model.layers.23.mlp.up_proj.weight,
model.layers.23.post_attention_layernorm.weight,
model.layers.23.self_attn.k_norm.weight,
model.layers.23.self_attn.k_proj.biases,
model.layers.23.self_attn.k_proj.scales,
model.layers.23.self_attn.k_proj.weight,
model.layers.23.self_attn.o_proj.biases,
model.layers.23.self_attn.o_proj.scales,
model.layers.23.self_attn.o_proj.weight,
model.layers.23.self_attn.q_norm.weight,
model.layers.23.self_attn.q_proj.biases,
model.layers.23.self_attn.q_proj.scales,
model.layers.23.self_attn.q_proj.weight,
model.layers.23.self_attn.v_proj.biases,
model.layers.23.self_attn.v_proj.scales,
model.layers.23.self_attn.v_proj.weight,
model.layers.24.input_layernorm.weight,
model.layers.24.mlp.down_proj.biases,
model.layers.24.mlp.down_proj.scales,
model.layers.24.mlp.down_proj.weight,
model.layers.24.mlp.gate_proj.biases,
model.layers.24.mlp.gate_proj.scales,
model.layers.24.mlp.gate_proj.weight,
model.layers.24.mlp.up_proj.biases,
model.layers.24.mlp.up_proj.scales,
model.layers.24.mlp.up_proj.weight,
model.layers.24.post_attention_layernorm.weight,
model.layers.24.self_attn.k_norm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.o_proj.biases,
model.layers.24.self_attn.o_proj.scales,
model.layers.24.self_attn.o_proj.weight,
model.layers.24.self_attn.q_norm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.input_layernorm.weight,
model.layers.25.mlp.down_proj.biases,
model.layers.25.mlp.down_proj.scales,
model.layers.25.mlp.down_proj.weight,
model.layers.25.mlp.gate_proj.biases,
model.layers.25.mlp.gate_proj.scales,
model.layers.25.mlp.gate_proj.weight,
model.layers.25.mlp.up_proj.biases,
model.layers.25.mlp.up_proj.scales,
model.layers.25.mlp.up_proj.weight,
model.layers.25.post_attention_layernorm.weight,
model.layers.25.self_attn.k_norm.weight,
model.layers.25.self_attn.k_proj.biases,
model.layers.25.self_attn.k_proj.scales,
model.layers.25.self_attn.k_proj.weight,
model.layers.25.self_attn.o_proj.biases,
model.layers.25.self_attn.o_proj.scales,
model.layers.25.self_attn.o_proj.weight,
model.layers.25.self_attn.q_norm.weight,
model.layers.25.self_attn.q_proj.biases,
model.layers.25.self_attn.q_proj.scales,
model.layers.25.self_attn.q_proj.weight,
model.layers.25.self_attn.v_proj.biases,
model.layers.25.self_attn.v_proj.scales,
model.layers.25.self_attn.v_proj.weight,
model.layers.26.input_layernorm.weight,
model.layers.26.mlp.down_proj.biases,
model.layers.26.mlp.down_proj.scales,
model.layers.26.mlp.down_proj.weight,
model.layers.26.mlp.gate_proj.biases,
model.layers.26.mlp.gate_proj.scales,
model.layers.26.mlp.gate_proj.weight,
model.layers.26.mlp.up_proj.biases,
model.layers.26.mlp.up_proj.scales,
model.layers.26.mlp.up_proj.weight,
model.layers.26.post_attention_layernorm.weight,
model.layers.26.self_attn.k_norm.weight,
model.layers.26.self_attn.k_proj.biases,
model.layers.26.self_attn.k_proj.scales,
model.layers.26.self_attn.k_proj.weight,
model.layers.26.self_attn.o_proj.biases,
model.layers.26.self_attn.o_proj.scales,
model.layers.26.self_attn.o_proj.weight,
model.layers.26.self_attn.q_norm.weight,
model.layers.26.self_attn.q_proj.biases,
model.layers.26.self_attn.q_proj.scales,
model.layers.26.self_attn.q_proj.weight,
model.layers.26.self_attn.v_proj.biases,
model.layers.26.self_attn.v_proj.scales,
model.layers.26.self_attn.v_proj.weight,
model.layers.27.input_layernorm.weight,
model.layers.27.mlp.down_proj.biases,
model.layers.27.mlp.down_proj.scales,
model.layers.27.mlp.down_proj.weight,
model.layers.27.mlp.gate_proj.biases,
model.layers.27.mlp.gate_proj.scales,
model.layers.27.mlp.gate_proj.weight,
model.layers.27.mlp.up_proj.biases,
model.layers.27.mlp.up_proj.scales,
model.layers.27.mlp.up_proj.weight,
model.layers.27.post_attention_layernorm.weight,
model.layers.27.self_attn.k_norm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.o_proj.biases,
model.layers.27.self_attn.o_proj.scales,
model.layers.27.self_attn.o_proj.weight,
model.layers.27.self_attn.q_norm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.input_layernorm.weight,
model.layers.28.mlp.down_proj.biases,
model.layers.28.mlp.down_proj.scales,
model.layers.28.mlp.down_proj.weight,
model.layers.28.mlp.gate_proj.biases,
model.layers.28.mlp.gate_proj.scales,
model.layers.28.mlp.gate_proj.weight,
model.layers.28.mlp.up_proj.biases,
model.layers.28.mlp.up_proj.scales,
model.layers.28.mlp.up_proj.weight,
model.layers.28.post_attention_layernorm.weight,
model.layers.28.self_attn.k_norm.weight,
model.layers.28.self_attn.k_proj.biases,
model.layers.28.self_attn.k_proj.scales,
model.layers.28.self_attn.k_proj.weight,
model.layers.28.self_attn.o_proj.biases,
model.layers.28.self_attn.o_proj.scales,
model.layers.28.self_attn.o_proj.weight,
model.layers.28.self_attn.q_norm.weight,
model.layers.28.self_attn.q_proj.biases,
model.layers.28.self_attn.q_proj.scales,
model.layers.28.self_attn.q_proj.weight,
model.layers.28.self_attn.v_proj.biases,
model.layers.28.self_attn.v_proj.scales,
model.layers.28.self_attn.v_proj.weight,
model.layers.29.input_layernorm.weight,
model.layers.29.mlp.down_proj.biases,
model.layers.29.mlp.down_proj.scales,
model.layers.29.mlp.down_proj.weight,
model.layers.29.mlp.gate_proj.biases,
model.layers.29.mlp.gate_proj.scales,
model.layers.29.mlp.gate_proj.weight,
model.layers.29.mlp.up_proj.biases,
model.layers.29.mlp.up_proj.scales,
model.layers.29.mlp.up_proj.weight,
model.layers.29.post_attention_layernorm.weight,
model.layers.29.self_attn.k_norm.weight,
model.layers.29.self_attn.k_proj.biases,
model.layers.29.self_attn.k_proj.scales,
model.layers.29.self_attn.k_proj.weight,
model.layers.29.self_attn.o_proj.biases,
model.layers.29.self_attn.o_proj.scales,
model.layers.29.self_attn.o_proj.weight,
model.layers.29.self_attn.q_norm.weight,
model.layers.29.self_attn.q_proj.biases,
model.layers.29.self_attn.q_proj.scales,
model.layers.29.self_attn.q_proj.weight,
model.layers.29.self_attn.v_proj.biases,
model.layers.29.self_attn.v_proj.scales,
model.layers.29.self_attn.v_proj.weight,
model.layers.3.input_layernorm.weight,
model.layers.3.mlp.down_proj.biases,
model.layers.3.mlp.down_proj.scales,
model.layers.3.mlp.down_proj.weight,
model.layers.3.mlp.gate_proj.biases,
model.layers.3.mlp.gate_proj.scales,
model.layers.3.mlp.gate_proj.weight,
model.layers.3.mlp.up_proj.biases,
model.layers.3.mlp.up_proj.scales,
model.layers.3.mlp.up_proj.weight,
model.layers.3.post_attention_layernorm.weight,
model.layers.3.self_attn.k_norm.weight,
model.layers.3.self_attn.k_proj.biases,
model.layers.3.self_attn.k_proj.scales,
model.layers.3.self_attn.k_proj.weight,
model.layers.3.self_attn.o_proj.biases,
model.layers.3.self_attn.o_proj.scales,
model.layers.3.self_attn.o_proj.weight,
model.layers.3.self_attn.q_norm.weight,
model.layers.3.self_attn.q_proj.biases,
model.layers.3.self_attn.q_proj.scales,
model.layers.3.self_attn.q_proj.weight,
model.layers.3.self_attn.v_proj.biases,
model.layers.3.self_attn.v_proj.scales,
model.layers.3.self_attn.v_proj.weight,
model.layers.30.input_layernorm.weight,
model.layers.30.mlp.down_proj.biases,
model.layers.30.mlp.down_proj.scales,
model.layers.30.mlp.down_proj.weight,
model.layers.30.mlp.gate_proj.biases,
model.layers.30.mlp.gate_proj.scales,
model.layers.30.mlp.gate_proj.weight,
model.layers.30.mlp.up_proj.biases,
model.layers.30.mlp.up_proj.scales,
model.layers.30.mlp.up_proj.weight,
model.layers.30.post_attention_layernorm.weight,
model.layers.30.self_attn.k_norm.weight,
model.layers.30.self_attn.k_proj.biases,
model.layers.30.self_attn.k_proj.scales,
model.layers.30.self_attn.k_proj.weight,
model.layers.30.self_attn.o_proj.biases,
model.layers.30.self_attn.o_proj.scales,
model.layers.30.self_attn.o_proj.weight,
model.layers.30.self_attn.q_norm.weight,
model.layers.30.self_attn.q_proj.biases,
model.layers.30.self_attn.q_proj.scales,
model.layers.30.self_attn.q_proj.weight,
model.layers.30.self_attn.v_proj.biases,
model.layers.30.self_attn.v_proj.scales,
model.layers.30.self_attn.v_proj.weight,
model.layers.31.input_layernorm.weight,
model.layers.31.mlp.down_proj.biases,
model.layers.31.mlp.down_proj.scales,
model.layers.31.mlp.down_proj.weight,
model.layers.31.mlp.gate_proj.biases,
model.layers.31.mlp.gate_proj.scales,
model.layers.31.mlp.gate_proj.weight,
model.layers.31.mlp.up_proj.biases,
model.layers.31.mlp.up_proj.scales,
model.layers.31.mlp.up_proj.weight,
model.layers.31.post_attention_layernorm.weight,
model.layers.31.self_attn.k_norm.weight,
model.layers.31.self_attn.k_proj.biases,
model.layers.31.self_attn.k_proj.scales,
model.layers.31.self_attn.k_proj.weight,
model.layers.31.self_attn.o_proj.biases,
model.layers.31.self_attn.o_proj.scales,
model.layers.31.self_attn.o_proj.weight,
model.layers.31.self_attn.q_norm.weight,
model.layers.31.self_attn.q_proj.biases,
model.layers.31.self_attn.q_proj.scales,
model.layers.31.self_attn.q_proj.weight,
model.layers.31.self_attn.v_proj.biases,
model.layers.31.self_attn.v_proj.scales,
model.layers.31.self_attn.v_proj.weight,
model.layers.32.input_layernorm.weight,
model.layers.32.mlp.down_proj.biases,
model.layers.32.mlp.down_proj.scales,
model.layers.32.mlp.down_proj.weight,
model.layers.32.mlp.gate_proj.biases,
model.layers.32.mlp.gate_proj.scales,
model.layers.32.mlp.gate_proj.weight,
model.layers.32.mlp.up_proj.biases,
model.layers.32.mlp.up_proj.scales,
model.layers.32.mlp.up_proj.weight,
model.layers.32.post_attention_layernorm.weight,
model.layers.32.self_attn.k_norm.weight,
model.layers.32.self_attn.k_proj.biases,
model.layers.32.self_attn.k_proj.scales,
model.layers.32.self_attn.k_proj.weight,
model.layers.32.self_attn.o_proj.biases,
model.layers.32.self_attn.o_proj.scales,
model.layers.32.self_attn.o_proj.weight,
model.layers.32.self_attn.q_norm.weight,
model.layers.32.self_attn.q_proj.biases,
model.layers.32.self_attn.q_proj.scales,
model.layers.32.self_attn.q_proj.weight,
model.layers.32.self_attn.v_proj.biases,
model.layers.32.self_attn.v_proj.scales,
model.layers.32.self_attn.v_proj.weight,
model.layers.33.input_layernorm.weight,
model.layers.33.mlp.down_proj.biases,
model.layers.33.mlp.down_proj.scales,
model.layers.33.mlp.down_proj.weight,
model.layers.33.mlp.gate_proj.biases,
model.layers.33.mlp.gate_proj.scales,
model.layers.33.mlp.gate_proj.weight,
model.layers.33.mlp.up_proj.biases,
model.layers.33.mlp.up_proj.scales,
model.layers.33.mlp.up_proj.weight,
model.layers.33.post_attention_layernorm.weight,
model.layers.33.self_attn.k_norm.weight,
model.layers.33.self_attn.k_proj.biases,
model.layers.33.self_attn.k_proj.scales,
model.layers.33.self_attn.k_proj.weight,
model.layers.33.self_attn.o_proj.biases,
model.layers.33.self_attn.o_proj.scales,
model.layers.33.self_attn.o_proj.weight,
model.layers.33.self_attn.q_norm.weight,
model.layers.33.self_attn.q_proj.biases,
model.layers.33.self_attn.q_proj.scales,
model.layers.33.self_attn.q_proj.weight,
model.layers.33.self_attn.v_proj.biases,
model.layers.33.self_attn.v_proj.scales,
model.layers.33.self_attn.v_proj.weight,
model.layers.34.input_layernorm.weight,
model.layers.34.mlp.down_proj.biases,
model.layers.34.mlp.down_proj.scales,
model.layers.34.mlp.down_proj.weight,
model.layers.34.mlp.gate_proj.biases,
model.layers.34.mlp.gate_proj.scales,
model.layers.34.mlp.gate_proj.weight,
model.layers.34.mlp.up_proj.biases,
model.layers.34.mlp.up_proj.scales,
model.layers.34.mlp.up_proj.weight,
model.layers.34.post_attention_layernorm.weight,
model.layers.34.self_attn.k_norm.weight,
model.layers.34.self_attn.k_proj.biases,
model.layers.34.self_attn.k_proj.scales,
model.layers.34.self_attn.k_proj.weight,
model.layers.34.self_attn.o_proj.biases,
model.layers.34.self_attn.o_proj.scales,
model.layers.34.self_attn.o_proj.weight,
model.layers.34.self_attn.q_norm.weight,
model.layers.34.self_attn.q_proj.biases,
model.layers.34.self_attn.q_proj.scales,
model.layers.34.self_attn.q_proj.weight,
model.layers.34.self_attn.v_proj.biases,
model.layers.34.self_attn.v_proj.scales,
model.layers.34.self_attn.v_proj.weight,
model.layers.35.input_layernorm.weight,
model.layers.35.mlp.down_proj.biases,
model.layers.35.mlp.down_proj.scales,
model.layers.35.mlp.down_proj.weight,
model.layers.35.mlp.gate_proj.biases,
model.layers.35.mlp.gate_proj.scales,
model.layers.35.mlp.gate_proj.weight,
model.layers.35.mlp.up_proj.biases,
model.layers.35.mlp.up_proj.scales,
model.layers.35.mlp.up_proj.weight,
model.layers.35.post_attention_layernorm.weight,
model.layers.35.self_attn.k_norm.weight,
model.layers.35.self_attn.k_proj.biases,
model.layers.35.self_attn.k_proj.scales,
model.layers.35.self_attn.k_proj.weight,
model.layers.35.self_attn.o_proj.biases,
model.layers.35.self_attn.o_proj.scales,
model.layers.35.self_attn.o_proj.weight,
model.layers.35.self_attn.q_norm.weight,
model.layers.35.self_attn.q_proj.biases,
model.layers.35.self_attn.q_proj.scales,
model.layers.35.self_attn.q_proj.weight,
model.layers.35.self_attn.v_proj.biases,
model.layers.35.self_attn.v_proj.scales,
model.layers.35.self_attn.v_proj.weight,
model.layers.4.input_layernorm.weight,
model.layers.4.mlp.down_proj.biases,
model.layers.4.mlp.down_proj.scales,
model.layers.4.mlp.down_proj.weight,
model.layers.4.mlp.gate_proj.biases,
model.layers.4.mlp.gate_proj.scales,
model.layers.4.mlp.gate_proj.weight,
model.layers.4.mlp.up_proj.biases,
model.layers.4.mlp.up_proj.scales,
model.layers.4.mlp.up_proj.weight,
model.layers.4.post_attention_layernorm.weight,
model.layers.4.self_attn.k_norm.weight,
model.layers.4.self_attn.k_proj.biases,
model.layers.4.self_attn.k_proj.scales,
model.layers.4.self_attn.k_proj.weight,
model.layers.4.self_attn.o_proj.biases,
model.layers.4.self_attn.o_proj.scales,
model.layers.4.self_attn.o_proj.weight,
model.layers.4.self_attn.q_norm.weight,
model.layers.4.self_attn.q_proj.biases,
model.layers.4.self_attn.q_proj.scales,
model.layers.4.self_attn.q_proj.weight,
model.layers.4.self_attn.v_proj.biases,
model.layers.4.self_attn.v_proj.scales,
model.layers.4.self_attn.v_proj.weight,
model.layers.5.input_layernorm.weight,
model.layers.5.mlp.down_proj.biases,
model.layers.5.mlp.down_proj.scales,
model.layers.5.mlp.down_proj.weight,
model.layers.5.mlp.gate_proj.biases,
model.layers.5.mlp.gate_proj.scales,
model.layers.5.mlp.gate_proj.weight,
model.layers.5.mlp.up_proj.biases,
model.layers.5.mlp.up_proj.scales,
model.layers.5.mlp.up_proj.weight,
model.layers.5.post_attention_layernorm.weight,
model.layers.5.self_attn.k_norm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.o_proj.biases,
model.layers.5.self_attn.o_proj.scales,
model.layers.5.self_attn.o_proj.weight,
model.layers.5.self_attn.q_norm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.input_layernorm.weight,
model.layers.6.mlp.down_proj.biases,
model.layers.6.mlp.down_proj.scales,
model.layers.6.mlp.down_proj.weight,
model.layers.6.mlp.gate_proj.biases,
model.layers.6.mlp.gate_proj.scales,
model.layers.6.mlp.gate_proj.weight,
model.layers.6.mlp.up_proj.biases,
model.layers.6.mlp.up_proj.scales,
model.layers.6.mlp.up_proj.weight,
model.layers.6.post_attention_layernorm.weight,
model.layers.6.self_attn.k_norm.weight,
model.layers.6.self_attn.k_proj.biases,
model.layers.6.self_attn.k_proj.scales,
model.layers.6.self_attn.k_proj.weight,
model.layers.6.self_attn.o_proj.biases,
model.layers.6.self_attn.o_proj.scales,
model.layers.6.self_attn.o_proj.weight,
model.layers.6.self_attn.q_norm.weight,
model.layers.6.self_attn.q_proj.biases,
model.layers.6.self_attn.q_proj.scales,
model.layers.6.self_attn.q_proj.weight,
model.layers.6.self_attn.v_proj.biases,
model.layers.6.self_attn.v_proj.scales,
model.layers.6.self_attn.v_proj.weight,
model.layers.7.input_layernorm.weight,
model.layers.7.mlp.down_proj.biases,
model.layers.7.mlp.down_proj.scales,
model.layers.7.mlp.down_proj.weight,
model.layers.7.mlp.gate_proj.biases,
model.layers.7.mlp.gate_proj.scales,
model.layers.7.mlp.gate_proj.weight,
model.layers.7.mlp.up_proj.biases,
model.layers.7.mlp.up_proj.scales,
model.layers.7.mlp.up_proj.weight,
model.layers.7.post_attention_layernorm.weight,
model.layers.7.self_attn.k_norm.weight,
model.layers.7.self_attn.k_proj.biases,
model.layers.7.self_attn.k_proj.scales,
model.layers.7.self_attn.k_proj.weight,
model.layers.7.self_attn.o_proj.biases,
model.layers.7.self_attn.o_proj.scales,
model.layers.7.self_attn.o_proj.weight,
model.layers.7.self_attn.q_norm.weight,
model.layers.7.self_attn.q_proj.biases,
model.layers.7.self_attn.q_proj.scales,
model.layers.7.self_attn.q_proj.weight,
model.layers.7.self_attn.v_proj.biases,
model.layers.7.self_attn.v_proj.scales,
model.layers.7.self_attn.v_proj.weight,
model.layers.8.input_layernorm.weight,
model.layers.8.mlp.down_proj.biases,
model.layers.8.mlp.down_proj.scales,
model.layers.8.mlp.down_proj.weight,
model.layers.8.mlp.gate_proj.biases,
model.layers.8.mlp.gate_proj.scales,
model.layers.8.mlp.gate_proj.weight,
model.layers.8.mlp.up_proj.biases,
model.layers.8.mlp.up_proj.scales,
model.layers.8.mlp.up_proj.weight,
model.layers.8.post_attention_layernorm.weight,
model.layers.8.self_attn.k_norm.weight,
model.layers.8.self_attn.k_proj.biases,
model.layers.8.self_attn.k_proj.scales,
model.layers.8.self_attn.k_proj.weight,
model.layers.8.self_attn.o_proj.biases,
model.layers.8.self_attn.o_proj.scales,
model.layers.8.self_attn.o_proj.weight,
model.layers.8.self_attn.q_norm.weight,
model.layers.8.self_attn.q_proj.biases,
model.layers.8.self_attn.q_proj.scales,
model.layers.8.self_attn.q_proj.weight,
model.layers.8.self_attn.v_proj.biases,
model.layers.8.self_attn.v_proj.scales,
model.layers.8.self_attn.v_proj.weight,
model.layers.9.input_layernorm.weight,
model.layers.9.mlp.down_proj.biases,
model.layers.9.mlp.down_proj.scales,
model.layers.9.mlp.down_proj.weight,
model.layers.9.mlp.gate_proj.biases,
model.layers.9.mlp.gate_proj.scales,
model.layers.9.mlp.gate_proj.weight,
model.layers.9.mlp.up_proj.biases,
model.layers.9.mlp.up_proj.scales,
model.layers.9.mlp.up_proj.weight,
model.layers.9.post_attention_layernorm.weight,
model.layers.9.self_attn.k_norm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.o_proj.biases,
model.layers.9.self_attn.o_proj.scales,
model.layers.9.self_attn.o_proj.weight,
model.layers.9.self_attn.q_norm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight,
model.norm.weight.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15198, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14183, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Received 904 parameters not in model: 
model.embed_tokens.biases,
model.embed_tokens.scales,
model.embed_tokens.weight,
model.layers.0.input_layernorm.weight,
model.layers.0.mlp.down_proj.biases,
model.layers.0.mlp.down_proj.scales,
model.layers.0.mlp.down_proj.weight,
model.layers.0.mlp.gate_proj.biases,
model.layers.0.mlp.gate_proj.scales,
model.layers.0.mlp.gate_proj.weight,
model.layers.0.mlp.up_proj.biases,
model.layers.0.mlp.up_proj.scales,
model.layers.0.mlp.up_proj.weight,
model.layers.0.post_attention_layernorm.weight,
model.layers.0.self_attn.k_norm.weight,
model.layers.0.self_attn.k_proj.biases,
model.layers.0.self_attn.k_proj.scales,
model.layers.0.self_attn.k_proj.weight,
model.layers.0.self_attn.o_proj.biases,
model.layers.0.self_attn.o_proj.scales,
model.layers.0.self_attn.o_proj.weight,
model.layers.0.self_attn.q_norm.weight,
model.layers.0.self_attn.q_proj.biases,
model.layers.0.self_attn.q_proj.scales,
model.layers.0.self_attn.q_proj.weight,
model.layers.0.self_attn.v_proj.biases,
model.layers.0.self_attn.v_proj.scales,
model.layers.0.self_attn.v_proj.weight,
model.layers.1.input_layernorm.weight,
model.layers.1.mlp.down_proj.biases,
model.layers.1.mlp.down_proj.scales,
model.layers.1.mlp.down_proj.weight,
model.layers.1.mlp.gate_proj.biases,
model.layers.1.mlp.gate_proj.scales,
model.layers.1.mlp.gate_proj.weight,
model.layers.1.mlp.up_proj.biases,
model.layers.1.mlp.up_proj.scales,
model.layers.1.mlp.up_proj.weight,
model.layers.1.post_attention_layernorm.weight,
model.layers.1.self_attn.k_norm.weight,
model.layers.1.self_attn.k_proj.biases,
model.layers.1.self_attn.k_proj.scales,
model.layers.1.self_attn.k_proj.weight,
model.layers.1.self_attn.o_proj.biases,
model.layers.1.self_attn.o_proj.scales,
model.layers.1.self_attn.o_proj.weight,
model.layers.1.self_attn.q_norm.weight,
model.layers.1.self_attn.q_proj.biases,
model.layers.1.self_attn.q_proj.scales,
model.layers.1.self_attn.q_proj.weight,
model.layers.1.self_attn.v_proj.biases,
model.layers.1.self_attn.v_proj.scales,
model.layers.1.self_attn.v_proj.weight,
model.layers.10.input_layernorm.weight,
model.layers.10.mlp.down_proj.biases,
model.layers.10.mlp.down_proj.scales,
model.layers.10.mlp.down_proj.weight,
model.layers.10.mlp.gate_proj.biases,
model.layers.10.mlp.gate_proj.scales,
model.layers.10.mlp.gate_proj.weight,
model.layers.10.mlp.up_proj.biases,
model.layers.10.mlp.up_proj.scales,
model.layers.10.mlp.up_proj.weight,
model.layers.10.post_attention_layernorm.weight,
model.layers.10.self_attn.k_norm.weight,
model.layers.10.self_attn.k_proj.biases,
model.layers.10.self_attn.k_proj.scales,
model.layers.10.self_attn.k_proj.weight,
model.layers.10.self_attn.o_proj.biases,
model.layers.10.self_attn.o_proj.scales,
model.layers.10.self_attn.o_proj.weight,
model.layers.10.self_attn.q_norm.weight,
model.layers.10.self_attn.q_proj.biases,
model.layers.10.self_attn.q_proj.scales,
model.layers.10.self_attn.q_proj.weight,
model.layers.10.self_attn.v_proj.biases,
model.layers.10.self_attn.v_proj.scales,
model.layers.10.self_attn.v_proj.weight,
model.layers.11.input_layernorm.weight,
model.layers.11.mlp.down_proj.biases,
model.layers.11.mlp.down_proj.scales,
model.layers.11.mlp.down_proj.weight,
model.layers.11.mlp.gate_proj.biases,
model.layers.11.mlp.gate_proj.scales,
model.layers.11.mlp.gate_proj.weight,
model.layers.11.mlp.up_proj.biases,
model.layers.11.mlp.up_proj.scales,
model.layers.11.mlp.up_proj.weight,
model.layers.11.post_attention_layernorm.weight,
model.layers.11.self_attn.k_norm.weight,
model.layers.11.self_attn.k_proj.biases,
model.layers.11.self_attn.k_proj.scales,
model.layers.11.self_attn.k_proj.weight,
model.layers.11.self_attn.o_proj.biases,
model.layers.11.self_attn.o_proj.scales,
model.layers.11.self_attn.o_proj.weight,
model.layers.11.self_attn.q_norm.weight,
model.layers.11.self_attn.q_proj.biases,
model.layers.11.self_attn.q_proj.scales,
model.layers.11.self_attn.q_proj.weight,
model.layers.11.self_attn.v_proj.biases,
model.layers.11.self_attn.v_proj.scales,
model.layers.11.self_attn.v_proj.weight,
model.layers.12.input_layernorm.weight,
model.layers.12.mlp.down_proj.biases,
model.layers.12.mlp.down_proj.scales,
model.layers.12.mlp.down_proj.weight,
model.layers.12.mlp.gate_proj.biases,
model.layers.12.mlp.gate_proj.scales,
model.layers.12.mlp.gate_proj.weight,
model.layers.12.mlp.up_proj.biases,
model.layers.12.mlp.up_proj.scales,
model.layers.12.mlp.up_proj.weight,
model.layers.12.post_attention_layernorm.weight,
model.layers.12.self_attn.k_norm.weight,
model.layers.12.self_attn.k_proj.biases,
model.layers.12.self_attn.k_proj.scales,
model.layers.12.self_attn.k_proj.weight,
model.layers.12.self_attn.o_proj.biases,
model.layers.12.self_attn.o_proj.scales,
model.layers.12.self_attn.o_proj.weight,
model.layers.12.self_attn.q_norm.weight,
model.layers.12.self_attn.q_proj.biases,
model.layers.12.self_attn.q_proj.scales,
model.layers.12.self_attn.q_proj.weight,
model.layers.12.self_attn.v_proj.biases,
model.layers.12.self_attn.v_proj.scales,
model.layers.12.self_attn.v_proj.weight,
model.layers.13.input_layernorm.weight,
model.layers.13.mlp.down_proj.biases,
model.layers.13.mlp.down_proj.scales,
model.layers.13.mlp.down_proj.weight,
model.layers.13.mlp.gate_proj.biases,
model.layers.13.mlp.gate_proj.scales,
model.layers.13.mlp.gate_proj.weight,
model.layers.13.mlp.up_proj.biases,
model.layers.13.mlp.up_proj.scales,
model.layers.13.mlp.up_proj.weight,
model.layers.13.post_attention_layernorm.weight,
model.layers.13.self_attn.k_norm.weight,
model.layers.13.self_attn.k_proj.biases,
model.layers.13.self_attn.k_proj.scales,
model.layers.13.self_attn.k_proj.weight,
model.layers.13.self_attn.o_proj.biases,
model.layers.13.self_attn.o_proj.scales,
model.layers.13.self_attn.o_proj.weight,
model.layers.13.self_attn.q_norm.weight,
model.layers.13.self_attn.q_proj.biases,
model.layers.13.self_attn.q_proj.scales,
model.layers.13.self_attn.q_proj.weight,
model.layers.13.self_attn.v_proj.biases,
model.layers.13.self_attn.v_proj.scales,
model.layers.13.self_attn.v_proj.weight,
model.layers.14.input_layernorm.weight,
model.layers.14.mlp.down_proj.biases,
model.layers.14.mlp.down_proj.scales,
model.layers.14.mlp.down_proj.weight,
model.layers.14.mlp.gate_proj.biases,
model.layers.14.mlp.gate_proj.scales,
model.layers.14.mlp.gate_proj.weight,
model.layers.14.mlp.up_proj.biases,
model.layers.14.mlp.up_proj.scales,
model.layers.14.mlp.up_proj.weight,
model.layers.14.post_attention_layernorm.weight,
model.layers.14.self_attn.k_norm.weight,
model.layers.14.self_attn.k_proj.biases,
model.layers.14.self_attn.k_proj.scales,
model.layers.14.self_attn.k_proj.weight,
model.layers.14.self_attn.o_proj.biases,
model.layers.14.self_attn.o_proj.scales,
model.layers.14.self_attn.o_proj.weight,
model.layers.14.self_attn.q_norm.weight,
model.layers.14.self_attn.q_proj.biases,
model.layers.14.self_attn.q_proj.scales,
model.layers.14.self_attn.q_proj.weight,
model.layers.14.self_attn.v_proj.biases,
model.layers.14.self_attn.v_proj.scales,
model.layers.14.self_attn.v_proj.weight,
model.layers.15.input_layernorm.weight,
model.layers.15.mlp.down_proj.biases,
model.layers.15.mlp.down_proj.scales,
model.layers.15.mlp.down_proj.weight,
model.layers.15.mlp.gate_proj.biases,
model.layers.15.mlp.gate_proj.scales,
model.layers.15.mlp.gate_proj.weight,
model.layers.15.mlp.up_proj.biases,
model.layers.15.mlp.up_proj.scales,
model.layers.15.mlp.up_proj.weight,
model.layers.15.post_attention_layernorm.weight,
model.layers.15.self_attn.k_norm.weight,
model.layers.15.self_attn.k_proj.biases,
model.layers.15.self_attn.k_proj.scales,
model.layers.15.self_attn.k_proj.weight,
model.layers.15.self_attn.o_proj.biases,
model.layers.15.self_attn.o_proj.scales,
model.layers.15.self_attn.o_proj.weight,
model.layers.15.self_attn.q_norm.weight,
model.layers.15.self_attn.q_proj.biases,
model.layers.15.self_attn.q_proj.scales,
model.layers.15.self_attn.q_proj.weight,
model.layers.15.self_attn.v_proj.biases,
model.layers.15.self_attn.v_proj.scales,
model.layers.15.self_attn.v_proj.weight,
model.layers.16.input_layernorm.weight,
model.layers.16.mlp.down_proj.biases,
model.layers.16.mlp.down_proj.scales,
model.layers.16.mlp.down_proj.weight,
model.layers.16.mlp.gate_proj.biases,
model.layers.16.mlp.gate_proj.scales,
model.layers.16.mlp.gate_proj.weight,
model.layers.16.mlp.up_proj.biases,
model.layers.16.mlp.up_proj.scales,
model.layers.16.mlp.up_proj.weight,
model.layers.16.post_attention_layernorm.weight,
model.layers.16.self_attn.k_norm.weight,
model.layers.16.self_attn.k_proj.biases,
model.layers.16.self_attn.k_proj.scales,
model.layers.16.self_attn.k_proj.weight,
model.layers.16.self_attn.o_proj.biases,
model.layers.16.self_attn.o_proj.scales,
model.layers.16.self_attn.o_proj.weight,
model.layers.16.self_attn.q_norm.weight,
model.layers.16.self_attn.q_proj.biases,
model.layers.16.self_attn.q_proj.scales,
model.layers.16.self_attn.q_proj.weight,
model.layers.16.self_attn.v_proj.biases,
model.layers.16.self_attn.v_proj.scales,
model.layers.16.self_attn.v_proj.weight,
model.layers.17.input_layernorm.weight,
model.layers.17.mlp.down_proj.biases,
model.layers.17.mlp.down_proj.scales,
model.layers.17.mlp.down_proj.weight,
model.layers.17.mlp.gate_proj.biases,
model.layers.17.mlp.gate_proj.scales,
model.layers.17.mlp.gate_proj.weight,
model.layers.17.mlp.up_proj.biases,
model.layers.17.mlp.up_proj.scales,
model.layers.17.mlp.up_proj.weight,
model.layers.17.post_attention_layernorm.weight,
model.layers.17.self_attn.k_norm.weight,
model.layers.17.self_attn.k_proj.biases,
model.layers.17.self_attn.k_proj.scales,
model.layers.17.self_attn.k_proj.weight,
model.layers.17.self_attn.o_proj.biases,
model.layers.17.self_attn.o_proj.scales,
model.layers.17.self_attn.o_proj.weight,
model.layers.17.self_attn.q_norm.weight,
model.layers.17.self_attn.q_proj.biases,
model.layers.17.self_attn.q_proj.scales,
model.layers.17.self_attn.q_proj.weight,
model.layers.17.self_attn.v_proj.biases,
model.layers.17.self_attn.v_proj.scales,
model.layers.17.self_attn.v_proj.weight,
model.layers.18.input_layernorm.weight,
model.layers.18.mlp.down_proj.biases,
model.layers.18.mlp.down_proj.scales,
model.layers.18.mlp.down_proj.weight,
model.layers.18.mlp.gate_proj.biases,
model.layers.18.mlp.gate_proj.scales,
model.layers.18.mlp.gate_proj.weight,
model.layers.18.mlp.up_proj.biases,
model.layers.18.mlp.up_proj.scales,
model.layers.18.mlp.up_proj.weight,
model.layers.18.post_attention_layernorm.weight,
model.layers.18.self_attn.k_norm.weight,
model.layers.18.self_attn.k_proj.biases,
model.layers.18.self_attn.k_proj.scales,
model.layers.18.self_attn.k_proj.weight,
model.layers.18.self_attn.o_proj.biases,
model.layers.18.self_attn.o_proj.scales,
model.layers.18.self_attn.o_proj.weight,
model.layers.18.self_attn.q_norm.weight,
model.layers.18.self_attn.q_proj.biases,
model.layers.18.self_attn.q_proj.scales,
model.layers.18.self_attn.q_proj.weight,
model.layers.18.self_attn.v_proj.biases,
model.layers.18.self_attn.v_proj.scales,
model.layers.18.self_attn.v_proj.weight,
model.layers.19.input_layernorm.weight,
model.layers.19.mlp.down_proj.biases,
model.layers.19.mlp.down_proj.scales,
model.layers.19.mlp.down_proj.weight,
model.layers.19.mlp.gate_proj.biases,
model.layers.19.mlp.gate_proj.scales,
model.layers.19.mlp.gate_proj.weight,
model.layers.19.mlp.up_proj.biases,
model.layers.19.mlp.up_proj.scales,
model.layers.19.mlp.up_proj.weight,
model.layers.19.post_attention_layernorm.weight,
model.layers.19.self_attn.k_norm.weight,
model.layers.19.self_attn.k_proj.biases,
model.layers.19.self_attn.k_proj.scales,
model.layers.19.self_attn.k_proj.weight,
model.layers.19.self_attn.o_proj.biases,
model.layers.19.self_attn.o_proj.scales,
model.layers.19.self_attn.o_proj.weight,
model.layers.19.self_attn.q_norm.weight,
model.layers.19.self_attn.q_proj.biases,
model.layers.19.self_attn.q_proj.scales,
model.layers.19.self_attn.q_proj.weight,
model.layers.19.self_attn.v_proj.biases,
model.layers.19.self_attn.v_proj.scales,
model.layers.19.self_attn.v_proj.weight,
model.layers.2.input_layernorm.weight,
model.layers.2.mlp.down_proj.biases,
model.layers.2.mlp.down_proj.scales,
model.layers.2.mlp.down_proj.weight,
model.layers.2.mlp.gate_proj.biases,
model.layers.2.mlp.gate_proj.scales,
model.layers.2.mlp.gate_proj.weight,
model.layers.2.mlp.up_proj.biases,
model.layers.2.mlp.up_proj.scales,
model.layers.2.mlp.up_proj.weight,
model.layers.2.post_attention_layernorm.weight,
model.layers.2.self_attn.k_norm.weight,
model.layers.2.self_attn.k_proj.biases,
model.layers.2.self_attn.k_proj.scales,
model.layers.2.self_attn.k_proj.weight,
model.layers.2.self_attn.o_proj.biases,
model.layers.2.self_attn.o_proj.scales,
model.layers.2.self_attn.o_proj.weight,
model.layers.2.self_attn.q_norm.weight,
model.layers.2.self_attn.q_proj.biases,
model.layers.2.self_attn.q_proj.scales,
model.layers.2.self_attn.q_proj.weight,
model.layers.2.self_attn.v_proj.biases,
model.layers.2.self_attn.v_proj.scales,
model.layers.2.self_attn.v_proj.weight,
model.layers.20.input_layernorm.weight,
model.layers.20.mlp.down_proj.biases,
model.layers.20.mlp.down_proj.scales,
model.layers.20.mlp.down_proj.weight,
model.layers.20.mlp.gate_proj.biases,
model.layers.20.mlp.gate_proj.scales,
model.layers.20.mlp.gate_proj.weight,
model.layers.20.mlp.up_proj.biases,
model.layers.20.mlp.up_proj.scales,
model.layers.20.mlp.up_proj.weight,
model.layers.20.post_attention_layernorm.weight,
model.layers.20.self_attn.k_norm.weight,
model.layers.20.self_attn.k_proj.biases,
model.layers.20.self_attn.k_proj.scales,
model.layers.20.self_attn.k_proj.weight,
model.layers.20.self_attn.o_proj.biases,
model.layers.20.self_attn.o_proj.scales,
model.layers.20.self_attn.o_proj.weight,
model.layers.20.self_attn.q_norm.weight,
model.layers.20.self_attn.q_proj.biases,
model.layers.20.self_attn.q_proj.scales,
model.layers.20.self_attn.q_proj.weight,
model.layers.20.self_attn.v_proj.biases,
model.layers.20.self_attn.v_proj.scales,
model.layers.20.self_attn.v_proj.weight,
model.layers.21.input_layernorm.weight,
model.layers.21.mlp.down_proj.biases,
model.layers.21.mlp.down_proj.scales,
model.layers.21.mlp.down_proj.weight,
model.layers.21.mlp.gate_proj.biases,
model.layers.21.mlp.gate_proj.scales,
model.layers.21.mlp.gate_proj.weight,
model.layers.21.mlp.up_proj.biases,
model.layers.21.mlp.up_proj.scales,
model.layers.21.mlp.up_proj.weight,
model.layers.21.post_attention_layernorm.weight,
model.layers.21.self_attn.k_norm.weight,
model.layers.21.self_attn.k_proj.biases,
model.layers.21.self_attn.k_proj.scales,
model.layers.21.self_attn.k_proj.weight,
model.layers.21.self_attn.o_proj.biases,
model.layers.21.self_attn.o_proj.scales,
model.layers.21.self_attn.o_proj.weight,
model.layers.21.self_attn.q_norm.weight,
model.layers.21.self_attn.q_proj.biases,
model.layers.21.self_attn.q_proj.scales,
model.layers.21.self_attn.q_proj.weight,
model.layers.21.self_attn.v_proj.biases,
model.layers.21.self_attn.v_proj.scales,
model.layers.21.self_attn.v_proj.weight,
model.layers.22.input_layernorm.weight,
model.layers.22.mlp.down_proj.biases,
model.layers.22.mlp.down_proj.scales,
model.layers.22.mlp.down_proj.weight,
model.layers.22.mlp.gate_proj.biases,
model.layers.22.mlp.gate_proj.scales,
model.layers.22.mlp.gate_proj.weight,
model.layers.22.mlp.up_proj.biases,
model.layers.22.mlp.up_proj.scales,
model.layers.22.mlp.up_proj.weight,
model.layers.22.post_attention_layernorm.weight,
model.layers.22.self_attn.k_norm.weight,
model.layers.22.self_attn.k_proj.biases,
model.layers.22.self_attn.k_proj.scales,
model.layers.22.self_attn.k_proj.weight,
model.layers.22.self_attn.o_proj.biases,
model.layers.22.self_attn.o_proj.scales,
model.layers.22.self_attn.o_proj.weight,
model.layers.22.self_attn.q_norm.weight,
model.layers.22.self_attn.q_proj.biases,
model.layers.22.self_attn.q_proj.scales,
model.layers.22.self_attn.q_proj.weight,
model.layers.22.self_attn.v_proj.biases,
model.layers.22.self_attn.v_proj.scales,
model.layers.22.self_attn.v_proj.weight,
model.layers.23.input_layernorm.weight,
model.layers.23.mlp.down_proj.biases,
model.layers.23.mlp.down_proj.scales,
model.layers.23.mlp.down_proj.weight,
model.layers.23.mlp.gate_proj.biases,
model.layers.23.mlp.gate_proj.scales,
model.layers.23.mlp.gate_proj.weight,
model.layers.23.mlp.up_proj.biases,
model.layers.23.mlp.up_proj.scales,
model.layers.23.mlp.up_proj.weight,
model.layers.23.post_attention_layernorm.weight,
model.layers.23.self_attn.k_norm.weight,
model.layers.23.self_attn.k_proj.biases,
model.layers.23.self_attn.k_proj.scales,
model.layers.23.self_attn.k_proj.weight,
model.layers.23.self_attn.o_proj.biases,
model.layers.23.self_attn.o_proj.scales,
model.layers.23.self_attn.o_proj.weight,
model.layers.23.self_attn.q_norm.weight,
model.layers.23.self_attn.q_proj.biases,
model.layers.23.self_attn.q_proj.scales,
model.layers.23.self_attn.q_proj.weight,
model.layers.23.self_attn.v_proj.biases,
model.layers.23.self_attn.v_proj.scales,
model.layers.23.self_attn.v_proj.weight,
model.layers.24.input_layernorm.weight,
model.layers.24.mlp.down_proj.biases,
model.layers.24.mlp.down_proj.scales,
model.layers.24.mlp.down_proj.weight,
model.layers.24.mlp.gate_proj.biases,
model.layers.24.mlp.gate_proj.scales,
model.layers.24.mlp.gate_proj.weight,
model.layers.24.mlp.up_proj.biases,
model.layers.24.mlp.up_proj.scales,
model.layers.24.mlp.up_proj.weight,
model.layers.24.post_attention_layernorm.weight,
model.layers.24.self_attn.k_norm.weight,
model.layers.24.self_attn.k_proj.biases,
model.layers.24.self_attn.k_proj.scales,
model.layers.24.self_attn.k_proj.weight,
model.layers.24.self_attn.o_proj.biases,
model.layers.24.self_attn.o_proj.scales,
model.layers.24.self_attn.o_proj.weight,
model.layers.24.self_attn.q_norm.weight,
model.layers.24.self_attn.q_proj.biases,
model.layers.24.self_attn.q_proj.scales,
model.layers.24.self_attn.q_proj.weight,
model.layers.24.self_attn.v_proj.biases,
model.layers.24.self_attn.v_proj.scales,
model.layers.24.self_attn.v_proj.weight,
model.layers.25.input_layernorm.weight,
model.layers.25.mlp.down_proj.biases,
model.layers.25.mlp.down_proj.scales,
model.layers.25.mlp.down_proj.weight,
model.layers.25.mlp.gate_proj.biases,
model.layers.25.mlp.gate_proj.scales,
model.layers.25.mlp.gate_proj.weight,
model.layers.25.mlp.up_proj.biases,
model.layers.25.mlp.up_proj.scales,
model.layers.25.mlp.up_proj.weight,
model.layers.25.post_attention_layernorm.weight,
model.layers.25.self_attn.k_norm.weight,
model.layers.25.self_attn.k_proj.biases,
model.layers.25.self_attn.k_proj.scales,
model.layers.25.self_attn.k_proj.weight,
model.layers.25.self_attn.o_proj.biases,
model.layers.25.self_attn.o_proj.scales,
model.layers.25.self_attn.o_proj.weight,
model.layers.25.self_attn.q_norm.weight,
model.layers.25.self_attn.q_proj.biases,
model.layers.25.self_attn.q_proj.scales,
model.layers.25.self_attn.q_proj.weight,
model.layers.25.self_attn.v_proj.biases,
model.layers.25.self_attn.v_proj.scales,
model.layers.25.self_attn.v_proj.weight,
model.layers.26.input_layernorm.weight,
model.layers.26.mlp.down_proj.biases,
model.layers.26.mlp.down_proj.scales,
model.layers.26.mlp.down_proj.weight,
model.layers.26.mlp.gate_proj.biases,
model.layers.26.mlp.gate_proj.scales,
model.layers.26.mlp.gate_proj.weight,
model.layers.26.mlp.up_proj.biases,
model.layers.26.mlp.up_proj.scales,
model.layers.26.mlp.up_proj.weight,
model.layers.26.post_attention_layernorm.weight,
model.layers.26.self_attn.k_norm.weight,
model.layers.26.self_attn.k_proj.biases,
model.layers.26.self_attn.k_proj.scales,
model.layers.26.self_attn.k_proj.weight,
model.layers.26.self_attn.o_proj.biases,
model.layers.26.self_attn.o_proj.scales,
model.layers.26.self_attn.o_proj.weight,
model.layers.26.self_attn.q_norm.weight,
model.layers.26.self_attn.q_proj.biases,
model.layers.26.self_attn.q_proj.scales,
model.layers.26.self_attn.q_proj.weight,
model.layers.26.self_attn.v_proj.biases,
model.layers.26.self_attn.v_proj.scales,
model.layers.26.self_attn.v_proj.weight,
model.layers.27.input_layernorm.weight,
model.layers.27.mlp.down_proj.biases,
model.layers.27.mlp.down_proj.scales,
model.layers.27.mlp.down_proj.weight,
model.layers.27.mlp.gate_proj.biases,
model.layers.27.mlp.gate_proj.scales,
model.layers.27.mlp.gate_proj.weight,
model.layers.27.mlp.up_proj.biases,
model.layers.27.mlp.up_proj.scales,
model.layers.27.mlp.up_proj.weight,
model.layers.27.post_attention_layernorm.weight,
model.layers.27.self_attn.k_norm.weight,
model.layers.27.self_attn.k_proj.biases,
model.layers.27.self_attn.k_proj.scales,
model.layers.27.self_attn.k_proj.weight,
model.layers.27.self_attn.o_proj.biases,
model.layers.27.self_attn.o_proj.scales,
model.layers.27.self_attn.o_proj.weight,
model.layers.27.self_attn.q_norm.weight,
model.layers.27.self_attn.q_proj.biases,
model.layers.27.self_attn.q_proj.scales,
model.layers.27.self_attn.q_proj.weight,
model.layers.27.self_attn.v_proj.biases,
model.layers.27.self_attn.v_proj.scales,
model.layers.27.self_attn.v_proj.weight,
model.layers.28.input_layernorm.weight,
model.layers.28.mlp.down_proj.biases,
model.layers.28.mlp.down_proj.scales,
model.layers.28.mlp.down_proj.weight,
model.layers.28.mlp.gate_proj.biases,
model.layers.28.mlp.gate_proj.scales,
model.layers.28.mlp.gate_proj.weight,
model.layers.28.mlp.up_proj.biases,
model.layers.28.mlp.up_proj.scales,
model.layers.28.mlp.up_proj.weight,
model.layers.28.post_attention_layernorm.weight,
model.layers.28.self_attn.k_norm.weight,
model.layers.28.self_attn.k_proj.biases,
model.layers.28.self_attn.k_proj.scales,
model.layers.28.self_attn.k_proj.weight,
model.layers.28.self_attn.o_proj.biases,
model.layers.28.self_attn.o_proj.scales,
model.layers.28.self_attn.o_proj.weight,
model.layers.28.self_attn.q_norm.weight,
model.layers.28.self_attn.q_proj.biases,
model.layers.28.self_attn.q_proj.scales,
model.layers.28.self_attn.q_proj.weight,
model.layers.28.self_attn.v_proj.biases,
model.layers.28.self_attn.v_proj.scales,
model.layers.28.self_attn.v_proj.weight,
model.layers.29.input_layernorm.weight,
model.layers.29.mlp.down_proj.biases,
model.layers.29.mlp.down_proj.scales,
model.layers.29.mlp.down_proj.weight,
model.layers.29.mlp.gate_proj.biases,
model.layers.29.mlp.gate_proj.scales,
model.layers.29.mlp.gate_proj.weight,
model.layers.29.mlp.up_proj.biases,
model.layers.29.mlp.up_proj.scales,
model.layers.29.mlp.up_proj.weight,
model.layers.29.post_attention_layernorm.weight,
model.layers.29.self_attn.k_norm.weight,
model.layers.29.self_attn.k_proj.biases,
model.layers.29.self_attn.k_proj.scales,
model.layers.29.self_attn.k_proj.weight,
model.layers.29.self_attn.o_proj.biases,
model.layers.29.self_attn.o_proj.scales,
model.layers.29.self_attn.o_proj.weight,
model.layers.29.self_attn.q_norm.weight,
model.layers.29.self_attn.q_proj.biases,
model.layers.29.self_attn.q_proj.scales,
model.layers.29.self_attn.q_proj.weight,
model.layers.29.self_attn.v_proj.biases,
model.layers.29.self_attn.v_proj.scales,
model.layers.29.self_attn.v_proj.weight,
model.layers.3.input_layernorm.weight,
model.layers.3.mlp.down_proj.biases,
model.layers.3.mlp.down_proj.scales,
model.layers.3.mlp.down_proj.weight,
model.layers.3.mlp.gate_proj.biases,
model.layers.3.mlp.gate_proj.scales,
model.layers.3.mlp.gate_proj.weight,
model.layers.3.mlp.up_proj.biases,
model.layers.3.mlp.up_proj.scales,
model.layers.3.mlp.up_proj.weight,
model.layers.3.post_attention_layernorm.weight,
model.layers.3.self_attn.k_norm.weight,
model.layers.3.self_attn.k_proj.biases,
model.layers.3.self_attn.k_proj.scales,
model.layers.3.self_attn.k_proj.weight,
model.layers.3.self_attn.o_proj.biases,
model.layers.3.self_attn.o_proj.scales,
model.layers.3.self_attn.o_proj.weight,
model.layers.3.self_attn.q_norm.weight,
model.layers.3.self_attn.q_proj.biases,
model.layers.3.self_attn.q_proj.scales,
model.layers.3.self_attn.q_proj.weight,
model.layers.3.self_attn.v_proj.biases,
model.layers.3.self_attn.v_proj.scales,
model.layers.3.self_attn.v_proj.weight,
model.layers.30.input_layernorm.weight,
model.layers.30.mlp.down_proj.biases,
model.layers.30.mlp.down_proj.scales,
model.layers.30.mlp.down_proj.weight,
model.layers.30.mlp.gate_proj.biases,
model.layers.30.mlp.gate_proj.scales,
model.layers.30.mlp.gate_proj.weight,
model.layers.30.mlp.up_proj.biases,
model.layers.30.mlp.up_proj.scales,
model.layers.30.mlp.up_proj.weight,
model.layers.30.post_attention_layernorm.weight,
model.layers.30.self_attn.k_norm.weight,
model.layers.30.self_attn.k_proj.biases,
model.layers.30.self_attn.k_proj.scales,
model.layers.30.self_attn.k_proj.weight,
model.layers.30.self_attn.o_proj.biases,
model.layers.30.self_attn.o_proj.scales,
model.layers.30.self_attn.o_proj.weight,
model.layers.30.self_attn.q_norm.weight,
model.layers.30.self_attn.q_proj.biases,
model.layers.30.self_attn.q_proj.scales,
model.layers.30.self_attn.q_proj.weight,
model.layers.30.self_attn.v_proj.biases,
model.layers.30.self_attn.v_proj.scales,
model.layers.30.self_attn.v_proj.weight,
model.layers.31.input_layernorm.weight,
model.layers.31.mlp.down_proj.biases,
model.layers.31.mlp.down_proj.scales,
model.layers.31.mlp.down_proj.weight,
model.layers.31.mlp.gate_proj.biases,
model.layers.31.mlp.gate_proj.scales,
model.layers.31.mlp.gate_proj.weight,
model.layers.31.mlp.up_proj.biases,
model.layers.31.mlp.up_proj.scales,
model.layers.31.mlp.up_proj.weight,
model.layers.31.post_attention_layernorm.weight,
model.layers.31.self_attn.k_norm.weight,
model.layers.31.self_attn.k_proj.biases,
model.layers.31.self_attn.k_proj.scales,
model.layers.31.self_attn.k_proj.weight,
model.layers.31.self_attn.o_proj.biases,
model.layers.31.self_attn.o_proj.scales,
model.layers.31.self_attn.o_proj.weight,
model.layers.31.self_attn.q_norm.weight,
model.layers.31.self_attn.q_proj.biases,
model.layers.31.self_attn.q_proj.scales,
model.layers.31.self_attn.q_proj.weight,
model.layers.31.self_attn.v_proj.biases,
model.layers.31.self_attn.v_proj.scales,
model.layers.31.self_attn.v_proj.weight,
model.layers.32.input_layernorm.weight,
model.layers.32.mlp.down_proj.biases,
model.layers.32.mlp.down_proj.scales,
model.layers.32.mlp.down_proj.weight,
model.layers.32.mlp.gate_proj.biases,
model.layers.32.mlp.gate_proj.scales,
model.layers.32.mlp.gate_proj.weight,
model.layers.32.mlp.up_proj.biases,
model.layers.32.mlp.up_proj.scales,
model.layers.32.mlp.up_proj.weight,
model.layers.32.post_attention_layernorm.weight,
model.layers.32.self_attn.k_norm.weight,
model.layers.32.self_attn.k_proj.biases,
model.layers.32.self_attn.k_proj.scales,
model.layers.32.self_attn.k_proj.weight,
model.layers.32.self_attn.o_proj.biases,
model.layers.32.self_attn.o_proj.scales,
model.layers.32.self_attn.o_proj.weight,
model.layers.32.self_attn.q_norm.weight,
model.layers.32.self_attn.q_proj.biases,
model.layers.32.self_attn.q_proj.scales,
model.layers.32.self_attn.q_proj.weight,
model.layers.32.self_attn.v_proj.biases,
model.layers.32.self_attn.v_proj.scales,
model.layers.32.self_attn.v_proj.weight,
model.layers.33.input_layernorm.weight,
model.layers.33.mlp.down_proj.biases,
model.layers.33.mlp.down_proj.scales,
model.layers.33.mlp.down_proj.weight,
model.layers.33.mlp.gate_proj.biases,
model.layers.33.mlp.gate_proj.scales,
model.layers.33.mlp.gate_proj.weight,
model.layers.33.mlp.up_proj.biases,
model.layers.33.mlp.up_proj.scales,
model.layers.33.mlp.up_proj.weight,
model.layers.33.post_attention_layernorm.weight,
model.layers.33.self_attn.k_norm.weight,
model.layers.33.self_attn.k_proj.biases,
model.layers.33.self_attn.k_proj.scales,
model.layers.33.self_attn.k_proj.weight,
model.layers.33.self_attn.o_proj.biases,
model.layers.33.self_attn.o_proj.scales,
model.layers.33.self_attn.o_proj.weight,
model.layers.33.self_attn.q_norm.weight,
model.layers.33.self_attn.q_proj.biases,
model.layers.33.self_attn.q_proj.scales,
model.layers.33.self_attn.q_proj.weight,
model.layers.33.self_attn.v_proj.biases,
model.layers.33.self_attn.v_proj.scales,
model.layers.33.self_attn.v_proj.weight,
model.layers.34.input_layernorm.weight,
model.layers.34.mlp.down_proj.biases,
model.layers.34.mlp.down_proj.scales,
model.layers.34.mlp.down_proj.weight,
model.layers.34.mlp.gate_proj.biases,
model.layers.34.mlp.gate_proj.scales,
model.layers.34.mlp.gate_proj.weight,
model.layers.34.mlp.up_proj.biases,
model.layers.34.mlp.up_proj.scales,
model.layers.34.mlp.up_proj.weight,
model.layers.34.post_attention_layernorm.weight,
model.layers.34.self_attn.k_norm.weight,
model.layers.34.self_attn.k_proj.biases,
model.layers.34.self_attn.k_proj.scales,
model.layers.34.self_attn.k_proj.weight,
model.layers.34.self_attn.o_proj.biases,
model.layers.34.self_attn.o_proj.scales,
model.layers.34.self_attn.o_proj.weight,
model.layers.34.self_attn.q_norm.weight,
model.layers.34.self_attn.q_proj.biases,
model.layers.34.self_attn.q_proj.scales,
model.layers.34.self_attn.q_proj.weight,
model.layers.34.self_attn.v_proj.biases,
model.layers.34.self_attn.v_proj.scales,
model.layers.34.self_attn.v_proj.weight,
model.layers.35.input_layernorm.weight,
model.layers.35.mlp.down_proj.biases,
model.layers.35.mlp.down_proj.scales,
model.layers.35.mlp.down_proj.weight,
model.layers.35.mlp.gate_proj.biases,
model.layers.35.mlp.gate_proj.scales,
model.layers.35.mlp.gate_proj.weight,
model.layers.35.mlp.up_proj.biases,
model.layers.35.mlp.up_proj.scales,
model.layers.35.mlp.up_proj.weight,
model.layers.35.post_attention_layernorm.weight,
model.layers.35.self_attn.k_norm.weight,
model.layers.35.self_attn.k_proj.biases,
model.layers.35.self_attn.k_proj.scales,
model.layers.35.self_attn.k_proj.weight,
model.layers.35.self_attn.o_proj.biases,
model.layers.35.self_attn.o_proj.scales,
model.layers.35.self_attn.o_proj.weight,
model.layers.35.self_attn.q_norm.weight,
model.layers.35.self_attn.q_proj.biases,
model.layers.35.self_attn.q_proj.scales,
model.layers.35.self_attn.q_proj.weight,
model.layers.35.self_attn.v_proj.biases,
model.layers.35.self_attn.v_proj.scales,
model.layers.35.self_attn.v_proj.weight,
model.layers.4.input_layernorm.weight,
model.layers.4.mlp.down_proj.biases,
model.layers.4.mlp.down_proj.scales,
model.layers.4.mlp.down_proj.weight,
model.layers.4.mlp.gate_proj.biases,
model.layers.4.mlp.gate_proj.scales,
model.layers.4.mlp.gate_proj.weight,
model.layers.4.mlp.up_proj.biases,
model.layers.4.mlp.up_proj.scales,
model.layers.4.mlp.up_proj.weight,
model.layers.4.post_attention_layernorm.weight,
model.layers.4.self_attn.k_norm.weight,
model.layers.4.self_attn.k_proj.biases,
model.layers.4.self_attn.k_proj.scales,
model.layers.4.self_attn.k_proj.weight,
model.layers.4.self_attn.o_proj.biases,
model.layers.4.self_attn.o_proj.scales,
model.layers.4.self_attn.o_proj.weight,
model.layers.4.self_attn.q_norm.weight,
model.layers.4.self_attn.q_proj.biases,
model.layers.4.self_attn.q_proj.scales,
model.layers.4.self_attn.q_proj.weight,
model.layers.4.self_attn.v_proj.biases,
model.layers.4.self_attn.v_proj.scales,
model.layers.4.self_attn.v_proj.weight,
model.layers.5.input_layernorm.weight,
model.layers.5.mlp.down_proj.biases,
model.layers.5.mlp.down_proj.scales,
model.layers.5.mlp.down_proj.weight,
model.layers.5.mlp.gate_proj.biases,
model.layers.5.mlp.gate_proj.scales,
model.layers.5.mlp.gate_proj.weight,
model.layers.5.mlp.up_proj.biases,
model.layers.5.mlp.up_proj.scales,
model.layers.5.mlp.up_proj.weight,
model.layers.5.post_attention_layernorm.weight,
model.layers.5.self_attn.k_norm.weight,
model.layers.5.self_attn.k_proj.biases,
model.layers.5.self_attn.k_proj.scales,
model.layers.5.self_attn.k_proj.weight,
model.layers.5.self_attn.o_proj.biases,
model.layers.5.self_attn.o_proj.scales,
model.layers.5.self_attn.o_proj.weight,
model.layers.5.self_attn.q_norm.weight,
model.layers.5.self_attn.q_proj.biases,
model.layers.5.self_attn.q_proj.scales,
model.layers.5.self_attn.q_proj.weight,
model.layers.5.self_attn.v_proj.biases,
model.layers.5.self_attn.v_proj.scales,
model.layers.5.self_attn.v_proj.weight,
model.layers.6.input_layernorm.weight,
model.layers.6.mlp.down_proj.biases,
model.layers.6.mlp.down_proj.scales,
model.layers.6.mlp.down_proj.weight,
model.layers.6.mlp.gate_proj.biases,
model.layers.6.mlp.gate_proj.scales,
model.layers.6.mlp.gate_proj.weight,
model.layers.6.mlp.up_proj.biases,
model.layers.6.mlp.up_proj.scales,
model.layers.6.mlp.up_proj.weight,
model.layers.6.post_attention_layernorm.weight,
model.layers.6.self_attn.k_norm.weight,
model.layers.6.self_attn.k_proj.biases,
model.layers.6.self_attn.k_proj.scales,
model.layers.6.self_attn.k_proj.weight,
model.layers.6.self_attn.o_proj.biases,
model.layers.6.self_attn.o_proj.scales,
model.layers.6.self_attn.o_proj.weight,
model.layers.6.self_attn.q_norm.weight,
model.layers.6.self_attn.q_proj.biases,
model.layers.6.self_attn.q_proj.scales,
model.layers.6.self_attn.q_proj.weight,
model.layers.6.self_attn.v_proj.biases,
model.layers.6.self_attn.v_proj.scales,
model.layers.6.self_attn.v_proj.weight,
model.layers.7.input_layernorm.weight,
model.layers.7.mlp.down_proj.biases,
model.layers.7.mlp.down_proj.scales,
model.layers.7.mlp.down_proj.weight,
model.layers.7.mlp.gate_proj.biases,
model.layers.7.mlp.gate_proj.scales,
model.layers.7.mlp.gate_proj.weight,
model.layers.7.mlp.up_proj.biases,
model.layers.7.mlp.up_proj.scales,
model.layers.7.mlp.up_proj.weight,
model.layers.7.post_attention_layernorm.weight,
model.layers.7.self_attn.k_norm.weight,
model.layers.7.self_attn.k_proj.biases,
model.layers.7.self_attn.k_proj.scales,
model.layers.7.self_attn.k_proj.weight,
model.layers.7.self_attn.o_proj.biases,
model.layers.7.self_attn.o_proj.scales,
model.layers.7.self_attn.o_proj.weight,
model.layers.7.self_attn.q_norm.weight,
model.layers.7.self_attn.q_proj.biases,
model.layers.7.self_attn.q_proj.scales,
model.layers.7.self_attn.q_proj.weight,
model.layers.7.self_attn.v_proj.biases,
model.layers.7.self_attn.v_proj.scales,
model.layers.7.self_attn.v_proj.weight,
model.layers.8.input_layernorm.weight,
model.layers.8.mlp.down_proj.biases,
model.layers.8.mlp.down_proj.scales,
model.layers.8.mlp.down_proj.weight,
model.layers.8.mlp.gate_proj.biases,
model.layers.8.mlp.gate_proj.scales,
model.layers.8.mlp.gate_proj.weight,
model.layers.8.mlp.up_proj.biases,
model.layers.8.mlp.up_proj.scales,
model.layers.8.mlp.up_proj.weight,
model.layers.8.post_attention_layernorm.weight,
model.layers.8.self_attn.k_norm.weight,
model.layers.8.self_attn.k_proj.biases,
model.layers.8.self_attn.k_proj.scales,
model.layers.8.self_attn.k_proj.weight,
model.layers.8.self_attn.o_proj.biases,
model.layers.8.self_attn.o_proj.scales,
model.layers.8.self_attn.o_proj.weight,
model.layers.8.self_attn.q_norm.weight,
model.layers.8.self_attn.q_proj.biases,
model.layers.8.self_attn.q_proj.scales,
model.layers.8.self_attn.q_proj.weight,
model.layers.8.self_attn.v_proj.biases,
model.layers.8.self_attn.v_proj.scales,
model.layers.8.self_attn.v_proj.weight,
model.layers.9.input_layernorm.weight,
model.layers.9.mlp.down_proj.biases,
model.layers.9.mlp.down_proj.scales,
model.layers.9.mlp.down_proj.weight,
model.layers.9.mlp.gate_proj.biases,
model.layers.9.mlp.gate_proj.scales,
model.layers.9.mlp.gate_proj.weight,
model.layers.9.mlp.up_proj.biases,
model.layers.9.mlp.up_proj.scales,
model.layers.9.mlp.up_proj.weight,
model.layers.9.post_attention_layernorm.weight,
model.layers.9.self_attn.k_norm.weight,
model.layers.9.self_attn.k_proj.biases,
model.layers.9.self_attn.k_proj.scales,
model.layers.9.self_attn.k_proj.weight,
model.layers.9.self_attn.o_proj.biases,
model.layers.9.self_attn.o_proj.scales,
model.layers.9.self_attn.o_proj.weight,
model.layers.9.self_attn.q_norm.weight,
model.layers.9.self_attn.q_proj.biases,
model.layers.9.self_attn.q_proj.scales,
model.layers.9.self_attn.q_proj.weight,
model.layers.9.self_attn.v_proj.biases,
model.layers.9.self_attn.v_proj.scales,
model.layers.9.self_attn.v_proj.weight,
model.norm.weight.

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
Fetching 9 files:   0%|          | 0/9 [00:00<?, ?it/s]
Fetching 9 files: 100%|##########| 9/9 [00:00<00:00, 3483.32it/s]
```

## Completed Runs with Observations

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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.4767209999845363
- *Peak memory at first token (GB):* 6.726575134
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.01261709
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2195
- *Prompt composition:* 2,195 = 403 text/template + 1,792 image tokens (82%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.69 GB vs 5.68 GB on disk)
- *Generation tokens:* 200
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 40.440120749990456
- *Peak memory at first token (GB):* 23.302106954
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.7; top_p 0.8; top_k 20; repetition_penalty 1.0
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: generation_config
- *Post-cleanup active memory (GB):* 0.00627636
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
  sample(s); mode snapshot

#### Complete output

```text
Title: Solitary swan on a calm river with moored boats
Description: A white swan glides across the calm waters of a river, framed by green foliage in the foreground. In the background, a row of moored motorboats and cruisers are docked beside a residential building with balconies, under a soft, overcast sky.
Keywords: swan, river, boats, mooring, motorboat, residential, building, balcony, trees, foliage, greenery, water, reflection, calm, wildlife, waterfowl, aquatic bird, architecture, canal, waterfront, marina, pier, waterway, leisure, nature, outdoor, serene, tranquil, peaceful, dusk, evening, overcast, cloudy, landscape, scene, view, perspective, foreground, background, composition, natural, wildlife, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal, bird, animal,
```

</details>

<a id="diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

<details>
<summary>mlx-community/Muse-Glimmer-30B-OptiQ-4bit — unusable — control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible</summary>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_special_token, role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Labelled fields not detected:* ["title", "description"]
- *Unexpected special tokens:* ["&lt;|message|&gt;"]
- *Role-boundary tokens in output:* ["&lt;|message|&gt;"]
- *Title word count:* 0
- *Keyword count:* 1
- *Token-cap degradation evidence:* ["missing_sections"]
- *Special tokens emitted (by token id):* ["&lt;|message|&gt;"]
- *Resolved model revision:* b4a74fa6001f1eca3b23eeeb702ffad2773a218f
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 9.097394791024271
- *Peak memory at first token (GB):* 25.462830536
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 1.0; top_p 0.95; top_k 64
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.004556016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4403
- *Prompt composition:* 4,403 = 347 text/template + 4,056 image tokens (92%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 23.85
- *Parameter count:* 30.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (23.86 GB vs 23.85 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 200001
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *Snapshot notes (neutral):* processor config missing from snapshot
  (preprocessor_config.json, processor_config.json)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

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

We need title 5-10 words. Concrete. Maybe "Mute Swan on River with Moored Boats and Apartments". Count words. Mute(1) Swan(2) on(3) River(4) with(5) Moored(6) Boats(7) and(8) Apartments(9). Good 9 words.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Capture date/time and GPS as authoritative facts but do not claim they are visible. So we can mention captured on 2026-09-12... at GPS...? Maybe include in description? It says combining relevant context with main visible subject, setting, action, lighting, and distinctive details. Context includes capture date/time and GPS. Should we include them in description? It says treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. So we can mention them as metadata in description? Possibly "Captured on 2026-09-12..." That is factual. Might be okay. Typically catalogue description includes date and location. Could incorporate. Must not claim they are visible. So we can say "Photographed on 12 September 2026 at 17:41 BST at 52.393850°N, 0.270830°E..."

We need factual description. Main visible subject: solitary white swan gliding on calm water. Boats moored: white boat named 'Samar II' with blue stripe, boat 'Magna Carta' with red cover, etc. Riverside residential buildings with brown facade, tiled roof, balconies. Foliage framing. Water reflection. Lighting appears overcast daylight.

Need 1-2 sentences. Let's craft one sentence: A mute swan swims on calm water in the foreground with moored motorboats and cruisers, including 'Samar II' and 'Magna Carta', along a riverbank with riverside apartments behind, framed by trees and foliage. The scene is captured on 2026-09-12 17:41:02 UTC+01:00 at 52.393850°N, 0.270830°E under overcast daylight with reflections on the water.

That's a bit long but okay. Maybe two sentences.

Make sure not claim date visible.

Keywords: 10-18 unique comma-separated keywords. Cover relevant context and visible details. Use British English spelling? Probably fine.

Keywords: mute swan, river, moored boats, motorboat, marina, riverbank, residential building, waterfront, water reflection, foliage, trees, vegetation, pier, mooring, waterfowl, aquatic bird, architecture

Need 10-18. Let's count: mute swan(1), river(2), moored boats(3), motorboat(4), marina(5), riverbank(6), residential building(7), waterfront(8), water reflection(9), foliage(10), trees(11), vegetation(12), pier(13), mooring(14), waterfowl(15), aquatic bird(16), architecture(17). Good.

Maybe include "Samar II", "Magna Carta"? Might be specific. Could include but maybe okay.

Ensure keywords comma-separated.

Return exactly these three sections and nothing else:

Title:
Description:
Keywords:

Let's produce.

Potential issues: title 5-10 words. Ensure concrete.

Description should be 
```

</details>

<a id="diagnostic-mlx-community-aya-vision-8b-4bit"></a>

<details>
<summary>mlx-community/aya-vision-8b-4bit — usable_with_caveats — control tokens visible</summary>

### mlx-community/aya-vision-8b-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type aya_vision)
- *Unexpected special tokens:* ["&lt;|START_RESPONSE|&gt;",
  "&lt;|END_RESPONSE|&gt;"]
- *Title word count:* 6
- *Keyword count:* 12
- *Resolved model revision:* 3e679b3e08f04d741c8965ebf0cb258295e8acf7
- *Processor class:* transformers.models.aya_vision.processing_aya_vision.AyaVisionProcessor
- *Tokenizer class:* transformers.models.cohere.tokenization_cohere.CohereTokenizer
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 2.069792583002709
- *Peak memory at first token (GB):* 6.460113054
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.009340216
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2090
- *Prompt composition:* 2,090 = 907 text/template + 1,183 image tokens (57%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.45
- *Parameter count:* 8.00B (name-estimate)
- *Quantization:* 4-bit, group 64
- *Load active memory vs checkpoint:* 1.00x (5.46 GB vs 5.45 GB on disk)
- *Generation tokens:* 96
- *Configured EOS token ID:* 255001
- *Configured EOS token:* &lt;|END_OF_TURN_TOKEN|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|START_RESPONSE|>Title: Solitary Swan Glides on Tranquil River
Description: A serene scene captures a graceful white swan effortlessly gliding across the calm waters of a river, framed by lush greenery. Leisure boats and cruisers are moored alongside riverside residential buildings, creating a picturesque setting under the soft lighting of the late afternoon.
Keywords: Swan, River, Greenery, Boats, Cruisers, Buildings, Lighting, Afternoon, Tranquil, Waterfowl, Architecture, Waterway<|END_RESPONSE|>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                         | Mechanical checks | Observations                 |
|-----------------------------------------------|-------------------|------------------------------|
| mlx-community/gemma-3n-E4B-it-4bit            | major concerns    | labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit              | major concerns    | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit  | major concerns    | cut off at token limit       |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16              | concerns detected | duplicate keywords           |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | duplicate keywords           |
| mlx-community/Molmo2-8B-4bit                  | concerns detected | duplicate keywords           |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                 | Runtime identity                                          | Performance                                           |
|-------------------------------------------------------|-----------------------------------------------------------|-------------------------------------------------------|
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed       | 2393 prompt / 151 generated; 29.7 tok/s; 23 GB peak   |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit      | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed    | 1634 prompt / 997 generated; 74.0 tok/s; 19 GB peak   |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed         | 590 prompt / 152 generated; 30.2 tok/s; 17 GB peak    |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed         | 595 prompt / 96 generated; 105 tok/s; 16 GB peak      |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed         | 595 prompt / 89 generated; 26.0 tok/s; 20 GB peak     |
| mlx-community/gemma-4-e4b-it-4bit                     | rev 475b9088d297; Gemma4Processor; stop completed         | 591 prompt / 73 generated; 122 tok/s; 5.9 GB peak     |
| mlx-community/GLM-4.6V-Flash-4bit                     | rev bd7b20686e8c; Glm46VProcessor; stop completed         | 6450 prompt / 140 generated; 75.2 tok/s; 8.7 GB peak  |
| mlx-community/GLM-4.6V-nvfp4                          | rev 2da6855d4e28; Glm46VMoEProcessor; stop completed      | 6450 prompt / 103 generated; 40.9 tok/s; 78 GB peak   |
| mlx-community/granite-4.0-3b-vision-4bit              | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed | 1379 prompt / 65 generated; 172 tok/s; 4.7 GB peak    |
| mlx-community/Idefics3-8B-Llama3-bf16                 | rev 8c2a30c48864; Idefics3Processor; stop completed       | 2612 prompt / 138 generated; 32.6 tok/s; 18 GB peak   |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed   | 2113 prompt / 77 generated; 34.9 tok/s; 17 GB peak    |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                 | rev 12c5ae493041; Lfm2VlProcessor; stop completed         | 2103 prompt / 80 generated; 205 tok/s; 4.0 GB peak    |
| mlx-community/MiniCPM-o-4_5-4bit                      | rev 592c09d85e7b; MiniCPMOProcessor; stop completed       | 391 prompt / 101 generated; 103 tok/s; 7.0 GB peak    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed       | 2926 prompt / 109 generated; 67.1 tok/s; 13 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed       | 2926 prompt / 187 generated; 64.1 tok/s; 13 GB peak   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed       | 2925 prompt / 102 generated; 190 tok/s; 7.8 GB peak   |
| mlx-community/North-Micro-Vision-Instruct-4bit        | rev 87466363e6c5; CohereCompassProcessor; stop completed  | 4083 prompt / 84 generated; 159 tok/s; 3.9 GB peak    |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit           | rev 5f31fcd089ce; Qwen3VLProcessor; stop completed        | 1289 prompt / 155 generated; 73.0 tok/s; 24 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed          | 1133 prompt / 107 generated; 36.6 tok/s; 9.3 GB peak  |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed        | 3116 prompt / 102 generated; 37.7 tok/s; 16 GB peak   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | rev c325e5ea14c2; Qwen3VLProcessor; stop completed        | 16549 prompt / 912 generated; 69.8 tok/s; 8.4 GB peak |
| mlx-community/Qwen3-VL-8B-Instruct-4bit               | rev defcdea7cc7a; Qwen3VLProcessor; stop completed        | 16547 prompt / 93 generated; 59.4 tok/s; 11 GB peak   |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed        | 16563 prompt / 97 generated; 74.4 tok/s; 25 GB peak   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed        | 16563 prompt / 86 generated; 88.2 tok/s; 11 GB peak   |
| mlx-community/Qwen3.8-27B-4bit                        | rev 3e6447f082e8; Qwen3VLProcessor; stop completed        | 16563 prompt / 124 generated; 14.4 tok/s; 21 GB peak  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | rev 844516024a1c; SmolVLMProcessor; stop completed        | 1426 prompt / 98 generated; 123 tok/s; 5.6 GB peak    |
| mlx-community/Step-3.7-Flash-oQ3e                     | rev 41d17ee00e16; Step3VLProcessor; stop completed        | 3491 prompt / 121 generated; 48.6 tok/s; 92 GB peak   |
| mlx-community/X-Reasoner-7B-8bit                      | rev 21732e74613b; Qwen2_5_VLProcessor; stop completed     | 16558 prompt / 136 generated; 57.2 tok/s; 14 GB peak  |

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

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-30B-A3B-4bit           | ed2ce3381528db1c5b70a2aad78a6390997e9250 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-4bit | 82f31be9840fa0d4c7e99257fe2e28b59a46df97 |
| mlx-community/Mage-VL-OptiQ-4bit                 | bde6c9c7146acff6af09e203245014f19306c5c5 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit            | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit     | 0555d34cb1ed80c0e61a5635194c70027b4c2ff3 |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit        | b4a74fa6001f1eca3b23eeeb702ffad2773a218f |
| mlx-community/aya-vision-8b-4bit                 | 3e679b3e08f04d741c8965ebf0cb258295e8acf7 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.0                                                                                                                                           |
| mlx-vlm source revision    | 45d6e125ab174cc279edea417f6be734870ff161                                                                                                        |
| mlx                        | 0.32.3.dev20260912+229f5b430                                                                                                                    |
| mlx source revision        | 229f5b430df7926743c5b6ac62068cae2ebc8978                                                                                                        |
| mlx-audio                  | 0.5.3                                                                                                                                           |
| transformers               | 5.17.0                                                                                                                                          |
| tokenizers                 | 0.23.2                                                                                                                                          |
| huggingface-hub            | 1.31.0                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (189,439,120 bytes, sha256=3e940a8eda4eb7bc22bd545f335ba774b504258eb8817cc56b86d156a8b574df) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,488,288 bytes, sha256=ab77edac8a1ec78b4f4a24648706d4703110d3e401b64e74761dabb81bd7df83)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
