# Crash: mlx-community/Llama-3.2-11B-Vision-Instruct-4bit

## Maintainer evidence

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

## Reproduction inputs

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
python -m mlx_vlm.generate --model mlx-community/Llama-3.2-11B-Vision-Instruct-4bit --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

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
Keywords:' --max-tokens 1000 --temperature 0.0 --revision 82f31be9840fa0d4c7e99257fe2e28b59a46df97 --trust-remote-code --seed 0 --prefill-step-size 2048
```

## Provenance and Environment

### Components

| Component       | Value                                                             |
|-----------------|-------------------------------------------------------------------|
| mlx-vlm         | 0.7.0                                                             |
| mlx             | 0.32.3.dev20260912+229f5b430                                      |
| transformers    | 5.17.0                                                            |
| tokenizers      | 0.23.2                                                            |
| huggingface-hub | 1.31.0                                                            |
| Pillow          | 12.3.0                                                            |
| Python Version  | 3.14.7                                                            |
| macOS Version   | 26.6.2                                                            |
| GPU/Chip        | Apple M5 Max                                                      |
| check_models    | 0.17.24; revision 09b5430fd4adb1ca5f371bf11544425753636488; clean |

### Full environment evidence

| Evidence | Link |
| --- | --- |
| Complete dependency and toolchain inventory | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
