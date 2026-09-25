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
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 49.4 MB

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 48    |
| Conclusive outcomes | 48    |
| Completed           | 46    |
| Crashed             | 2     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 2     |
| none                           | 38    |
| observation needs reproduction | 8     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| not assessed         | 2     |
| major concerns       | 13    |
| no concerns detected | 31    |
| concerns detected    | 2     |

Observation counts

| Observation                                                  | Count |
|--------------------------------------------------------------|-------|
| Response repeats the same text                               | 2     |
| Generation was stopped early after sustained repeated output | 4     |
| Unrecognised model control tokens remain visible             | 2     |
| Required labelled fields not detected                        | 8     |
| Response appears cut off at the token limit                  | 2     |
| Internal reasoning block appears incomplete                  | 2     |
| Conversation-role control tokens remain visible              | 1     |
| Repeated keyword entries                                     | 6     |

## Triage

| Model                                                                                                           | Execution | Mechanical checks | Maintainer status              | Observations                                                                                      |
|-----------------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|---------------------------------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)                               | crashed   | not assessed      | actionable_failure             | none                                                                                              |
| [mlx-community/Mage-VL-OptiQ-4bit](#diagnostic-mlx-community-mage-vl-optiq-4bit)                                | crashed   | not assessed      | actionable_failure             | none                                                                                              |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed | major concerns    | observation_needs_reproduction | repeated text; cut off at token limit; duplicate keywords                                         |
| [mlx-community/Qwen2-VL-7B-Instruct-4bit](#diagnostic-mlx-community-qwen2-vl-7b-instruct-4bit)                  | completed | major concerns    | observation_needs_reproduction | repeated text; stopped early: repeating; duplicate keywords                                       |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit) | completed | major concerns    | observation_needs_reproduction | stopped early: repeating; labelled fields not detected; incomplete thinking block                 |
| [mlx-community/Molmo2-8B-4bit](#diagnostic-mlx-community-molmo2-8b-4bit)                                        | completed | major concerns    | observation_needs_reproduction | stopped early: repeating; duplicate keywords                                                      |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed | major concerns    | observation_needs_reproduction | stopped early: repeating; duplicate keywords                                                      |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | completed | major concerns    | observation_needs_reproduction | control tokens visible; labelled fields not detected                                              |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit)              | completed | major concerns    | observation_needs_reproduction | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [mlx-community/MiniCPM-V-4.6-4bit](#diagnostic-mlx-community-minicpm-v-46-4bit)                                 | completed | major concerns    | observation_needs_reproduction | incomplete thinking block                                                                         |

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
- *Post-cleanup active memory (GB):* 0.002443844
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14490, in _run_model_generation
    model, processor, config = _load_model(params)
                               ~~~~~~~~~~~^^^^^^^^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13446, in _load_model
    model, processor = load(
                       ~~~~^
        path_or_hf_repo=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
        quantize_activations=params.quantize_activations,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 842, in _typed_mlx_vlm_load
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15608, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14505, in _run_model_generation
    raise _tag_exception_failure_phase(ValueError(error_details), "model_load") from load_err
ValueError: Model loading failed: Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
[23:54:09] INFO     Loading model weights and processor...
Fetching 14 files:   0%|          | 0/14 [00:00<?, ?it/s]
Fetching 14 files: 100%|██████████| 14/14 [00:00<00:00, 3749.22it/s]
ERROR:root:Model type internvl not supported. Error: No module named 'mlx_vlm.speculative.drafters.internvl'
[23:54:09] DEBUG    HF Cache Info for mlx-community/InternVL3_5-1B-4bit: size=1046.5 MB, files=16
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
- *Post-cleanup active memory (GB):* 0.003885684
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13780, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14412, in _generate_once
    return _generate_with_repetition_guard(
        model=prepared.model,
    ...<6 lines>...
        **prepared.generate_kwargs,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14316, in _generate_with_repetition_guard
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 15608, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14517, in _run_model_generation
    output, duration = _execute_prepared_generation(
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        params,
        ^^^^^^^
    ...<2 lines>...
        phase_timer=phase_timer,
        ^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 14433, in _execute_prepared_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 13789, in _run_generation_guarded
    raise _tag_exception_failure_phase(
        ValueError(msg), _generation_failure_phase(gen_known_err)
    ) from gen_known_err
ValueError: Model generation failed for mlx-community/Mage-VL-OptiQ-4bit: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1, 50, 76)]

```

</details>

#### Captured stdout/stderr

```text
=== STDERR ===
[23:55:28] INFO     Loading model weights and processor...
Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]
Fetching 10 files: 100%|██████████| 10/10 [00:00<00:00, 3143.45it/s]
[23:55:29] DEBUG    Legacy snapshot note for mlx-community/Mage-VL-OptiQ-4bit: processor config
                    missing from snapshot (preprocessor_config.json, processor_config.json).
[23:55:29] INFO     Rendering the prompt...
[23:55:29] INFO     Preparing image inputs and prefilling...
[23:55:30] ERROR    Generation error for mlx-community/Mage-VL-OptiQ-4bit
                    ValueError: cu_seqlens mismatch: total_patches=7600 calculated=3800 grid=[(1,
                    50, 76)]
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — unusable — repeated text; cut off at token limit; duplicate keywords</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Repeated fragment:* keyword: "adventure"
- *Title word count:* 9
- *Keyword count:* 343
- *Duplicate keywords:* ["adventure", "learning", "education", "training",
  "practice", "improvement", "progress", "success", "achievement",
  "accomplishment", "pride", "satisfaction", "happiness", "joy", "laughter",
  "smiles", "gratitude", "appreciation", "wonder", "awe", "amazement",
  "enthusiasm", "excitement", "exploration", "discovery"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 2.1173165419895668
- *Peak memory at first token (GB):* 14.979459316
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.003885684
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 308
- *Prompt composition:* 308 = 307 text/template + 1 image tokens (0%; exact,
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
Title: Two Sailors Navigate Calm Waters in a Forested Estuary

Description: On a sunny day in September 2026, two sailors, one in a Vortex catamaran and the other in a Laser dinghy, navigate their boats across the calm waters of an estuary, surrounded by dense green woodland and a blue sky with clouds. The Vortex catamaran, with its distinctive orange and grey sail, is sail number 1067, while the Laser dinghy, with its white sail bearing the GBR 188572 number, is a British vessel.

Keywords: Sailboat, Catamaran, Estuary, Forest, Sailing, Sailors, Vortex, Laser, GBR, 1067, 188572, Sail, Boat, Water, Trees, Clouds, Sky, Blue, Green, Woodland, Nature, Outdoor, Recreation, Sport, Activity, Adventure, Fun, Relaxation, Peaceful, Calm, Serene, Scenic, Breathtaking, Beautiful, Picturesque, Idyllic, Tranquil, Soothing, Uplifting, Inspiring, Motivational, Encouraging, Empowering, Confidence-Boosting, Self-Improvement, Personal-Growth, Development, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, Enthusiasm, Excitement, Adventure, Exploration, Discovery, Learning, Education, Training, Practice, Improvement, Progress, Success, Achievement, Accomplishment, Pride, Satisfaction, Happiness, Joy, Laughter, Smiles, Gratitude, Appreciation, Wonder, Awe, Amazement, En
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-7b-instruct-4bit"></a>

<details>
<summary>mlx-community/Qwen2-VL-7B-Instruct-4bit — unusable — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/Qwen2-VL-7B-Instruct-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Repeated fragment:* keyword: "trees"
- *Title word count:* 2
- *Keyword count:* 67
- *Duplicate keywords:* ["trees", "forest", "mast", "life jacket", "river",
  "estuary", "sky", "shoreline", "sail", "boat", "boating"]
- *Resolved model revision:* 1c638e970be36948e087e77d103b9147e1a9a1dd
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 39.707785375008825
- *Peak memory at first token (GB):* 9.283160132
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.007359262
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16560
- *Prompt composition:* 16,560 = 336 text/template + 16,224 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 4.67
- *Parameter count:* 7.00B (name-estimate)
- *Quantization:* 4-bit, group 64
- *Declared context length:* 32,768 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (4.68 GB vs 4.67 GB on disk)
- *Generation tokens:* 225
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Title: Sailing Adventure

Description: Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) across calm waters, with dense green woodland in the background.

Keywords: Sailing, Catamaran, Laser dinghy, Vortex, Sail number, Water, Trees, Forest, Woodland, Adventure, Outdoor, Action, Sailboat, Mast, Life jacket, River, Estuary, Sky, Clouds, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat, Boating, Estuary, Forest, Life jacket, Mast, River, Sky, Trees, Shoreline, Sail, Boat,
```

</details>

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-4bit"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit — unusable — stopped early: repeating; labelled fields not detected; incomplete thinking block</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repetition_abort, missing_requested_sections,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Resolved model revision:* 846ea5576854468f25af3767d769b0805b1b08b6
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.3287507079949137
- *Peak memory at first token (GB):* 18.798076473
- *Checkpoint-declared sampling (generation_config.json):* temperature 0.6;
  top_p 0.95; repetition_penalty 1.0
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: default; min_p: default; repetition_penalty:
  generation_config
- *Post-cleanup active memory (GB):* 0.000625182
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1639
- *Prompt composition:* 1,639 = 392 text/template + 1,247 image tokens (76%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 17.44
- *Parameter count:* 28.00B total, 3.00B active (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 131,072 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (17.45 GB vs 17.44 GB on disk)
- *Generation tokens:* 425
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;
- *Configured thinking start token:* &lt;think&gt;
- *Configured thinking end token:* &lt;/think&gt;
- *Configured thinking budget:* 800
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Alright, let's get this done. I need to create some metadata for this image, and it's my job to be precise.

First, I need to figure out a good title. "Sailboats on the Water" is too generic. "Two Sailors on Dinghies" is better, but I want something more specific. "Sailors on Catamaran and Laser Dinghy" - that's it! Concise and accurate.

Next, the description. I'll start with the basics: date, time, and location. Then I'll describe the scene. I'll mention the two boats, the sailors, and the setting. The hint about the boats is great. Let me work that in. Okay, I'm going to use the hint about the catamaran and laser. I'll refine that. "On a calm day, two sailors navigate a Vortex catamaran and a Laser dinghy on a lake in the UK." That's good, it's accurate, and it's a good length.

Finally, the keywords. I'll go through the hints and add in some keywords that are relevant. Let's see, "sailboat," "dinghy," "catamaran," "laser," "sailor," "boat," "water," "lake," "sky," "trees," "shoreline," "recreation," "sport," "vessel," "wind," "man," "life jacket," "hull," "rigging," "sail number," "1067," "GBR 188572," "green," "calm," "daytime," "September," "UK," "England," "solent," "estuary," "marina," "hull number," "hull number," "hull number," "hull number," "hull number," "hull number," "hull number," "hull number," "hull
```

</details>

<a id="diagnostic-mlx-community-molmo2-8b-4bit"></a>

<details>
<summary>mlx-community/Molmo2-8B-4bit — unusable — stopped early: repeating; duplicate keywords</summary>

### mlx-community/Molmo2-8B-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type molmo2)
- *Title word count:* 8
- *Keyword count:* 52
- *Duplicate keywords:* ["blue stripes", "white boat", "blue canopy", "white
  hull"]
- *Resolved model revision:* 4fcbe926577641af12aa5086cf5c292db9dd9730
- *Processor class:* mlx_vlm.models.molmo2.processing.Molmo2Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.3944302500167396
- *Peak memory at first token (GB):* 8.183413476
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.005540484
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1526
- *Prompt composition:* 1,526 = 324 text/template (tokenizer-exact) + 1,202
  non-text (79%, image/audio expansion)
- *Checkpoint weights (GB):* 6.44
- *Parameter count:* 8.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 36,864 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (6.45 GB vs 6.44 GB on disk)
- *Generation tokens:* 225
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Title: Catamaran and Laser Dinghy Sail on Calm River

Description: Two sailors navigate small dinghies across a tranquil river, with a Vortex catamaran on the left and a Laser dinghy on the right. The scene features calm waters, dense green woodland, and a clear blue sky with scattered clouds.

Keywords: Boat, Boating, Catamaran, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water, Sail number, Vortex, Calm waters, Dense woodland, Clear sky, Scattered clouds, Small dinghies, Two sailors, White hulls, Blue life jackets, White sails, Orange stripes, White text, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull, Blue stripes, White boat, Blue canopy, White hull,
```

</details>

<a id="diagnostic-mlx-community-x-reasoner-7b-8bit"></a>

<details>
<summary>mlx-community/X-Reasoner-7B-8bit — unusable — stopped early: repeating; duplicate keywords</summary>

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Title word count:* 8
- *Keyword count:* 39
- *Duplicate keywords:* ["blue and white forest", "blue and white sky", "blue
  and white water", "blue and white sail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 14.266369332995964
- *Peak memory at first token (GB):* 14.286107454
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.0109474
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16560
- *Prompt composition:* 16,560 = 336 text/template + 16,224 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 9.45
- *Parameter count:* 7.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 128,000 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (9.46 GB vs 9.45 GB on disk)
- *Generation tokens:* 250
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
Title:
Sailing Catamaran and Laser Dinghy on Calm Waters

Description:
Two sailors navigate a Vortex catamaran (sail number 1067) and a Laser dinghy (sail number GBR 188572) across tranquil waters, set against a backdrop of dense green woodland under a partly cloudy sky.

Keywords: Sailing, Catamaran, Laser dinghy, Vortex, Sail number 1067, Sail number GBR 188572, Calm waters, Distant shoreline, Forest, Cloudy sky, Life jackets, Outdoor recreation, River, Shoreline, Trees, Water, Man, Mast, Sunburst logo, Blue and white sails, Orange and black sail, Blue and white catamaran, White Laser dinghy, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail, Blue and white forest, Blue and white sky, Blue and white water, Blue and white sail,
```

</details>

<a id="diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit"></a>

<details>
<summary>mlx-community/llm-jp-4-vl-9b-mlx-4bit — unusable — control tokens visible; labelled fields not detected</summary>

### mlx-community/llm-jp-4-vl-9b-mlx-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Unexpected special tokens:* ["&lt;|channel|&gt;", "&lt;|message|&gt;"]
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Processor class:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.5211921670124866
- *Peak memory at first token (GB):* 6.729361982
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.015387538
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2197
- *Prompt composition:* 2,197 = 405 text/template + 1,792 image tokens (82%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.70 GB vs 5.68 GB on disk)
- *Generation tokens:* 16
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows two small sailboats racing on a river.
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
- *Resolved model revision:* 98377360cbc84f982e90336f956b08adb46cad88
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 9.610304666974116
- *Peak memory at first token (GB):* 25.464306662
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 1.0; top_p 0.95; top_k 64
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.005950222
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4409
- *Prompt composition:* 4,409 = 353 text/template + 4,056 image tokens (92%;
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
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

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

Need title 5-10 words. Let's craft: "Two Sailors on Vortex Catamaran and Laser Dinghy" That's 8 words? Two(1) Sailors2 on3 Vortex4 Catamaran5 and6 Laser7 Dinghy8. Good.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Include capture date/time? It says treat capture date/time as authoritative facts, but do not claim they are visible. So can mention captured on 2026-09-19 17:12:46 UTC+01:00? Might be okay to include. Probably include date.

We need combine relevant context with main visible subject, setting, action, lighting, distinctive details.

Image shows two small sailboats on water. Left is Vortex with sail number 1067, grey sail with orange stripes, hull orange/white. Sailor wearing blue top and red life jacket standing. Right is Laser with sail number 188572 GBR, white sail with blue corners, red sunburst logo. Sailor sitting wearing black jacket and life jacket. Background dense green trees, shoreline, partly cloudy sky. Another boat in background: white boat with blue cover and mast. Also a small white boat with blue trim mid background, buoy.

Lighting is daylight, overcast partly.

Make description 1-2 sentences.

Maybe: On 2026-09-19 at 17:12:46 UTC+01:00, a Vortex catamaran sail number 1067 and a Laser dinghy sail number GBR 188572 sail on calm water with wooded shoreline and partly cloudy sky behind them; the Vortex sailor stands at the helm in a life jacket while the Laser sailor sits. That's one sentence maybe long.

Need British English. Use "behaviour"? Okay.

Keywords: 10-18 unique comma-separated. Use hints but correct. Boat, Boating, Catamaran, Clouds, Dinghy, Estuary, Forest, Laser dinghy, Life jacket, Man, Mast, Outdoor recreation, River, Sailboat, Sailing, Sailor, Shoreline, Sky, Trees, Water

We need unique. Possibly include Vortex catamaran, sail number, etc. But keywords list likely generic.

Make 15 keywords.

Let's produce:

Title:
Two Sailors on Vortex Catamaran and Laser Dinghy

Description:
Captured on 2026-09-19 17:12:46 UTC+01:00, a Vortex catamaran numbered 1067 and a Laser dinghy numbered GBR 188572 are under sail on calm water with a wooded shoreline and partly cloudy sky beyond; the Vortex sailor stands wearing a life jacket at the helm while the Laser sailor sits in the cockpit, with additional moored vessels in the background.

Keywords: sailboat, sailing, dinghy, catamaran, Vortex, Laser dinghy, sailor, life jacket, water, shoreline, trees, forest, sky, clouds, outdoor recreation

That's 15.

Ensure format exactly:

Title:
Description:
Keywords:

Return exactly these three sections and nothing else.

Make sure Title is 5-10 words. Ours is 8.

Description is 1-2 sentences. It's one sentence? It's maybe two sentences due to semicolon? Might be okay. Could make one sentence. Let's keep as one sentence maybe with
```

</details>

<a id="diagnostic-mlx-community-minicpm-v-46-4bit"></a>

<details>
<summary>mlx-community/MiniCPM-V-4.6-4bit — unusable — incomplete thinking block</summary>

### mlx-community/MiniCPM-V-4.6-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 5
- *Keyword count:* 12
- *Resolved model revision:* 86cd463d33a946e4481b77e3c10fc63121b60a19
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.9244277920224704
- *Peak memory at first token (GB):* 3.257943688
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.7; top_p 1.0; top_k 0; repetition_penalty 1.0
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: generation_config
- *Post-cleanup active memory (GB):* 0.0043936
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 934
- *Prompt composition:* 934 = 345 text/template (tokenizer-exact) + 589
  non-text (63%, image/audio expansion)
- *Checkpoint weights (GB):* 2.16
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (2.16 GB vs 2.16 GB on disk)
- *Generation tokens:* 84
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); mode snapshot

#### Complete output

```text
<think>
Here's the response following the instructions:

Title: Sailing vessels on calm waters

Description: Two sailors are sailing small dinghies, a Vortex catamaran and a Laser dinghy, across calm water with green forest in the background.

Keywords: boats, sailing, dinghy, catamaran, water, nature, forest, sail, person, life jacket, outdoor, recreation
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                     | Mechanical checks | Observations                 |
|-------------------------------------------|-------------------|------------------------------|
| mlx-community/FastVLM-0.5B-bf16           | major concerns    | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit        | major concerns    | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns    | labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit          | major concerns    | labelled fields not detected |
| mlx-community/SmolVLM-256M-Instruct-4bit  | major concerns    | labelled fields not detected |
| mlx-community/gemma-4-12B-it-4bit         | concerns detected | duplicate keywords           |
| mlx-community/Idefics3-8B-Llama3-bf16     | concerns detected | duplicate keywords           |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                 | Runtime identity                                           | Performance                                           |
|-------------------------------------------------------|------------------------------------------------------------|-------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | rev ed71acdae079; Lfm2VlProcessor; stop completed          | 2119 prompt / 141 generated; 480 tok/s; 1.9 GB peak   |
| mlx-community/aya-vision-8b-4bit                      | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed | 2089 prompt / 112 generated; 98.5 tok/s; 6.5 GB peak  |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed        | 2394 prompt / 123 generated; 30.0 tok/s; 23 GB peak   |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | rev ded389e478f8; DiffusionGemma4Processor; stop completed | 592 prompt / 85 generated; 46.5 tok/s; 28 GB peak     |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed          | 591 prompt / 175 generated; 30.4 tok/s; 17 GB peak    |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed          | 596 prompt / 130 generated; 104 tok/s; 16 GB peak     |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed          | 596 prompt / 89 generated; 26.3 tok/s; 20 GB peak     |
| mlx-community/gemma-4-e4b-it-4bit                     | rev 475b9088d297; Gemma4Processor; stop completed          | 592 prompt / 94 generated; 121 tok/s; 5.9 GB peak     |
| mlx-community/GLM-4.6V-Flash-4bit                     | rev bd7b20686e8c; Glm46VProcessor; stop completed          | 6454 prompt / 139 generated; 73.1 tok/s; 8.7 GB peak  |
| mlx-community/GLM-4.6V-nvfp4                          | rev 2da6855d4e28; Glm46VMoEProcessor; stop completed       | 6454 prompt / 142 generated; 39.5 tok/s; 78 GB peak   |
| mlx-community/granite-4.0-3b-vision-4bit              | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed  | 1383 prompt / 99 generated; 166 tok/s; 4.7 GB peak    |
| mlx-community/InternVL3-14B-4bit                      | rev 26328eaab82c; InternVLChatProcessor; stop completed    | 2115 prompt / 116 generated; 55.4 tok/s; 10 GB peak   |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed    | 2115 prompt / 103 generated; 36.8 tok/s; 17 GB peak   |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit          | rev e5abbe34cbfa; KimiVLProcessor; stop completed          | 1331 prompt / 729 generated; 61.5 tok/s; 20 GB peak   |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                 | rev 7886c0b4a4b5; Lfm2VlProcessor; stop completed          | 2111 prompt / 110 generated; 209 tok/s; 4.0 GB peak   |
| mlx-community/MiniCPM-o-4_5-4bit                      | rev 592c09d85e7b; MiniCPMOProcessor; stop completed        | 393 prompt / 114 generated; 104 tok/s; 7.0 GB peak    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed        | 2927 prompt / 160 generated; 66.3 tok/s; 13 GB peak   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed        | 2926 prompt / 152 generated; 186 tok/s; 7.8 GB peak   |
| mlx-community/North-Micro-Vision-Instruct-4bit        | rev 87466363e6c5; CohereCompassProcessor; stop completed   | 4085 prompt / 112 generated; 147 tok/s; 3.9 GB peak   |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit           | rev 4620fdbbd1e7; Qwen3VLProcessor; stop completed         | 1291 prompt / 151 generated; 73.1 tok/s; 24 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16            | rev d8da684308c2; Phi3VProcessor; stop completed           | 1141 prompt / 136 generated; 55.7 tok/s; 9.3 GB peak  |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed         | 3117 prompt / 114 generated; 39.5 tok/s; 16 GB peak   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | rev c325e5ea14c2; Qwen3VLProcessor; stop completed         | 16551 prompt / 918 generated; 78.2 tok/s; 8.4 GB peak |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit          | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed         | 16549 prompt / 128 generated; 74.8 tok/s; 23 GB peak  |
| mlx-community/Qwen3-VL-32B-Instruct-4bit              | rev 6e5644d3ea4b; Qwen3VLProcessor; stop completed         | 16549 prompt / 181 generated; 19.8 tok/s; 26 GB peak  |
| mlx-community/Qwen3-VL-8B-Instruct-4bit               | rev defcdea7cc7a; Qwen3VLProcessor; stop completed         | 16549 prompt / 104 generated; 67.4 tok/s; 11 GB peak  |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed         | 16565 prompt / 111 generated; 68.6 tok/s; 25 GB peak  |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed         | 16565 prompt / 142 generated; 88.1 tok/s; 11 GB peak  |
| mlx-community/Qwen3.8-27B-nvfp4                       | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed         | 16565 prompt / 144 generated; 28.0 tok/s; 21 GB peak  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | rev 844516024a1c; SmolVLMProcessor; stop completed         | 1433 prompt / 78 generated; 124 tok/s; 5.6 GB peak    |
| mlx-community/Step-3.7-Flash-oQ3e                     | rev 41d17ee00e16; Step3VLProcessor; stop completed         | 3494 prompt / 147 generated; 46.4 tok/s; 92 GB peak   |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

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
python -m mlx_vlm.generate --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

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
Keywords:' --max-tokens 1000 --temperature 0.0 --revision RESOLVED_REVISION --trust-remote-code --seed 0 --prefill-step-size 2048
```

The shared command omits per-model automatic thinking flags. When substituting
these models, append the flags recorded in their diagnostics blocks:
`mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit` (--enable-thinking
--thinking-budget 800).

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-1B-4bit                | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/Mage-VL-OptiQ-4bit                 | c98dad5f92f13334cc679c93fb9f185ab49ed626 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/Qwen2-VL-7B-Instruct-4bit          | 1c638e970be36948e087e77d103b9147e1a9a1dd |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | 846ea5576854468f25af3767d769b0805b1b08b6 |
| mlx-community/Molmo2-8B-4bit                     | 4fcbe926577641af12aa5086cf5c292db9dd9730 |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit            | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit        | 98377360cbc84f982e90336f956b08adb46cad88 |
| mlx-community/MiniCPM-V-4.6-4bit                 | 86cd463d33a946e4481b77e3c10fc63121b60a19 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.3                                                                                                                                           |
| mlx-vlm source revision    | 990a028763b2f8c605329f29ae4e747f9285902d                                                                                                        |
| mlx                        | 0.32.3.dev20260925+073d2252c                                                                                                                    |
| mlx source revision        | 073d2252c                                                                                                                                       |
| mlx-audio                  | 0.5.6                                                                                                                                           |
| transformers               | 5.17.0                                                                                                                                          |
| tokenizers                 | 0.23.2                                                                                                                                          |
| huggingface-hub            | 1.33.0                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (192,479,632 bytes, sha256=bd049de2432f971b6f50b872f6e15f0c73477ed4bdab4208beaf921423ae197d) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,850,640 bytes, sha256=30c725ed1dc696853d8cd25b36b6f5758e833264048a33cac57c265534b6c8bd)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
