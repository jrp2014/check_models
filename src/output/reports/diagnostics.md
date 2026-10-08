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
- *Input image:* JPEG, 9,641 x 6,427 pixels (62.0 MP), 58.1 MB

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 50    |
| Conclusive outcomes | 50    |
| Completed           | 50    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 43    |
| observation needs reproduction | 7     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 13    |
| no concerns detected | 29    |
| concerns detected    | 8     |

Observation counts

| Observation                                                               | Count |
|---------------------------------------------------------------------------|-------|
| Response repeats the same text                                            | 4     |
| Generation was stopped early after sustained repeated output              | 3     |
| Unrecognised model control tokens remain visible                          | 2     |
| Required labelled fields not detected                                     | 9     |
| Response appears cut off at the token limit                               | 2     |
| Conversation-role control tokens remain visible                           | 1     |
| Repeated keyword entries                                                  | 6     |
| Output repeats the prompt's own hint text instead of describing the image | 8     |

## Triage

| Model                                                                                                           | Execution | Mechanical checks | Maintainer status              | Observations                                                                                      |
|-----------------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|---------------------------------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)                               | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                       |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed | major concerns    | observation needs reproduction | repeated text; duplicate keywords                                                                 |
| [mlx-community/nanoLLaVA-1.5-4bit](#diagnostic-mlx-community-nanollava-15-4bit)                                 | completed | major concerns    | observation needs reproduction | repeated text; labelled fields not detected; cut off at token limit                               |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                       |
| [mlx-community/Qwen2-VL-2B-mlx](#diagnostic-mlx-community-qwen2-vl-2b-mlx)                                      | completed | major concerns    | observation needs reproduction | stopped early: repeating; duplicate keywords                                                      |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected                                              |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit)              | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-internvl35-1b-4bit"></a>

<details>
<summary>mlx-community/InternVL3_5-1B-4bit — major concerns — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/InternVL3_5-1B-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type internvl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "dover"
- *Title word count:* 9
- *Keyword count:* 59
- *Keywords taken verbatim from the prompt's keyword hints:* 16
- *Percent of words in four-word runs copied from the description hint:* 25
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["dover"]
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Processor class:* mlx_vlm.models.internvl.processing_internvl.InternVLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.891
- *Peak memory at first token (GB):* 2.145
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.003
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2094
- *Prompt composition:* 2,094 = 302 text/template + 1,792 image tokens (86%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 1.08
- *Parameter count:* 1.00B (name-estimate)
- *Quantization:* 4-bit, group 32, affine
- *Declared context length:* 40,960 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (1.08 GB vs 1.08 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 1.08 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max nominal
  over 2 sample(s); mode snapshot

#### Complete output

```text
Title: Moored Patrol Vessels at Ramsgate Harbour, Kent, During Sunset

Description: The image captures a serene sunset over Ramsgate Harbour, featuring BSC patrol vessels, including the BSC Defender and BSC Volunteer, moored alongside other boats, with a dramatic skyline of buildings and a lifeguard lifebuoy visible.

Keywords: Ramsgate, Kent, Sunset, BSC patrol, harbor, mooring, Dover, Fleet, Maritime, Coast, Dusk, England, Lifebuoy, Marina, Patrol boats, Pier, Port, Dover, Horizon, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover, Dover,
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — major concerns — repeated text; duplicate keywords</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* phrase: "sea and sky view..."
- *Title word count:* 7
- *Keyword count:* 45
- *Keywords taken verbatim from the prompt's keyword hints:* 6
- *Percent of words in four-word runs copied from the description hint:* 26
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["sea"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 2.158
- *Peak memory at first token (GB):* 14.969
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": [128001, 128008,
  128009]}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.004
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 290
- *Prompt composition:* 290 = 289 text/template + 1 image tokens (0%; exact,
  counted by token id in the prepared input)
- *Checkpoint weights (GB):* 11.34
- *Parameter count:* 11.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (11.35 GB vs 11.34 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 11.35 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 667
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max nominal
  over 2 sample(s); mode snapshot

#### Complete output

```text
Title: Ramsgate Harbour Sunset with Border Security Vessels

Description: The image captures a serene sunset scene in Ramsgate Harbour, featuring a fleet of border security vessels, including the BSC Defender and BSC Volunteer, moored alongside each other. The vessels are situated in the foreground, with the town's cliffside skyline and buildings visible in the background.

Keywords: Border security vessels, Ramsgate Harbour, sunset, BSC Defender, BSC Volunteer, moored vessels, cliffside skyline, buildings, harbour, sea, boats, vessels, security, maritime, port, marina, water, reflection, sky, clouds, sun, sea, seafront, sea view, sea and sky, sea and sky view, sea and sky view at sunset, sea and sky view at sunset with boats, sea and sky view at sunset with boats and buildings, sea and sky view at sunset with boats and buildings and cliffside, sea and sky view at sunset with boats and buildings and cliffside and harbour, sea and sky view at sunset with boats and buildings and cliffside and harbour and security, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky and clouds, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky and clouds and sun, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky and clouds and sun and sea, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky and clouds and sun and sea and seafront, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky and clouds and sun and sea and seafront and sea view, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky and clouds and sun and sea and seafront and sea view and sea and sky, sea and sky view at sunset with boats and buildings and cliffside and harbour and security and maritime and port and marina and water and reflection and sky and clouds and sun and sea and seafront and sea view and sea and sky and sea and sky view.
```

</details>

<a id="diagnostic-mlx-community-nanollava-15-4bit"></a>

<details>
<summary>mlx-community/nanoLLaVA-1.5-4bit — major concerns — repeated text; labelled fields not detected; cut off at token limit</summary>

### mlx-community/nanoLLaVA-1.5-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["description", "keywords"]
- *Repeated fragment:* phrase: "dusk, england, fleet, harbor,..."
- *Title word count:* 447
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail"]
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Processor class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.707
- *Peak memory at first token (GB):* 1.781
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 308
- *Prompt composition:* 308 = 307 text/template + 1 image tokens (0%; exact,
  counted by token id in the prepared input)
- *Checkpoint weights (GB):* 0.59
- *Quantization:* 4-bit, group 64
- *Declared context length:* 32,768 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.03x (0.61 GB vs 0.59 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 0.60 GB; other
  load-time allocations 0.02 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Snapshot notes (neutral):* processor config missing from snapshot
  (preprocessor_config.json, processor_config.json)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); mode snapshot

#### Complete output

```text
Title: "UK Border Security Command Patrol Vessels in Ramsgate Harbour, Kent, Dusk, England, Fleet, Harbor, Maritime, Mooring, Patrol Boat, Pier, Port, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Dusk, Coast, Dover, Dusk, England, Fleet, Harbor
```

</details>

<a id="diagnostic-mlx-community-x-reasoner-7b-8bit"></a>

<details>
<summary>mlx-community/X-Reasoner-7B-8bit — major concerns — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "horizon"
- *Title word count:* 6
- *Keyword count:* 53
- *Keywords taken verbatim from the prompt's keyword hints:* 12
- *Percent of words in four-word runs copied from the description hint:* 67
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["horizon"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 14.172
- *Peak memory at first token (GB):* 14.286
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja
- *Post-cleanup active memory (GB):* 0.011
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16536
- *Prompt composition:* 16,536 = 312 text/template + 16,224 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 9.45
- *Parameter count:* 7.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 128,000 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (9.46 GB vs 9.45 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 9.45 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); mode snapshot

#### Complete output

```text
Title:
Dusk at Ramsgate: Border Security Vessels

Description:
UK Border Security Command patrol boats, including the BSC Defender and Volunteer, are moored side-by-side in Ramsgate Harbour, Kent, against a dramatic sunset and the town's cliffside skyline, with reflections on the water and a mix of warm and cool lighting.

Keywords: UK Border Security Command, BSC Defender, BSC Volunteer, Ramsgate Harbour, Kent, Dusk, Sunset, Cliffside, Buildings, Marina, Mooring, Patrol boats, Lifebuoys, Reflections, Water, Horizon, England, Coast, Port, Maritime, Horizon, Pier, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon, Horizon,
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-mlx"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-mlx — major concerns — stopped early: repeating; duplicate keywords</summary>

### mlx-community/Qwen2-VL-2B-mlx

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Family README in installed mlx-vlm:* none
- *Title word count:* 9
- *Keyword count:* 22
- *Keywords taken verbatim from the prompt's keyword hints:* 6
- *Percent of words in four-word runs copied from the description hint:* 5
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["lifeboat station"]
- *Resolved model revision:* d8c7c767e2e2c62cda8a51943276458ea6ad43bc
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 37.112
- *Peak memory at first token (GB):* 9.361
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.1; top_p 0.001; top_k 1; repetition_penalty 1.05
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"repetition_penalty":
  1.05}
- *EOS ids declared, by file:* {"config.json": [151645, 151643],
  "generation_config.json": [151645, 151643]}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja
- *Post-cleanup active memory (GB):* 0.008
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16536
- *Prompt composition:* 16,536 = 312 text/template + 16,224 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 4.42
- *Parameter count:* 2.00B (name-estimate)
- *Declared context length:* 32,768 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (4.43 GB vs 4.42 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 4.42 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 225
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); mode snapshot

#### Complete output

```text
Title: Border Security Command vessels in Ramsgate Harbour at sunset

Description: The image shows a marina in Ramsgate, Kent, UK. There are several large boats moored in the harbour, including Border Security Command vessels such as the BSC Defender and BSC Volunteer. The boats are moored alongside each other, with the harbour in the foreground and the town skyline in the background. The sky is illuminated by the setting sun, creating a dramatic and colourful sunset over the buildings and the harbour. The boats are silhouetted against the sunset, with the reflection of the water on the water surface adding to the overall composition. The image captures the contrast between the modern, high-tech vessels and the traditional, historic buildings in the background.

Keywords: Border Security Command vessels, Ramsgate Harbour, sunset, boats, buildings, reflection, Kent, UK, sea, harbour, mooring, lifebuoy, pier, lifeboat, lifeboat station, lifeboat station, lifeboat station, lifeboat station, lifeboat station, lifeboat station, lifeboat station, lifeboat station,
```

</details>

<a id="diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit"></a>

<details>
<summary>mlx-community/llm-jp-4-vl-9b-mlx-4bit — major concerns — control tokens visible; labelled fields not detected</summary>

### mlx-community/llm-jp-4-vl-9b-mlx-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* missing_requested_sections, unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Unexpected special tokens:* ["&lt;|channel|&gt;", "&lt;|message|&gt;"]
- *Percent of words in four-word runs copied from the description hint:* 0
- *Text compared with the description hint:* answer
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Processor class:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.539
- *Peak memory at first token (GB):* 6.729
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.015
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2174
- *Prompt composition:* 2,174 = 382 text/template + 1,792 image tokens (82%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.70 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.02 GB
- *Generation tokens:* 18
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a harbor with several boats docked in a row.
```

</details>

<a id="diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

<details>
<summary>mlx-community/Muse-Glimmer-30B-OptiQ-4bit — major concerns — control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible</summary>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_special_token, role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["title", "description"]
- *Unexpected special tokens:* ["&lt;|message|&gt;"]
- *Role-boundary tokens in output:* ["&lt;|message|&gt;"]
- *Title word count:* 0
- *Keyword count:* 2
- *Keywords taken verbatim from the prompt's keyword hints:* 0
- *Token-cap degradation evidence:* ["missing_sections"]
- *Special tokens emitted (by token id):* ["&lt;|message|&gt;"]
- *Resolved model revision:* 98377360cbc84f982e90336f956b08adb46cad88
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 8.873
- *Peak memory at first token (GB):* 25.465
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 1.0; top_p 0.95; top_k 64
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_length":
  131072}
- *EOS ids declared, by file:* {"config.json": [200001, 200008], "config.json
  text_config": 200001, "generation_config.json": [200001, 200008]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.007
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4390
- *Prompt composition:* 4,390 = 334 text/template + 4,056 image tokens (92%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 23.85
- *Parameter count:* 30.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (23.86 GB vs 23.85 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 23.85 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 200001
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *Snapshot notes (neutral):* processor config missing from snapshot
  (preprocessor_config.json, processor_config.json)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); mode snapshot

#### Complete output

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-03 18:18:19 UTC+01:00

Descriptive hints:
- Description hint: UK Border Security Command patrol vessels, including the BSC Defender and BSC Volunteer, are moored side-by-side in Ramsgate Harbour, Kent, against a dramatic sunset and the town's cliffside skyline.
- Keyword hints: Border security vessels, Buildings, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Reflection

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

Need title 5-10 words.

Maybe "Border Security Command Vessels Moored at Ramsgate Harbour Sunset". That's 9 words? Count: Border(1) Security(2) Command(3) Vessels(4) Moored(5) at(6) Ramsgate(7) Harbour(8) Sunset(9). Good.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details.

We have capture date/time: 2026-10-03 18:18:19 UTC+01:00. Should include? Could mention time? Maybe not claim visible. "Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible." So we can say captured on ...? Might be okay to include date in description? Probably factual description about image. Might incorporate date? Usually description of scene, not metadata. Safer to not mention date as visible. Could just describe scene. Maybe include "on 3 October 2026" as authoritative fact. Hmm instruction: Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. So we can mention capture date in description? Might be okay. Probably include as context. But description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. Could mention sunset dusk.

We have vessels including BSC Defender and BSC Volunteer moored side-by-side in Ramsgate Harbour, Kent, against dramatic sunset and town's cliffside skyline. Image shows multiple patrol boats, white vessels with "Border Security Command" markings, lifebuoys, one labeled DOVER on stern, BSC DEFENDER visible. Also buildings on cliff, white chalk cliff, water reflection golden.

Keyword hints include Dover but image shows Dover on vessel maybe? The hint says Dover. Could be confusion. The harbour is Ramsgate. The vessel has DOVER. The keyword hints include Dover. Maybe keep.

We need 10-18 unique comma-separated keywords.

Let's produce keywords: Border Security Command, patrol vessels, Ramsgate Harbour, Kent, sunset, dusk, harbour, moored boats, cliffside buildings, white cliffs, water reflection, lifebuoys, maritime, England

That's 14.

Ensure unique.

Return exactly three sections.

Title:
Description:
Keywords:

No extra text.

Let's craft description: "UK Border Security Command patrol vessels, including BSC Defender and BSC Volunteer, are moored side-by-side in Ramsgate Harbour, Kent, at dusk with a golden sunset reflecting on the water. The fleet is lined along the quay beneath the town’s cliffside terraced houses and white chalk cliffs under a dramatic orange-purple sky."

That's one sentence? It's two sentences. Acceptable 1-2 sentence.

Make sure not claim date visible. Might not mention date. Could mention capture date? Might be okay to not mention. The instruction says treat capture date/time as authoritative facts, but do not claim they are visible. Could incorporate in description? Maybe not needed. Safer to leave out? The context is authoritative, so maybe incorporate. Hmm.

It says combining relevant context with main visible subject. Context includes capture date/time. Might be relevant. Could add "Captured on 3 October 2026 at 18:18 BST". But that's metadata. Might be okay.

Better keep description about visual. Maybe we can include date in a separate way? Might be confusing.

I think include date not required but allowed. I'll include a brief
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Mechanical checks | Observations                                       |
|-------------------------------------------------------|-------------------|----------------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                       | major concerns    | labelled fields not detected                       |
| mlx-community/gemma-3n-E4B-it-4bit                    | major concerns    | labelled fields not detected                       |
| mlx-community/granite-vision-3.2-2b-nvfp4             | major concerns    | labelled fields not detected; duplicate keywords   |
| mlx-community/MolmoPoint-8B-4bit                      | major concerns    | labelled fields not detected                       |
| mlx-community/SmolVLM-256M-Instruct-4bit              | major concerns    | labelled fields not detected                       |
| vikhyatk/moondream2                                   | major concerns    | labelled fields not detected; prompt hint repeated |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit      | concerns detected | duplicate keywords                                 |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | prompt hint repeated                               |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | concerns detected | prompt hint repeated                               |
| mlx-community/GLM-4.6V-nvfp4                          | concerns detected | prompt hint repeated                               |
| mlx-community/granite-4.0-3b-vision-4bit              | concerns detected | prompt hint repeated                               |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                 | concerns detected | prompt hint repeated                               |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | concerns detected | prompt hint repeated                               |
| mlx-community/Step-3.7-Flash-oQ3e                     | concerns detected | prompt hint repeated                               |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                          |
|-------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 2103 prompt / 58 generated; 493 tok/s; 1.9 GB peak   |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2070 prompt / 132 generated; 102 tok/s; 6.5 GB peak  |
| mlx-community/gemma-3-27b-it-qat-4bit                       | rev fc4e000f32af; Gemma3Processor; stop completed            | 572 prompt / 138 generated; 31.5 tok/s; 17 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 577 prompt / 108 generated; 61.3 tok/s; 7.6 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 577 prompt / 106 generated; 125 tok/s; 16 GB peak    |
| mlx-community/gemma-4-31b-it-4bit                           | rev 696d436c4047; Gemma4Processor; stop completed            | 577 prompt / 107 generated; 26.3 tok/s; 20 GB peak   |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 573 prompt / 79 generated; 126 tok/s; 5.9 GB peak    |
| mlx-community/GLM-4.6V-Flash-4bit                           | rev bd7b20686e8c; Glm46VProcessor; stop completed            | 6339 prompt / 98 generated; 78.7 tok/s; 8.7 GB peak  |
| mlx-community/Idefics3-8B-Llama3-bf16                       | rev 8c2a30c48864; Idefics3Processor; stop completed          | 2601 prompt / 130 generated; 35.4 tok/s; 18 GB peak  |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 2091 prompt / 113 generated; 56.9 tok/s; 10 GB peak  |
| mlx-community/InternVL3-8B-bf16                             | rev e0df3dd79263; InternVLChatProcessor; stop completed      | 2091 prompt / 80 generated; 37.5 tok/s; 17 GB peak   |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit                | rev e5abbe34cbfa; KimiVLProcessor; stop completed            | 1312 prompt / 995 generated; 66.1 tok/s; 20 GB peak  |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 369 prompt / 88 generated; 106 tok/s; 7.0 GB peak    |
| mlx-community/MiniCPM-V-4.6-4bit                            | rev 86cd463d33a9; MiniCPMVProcessor; stop completed          | 910 prompt / 662 generated; 302 tok/s; 3.2 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 2905 prompt / 140 generated; 66.2 tok/s; 13 GB peak  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 2904 prompt / 132 generated; 188 tok/s; 7.8 GB peak  |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 1502 prompt / 148 generated; 71.7 tok/s; 8.5 GB peak |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 4065 prompt / 148 generated; 209 tok/s; 3.9 GB peak  |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit                 | rev 4620fdbbd1e7; Qwen3VLProcessor; stop completed           | 1267 prompt / 129 generated; 106 tok/s; 24 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16                  | rev d8da684308c2; Phi3VProcessor; stop completed             | 1115 prompt / 144 generated; 59.1 tok/s; 9.3 GB peak |
| mlx-community/pixtral-12b-8bit                              | rev 79e24b66302d; PixtralProcessor; stop completed           | 3095 prompt / 119 generated; 40.0 tok/s; 16 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 12768 prompt / 136 generated; 65.4 tok/s; 26 GB peak |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 16525 prompt / 133 generated; 83.0 tok/s; 23 GB peak |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 16525 prompt / 112 generated; 67.9 tok/s; 11 GB peak |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 16541 prompt / 152 generated; 102 tok/s; 25 GB peak  |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 16541 prompt / 121 generated; 27.4 tok/s; 21 GB peak |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 4188 prompt / 118 generated; 129 tok/s; 5.4 GB peak  |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 1251 prompt / 114 generated; 36.8 tok/s; 18 GB peak  |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 3606 prompt / 132 generated; 156 tok/s; 23 GB peak   |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,641 x 6,427 pixels
- *Image size:* 58,125,687 bytes
- *Image SHA-256:* 9f0e8d795514c6c4d18ec033dde42b6865c30a5c45ad85343e85dc24b75bcb63

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-03 18:18:19 UTC+01:00

Descriptive hints:
- Description hint: UK Border Security Command patrol vessels, including the BSC Defender and BSC Volunteer, are moored side-by-side in Ramsgate Harbour, Kent, against a dramatic sunset and the town's cliffside skyline.
- Keyword hints: Border security vessels, Buildings, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Reflection

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-c688458ce4ed66ce.jpg>
- *Preview dimensions:* 1,024 x 683 pixels
- *Preview size:* 117,105 bytes
- *Preview SHA-256:* c688458ce4ed66ce78ed6e7bbc72fbfd28edfce5f0d1160bdf3322c4edc910be

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-c688458ce4ed66ce.jpg
printf '%s\n' 'c688458ce4ed66ce78ed6e7bbc72fbfd28edfce5f0d1160bdf3322c4edc910be  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-03 18:18:19 UTC+01:00

Descriptive hints:
- Description hint: UK Border Security Command patrol vessels, including the BSC Defender and BSC Volunteer, are moored side-by-side in Ramsgate Harbour, Kent, against a dramatic sunset and the town'"'"'s cliffside skyline.
- Keyword hints: Border security vessels, Buildings, Coast, Dover, Dusk, England, Fleet, Harbor, Horizon, Kent, Lifebuoy, Marina, Maritime, Mooring, Patrol boat, Patrol boats, Pier, Port, Ramsgate, Reflection

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
| mlx-community/InternVL3_5-1B-4bit                | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/nanoLLaVA-1.5-4bit                 | 5240204744963d72823e5de933c528c4aa82dfca |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/Qwen2-VL-2B-mlx                    | d8c7c767e2e2c62cda8a51943276458ea6ad43bc |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit            | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit        | 98377360cbc84f982e90336f956b08adb46cad88 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.7                                                                                                                                           |
| mlx-vlm source revision    | 1cc6025432f61b146cb72c7516209f7259fd45d0                                                                                                        |
| mlx                        | 0.32.4.dev20261008+3c40e8f92                                                                                                                    |
| mlx source revision        | 3c40e8f92                                                                                                                                       |
| mlx-audio                  | 0.5.8                                                                                                                                           |
| transformers               | 5.19.0                                                                                                                                          |
| tokenizers                 | 0.23.2                                                                                                                                          |
| huggingface-hub            | 1.33.0                                                                                                                                          |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 27.0.0                                                                                                                                   |
| macOS Version              | 27.0.1                                                                                                                                          |
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
| Recommended Working Set    | 115 GB                                                                                                                                          |
| Fused Attention            | Available                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-314-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (204,770,592 bytes, sha256=1fea48780ba22f210a1222a4933b92178daccb3c0c853ba855dece1030f4907a) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,720,080 bytes, sha256=7b5c77e837b904a2c1db1584804df9ae7fc417fbdd0427fffb823b9a61c951fa)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
