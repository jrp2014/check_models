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
- *Input image:* JPEG, 5,800 x 8,389 pixels (48.7 MP), 40.8 MB

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 53    |
| Conclusive outcomes | 53    |
| Completed           | 53    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 45    |
| observation needs reproduction | 8     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 14    |
| no concerns detected | 30    |
| concerns detected    | 9     |

Observation counts

| Observation                                                  | Count |
|--------------------------------------------------------------|-------|
| Response repeats the same text                               | 5     |
| Generation was stopped early after sustained repeated output | 3     |
| Unrecognised model control tokens remain visible             | 2     |
| Required labelled fields not detected                        | 9     |
| Response appears cut off at the token limit                  | 5     |
| Internal reasoning block appears incomplete                  | 1     |
| Conversation-role control tokens remain visible              | 1     |
| Repeated keyword entries                                     | 9     |
| Output repeats the prompt's own hint text                    | 4     |
| Names a place the prompt did not supply                      | 2     |

## Triage

| Model                                                                                                    | Execution | Mechanical checks | Maintainer status              | Observations                                                                                      |
|----------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|---------------------------------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)                        | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                       |
| [mlx-community/MolmoPoint-8B-4bit](#diagnostic-mlx-community-molmopoint-8b-4bit)                         | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                       |
| [mlx-community/nanoLLaVA-1.5-4bit](#diagnostic-mlx-community-nanollava-15-4bit)                          | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; labelled fields not detected                             |
| [mlx-community/Qwen2-VL-2B-mlx](#diagnostic-mlx-community-qwen2-vl-2b-mlx)                               | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit                                                             |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                         | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit; duplicate keywords                                         |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)               | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected                                              |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit)       | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) | completed | major concerns    | observation needs reproduction | labelled fields not detected; cut off at token limit; incomplete thinking block                   |

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
- *Repeated fragment:* keyword: "photo"
- *Title word count:* 6
- *Keyword count:* 49
- *Keywords taken verbatim from the prompt's keyword hints:* 9
- *Percent of words in four-word runs copied from the description hint:* 33
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["signage", "photo"]
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Processor class:* mlx_vlm.models.internvl.processing_internvl.InternVLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.721
- *Peak memory at first token (GB):* 2.147
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.004
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2145
- *Prompt composition:* 2,145 = 353 text/template + 1,792 image tokens (84%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 1.08
- *Parameter count:* 1.00B (name-estimate)
- *Quantization:* 4-bit, group 32, affine
- *Declared context length:* 40,960 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (1.09 GB vs 1.08 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 1.08 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: Architectural view of The Shopkeeper Store

Description: The image showcases a traditional British brick building with a black-painted storefront featuring gold detailing, located at No. 76, with a middle story and three sash windows, and a gabled dormer on the upper level.

Keywords: The Shopkeeper Store, No. 76, black-painted, gold detailing, red brick, sash window, gable, entrance, European, boutique, signage, British, architectural detail, building exterior, store, clock, GPS, 52.629112°N, 1.288265°E, photo, photo editing, vintage, UK, street view, signage, shop, signage, building, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo, photo,
```

</details>

<a id="diagnostic-mlx-community-molmopoint-8b-4bit"></a>

<details>
<summary>mlx-community/MolmoPoint-8B-4bit — major concerns — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/MolmoPoint-8B-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type molmo_point)
- *Family README in installed mlx-vlm:* mlx_vlm/models/molmo_point/README.md
  (read before filing)
- *Repeated fragment:* keyword: "united kingdom"
- *Title word count:* 9
- *Keyword count:* 37
- *Keywords taken verbatim from the prompt's keyword hints:* 1
- *Percent of words in four-word runs copied from the description hint:* 40
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["united kingdom"]
- *Resolved model revision:* 9bab196f867ceedecdb1eab071395bfedcd62538
- *Processor class:* mlx_vlm.models.molmo_point.processing_molmo_point.MolmoPointProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 2.869
- *Peak memory at first token (GB):* 12.75
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.008
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3174
- *Prompt composition:* 3,174 = 351 text/template (tokenizer-exact) + 2,823
  non-text (89%, image/audio expansion)
- *Checkpoint weights (GB):* 7.22
- *Parameter count:* 8.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 37,376 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (7.23 GB vs 7.22 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 7.22 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: Traditional British Shopfront with Gold Detailing and Red Brick

Description: A street-level view of The Shopkeeper Store at No. 76 in the United Kingdom, featuring a black-painted ground floor with gold detailing, a red brick middle story with three sash windows, and twin slate-grey gabled dormers on the upper level. The image captures the building's traditional architecture and boutique aesthetic on a clear day.

Keywords: United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United Kingdom, United
```

</details>

<a id="diagnostic-mlx-community-nanollava-15-4bit"></a>

<details>
<summary>mlx-community/nanoLLaVA-1.5-4bit — major concerns — repeated text; stopped early: repeating; labelled fields not detected</summary>

### mlx-community/nanoLLaVA-1.5-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort,
  missing_requested_sections
- *Arch supported by installed mlx-vlm:* yes (model_type llava-qwen2 via
  llava_bunny)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["keywords"]
- *Repeated fragment:* phrase: "no. 76, no. 76,..."
- *Title word count:* 10
- *Percent of words in four-word runs copied from the description hint:* 44
- *Text compared with the description hint:* description
- *Resolved model revision:* 5240204744963d72823e5de933c528c4aa82dfca
- *Processor class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.588
- *Peak memory at first token (GB):* 1.813
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.018
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 359
- *Prompt composition:* 359 = 358 text/template + 1 image tokens (0%; exact,
  counted by token id in the prepared input)
- *Checkpoint weights (GB):* 0.59
- *Quantization:* 4-bit, group 64
- *Declared context length:* 32,768 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.04x (0.62 GB vs 0.59 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 0.60 GB; other
  load-time allocations 0.02 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Snapshot notes (neutral):* processor config missing from snapshot
  (preprocessor_config.json, processor_config.json)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: "The Shopkeeper Store: A Street-Level Architectural View of The Shopfront"
Description: A detailed view of the shopfront of The Shopkeeper Store, featuring a traditional black-painted storefront with gold detailing on the ground floor, red brick middle story with three sash windows, and twin slate-grey gabled dormers on the upper level. The building is situated at No. 76, located at No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76, No. 76,
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-mlx"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-mlx — major concerns — repeated text; cut off at token limit</summary>

### mlx-community/Qwen2-VL-2B-mlx

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* phrase: "view of the shopkeeper..."
- *Title word count:* 8
- *Keyword count:* 12
- *Keywords taken verbatim from the prompt's keyword hints:* 6
- *Percent of words in four-word runs copied from the description hint:* 0
- *Text compared with the description hint:* description
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* d8c7c767e2e2c62cda8a51943276458ea6ad43bc
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 32.887
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
- *Post-cleanup active memory (GB):* 0.01
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16581
- *Prompt composition:* 16,581 = 363 text/template + 16,218 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 4.42
- *Parameter count:* 2.00B (name-estimate)
- *Declared context length:* 32,768 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (4.43 GB vs 4.42 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 4.42 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: A Street-Level Architectural View of The Shopkeeper Store

Description: A 5-10-word title that captures the essence of the image, focusing on the main subject and setting.

Keywords: Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 10-18 unique, comma-separated keywords covering relevant context and visible details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 5-10-word title that captures the essence of the image, focusing on the main subject and setting.

Keywords: Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 10-18 unique, comma-separated keywords covering relevant context and visible details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 5-10-word title that captures the essence of the image, focusing on the main subject and setting.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 10-18 unique, comma-separated keywords covering relevant context and visible details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 5-10-word title that captures the essence of the image, focusing on the main subject and setting.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 10-18 unique, comma-separated keywords covering relevant context and visible details.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick, Sash window, Twin dormers, Closed, Shopfront, Signage, United Kingdom

Title: A Street-Level View of The Shopkeeper Store

Description: A 5-10-word title that captures the essence of the image, focusing on the main subject and setting.

Keywords: Street-level, Exterior, Traditional, Black-painted, Gold detailing, Brick wall, Red brick,
```

</details>

<a id="diagnostic-mlx-community-x-reasoner-7b-8bit"></a>

<details>
<summary>mlx-community/X-Reasoner-7B-8bit — major concerns — repeated text; cut off at token limit; duplicate keywords</summary>

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, token_cap_truncation, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "closed building front door"
- *Title word count:* 6
- *Keyword count:* 223
- *Keywords taken verbatim from the prompt's keyword hints:* 14
- *Percent of words in four-word runs copied from the description hint:* 28
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["closed sign", "boutique", "traditional", "entrance",
  "building", "closed", "closed boutique", "closed building front door",
  "closed shop front door", "closed store front door", "closed boutique front
  door", "closed building entrance door", "closed shop entrance door", "closed
  store entrance door", "closed boutique entrance door", "closed building
  exterior door", "closed shop exterior door", "closed store exterior door",
  "closed boutique exterior door"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 15.506
- *Peak memory at first token (GB):* 14.286
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja
- *Post-cleanup active memory (GB):* 0.013
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16581
- *Prompt composition:* 16,581 = 363 text/template + 16,218 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 9.45
- *Parameter count:* 7.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 128,000 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (9.46 GB vs 9.45 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 9.45 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
The Shopkeeper Store, No.76, UK

Description:
A traditional British storefront at No.76, featuring a black-painted ground floor with gold detailing, red brick middle story, and twin slate-grey gabled dormers, with closed signage and merchandise displayed in the windows. The setting is a street-level view under a partly cloudy sky, with the GPS coordinates provided.

Keywords: The Shopkeeper Store, No.76, UK, closed sign, black storefront, gold detailing, red brick, gabled dormers, sash windows, street-level view, partly cloudy, GPS coordinates, architectural detail, boutique, brick building, building exterior, United Kingdom, Europe, architectural, traditional, shopfront, signage, merchandise, entrance, chimney, roof, red brick wall, boutique, shop, store, exterior, building, architecture, street, closed, day, lighting, windows, doors, door, entrance, facade, gable, dormer, slate, grey, brick, red, black, gold, traditional, European, architectural style, building, structure, exterior detail, shop window, display, closed, sign, closed sign, closed door, closed storefront, closed shop, closed business, closed store, closed boutique, closed shopfront, closed building, closed structure, closed architecture, closed building exterior, closed shop exterior, closed store exterior, closed boutique exterior, closed building front, closed shop front, closed store front, closed boutique front, closed building facade, closed shop facade, closed store facade, closed boutique facade, closed building exterior detail, closed shop exterior detail, closed store exterior detail, closed boutique exterior detail, closed building architectural detail, closed shop architectural detail, closed store architectural detail, closed boutique architectural detail, closed building window, closed shop window, closed store window, closed boutique window, closed building door, closed shop door, closed store door, closed boutique door, closed building entrance, closed shop entrance, closed store entrance, closed boutique entrance, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique exterior door, closed building front door, closed shop front door, closed store front door, closed boutique front door, closed building entrance door, closed shop entrance door, closed store entrance door, closed boutique entrance door, closed building exterior door, closed shop exterior door, closed store exterior door, closed boutique
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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.391
- *Peak memory at first token (GB):* 6.731
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.017
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2234
- *Prompt composition:* 2,234 = 442 text/template + 1,792 image tokens (80%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.70 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.02 GB
- *Generation tokens:* 26
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a brick building with a black storefront and a grey upper level with two dormers.
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
- *Keyword count:* 11
- *Keywords taken verbatim from the prompt's keyword hints:* 0
- *Token-cap degradation evidence:* ["missing_sections"]
- *Special tokens emitted (by token id):* ["&lt;|message|&gt;"]
- *Resolved model revision:* 98377360cbc84f982e90336f956b08adb46cad88
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 8.225
- *Peak memory at first token (GB):* 25.468
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
- *Post-cleanup active memory (GB):* 0.008
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4458
- *Prompt composition:* 4,458 = 377 text/template + 4,081 image tokens (92%;
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
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 17:21:00 UTC+01:00
- GPS: 52.629112°N, 1.288265°E

Descriptive hints:
- Description hint: A street-level architectural view of the exterior of The Shopkeeper Store, located at No. 76, featuring a traditional black-painted storefront adorned with gold detailing on the ground floor, a red brick middle story with three sash windows, and twin slate-grey gabled dormers on the upper level.
- Keyword hints: Adobe Stock, Any Vision, Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick building, brick wall, british, building exterior

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need title 5-10 words. Concrete. Maybe "The Shopkeeper Store No 76 Exterior Norwich". Need check GPS 52.629112°N, 1.288265°E. That's Norwich, England. The Shopkeeper at 76? Possibly 76a entrance left. The store is The Shopkeeper, No 76. Upper twin gabled dormers slate-grey. Red brick middle story with three windows. Black storefront with gold detailing. Closed sign on door. Entrance 76a to left.

Description 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Capture date/time 2026-10-10 17:21:00 UTC+01:00. GPS given. Don't claim visible. Should mention street-level architectural view of exterior of The Shopkeeper at No.76, etc. Daylight, partly cloudy sky. No people visible? Possibly none. Should mention closed sign.

Keywords: 10-18 unique comma-separated. Include Adobe Stock, Any Vision? Maybe those are hints for source? Probably include. Keywords hints list includes Adobe Stock, Any Vision, Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick building, brick wall, british, building exterior.

We should include relevant visible details: dormer, gable, chimneys, 76a door, bay windows, display, etc.

Make sure keywords unique comma-separated.

Title: maybe 7 words: "The Shopkeeper Store Exterior No 76 Norwich". That's 7? The(1) Shopkeeper(2) Store(3) Exterior(4) No(5) 76(6) Norwich(7). Good.

Description: Street-level view of the exterior of The Shopkeeper boutique at No. 76, captured on 2026-10-10 at 17:21 UTC+01:00 at GPS 52.629112°N, 1.288265°E. The three-storey red-brick building has a black-painted shopfront with gold detailing, large bow windows displaying clothing and accessories, a central door marked Closed, and twin slate-grey gabled dormers above three sash windows under a partly cloudy sky.

That's one sentence? Could be two sentences. Keep 1-2 sentences.

Make sure not claim capture date/time visible. It's okay to state captured on... Might be okay.

Keywords list: maybe 15 items.

Keywords: The Shopkeeper, No 76, Norwich, United Kingdom, building exterior, red brick, sash window, shopfront, boutique, gabled dormer, chimney, black facade, gold detailing, signage, entrance, architectural detail

Need comma-separated. Ensure 10-18 unique.

Maybe include Adobe Stock, Any Vision? Hints include them maybe as source metadata. Might be appropriate to include. But keywords should cover relevant context and visible details. Might include Adobe Stock as per hint. Could include. Safer to follow hints.

The hint list includes Adobe Stock, Any Vision. Could include both.

Let's produce:

Title:
The Shopkeeper Store Exterior No 76 Norwich

Description:
Street-level view of the exterior of The Shopkeeper boutique at No. 
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-8bit — major concerns — labelled fields not detected; cut off at token limit; incomplete thinking block</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["keywords"]
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Title word count:* 127
- *Percent of words in four-word runs copied from the description hint:* 22
- *Text compared with the description hint:* description
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace"]
- *Resolved model revision:* e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.225
- *Peak memory at first token (GB):* 20.063
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.6
- *Sampling settings source:* temperature: generation_config; top_p: default;
  top_k: default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": [163584, 163586], "config.json
  text_config": 163585, "generation_config.json": [163585]}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja
- *Post-cleanup active memory (GB):* 0.005
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1326
- *Prompt composition:* 1,326 = 338 text/template + 988 image tokens (75%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 17.83
- *Active parameter count:* 3.00B (name-estimate; total not stated in the
  name)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (17.83 GB vs 17.83 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 17.83 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Okay, let's tackle this task step by step. First, I need to create a British-English catalogue metadata based on the provided context and image. The key is to stay true to the authoritative information and the descriptive hints while also incorporating visible details from the image.

Starting with the title: The image shows a street-level view of a building at No. 76, which is "The Shopkeeper Store" as indicated by the signage. The building has a traditional black storefront with gold detailing, red brick, slate grey upper floors, and dormers. The date is 2026-10-10 17:21:00 UTC+01:00, and the location is 52.629112°N, 1.288265°E in the UK. The keywords include elements like chimney, entrance, sash windows, brick, etc.

For the description, I need to combine the context with visible details. The store is a boutique shop with a black storefront and red brick facade. The building has three sash windows, a chimney, and is in the UK. The storefront is closed, and there are display windows with items inside. The lighting is bright, suggesting it's daytime. Also, the architectural style is traditional British.

Now, the keywords should be 10-18 unique terms. Let's list them: "Adobe Stock, Any Vision, Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick wall, british, building exterior, closed, display windows, daylight, timber door." Wait, but "Adobe Stock" and "Any Vision" are probably watermarks, but the user said to treat capture date/time and GPS as authoritative, but not claim they are visible. Maybe include "Adobe Stock" as a keyword if it's part of the watermark, but perhaps they are not visible. Wait, the user's instructions say to omit uncertain details. So maybe exclude those watermarks. Let me check the hints again. The hints include "Adobe Stock, Any Vision" as keywords but the image might have watermarks. Since the user says to omit uncertain details, maybe exclude those. So adjust keywords to visible elements: "Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick wall, british, building exterior, closed, display windows, daylight."

Wait, the user's instructions say to prefer image evidence over hints. The image shows "No 76" on the door and the store name "The Shopkeeper". So include "No 76" and "The Shopkeeper Store" as visible details. Also, the store is closed (sign says "CLOSED"). So update keywords to include those. Let's check: "Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick wall, british, building exterior, closed, display windows, daylight, No 76, The Shopkeeper Store". That's 18 keywords. 

Title: "Traditional British boutique shopfront at No. 76, featuring red brick facade, black storefront with gold detailing, and gabled roof". Wait, the roof is slate grey, not sure if "gabled roof" is correct. The upper part is slate grey, so maybe "slate-tiled roof". But the hints mention "Gable" and "Roof". The image has dormers on the upper level, so "gabled dormers" might be better. Let me check the image again. The upper part is a grey slate roof with dormers. So "Gabled dormers on slate roof" could be accurate. But the user's hints mention "Roof" and "Gable". So maybe "Gabled dormers on slate roof". But the hints say to retain details supported by the image. The dormers are part of the upper floor, so "Gabled dormers" is correct. 

Title: "Traditional British boutique shopfront at No. 76 with red brick facade, black storefront, and gabled dormers on slate roof". Then the description should mention the store being closed, the entrance, sash windows, etc. 

Putting it all together:

Title: "Traditional British boutique shopfront at No. 76 with red brick facade, black storefront, and gabled dormers on slate roof"

Description: "Street-level view of The Shopkeeper Store, a traditional British boutique located at No. 76, featuring a red brick facade, black storefront with gold detailing, three sash windows, a timber door, and gabled dormers on a slate roof, captured on 2026-10-10 at 17:
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                       | Mechanical checks | Observations                                     |
|-------------------------------------------------------------|-------------------|--------------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                             | major concerns    | labelled fields not detected                     |
| mlx-community/gemma-3n-E4B-it-4bit                          | major concerns    | labelled fields not detected                     |
| mlx-community/granite-vision-3.2-2b-nvfp4                   | major concerns    | labelled fields not detected; duplicate keywords |
| mlx-community/SmolVLM-256M-Instruct-4bit                    | major concerns    | labelled fields not detected                     |
| vikhyatk/moondream2                                         | major concerns    | labelled fields not detected                     |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit            | major concerns    | cut off at token limit; duplicate keywords       |
| mlx-community/GLM-4.6V-Flash-4bit                           | concerns detected | duplicate keywords                               |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit            | concerns detected | duplicate keywords                               |
| mlx-community/Molmo2-8B-4bit                                | concerns detected | duplicate keywords                               |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | concerns detected | duplicate keywords                               |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                       | concerns detected | prompt hint repeated; unsupplied place name      |
| mlx-community/North-Micro-Vision-Instruct-4bit              | concerns detected | prompt hint repeated                             |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx                    | concerns detected | prompt hint repeated                             |
| sahilchachra/LensVLM-9B-MXFP4                               | concerns detected | prompt hint repeated                             |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | concerns detected | unsupplied place name                            |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                    | Runtime identity                                           | Performance                                          |
|----------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                         | rev ed71acdae079; Lfm2VlProcessor; stop completed          | 2144 prompt / 83 generated; 484 tok/s; 1.9 GB peak   |
| mlx-community/AREX-2-4bit                                | rev 02551ee54839; Qwen3VLProcessor; stop completed         | 16586 prompt / 137 generated; 30.7 tok/s; 21 GB peak |
| mlx-community/aya-vision-8b-4bit                         | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed | 2123 prompt / 125 generated; 101 tok/s; 6.5 GB peak  |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit    | rev 0a970d20ad7d; Mistral3Processor; stop completed        | 2498 prompt / 134 generated; 30.1 tok/s; 23 GB peak  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8            | rev ded389e478f8; DiffusionGemma4Processor; stop completed | 626 prompt / 95 generated; 49.6 tok/s; 28 GB peak    |
| mlx-community/gemma-3-27b-it-qat-4bit                    | rev fc4e000f32af; Gemma3Processor; stop completed          | 625 prompt / 161 generated; 29.9 tok/s; 17 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                        | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed   | 630 prompt / 106 generated; 60.0 tok/s; 7.7 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                    | rev 0d77464eeb23; Gemma4Processor; stop completed          | 630 prompt / 104 generated; 110 tok/s; 16 GB peak    |
| mlx-community/gemma-4-31b-it-4bit                        | rev 696d436c4047; Gemma4Processor; stop completed          | 630 prompt / 109 generated; 26.3 tok/s; 20 GB peak   |
| mlx-community/gemma-4-e4b-it-4bit                        | rev 475b9088d297; Gemma4Processor; stop completed          | 626 prompt / 104 generated; 125 tok/s; 6.0 GB peak   |
| mlx-community/GLM-4.6V-nvfp4                             | rev 2da6855d4e28; Glm46VMoEProcessor; stop completed       | 6445 prompt / 145 generated; 44.1 tok/s; 78 GB peak  |
| mlx-community/granite-4.0-3b-vision-4bit                 | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed  | 1420 prompt / 160 generated; 174 tok/s; 4.6 GB peak  |
| mlx-community/Idefics3-8B-Llama3-bf16                    | rev 8c2a30c48864; Idefics3Processor; stop completed        | 2641 prompt / 200 generated; 34.4 tok/s; 18 GB peak  |
| mlx-community/InternVL3-14B-4bit                         | rev 26328eaab82c; InternVLChatProcessor; stop completed    | 2142 prompt / 134 generated; 56.3 tok/s; 10 GB peak  |
| mlx-community/InternVL3-8B-bf16                          | rev e0df3dd79263; InternVLChatProcessor; stop completed    | 2142 prompt / 104 generated; 36.8 tok/s; 17 GB peak  |
| mlx-community/MiniCPM-o-4_5-4bit                         | rev 592c09d85e7b; MiniCPMOProcessor; stop completed        | 420 prompt / 94 generated; 104 tok/s; 7.0 GB peak    |
| mlx-community/MiniCPM-V-4.6-4bit                         | rev 86cd463d33a9; MiniCPMVProcessor; stop completed        | 963 prompt / 826 generated; 300 tok/s; 3.2 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4        | rev 7c992876448f; Mistral3Processor; stop completed        | 3031 prompt / 244 generated; 65.7 tok/s; 13 GB peak  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit          | rev a962dcb09eee; Mistral3Processor; stop completed        | 3030 prompt / 166 generated; 184 tok/s; 8.1 GB peak  |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit              | rev 4620fdbbd1e7; Qwen3VLProcessor; stop completed         | 1330 prompt / 127 generated; 101 tok/s; 25 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16               | rev d8da684308c2; Phi3VProcessor; stop completed           | 1169 prompt / 147 generated; 56.0 tok/s; 9.3 GB peak |
| mlx-community/pixtral-12b-8bit                           | rev 79e24b66302d; PixtralProcessor; stop completed         | 3297 prompt / 133 generated; 34.7 tok/s; 16 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit           | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed    | 12814 prompt / 141 generated; 71.1 tok/s; 26 GB peak |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                  | rev defcdea7cc7a; Qwen3VLProcessor; stop completed         | 16570 prompt / 87 generated; 68.9 tok/s; 11 GB peak  |
| mlx-community/Qwen3.5-35B-A3B-4bit                       | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed         | 16586 prompt / 140 generated; 108 tok/s; 25 GB peak  |
| mlx-community/Qwen3.8-27B-nvfp4                          | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed         | 16586 prompt / 128 generated; 29.0 tok/s; 21 GB peak |
| mlx-community/Step-3.7-Flash-oQ3e                        | rev 41d17ee00e16; Step3VLProcessor; stop completed         | 3522 prompt / 117 generated; 47.4 tok/s; 92 GB peak  |
| nativ-community/Mage-VL-OptiQ-4bit                       | rev 4f0a424370e5; MageVLProcessor; stop completed          | 4188 prompt / 198 generated; 119 tok/s; 5.4 GB peak  |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | rev bdbeb0d8c89e; Mistral3Processor; stop completed        | 1353 prompt / 136 generated; 35.1 tok/s; 18 GB peak  |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit              | rev 498775b03b55; Qwen3VLProcessor; stop completed         | 16586 prompt / 182 generated; 36.2 tok/s; 17 GB peak |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 5,800 x 8,389 pixels
- *Image size:* 40,750,483 bytes
- *Image SHA-256:* 95f6022daf7bb25ac113cd253b16bb914048dc8e74c4e8c00530bc742e50a21e

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 17:21:00 UTC+01:00
- GPS: 52.629112°N, 1.288265°E

Descriptive hints:
- Description hint: A street-level architectural view of the exterior of The Shopkeeper Store, located at No. 76, featuring a traditional black-painted storefront adorned with gold detailing on the ground floor, a red brick middle story with three sash windows, and twin slate-grey gabled dormers on the upper level.
- Keyword hints: Adobe Stock, Any Vision, Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick building, brick wall, british, building exterior

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-fa374f03acd127a3.jpg>
- *Preview dimensions:* 708 x 1,024 pixels
- *Preview size:* 140,906 bytes
- *Preview SHA-256:* fa374f03acd127a3bd6cbcd73aa11c316de0a3df43e22644c1424fcc8639bab5

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-fa374f03acd127a3.jpg
printf '%s\n' 'fa374f03acd127a3bd6cbcd73aa11c316de0a3df43e22644c1424fcc8639bab5  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 17:21:00 UTC+01:00
- GPS: 52.629112°N, 1.288265°E

Descriptive hints:
- Description hint: A street-level architectural view of the exterior of The Shopkeeper Store, located at No. 76, featuring a traditional black-painted storefront adorned with gold detailing on the ground floor, a red brick middle story with three sash windows, and twin slate-grey gabled dormers on the upper level.
- Keyword hints: Adobe Stock, Any Vision, Chimney, Entrance, Europe, Gable, Objects, Red brick, Roof, Sash Window, Shopfront, Signage, United Kingdom, architectural detail, architecture, boutique, brick building, brick wall, british, building exterior

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

| Model                                        | Resolved revision                        |
|----------------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-1B-4bit            | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/MolmoPoint-8B-4bit             | 9bab196f867ceedecdb1eab071395bfedcd62538 |
| mlx-community/nanoLLaVA-1.5-4bit             | 5240204744963d72823e5de933c528c4aa82dfca |
| mlx-community/Qwen2-VL-2B-mlx                | d8c7c767e2e2c62cda8a51943276458ea6ad43bc |
| mlx-community/X-Reasoner-7B-8bit             | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit        | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit    | 98377360cbc84f982e90336f956b08adb46cad88 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | e5abbe34cbfabd829fafd0362856e5b468d19f85 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.7                                                                                                                                           |
| mlx-vlm source revision    | 952d4f6bc65bd5095e74abe71c78038764ee08aa                                                                                                        |
| mlx                        | 0.32.4.dev20261010+06eb7483f                                                                                                                    |
| mlx source revision        | 06eb7483f                                                                                                                                       |
| mlx-audio                  | 0.5.8                                                                                                                                           |
| transformers               | 5.19.0                                                                                                                                          |
| tokenizers                 | 0.23.3                                                                                                                                          |
| huggingface-hub            | 2.2.0                                                                                                                                           |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (215,965,088 bytes, sha256=46366f90fed6bb8ddd20e811c14cb991b791018d1e1556dae974bd5d33a610dc) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,056,416 bytes, sha256=7d6f4829617d878663d8366327b36d3ca3e41cfb4399b59df5524069534978ca)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
