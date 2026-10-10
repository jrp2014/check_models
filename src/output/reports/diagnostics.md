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
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 33.5 MB

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 47    |
| Conclusive outcomes | 47    |
| Completed           | 47    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 39    |
| observation needs reproduction | 8     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 13    |
| no concerns detected | 29    |
| concerns detected    | 5     |

Observation counts

| Observation                                                  | Count |
|--------------------------------------------------------------|-------|
| Response repeats the same text                               | 4     |
| Generation was stopped early after sustained repeated output | 4     |
| Unrecognised model control tokens remain visible             | 2     |
| Required labelled fields not detected                        | 9     |
| Response appears cut off at the token limit                  | 2     |
| Internal reasoning block appears incomplete                  | 2     |
| Conversation-role control tokens remain visible              | 1     |
| Repeated keyword entries                                     | 7     |
| Output repeats the prompt's own hint text                    | 5     |

## Triage

| Model                                                                                                           | Execution | Mechanical checks | Maintainer status              | Observations                                                                                                          |
|-----------------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)                               | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                                           |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                                           |
| [mlx-community/Qwen2-VL-2B-mlx](#diagnostic-mlx-community-qwen2-vl-2b-mlx)                                      | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                                           |
| [mlx-community/SmolVLM2-2.2B-Instruct-mlx](#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx)                 | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; labelled fields not detected                                                 |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected                                                                  |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit)              | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; duplicate keywords |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit)        | completed | major concerns    | observation needs reproduction | labelled fields not detected; cut off at token limit; incomplete thinking block                                       |
| [mlx-community/MiniCPM-V-4.6-4bit](#diagnostic-mlx-community-minicpm-v-46-4bit)                                 | completed | major concerns    | observation needs reproduction | incomplete thinking block                                                                                             |

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
- *Repeated fragment:* keyword: "norwich"
- *Title word count:* 8
- *Keyword count:* 47
- *Keywords taken verbatim from the prompt's keyword hints:* 12
- *Percent of words in four-word runs copied from the description hint:* 36
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["norwich guildhall", "lion", "norwich", "southeast
  europe", "southeast"]
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Processor class:* mlx_vlm.models.internvl.processing_internvl.InternVLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.863
- *Peak memory at first token (GB):* 2.145
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.003
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2127
- *Prompt composition:* 2,127 = 335 text/template + 1,792 image tokens (84%;
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
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: "Bronze Lion Sculpture at Norwich Guildhall, Norwich, England"

Description: The image features a bronze lion sculpture by Alfred Hardiman, positioned outside Norwich City Hall, with the historic 15th-century flint Norwich Guildhall visible in the background. The scene captures a sunny day with a blue sky, showcasing a street scene with pedestrians and a car, highlighting British heritage and Gothic architecture.

Keywords: Alfred Hardiman, Norwich Guildhall, Gothic Architecture, British heritage, Lion, Norwich, Norwich City Hall, Southeast Europe, East Anglia, Car, Pedestrian, Statue, Historic Landmark, Blue sky, Southeast, Southeast Europe, Norwich, England, Gothic, British, City Hall, Southeast, Norwich, Norwich, Norwich Guildhall, Lion, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich, Norwich,
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — major concerns — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "lion statue"
- *Title word count:* 8
- *Keyword count:* 44
- *Keywords taken verbatim from the prompt's keyword hints:* 8
- *Percent of words in four-word runs copied from the description hint:* 0
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["statue of a lion", "lion statue"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 2.209
- *Peak memory at first token (GB):* 14.982
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": [128001, 128008,
  128009]}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.004
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 311
- *Prompt composition:* 311 = 310 text/template + 1 image tokens (0%; exact,
  counted by token id in the prepared input)
- *Checkpoint weights (GB):* 11.34
- *Parameter count:* 11.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (11.35 GB vs 11.34 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 11.35 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: Bronze Lion Statue in Front of Norwich Guildhall

Description: A bronze lion statue stands proudly in front of the historic Norwich Guildhall, set against the backdrop of a sunny day in the city of Norwich, Norfolk, England. The statue's intricate details and the building's Gothic architecture are prominent features of the scene.

Keywords: Bronze lion statue, Norwich Guildhall, historic building, Gothic architecture, sunny day, city of Norwich, Norfolk, England, outdoor setting, statue, building, architecture, cityscape, outdoor scene, daytime scene, blue sky, green trees, street scene, people walking, cars parked, street lamps, statue of a lion, lion statue, bronze statue, statue of a lion, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion statue, lion
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-mlx"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-mlx — major concerns — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/Qwen2-VL-2B-mlx

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "medieval town square"
- *Title word count:* 6
- *Keyword count:* 39
- *Keywords taken verbatim from the prompt's keyword hints:* 9
- *Percent of words in four-word runs copied from the description hint:* 30
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["medieval town", "medieval town square"]
- *Resolved model revision:* d8c7c767e2e2c62cda8a51943276458ea6ad43bc
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 46.7
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
- *Prompt tokens:* 16569
- *Prompt composition:* 16,569 = 345 text/template + 16,224 image tokens (98%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 4.42
- *Parameter count:* 2.00B (name-estimate)
- *Declared context length:* 32,768 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (4.43 GB vs 4.42 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 4.42 GB; other
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
Title: Bronze Lion Statue in Norwich, England

Description:
The bronze lion statue stands on the bridge in Norwich, Norfolk, England. It is a 15th-century flint Norwich Guildhall, with the historic building visible in the background. The statue is a symbol of the city and is a popular tourist attraction.

Keywords: Bronze Lion Statue, Norwich, Norfolk, England, Guildhall, Historic Landmark, Lion, Statue, Sightseeing, Street Scene, Bridge, City Hall, Flint, Medieval, Medieval Architecture, Medieval City, Medieval Guildhall, Medieval Statue, Medieval Town, Medieval Town Hall, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town Square, Medieval Town
```

</details>

<a id="diagnostic-mlx-community-smolvlm2-22b-instruct-mlx"></a>

<details>
<summary>mlx-community/SmolVLM2-2.2B-Instruct-mlx — major concerns — repeated text; stopped early: repeating; labelled fields not detected</summary>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort,
  missing_requested_sections
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "sculpture, sculpture, sculptur..."
- *Percent of words in four-word runs copied from the description hint:* 0
- *Text compared with the description hint:* answer
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Processor class:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.031
- *Peak memory at first token (GB):* 5.563
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 2}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.01
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1439
- *Prompt composition:* 1,439 = 386 text/template + 1,053 image tokens (73%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 4.49
- *Parameter count:* 2.20B (name-estimate)
- *Declared context length:* 8,192 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (4.50 GB vs 4.49 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 4.49 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 49279
- *Configured EOS token:* &lt;end_of_utterance&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
 Lion, City Hall, Norwich Guildhall, Sculpture, Sightseeing, Statue, Street Scene, Sculpture, Sightseeing, Statue, Sculpture, Sightseeing, Sculpture, Sightseeing, Sculpture, Sightseeing, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture, Sculpture,
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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.428
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
- *Prompt tokens:* 2205
- *Prompt composition:* 2,205 = 413 text/template + 1,792 image tokens (81%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.69 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 15
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a bronze statue of a roaring lion.
```

</details>

<a id="diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

<details>
<summary>mlx-community/Muse-Glimmer-30B-OptiQ-4bit — major concerns — control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; duplicate keywords</summary>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_special_token, role_boundary_token_present, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["title", "description"]
- *Unexpected special tokens:* ["&lt;|message|&gt;"]
- *Role-boundary tokens in output:* ["&lt;|message|&gt;"]
- *Title word count:* 0
- *Keyword count:* 32
- *Keywords taken verbatim from the prompt's keyword hints:* 3
- *Duplicate keywords:* ["correct conflicts", "and add important visible
  details prefer image evidence when a hint conflicts", "and omit uncertain
  details"]
- *Token-cap degradation evidence:* ["missing_sections"]
- *Special tokens emitted (by token id):* ["&lt;|message|&gt;"]
- *Resolved model revision:* 98377360cbc84f982e90336f956b08adb46cad88
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 9.191
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
- *Prompt tokens:* 4413
- *Prompt composition:* 4,413 = 357 text/template + 4,056 image tokens (92%;
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

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 16:41:54 UTC+01:00
- GPS: 52.628900°N, 1.292500°E

Descriptive hints:
- Description hint: A bronze lion sculpture by Alfred Hardiman stands outside City Hall overlooking the Market Place, with the historic 15th-century flint Norwich Guildhall visible in the background in Norwich, Norfolk, England.
- Keyword hints: Adobe Stock, Any Vision, Blue sky, British heritage, Car, East Anglia, England, Europe, Gothic Architecture, Guildhall, Historic Landmark, Lion, Norfolk, Norwich, Norwich Guildhall, Pedestrian, Sculpture, Sightseeing, Statue, Street Scene

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need title 5-10 words. Concrete.

Bronze lion statue on plinth with Norwich Guildhall behind. The statue is maybe the Norwich lion? Actually the lion sculpture by Alfred Hardiman stands outside City Hall overlooking Market Place? The hint says bronze lion sculpture by Alfred Hardiman stands outside City Hall overlooking Market Place, with historic 15th-century flint Norwich Guildhall visible in background in Norwich, Norfolk, England.

Image shows a bronze lion with green patina, rearing, mouth open, on a stone plinth. Background is stone building with Gothic windows, crenellated roof, arched doorways. There's a street lamp, blue railing, car, pedestrians.

GPS 52.6289 N 1.2925 E is Norwich. So fine.

We must not claim capture date/time visible. So don't mention date in description. Can use context but not claim visible.

We should retain details supported by image. The lion sculpture: bronze, green patina, rearing with one paw raised. Background building is Norwich Guildhall? The building looks like Guildhall. Might be City Hall? The hint says outside City Hall overlooking Market Place, with Guildhall visible. Could be correct. The image shows the building with crenellations and Gothic windows. Could be Norwich Guildhall. Might be consistent.

We should not claim author Alfred Hardiman? Hint says sculpture by Alfred Hardiman. Is that correct? Might be uncertain. The hint says description hint: A bronze lion sculpture by Alfred Hardiman stands outside City Hall... We can retain? The instruction: Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

We cannot verify author from image. Might be uncertain. Probably safer to omit artist name? The hint is authoritative context? It says descriptive hints. Could be wrong. Might be risky. Possibly the lion is the "Norwich Lion" or "City Hall lions". Actually Norwich City Hall has two lions by Alfred Hardiman? Might be correct. Hmm.

The instruction says retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Artist name is not visible. So maybe omit. But it's context maybe accepted. The description hint includes it. Might be okay to include? Might be uncertain. Safer to say bronze lion sculpture. Not name artist.

Also "stands outside City Hall overlooking the Market Place" - image shows building behind. Could be Guildhall. Let's just say in Norwich city centre with historic Guildhall in background.

Make title 5-10 words. e.g. "Bronze Lion Statue with Norwich Guildhall Background". That's 6 words? Bronze(1) Lion(2) Statue(3) with(4) Norwich(5) Guildhall(6) Background(7). Good.

Description: 1-2 sentences factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details.

We can say: A weathered bronze lion statue with a green patina stands on a stone plinth in the foreground, one forepaw raised, mouth open, against a bright blue sky. Behind it the historic flint Norwich Guildhall with Gothic windows and crenellated parapet is visible, with street scene elements including a lamp post, vehicles and pedestrians
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
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Percent of words in four-word runs copied from the description hint:* 4
- *Text compared with the description hint:* answer
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace"]
- *Resolved model revision:* e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.293
- *Peak memory at first token (GB):* 20.071
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.6
- *Sampling settings source:* temperature: generation_config; top_p: default;
  top_k: default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": [163584, 163586], "config.json
  text_config": 163585, "generation_config.json": [163585]}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja
- *Post-cleanup active memory (GB):* 0.004
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1334
- *Prompt composition:* 1,334 = 320 text/template + 1,014 image tokens (76%;
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
◁think▷Okay, let's tackle this task step by step. First, I need to create a British-English catalogue metadata based on the provided context and image. The key elements from the context are the capture date/time (2026-10-10 16:41:54 UTC+01:00) and GPS coordinates (52.628900°N, 1.292500°E). The descriptive hints mention a bronze lion sculpture by Alfred Hardiman outside City Hall in Norwich, with the 15th-century flint Norwich Guildhall in the background. The keywords include Adobe Stock, Any Vision, Blue sky, British heritage, Car, East Anglia, England, Europe, Gothic Architecture, Guildhall, Historic Landmark, Lion, Norfolk, Norwich, Norwich Guildhall, Pedestrian, Sculpture, Sightseeing, Statue, Street Scene.

Now, looking at the image: the main subject is a bronze lion statue in the foreground, positioned on a pedestal. The background features a historic stone building (likely the Guildhall) with Gothic architecture, a blue sky, and some cars, pedestrians, and a street lamp. The lighting suggests it might be late afternoon or early evening. 

For the title, something concise that captures the main elements: "Bronze Lion Statue at Norwich Guildhall, Historic Landmark in City Hall, East Anglia." That's 10 words. 

The description should combine the context and visible details. Mention the lion sculpture by Alfred Hardiman, the Guildhall, the setting (Norwich, Norfolk), the time of day (implied by lighting), and key elements like the blue sky and Gothic architecture. So, "A bronze lion sculpture by Alfred Hardiman stands prominently on a pedestal overlooking the historic 15th-century flint Norwich Guildhall in Norwich, Norfolk, under a clear blue sky, with Gothic architecture and a bustling street scene in the background."

For keywords, list unique ones from the hints and visible details: Bronze Lion Statue, Alfred Hardiman, Norwich Guildhall, Gothic Architecture, Historic Landmark, City Hall, East Anglia, England, European Heritage, Pedestrian Area, Street Scene, Blue Sky. Let me check that they’re all present in the image or hints. Yes, most are covered. Also, include the GPS coordinates and date as authoritative facts. Wait, the user said to treat capture date/time and GPS as authoritative facts but not as visible. So include them as metadata but not in the description. So keywords should be visible details. Let me adjust. Maybe add "2026-10-10" but the user said not to include as visible. So focus on the image elements. Maybe "Blue Sky" is already covered. Let me check the hints again. The hints include "Blue sky" and "Street Scene". So the keywords should be unique and visible. Let me list them again:

Bronze Lion Statue, Alfred Hardiman, Norwich Guildhall, Gothic Architecture, Historic Landmark, City Hall, East Anglia, England, Pedestrian Area, Street Scene, Blue Sky. That's 10 keywords. Wait, the user asked for 10-18. Maybe include "Sculpture" as "Bronze Lion Statue" is already there. Maybe add "Norwich" and "Norfolk" as they are part of the context. Wait, the hints include "Norwich Guildhall" and "Norfolk". So perhaps include "Norwich" and "Norfolk" as separate keywords. Let me check the image: the background has "Norwich" in "Norwich Guildhall". So maybe add "Norwich" as a keyword. Similarly, "Guildhall" is already there. So revised keywords: Bronze Lion Statue, Alfred Hardiman, Norwich Guildhall, Gothic Architecture, Historic Landmark, City Hall, East Anglia, England, Pedestrian Area, Street Scene, Blue Sky, Norwich. That's 11. Maybe add "Historic Building" but that's not in the hints. Wait, the hints mention "Historic Landmark". So keep it as "Historic Landmark". Maybe add "Sculpture" but "Bronze Lion Statue" covers that. Let's stick to the 10-18 keywords. Let me check again. The image has a lion statue, the Guildhall, Gothic architecture, blue sky, street scene, pedestrians, cars, lamp post. So maybe include "Car" as a keyword. But "Street Scene" covers that. Maybe "Cars" is redundant. The hints have "Car" but not visible. Wait, the image shows a car in the background. Since the user said to retain visible details and prefer image evidence over conflicting hints, if the car is visible, include it. So add "Car". Then keywords: Bronze Lion Statue, Alfred Hardiman, Norwich Guildhall, Gothic Architecture, Historic Landmark, City Hall, East Anglia, England, Ped
```

</details>

<a id="diagnostic-mlx-community-minicpm-v-46-4bit"></a>

<details>
<summary>mlx-community/MiniCPM-V-4.6-4bit — major concerns — incomplete thinking block</summary>

### mlx-community/MiniCPM-V-4.6-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type minicpmv4_6)
- *Family README in installed mlx-vlm:* mlx_vlm/models/minicpmv4_6/README.md
  (read before filing)
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 8
- *Keyword count:* 21
- *Keywords taken verbatim from the prompt's keyword hints:* 14
- *Percent of words in four-word runs copied from the description hint:* 0
- *Text compared with the description hint:* description
- *Resolved model revision:* 86cd463d33a946e4481b77e3c10fc63121b60a19
- *Processor class:* mlx_vlm.models.minicpmv4_6.processing_minicpmv4_6.MiniCPMVProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.883
- *Peak memory at first token (GB):* 3.101
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.7; top_p 1.0; top_k 0; repetition_penalty 1.0
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"repetition_penalty":
  1.0}
- *EOS ids declared, by file:* {"config.json": [248044, 248046],
  "generation_config.json": [248044, 248046]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.005
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 943
- *Prompt composition:* 943 = 354 text/template (tokenizer-exact) + 589
  non-text (62%, image/audio expansion)
- *Checkpoint weights (GB):* 2.16
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (2.16 GB vs 2.16 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 2.16 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 124
- *Configured EOS token ID:* 248046
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
<think>
Here's the response following the instructions:

Title:
Bronze Lion Statue at City Hall, Norwich, England

Description:
A grand bronze lion sculpture stands prominently outside City Hall, set against a historic backdrop with medieval architecture. The scene is illuminated by bright daylight, highlighting the sculpture's details and the surrounding area.

Keywords: British-English, bronze lion, City Hall, Norwich, England, historic, Gothic Architecture, Guildhall, Lion, Norfolk, Norwich Guildhall, statue, sightseeing, street scene, blue sky, heritage, architecture, market place, pedestrian, sculpture, Europe
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                            | Mechanical checks | Observations                                       |
|--------------------------------------------------|-------------------|----------------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                  | major concerns    | labelled fields not detected; prompt hint repeated |
| mlx-community/gemma-3n-E4B-it-4bit               | major concerns    | labelled fields not detected                       |
| mlx-community/nanoLLaVA-1.5-4bit                 | major concerns    | labelled fields not detected                       |
| mlx-community/SmolVLM-256M-Instruct-4bit         | major concerns    | labelled fields not detected; prompt hint repeated |
| vikhyatk/moondream2                              | major concerns    | labelled fields not detected; prompt hint repeated |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | duplicate keywords                                 |
| mlx-community/gemma-4-12B-it-4bit                | concerns detected | duplicate keywords                                 |
| mlx-community/granite-vision-3.2-2b-nvfp4        | concerns detected | duplicate keywords                                 |
| mlx-community/granite-4.0-3b-vision-4bit         | concerns detected | prompt hint repeated                               |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit            | concerns detected | prompt hint repeated                               |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                          |
|-------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 2128 prompt / 90 generated; 409 tok/s; 1.9 GB peak   |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2100 prompt / 119 generated; 96.1 tok/s; 6.5 GB peak |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 605 prompt / 87 generated; 77.5 tok/s; 28 GB peak    |
| mlx-community/gemma-3-27b-it-qat-4bit                       | rev fc4e000f32af; Gemma3Processor; stop completed            | 604 prompt / 140 generated; 19.6 tok/s; 17 GB peak   |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 609 prompt / 106 generated; 99.4 tok/s; 16 GB peak   |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 605 prompt / 87 generated; 121 tok/s; 5.9 GB peak    |
| mlx-community/GLM-4.6V-Flash-4bit                           | rev bd7b20686e8c; Glm46VProcessor; stop completed            | 6460 prompt / 99 generated; 76.4 tok/s; 8.7 GB peak  |
| mlx-community/GLM-4.6V-nvfp4                                | rev 2da6855d4e28; Glm46VMoEProcessor; stop completed         | 6460 prompt / 108 generated; 37.0 tok/s; 78 GB peak  |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 2124 prompt / 123 generated; 53.5 tok/s; 10 GB peak  |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 402 prompt / 91 generated; 105 tok/s; 7.0 GB peak    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 2935 prompt / 130 generated; 64.4 tok/s; 13 GB peak  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 2934 prompt / 128 generated; 169 tok/s; 7.8 GB peak  |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 1535 prompt / 181 generated; 65.9 tok/s; 8.1 GB peak |
| mlx-community/MolmoPoint-8B-4bit                            | rev 9bab196f867c; MolmoPointProcessor; stop completed        | 3137 prompt / 129 generated; 29.8 tok/s; 13 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 4095 prompt / 171 generated; 143 tok/s; 3.9 GB peak  |
| mlx-community/Phi-3.5-vision-instruct-bf16                  | rev d8da684308c2; Phi3VProcessor; stop completed             | 1149 prompt / 128 generated; 50.4 tok/s; 9.3 GB peak |
| mlx-community/pixtral-12b-8bit                              | rev 79e24b66302d; PixtralProcessor; stop completed           | 3125 prompt / 110 generated; 40.2 tok/s; 16 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 12801 prompt / 109 generated; 50.7 tok/s; 26 GB peak |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 16558 prompt / 161 generated; 75.5 tok/s; 23 GB peak |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 16558 prompt / 125 generated; 62.3 tok/s; 11 GB peak |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 16574 prompt / 148 generated; 49.1 tok/s; 25 GB peak |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 16574 prompt / 124 generated; 27.6 tok/s; 21 GB peak |
| mlx-community/Step-3.7-Flash-oQ3e                           | rev 41d17ee00e16; Step3VLProcessor; stop completed           | 3502 prompt / 122 generated; 48.7 tok/s; 92 GB peak  |
| mlx-community/X-Reasoner-7B-8bit                            | rev 21732e74613b; Qwen2_5_VLProcessor; stop completed        | 16569 prompt / 116 generated; 52.8 tok/s; 14 GB peak |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 4221 prompt / 157 generated; 126 tok/s; 5.4 GB peak  |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 1281 prompt / 133 generated; 13.4 tok/s; 18 GB peak  |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 3636 prompt / 113 generated; 122 tok/s; 23 GB peak   |
| sahilchachra/LensVLM-9B-MXFP4                               | rev 23ae80ae9a7d; Qwen3VLProcessor; stop completed           | 1886 prompt / 223 generated; 101 tok/s; 7.5 GB peak  |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 16574 prompt / 125 generated; 32.5 tok/s; 17 GB peak |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 6,656 pixels
- *Image size:* 33,496,486 bytes
- *Image SHA-256:* 712a5faa6eab3fe302b2a217258979846efc8ae2aa86a8fe7b14b79647a2a44d

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 16:41:54 UTC+01:00
- GPS: 52.628900°N, 1.292500°E

Descriptive hints:
- Description hint: A bronze lion sculpture by Alfred Hardiman stands outside City Hall overlooking the Market Place, with the historic 15th-century flint Norwich Guildhall visible in the background in Norwich, Norfolk, England.
- Keyword hints: Adobe Stock, Any Vision, Blue sky, British heritage, Car, East Anglia, England, Europe, Gothic Architecture, Guildhall, Historic Landmark, Lion, Norfolk, Norwich, Norwich Guildhall, Pedestrian, Sculpture, Sightseeing, Statue, Street Scene

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-142e2341731f9ce5.jpg>
- *Preview dimensions:* 1,024 x 683 pixels
- *Preview size:* 84,474 bytes
- *Preview SHA-256:* 142e2341731f9ce58a5268765c820c8a7ee7615c8034dbbee21a7da5a5033260

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-142e2341731f9ce5.jpg
printf '%s\n' '142e2341731f9ce58a5268765c820c8a7ee7615c8034dbbee21a7da5a5033260  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 16:41:54 UTC+01:00
- GPS: 52.628900°N, 1.292500°E

Descriptive hints:
- Description hint: A bronze lion sculpture by Alfred Hardiman stands outside City Hall overlooking the Market Place, with the historic 15th-century flint Norwich Guildhall visible in the background in Norwich, Norfolk, England.
- Keyword hints: Adobe Stock, Any Vision, Blue sky, British heritage, Car, East Anglia, England, Europe, Gothic Architecture, Guildhall, Historic Landmark, Lion, Norfolk, Norwich, Norwich Guildhall, Pedestrian, Sculpture, Sightseeing, Statue, Street Scene

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
| mlx-community/Qwen2-VL-2B-mlx                    | d8c7c767e2e2c62cda8a51943276458ea6ad43bc |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx         | 844516024a1c4400d34489b89ee067d794e432ed |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit            | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit        | 98377360cbc84f982e90336f956b08adb46cad88 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit     | e5abbe34cbfabd829fafd0362856e5b468d19f85 |
| mlx-community/MiniCPM-V-4.6-4bit                 | 86cd463d33a946e4481b77e3c10fc63121b60a19 |

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
