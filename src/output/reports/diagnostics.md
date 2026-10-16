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
- *Input image:* JPEG, 6,213 x 9,320 pixels (57.9 MP), 52.1 MB

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 51    |
| Conclusive outcomes | 51    |
| Completed           | 51    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 47    |
| observation needs reproduction | 4     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 11    |
| no concerns detected | 29    |
| concerns detected    | 11    |

Observation counts

| Observation                                                  | Count |
|--------------------------------------------------------------|-------|
| Response repeats the same text                               | 3     |
| Generation was stopped early after sustained repeated output | 2     |
| Unrecognised model control tokens remain visible             | 2     |
| Required labelled fields not detected                        | 10    |
| Response appears cut off at the token limit                  | 2     |
| Repeated keyword entries                                     | 4     |
| Output repeats the prompt's own hint text                    | 13    |

## Triage

| Model                                                                                               | Execution | Mechanical checks | Maintainer status              | Observations                                                                                |
|-----------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|---------------------------------------------------------------------------------------------|
| [mlx-community/granite-vision-3.2-2b-nvfp4](#diagnostic-mlx-community-granite-vision-32-2b-nvfp4)   | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; labelled fields not detected; duplicate keywords   |
| [mlx-community/Phi-3.5-vision-instruct-bf16](#diagnostic-mlx-community-phi-35-vision-instruct-bf16) | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                                 |
| [mlx-community/Qwen2-VL-2B-mlx](#diagnostic-mlx-community-qwen2-vl-2b-mlx)                          | completed | major concerns    | observation needs reproduction | repeated text; control tokens visible; labelled fields not detected; cut off at token limit |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)          | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected                                        |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-granite-vision-32-2b-nvfp4"></a>

<details>
<summary>mlx-community/granite-vision-3.2-2b-nvfp4 — major concerns — repeated text; stopped early: repeating; labelled fields not detected; duplicate keywords</summary>

### mlx-community/granite-vision-3.2-2b-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort,
  missing_requested_sections, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type granite_vision)
- *Family README in installed mlx-vlm:* mlx_vlm/models/granite_vision/README.md
  (read before filing)
- *Labelled fields not detected:* ["description"]
- *Repeated fragment:* keyword: "trees"
- *Title word count:* 12
- *Keyword count:* 48
- *Keywords taken verbatim from the prompt's keyword hints:* 14
- *Duplicate keywords:* ["norwich", "trees"]
- *Resolved model revision:* f0c70965cdae87ebf38e84ff958e841ad943e8e6
- *Processor class:* mlx_vlm.models.granite_vision.processing_granite_vision.GraniteVisionProcessor
- *Tokenizer class:* transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 3.022
- *Peak memory at first token (GB):* 4.324
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 0,
  "generation_config.json": 0}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja, tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 5688
- *Prompt composition:* 5,688 = 449 text/template + 5,239 image tokens (92%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 2.35
- *Parameter count:* 2.00B (name-estimate)
- *Quantization:* 4-bit, group 16, nvfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.01x (2.36 GB vs 2.35 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 2.35 GB; other
  load-time allocations 0.02 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 0
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
Title: "Twilight at St Peter Mancroft: A Gothic Symphony of Light and Shadow"
Keywords: Gothic Architecture, Historic Landmark, Norwich, Pedestrians, Perpendicular Gothic, Stone Masonry, Stone facade, Trees, United Kingdom, Walking, Evening Light, Norwich, England, Church Tower, Bell Tower, St Peter Mancroft, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees, Trees,
```

</details>

<a id="diagnostic-mlx-community-phi-35-vision-instruct-bf16"></a>

<details>
<summary>mlx-community/Phi-3.5-vision-instruct-bf16 — major concerns — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/Phi-3.5-vision-instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type phi3_v)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "stone architecture"
- *Title word count:* 9
- *Keyword count:* 37
- *Keywords taken verbatim from the prompt's keyword hints:* 16
- *Percent of words in four-word runs copied from the description hint:* 50
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["historic architecture", "stone architecture"]
- *Resolved model revision:* d8da684308c275a86659e2b36a9189b2f4aec8ea
- *Processor class:* mlx_vlm.models.phi3_v.processing_phi3_v.Phi3VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.796
- *Peak memory at first token (GB):* 9.313
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": [2, 32000, 32007]}
- *Chat template found in (the first is the one the processor uses):* tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.009
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1151
- *Prompt composition:* 1,151 = 394 text/template + 757 image tokens (66%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 8.29
- *Declared context length:* 131,072 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (8.30 GB vs 8.29 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 8.29 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 32000
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
Title: Gothic Church Tower of St Peter Mancroft in Norwich

Description: The Gothic Church Tower of St Peter Mancroft in Norwich, Norfolk, England, stands tall with its Victorian lead flèche illuminated in the evening light, as pedestrians walk along the street below.

Keywords: Gothic architecture, Victorian lead flèche, Church Tower, St Peter Mancroft, Norwich, Norfolk, England, historic landmark, stone masonry, stone facade, clear sky, twilight, United Kingdom, walking, pedestrians, perpendicular gothic, spire, historic architecture, historic architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone architecture, stone
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-mlx"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-mlx — major concerns — repeated text; control tokens visible; labelled fields not detected; cut off at token limit</summary>

### mlx-community/Qwen2-VL-2B-mlx

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["keywords"]
- *Repeated fragment:* phrase: "assistant assistant assistant ..."
- *Unexpected special tokens:* ["&lt;|im_start|&gt;"]
- *Title word count:* 32
- *Percent of words in four-word runs copied from the description hint:* 9
- *Text compared with the description hint:* description
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail"]
- *Special tokens emitted (by token id):* ["&lt;|im_start|&gt;"]
- *Resolved model revision:* d8c7c767e2e2c62cda8a51943276458ea6ad43bc
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 39.805
- *Peak memory at first token (GB):* 9.362
- *Text delivery:* 853 of 1,001 streamed chunks released no text (longest
  consecutive stretch 796); the last released 5,841 characters
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
- *Post-cleanup active memory (GB):* 0.009
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16577
- *Prompt composition:* 16,577 = 353 text/template + 16,224 image tokens (98%;
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
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
Title: Low-angle view of the medieval tower and Victorian lead flèche of the Church of St Peter Mancroft illuminated in evening light, with pedestrians walking along the street below in Norwich, Norfolk, England.

Description: The image shows a low-angle view of the medieval tower and Victorian lead flèche of the Church of St Peter Mancroft in Norwich, Norfolk, England. The tower is illuminated in evening light, with pedestrians walking along the street below. The church has a gothic architectural style with intricate details and a spire. The sky is clear, and there are trees in the foreground. The image is taken during twilight, with the sky transitioning from day to night. The church is a historic landmark and a popular tourist destination. The keywords include Gothic Architecture, Historic Landmark, Historic architecture, Landmark, Norwich, Pedestrians, Perpendicular Gothic, Spire, St Peter Mancroft, Stone Masonry, Stone facade, Trees, Twilight, United Kingdom, Walking.

<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant
<|im_start|>assistant

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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.465
- *Peak memory at first token (GB):* 6.73
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2211
- *Prompt composition:* 2,211 = 419 text/template + 1,792 image tokens (81%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.70 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.02 GB
- *Generation tokens:* 19
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a large church tower with a spire and a pointed roof.
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                    | Mechanical checks | Observations                                         |
|----------------------------------------------------------|-------------------|------------------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                          | major concerns    | labelled fields not detected; prompt hint repeated   |
| mlx-community/gemma-3n-E4B-it-4bit                       | major concerns    | labelled fields not detected                         |
| mlx-community/MolmoPoint-8B-4bit                         | major concerns    | labelled fields not detected                         |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit                | major concerns    | labelled fields not detected; cut off at token limit |
| mlx-community/nanoLLaVA-1.5-4bit                         | major concerns    | labelled fields not detected                         |
| mlx-community/SmolVLM-256M-Instruct-4bit                 | major concerns    | labelled fields not detected; prompt hint repeated   |
| vikhyatk/moondream2                                      | major concerns    | labelled fields not detected; prompt hint repeated   |
| mlx-community/Molmo2-8B-4bit                             | concerns detected | duplicate keywords                                   |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx                 | concerns detected | duplicate keywords; prompt hint repeated             |
| mlx-community/AREX-2-4bit                                | concerns detected | prompt hint repeated                                 |
| mlx-community/GLM-4.6V-Flash-4bit                        | concerns detected | prompt hint repeated                                 |
| mlx-community/granite-4.0-3b-vision-4bit                 | concerns detected | prompt hint repeated                                 |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                    | concerns detected | prompt hint repeated                                 |
| mlx-community/pixtral-12b-8bit                           | concerns detected | prompt hint repeated                                 |
| mlx-community/Step-3.7-Flash-oQ3e                        | concerns detected | prompt hint repeated                                 |
| nativ-community/Mage-VL-OptiQ-4bit                       | concerns detected | prompt hint repeated                                 |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | concerns detected | prompt hint repeated                                 |
| sahilchachra/LensVLM-9B-MXFP4                            | concerns detected | prompt hint repeated                                 |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                          |
|-------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 2131 prompt / 83 generated; 493 tok/s; 1.9 GB peak   |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2108 prompt / 102 generated; 102 tok/s; 6.5 GB peak  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 611 prompt / 92 generated; 75.8 tok/s; 28 GB peak    |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit            | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed       | 1656 prompt / 617 generated; 124 tok/s; 19 GB peak   |
| mlx-community/gemma-3-27b-it-qat-4bit                       | rev fc4e000f32af; Gemma3Processor; stop completed            | 610 prompt / 145 generated; 31.5 tok/s; 17 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 615 prompt / 118 generated; 62.6 tok/s; 7.6 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 615 prompt / 105 generated; 123 tok/s; 16 GB peak    |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 611 prompt / 108 generated; 125 tok/s; 6.0 GB peak   |
| mlx-community/GLM-4.6V-nvfp4                                | rev 2da6855d4e28; Glm46VMoEProcessor; stop completed         | 6372 prompt / 112 generated; 44.7 tok/s; 78 GB peak  |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 2132 prompt / 98 generated; 57.6 tok/s; 10 GB peak   |
| mlx-community/InternVL3_5-1B-4bit                           | rev f9d179a8be8a; InternVLProcessor; stop completed          | 2135 prompt / 126 generated; 423 tok/s; 2.1 GB peak  |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit                | rev e5abbe34cbfa; KimiVLProcessor; stop completed            | 1342 prompt / 750 generated; 66.6 tok/s; 20 GB peak  |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit            | rev 8451adc50203; MllamaProcessor; stop completed            | 320 prompt / 101 generated; 22.0 tok/s; 15 GB peak   |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 410 prompt / 104 generated; 104 tok/s; 7.1 GB peak   |
| mlx-community/MiniCPM-V-4.6-4bit                            | rev 86cd463d33a9; MiniCPMVProcessor; stop completed          | 950 prompt / 333 generated; 305 tok/s; 3.2 GB peak   |
| mlx-community/MiniCPM-V-4.7-1B-4bit                         | rev 13caa569f70e; MiniCPMV4_7Processor; stop completed       | 952 prompt / 77 generated; 311 tok/s; 3.7 GB peak    |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit                    | rev 0e3b706c1599; MiniCPMV4_7Processor; stop completed       | 952 prompt / 120 generated; 88.9 tok/s; 25 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 2959 prompt / 122 generated; 66.1 tok/s; 13 GB peak  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 2958 prompt / 113 generated; 187 tok/s; 7.8 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 4101 prompt / 158 generated; 209 tok/s; 3.9 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 12809 prompt / 120 generated; 72.0 tok/s; 26 GB peak |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 16566 prompt / 144 generated; 85.9 tok/s; 23 GB peak |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 16566 prompt / 91 generated; 69.0 tok/s; 11 GB peak  |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 16579 prompt / 130 generated; 108 tok/s; 25 GB peak  |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 16579 prompt / 133 generated; 29.8 tok/s; 21 GB peak |
| mlx-community/X-Reasoner-7B-8bit                            | rev 21732e74613b; Qwen2_5_VLProcessor; stop completed        | 16577 prompt / 127 generated; 59.7 tok/s; 14 GB peak |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 3642 prompt / 121 generated; 156 tok/s; 23 GB peak   |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit       | rev 4bc0a247d78e; NemotronHNanoOmniProcessor; stop completed | 3641 prompt / 153 generated; 69.6 tok/s; 52 GB peak  |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 16579 prompt / 123 generated; 36.2 tok/s; 17 GB peak |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,213 x 9,320 pixels
- *Image size:* 52,096,760 bytes
- *Image SHA-256:* 5526381f05071b9832b086adc634888de171e314a3d5d1acc4c80c3afc95a8a4

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 18:09:17 UTC+01:00
- GPS: 52.628000°N, 1.293600°E

Descriptive hints:
- Description hint: A low-angle view of the medieval tower and Victorian lead flèche of the Church of St Peter Mancroft illuminated in evening light, with pedestrians walking along the street below in Norwich, Norfolk, England.
- Keyword hints: Archway, Bell Tower, Church Tower, Clear Sky, England, Gothic Architecture, Historic Landmark, Historic architecture, Landmark, Norwich, Pedestrians, Perpendicular Gothic, Spire, St Peter Mancroft, Stone Masonry, Stone facade, Trees, Twilight, United Kingdom, Walking

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-f00ae35004e26a90.jpg>
- *Preview dimensions:* 683 x 1,024 pixels
- *Preview size:* 99,535 bytes
- *Preview SHA-256:* f00ae35004e26a90c102e7357a69f13b84009d2416fe3bdb9ba7d7c4b3a6ee3b

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-f00ae35004e26a90.jpg
printf '%s\n' 'f00ae35004e26a90c102e7357a69f13b84009d2416fe3bdb9ba7d7c4b3a6ee3b  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-10 18:09:17 UTC+01:00
- GPS: 52.628000°N, 1.293600°E

Descriptive hints:
- Description hint: A low-angle view of the medieval tower and Victorian lead flèche of the Church of St Peter Mancroft illuminated in evening light, with pedestrians walking along the street below in Norwich, Norfolk, England.
- Keyword hints: Archway, Bell Tower, Church Tower, Clear Sky, England, Gothic Architecture, Historic Landmark, Historic architecture, Landmark, Norwich, Pedestrians, Perpendicular Gothic, Spire, St Peter Mancroft, Stone Masonry, Stone facade, Trees, Twilight, United Kingdom, Walking

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

| Model                                      | Resolved revision                        |
|--------------------------------------------|------------------------------------------|
| mlx-community/granite-vision-3.2-2b-nvfp4  | f0c70965cdae87ebf38e84ff958e841ad943e8e6 |
| mlx-community/Phi-3.5-vision-instruct-bf16 | d8da684308c275a86659e2b36a9189b2f4aec8ea |
| mlx-community/Qwen2-VL-2B-mlx              | d8c7c767e2e2c62cda8a51943276458ea6ad43bc |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit      | 9c056d48b1e611dc586139a5deb927ae363cfe6f |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.8                                                                                                                                           |
| mlx-vlm source revision    | 259802ec00b26cd73d1a138cd0ed2bb72d85e205                                                                                                        |
| mlx                        | 0.32.5.dev20261016+b6642e604                                                                                                                    |
| mlx source revision        | b6642e604                                                                                                                                       |
| mlx-audio                  | 0.5.9                                                                                                                                           |
| transformers               | 5.20.1                                                                                                                                          |
| tokenizers                 | 0.23.3                                                                                                                                          |
| huggingface-hub            | 2.2.0                                                                                                                                           |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 27.0.0                                                                                                                                   |
| macOS Version              | 27.0.1                                                                                                                                          |
| SDK Version                | 27.0                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk                                              |
| Xcode Version              | 27.1                                                                                                                                            |
| Xcode Build                | 27A9275                                                                                                                                         |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX27.0.sdk                                                                                                                                  |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (215,979,424 bytes, sha256=aba48559c501dcd8307a195ec30bccb0280cac39452241a4447f11110bf6ca8e) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,057,456 bytes, sha256=0caf4de04f76c025462924d0c01707f228c29079092a8c861bf078074121cfe3)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
