# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

This run records model responses to one shared image and prompt (evaluation
lane: blind). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run Summary

- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB

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
| none                           | 45    |
| observation needs reproduction | 6     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 10    |
| no concerns detected | 41    |

Observation counts

| Observation                                                  | Count |
|--------------------------------------------------------------|-------|
| Response repeats the same text                               | 4     |
| Generation was stopped early after sustained repeated output | 1     |
| Final answer emitted twice                                   | 1     |
| Unrecognised model control tokens remain visible             | 2     |
| Required labelled fields not detected                        | 6     |
| Response appears cut off at the token limit                  | 3     |
| Repeated keyword entries                                     | 3     |

## Triage

| Model                                                                                                           | Execution | Mechanical checks | Maintainer status              | Observations                                                        |
|-----------------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|---------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)                               | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit; duplicate keywords           |
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit; duplicate keywords           |
| [mlx-community/Qwen2-VL-2B-mlx](#diagnostic-mlx-community-qwen2-vl-2b-mlx)                                      | completed | major concerns    | observation needs reproduction | repeated text; labelled fields not detected; cut off at token limit |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords         |
| [mlx-community/Step-3.7-Flash-oQ3e](#diagnostic-mlx-community-step-37-flash-oq3e)                               | completed | major concerns    | observation needs reproduction | answer emitted twice; control tokens visible                        |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)                      | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected                |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-internvl35-1b-4bit"></a>

<details>
<summary>mlx-community/InternVL3_5-1B-4bit — major concerns — repeated text; cut off at token limit; duplicate keywords</summary>

### mlx-community/InternVL3_5-1B-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, token_cap_truncation, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type internvl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "british english examples of me"
- *Title word count:* 7
- *Keyword count:* 144
- *Duplicate keywords:* ["british english", "british english examples of
  metadata", "british english examples of details", "british english examples
  of context", "british english examples of information", "british english
  examples of data", "british english examples of content", "british english
  examples of examples"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Processor class:* mlx_vlm.models.internvl.processing_internvl.InternVLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.372
- *Peak memory at first token (GB):* 2.307
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.003
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3438
- *Prompt composition:* 3,438 = 110 text/template + 3,328 image tokens (97%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 1.08
- *Parameter count:* 1.00B (name-estimate)
- *Quantization:* 4-bit, group 32, affine
- *Declared context length:* 40,960 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (1.08 GB vs 1.08 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 1.08 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max nominal
  over 2 sample(s); GPU activity outside harness inference: max 0% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
Title: Two cats sleeping on a pink couch

Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a remote control.

Keywords: cats, sleeping, pink couch, remote control, striped, British English, British, British-English, British-English catalogue, British-English metadata, British-English details, British-English context, British-English information, British-English data, British-English content, British-English examples, British-English examples of metadata, British-English examples of keywords, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata
```

</details>

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — major concerns — repeated text; cut off at token limit; duplicate keywords</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, token_cap_truncation, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "feline household domesticated "
- *Title word count:* 6
- *Keyword count:* 141
- *Duplicate keywords:* ["feline household domesticated animals", "feline
  household domesticated companions", "feline household domesticated friends",
  "feline household domesticated comfort", "feline household domesticated
  relaxation", "feline household domesticated comfort and relaxation"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.513
- *Peak memory at first token (GB):* 14.953
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": [128001, 128008,
  128009]}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  tokenizer_config.json
- *Post-cleanup active memory (GB):* 0.005
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 107
- *Prompt composition:* 107 = 106 text/template + 1 image tokens (1%; exact,
  counted by token id in the prepared input)
- *Checkpoint weights (GB):* 11.34
- *Parameter count:* 11.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (11.35 GB vs 11.34 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 11.35 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max nominal
  over 2 sample(s); GPU activity outside harness inference: max 0% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
Here is the metadata for the image:

**Title:** Two Cats on a Pink Blanket

**Description:** Two tabby cats are sleeping on a pink blanket, with two remote controls nearby. The cats are lying on their backs, with their paws stretched out and their eyes closed. The blanket is a deep pink color and appears to be made of a soft, plush material.

**Keywords:** Tabby cats, pink blanket, remote controls, sleeping, feline, domestic, household, comfort, relaxation, domesticated, household pets, feline companions, household items, comfort and relaxation, domesticated animals, household companions, feline friends, household comfort, feline relaxation, household felines, feline household, household feline, feline household items, feline household comfort, feline household relaxation, feline household companions, feline household friends, feline household comfort and relaxation, feline household domesticated, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation, feline household domesticated animals, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort, feline household domesticated relaxation, feline household domesticated companions, feline household domesticated friends, feline household domesticated comfort and relaxation,
```

</details>

<a id="diagnostic-mlx-community-qwen2-vl-2b-mlx"></a>

<details>
<summary>mlx-community/Qwen2-VL-2B-mlx — major concerns — repeated text; labelled fields not detected; cut off at token limit</summary>

### mlx-community/Qwen2-VL-2B-mlx

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_vl)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["keywords"]
- *Repeated fragment:* phrase: "the cats are not..."
- *Title word count:* 7
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail"]
- *Resolved model revision:* d8c7c767e2e2c62cda8a51943276458ea6ad43bc
- *Processor class:* mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.17
- *Peak memory at first token (GB):* 5.433
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
- *Prompt tokens:* 511
- *Prompt composition:* 511 = 120 text/template + 391 image tokens (77%;
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
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); GPU activity outside harness inference: max 0% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
Title: Two cats sleeping on a pink couch

Description: Two cats, one with a green collar, are sleeping on a pink couch. One cat is lying on its side, while the other cat is lying on its back. Both cats are striped and have their eyes closed. There are two remote controls on the couch next to the cats. The cats are lying on a pink blanket that covers the couch. The cats are not interacting with each other. The cats are not looking at the camera. The cats are not wearing any other accessories. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry. The cats are not wearing any other makeup. The cats are not wearing any other clothes. The cats are not wearing any other shoes. The cats are not wearing any other jewelry.
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
- *Repeated fragment:* keyword: "feline behavior"
- *Title word count:* 7
- *Keyword count:* 54
- *Duplicate keywords:* ["feline behavior", "feline companions", "feline
  comfort", "feline companionship", "feline lifestyle", "feline rest", "feline
  home"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.346
- *Peak memory at first token (GB):* 10.221
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja
- *Post-cleanup active memory (GB):* 0.012
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 511
- *Prompt composition:* 511 = 120 text/template + 391 image tokens (77%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 9.45
- *Parameter count:* 7.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 128,000 (max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (9.46 GB vs 9.45 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 9.45 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 275
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max nominal
  over 2 sample(s); GPU activity outside harness inference: max 0% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
Title: Cats Sleeping on Pink Couch with Remotes

Description: Two tabby cats are curled up on a pink couch, one slightly larger than the other, both appearing relaxed. A remote control rests between them, adding a casual, lived-in feel to the scene. The lighting is soft and natural, highlighting the cats' striped fur and the vibrant pink fabric.

Keywords: cats, sleeping, pink couch, tabby, remote control, relaxed, striped fur, natural light, cozy, domestic, feline, comfort, home setting, feline behavior, rest, domestic animals, feline companions, feline friends, feline relaxation, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companions, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior, feline home, feline comfort, feline companionship, feline lifestyle, feline rest, feline behavior,
```

</details>

<a id="diagnostic-mlx-community-step-37-flash-oq3e"></a>

<details>
<summary>mlx-community/Step-3.7-Flash-oQ3e — major concerns — answer emitted twice; control tokens visible</summary>

### mlx-community/Step-3.7-Flash-oQ3e

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* final_answer_duplicated, unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type step3p7)
- *Family README in installed mlx-vlm:* none
- *Unexpected special tokens:* ["&lt;/think&gt;"]
- *Title word count:* 7
- *Keyword count:* 19
- *Text between the two answer copies:* &lt;/think&gt;
- *Resolved model revision:* 41d17ee00e168a2918bb839e4a7b6e445c6f03f1
- *Processor class:* mlx_vlm.models.step3p7.processing_step3p7.Step3VLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 13.103
- *Peak memory at first token (GB):* 86.896
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 128007, "config.json
  text_config": [1, 2, 128007]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.012
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 286
- *Prompt composition:* 286 = 117 text/template + 169 image tokens (59%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 84.95
- *Quantization:* 3-bit, group 128, affine
- *Declared context length:* 262,144 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (84.96 GB vs 84.95 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 84.95 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 236
- *Configured EOS token ID:* 128007
- *Configured EOS token:* &lt;|im_end|&gt;
- *Snapshot notes (neutral):* processor config missing from snapshot
  (preprocessor_config.json, processor_config.json)
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); GPU activity outside harness inference: max 0% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
Title:
Two cats sleeping on a pink blanket

Description:
Two tabby cats are lying on a bright pink blanket on a red sofa. Both cats appear to be asleep, with one cat on the left having a green collar and the other cat on the right being larger. Two remote controls are placed near the cats on the blanket.

Keywords:
cats, sleeping, pink blanket, red sofa, tabby cats, remote controls, green collar, large cat, small cat, relaxed, pets, home interior, soft fabric, lying down, close-up, daytime, comfortable, cozy
</think>
Title:
Two cats sleeping on a pink blanket

Description:
Two tabby cats are lying on a bright pink blanket on a red sofa. Both cats appear to be asleep, with one cat on the left having a green collar and the other cat on the right being larger. Two remote controls are placed near the cats on the blanket.

Keywords:
cats, sleeping, pink blanket, red sofa, tabby cats, remote controls, green collar, large cat, small cat, relaxed, pets, home interior, soft fabric, lying down, close-up, daytime, comfortable, cozy
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
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Processor class:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.366
- *Peak memory at first token (GB):* 7.013
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.016
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3512
- *Prompt composition:* 3,512 = 184 text/template + 3,328 image tokens (95%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.70 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.02 GB
- *Generation tokens:* 14
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max nominal
  over 2 sample(s); GPU activity outside harness inference: max 0% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The two cats are sleeping on a pink couch.
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                    | Mechanical checks | Observations                 |
|------------------------------------------|-------------------|------------------------------|
| mlx-community/FastVLM-0.5B-bf16          | major concerns    | labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit         | major concerns    | labelled fields not detected |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns    | labelled fields not detected |
| vikhyatk/moondream2                      | major concerns    | labelled fields not detected |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                         |
|-------------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 350 prompt / 47 generated; 521 tok/s; 1.2 GB peak   |
| mlx-community/AREX-2-4bit                                   | rev 02551ee54839; Qwen3VLProcessor; stop completed           | 421 prompt / 80 generated; 33.3 tok/s; 17 GB peak   |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2909 prompt / 82 generated; 100 tok/s; 6.5 GB peak  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 386 prompt / 67 generated; 54.0 tok/s; 28 GB peak   |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit            | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed       | 545 prompt / 478 generated; 109 tok/s; 18 GB peak   |
| mlx-community/gemma-3-27b-it-qat-4bit                       | rev fc4e000f32af; Gemma3Processor; stop completed            | 379 prompt / 89 generated; 31.5 tok/s; 17 GB peak   |
| mlx-community/gemma-3n-E4B-it-4bit                          | rev 505468a22e57; Gemma3nProcessor; stop completed           | 378 prompt / 63 generated; 72.3 tok/s; 7.2 GB peak  |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 390 prompt / 77 generated; 62.6 tok/s; 7.5 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 390 prompt / 81 generated; 115 tok/s; 16 GB peak    |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 386 prompt / 62 generated; 125 tok/s; 5.9 GB peak   |
| mlx-community/GLM-4.6V-Flash-4bit                           | rev bd7b20686e8c; Glm46VProcessor; stop completed            | 499 prompt / 49 generated; 86.5 tok/s; 8.0 GB peak  |
| mlx-community/GLM-4.6V-nvfp4                                | rev 2da6855d4e28; Glm46VMoEProcessor; stop completed         | 499 prompt / 85 generated; 49.9 tok/s; 63 GB peak   |
| mlx-community/granite-4.0-3b-vision-4bit                    | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed    | 734 prompt / 66 generated; 179 tok/s; 4.7 GB peak   |
| mlx-community/granite-vision-3.2-2b-nvfp4                   | rev f0c70965cdae; GraniteVisionProcessor; stop completed     | 3095 prompt / 119 generated; 174 tok/s; 3.9 GB peak |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 3435 prompt / 90 generated; 56.1 tok/s; 10 GB peak  |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit                | rev e5abbe34cbfa; KimiVLProcessor; stop completed            | 489 prompt / 435 generated; 67.8 tok/s; 20 GB peak  |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                       | rev 7886c0b4a4b5; Lfm2VlProcessor; stop completed            | 344 prompt / 74 generated; 214 tok/s; 3.6 GB peak   |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 177 prompt / 103 generated; 103 tok/s; 7.0 GB peak  |
| mlx-community/MiniCPM-V-4.6-4bit                            | rev 86cd463d33a9; MiniCPMVProcessor; stop completed          | 330 prompt / 478 generated; 255 tok/s; 3.1 GB peak  |
| mlx-community/MiniCPM-V-4.7-1B-4bit                         | rev 13caa569f70e; MiniCPMV4_7Processor; stop completed       | 332 prompt / 64 generated; 261 tok/s; 3.6 GB peak   |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit                    | rev 0e3b706c1599; MiniCPMV4_7Processor; stop completed       | 332 prompt / 86 generated; 78.9 tok/s; 24 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 1070 prompt / 72 generated; 68.7 tok/s; 9.8 GB peak |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 1069 prompt / 100 generated; 198 tok/s; 4.5 GB peak |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 873 prompt / 95 generated; 71.3 tok/s; 8.5 GB peak  |
| mlx-community/MolmoPoint-8B-4bit                            | rev 9bab196f867c; MolmoPointProcessor; stop completed        | 860 prompt / 70 generated; 32.0 tok/s; 12 GB peak   |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit                   | rev 865ee8de6579; MuseGlimmerProcessor; stop completed       | 545 prompt / 937 generated; 23.1 tok/s; 25 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 418 prompt / 78 generated; 164 tok/s; 3.5 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16                  | rev d8da684308c2; Phi3VProcessor; stop completed             | 883 prompt / 94 generated; 59.4 tok/s; 9.1 GB peak  |
| mlx-community/pixtral-12b-8bit                              | rev 79e24b66302d; PixtralProcessor; stop completed           | 1335 prompt / 89 generated; 41.1 tok/s; 15 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 409 prompt / 87 generated; 84.5 tok/s; 23 GB peak   |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 409 prompt / 83 generated; 88.4 tok/s; 19 GB peak   |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 409 prompt / 60 generated; 102 tok/s; 6.8 GB peak   |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 421 prompt / 102 generated; 79.3 tok/s; 21 GB peak  |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 421 prompt / 113 generated; 30.5 tok/s; 17 GB peak  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx                    | rev 844516024a1c; SmolVLMProcessor; stop completed           | 1204 prompt / 31 generated; 131 tok/s; 5.6 GB peak  |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 420 prompt / 89 generated; 149 tok/s; 4.9 GB peak   |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 538 prompt / 87 generated; 37.3 tok/s; 17 GB peak   |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 421 prompt / 65 generated; 141 tok/s; 21 GB peak    |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit       | rev 4bc0a247d78e; NemotronHNanoOmniProcessor; stop completed | 420 prompt / 105 generated; 61.4 tok/s; 46 GB peak  |
| sahilchachra/LensVLM-9B-MXFP4                               | rev 23ae80ae9a7d; Qwen3VLProcessor; stop completed           | 421 prompt / 111 generated; 105 tok/s; 6.8 GB peak  |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 421 prompt / 94 generated; 40.9 tok/s; 11 GB peak   |

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
Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:
```

</details>

The original local input is not published, so this report does not claim a
complete reproduction command. Use a shareable equivalent image or add the
original image before filing.

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-c4aeb8d18432656f.jpg>
- *Preview dimensions:* 640 x 480 pixels
- *Preview size:* 63,187 bytes
- *Preview SHA-256:* c4aeb8d18432656f38f690facffbfa90ad3966dc2113a675025d1231a7c5747b

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-c4aeb8d18432656f.jpg
printf '%s\n' 'c4aeb8d18432656f38f690facffbfa90ad3966dc2113a675025d1231a7c5747b  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

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
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/Step-3.7-Flash-oQ3e                | 41d17ee00e168a2918bb839e4a7b6e445c6f03f1 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit            | 9c056d48b1e611dc586139a5deb927ae363cfe6f |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.8                                                                                                                                           |
| mlx-vlm source revision    | b38cfa7b84b7824400290e6a2bb7d9fd3fe3690f                                                                                                        |
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
