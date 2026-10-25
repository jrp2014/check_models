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
| Attempted           | 43    |
| Conclusive outcomes | 43    |
| Completed           | 43    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 40    |
| observation needs reproduction | 3     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 5     |
| no concerns detected | 37    |
| concerns detected    | 1     |

Observation counts

| Observation                                      | Count |
|--------------------------------------------------|-------|
| Response repeats the same text                   | 2     |
| Unrecognised model control tokens remain visible | 1     |
| Required labelled fields not detected            | 3     |
| Response appears cut off at the token limit      | 2     |
| Repeated keyword entries                         | 3     |

## Triage

| Model                                                                                      | Execution | Mechanical checks | Maintainer status              | Observations                                              |
|--------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|-----------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)          | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit; duplicate keywords |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)           | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit; duplicate keywords |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected      |

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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.544
- *Peak memory at first token (GB):* 2.306
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
- *Repeated fragment:* keyword: "feline behavior study"
- *Title word count:* 7
- *Keyword count:* 189
- *Duplicate keywords:* ["feline behavior study", "feline comfort study",
  "feline rest study", "feline lifestyle study", "feline domestic life study",
  "feline behavior analysis", "feline comfort analysis", "feline rest
  analysis", "feline lifestyle analysis", "feline domestic life analysis",
  "feline behavior observation", "feline comfort observation", "feline rest
  observation", "feline lifestyle observation", "feline domestic life
  observation", "feline behavior research", "feline comfort research", "feline
  rest research", "feline lifestyle research", "feline domestic life
  research"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.398
- *Peak memory at first token (GB):* 10.219
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.json,
  chat_template.jinja
- *Post-cleanup active memory (GB):* 0.011
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
Title: Cats Sleeping on Pink Couch with Remotes

Description: Two tabby cats rest on a pink couch, one curled up with a remote, the other stretched out. The setting is indoors, with soft lighting highlighting their relaxed postures. The couch's vibrant color contrasts with the cats' striped fur.

Keywords: cats, sleeping, pink couch, tabby, remote control, indoor, relaxation, striped fur, comfortable, domestic, feline, rest, home setting, cozy, feline behavior, feline companions, feline rest, feline comfort, feline lifestyle, feline relaxation, feline companionship, feline domestic life, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis, feline rest analysis, feline lifestyle analysis, feline domestic life analysis, feline behavior observation, feline comfort observation, feline rest observation, feline lifestyle observation, feline domestic life observation, feline behavior research, feline comfort research, feline rest research, feline lifestyle research, feline domestic life research, feline behavior study, feline comfort study, feline rest study, feline lifestyle study, feline domestic life study, feline behavior analysis, feline comfort analysis,
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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.908
- *Peak memory at first token (GB):* 7.01
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.013
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3512
- *Prompt composition:* 3,512 = 184 text/template + 3,328 image tokens (95%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.69 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 14
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); GPU activity outside harness inference: max 0% at 2
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

| Model                                            | Mechanical checks | Observations                 |
|--------------------------------------------------|-------------------|------------------------------|
| mlx-community/FastVLM-0.5B-bf16                  | major concerns    | labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit                 | major concerns    | labelled fields not detected |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | duplicate keywords           |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                         |
|-------------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------|
| ibm-granite/granite-vision-4.1-4b                           | rev 37d591f06319; Granite4VisionProcessor; stop completed    | 721 prompt / 55 generated; 68.0 tok/s; 9.7 GB peak  |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 350 prompt / 47 generated; 532 tok/s; 1.3 GB peak   |
| mlx-community/AREX-2-4bit                                   | rev 02551ee54839; Qwen3VLProcessor; stop completed           | 421 prompt / 80 generated; 33.3 tok/s; 17 GB peak   |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2909 prompt / 82 generated; 98.9 tok/s; 6.5 GB peak |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 386 prompt / 72 generated; 37.4 tok/s; 28 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 390 prompt / 88 generated; 62.0 tok/s; 7.5 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 390 prompt / 88 generated; 110 tok/s; 16 GB peak    |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 386 prompt / 67 generated; 122 tok/s; 5.9 GB peak   |
| mlx-community/GLM-4.6V-Flash-4bit                           | rev bd7b20686e8c; Glm46VProcessor; stop completed            | 499 prompt / 49 generated; 86.2 tok/s; 8.0 GB peak  |
| mlx-community/granite-4.0-3b-vision-4bit                    | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed    | 734 prompt / 58 generated; 171 tok/s; 4.7 GB peak   |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 3435 prompt / 90 generated; 56.1 tok/s; 10 GB peak  |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                       | rev b6311011d95c; Lfm2VlProcessor; stop completed            | 344 prompt / 79 generated; 217 tok/s; 3.6 GB peak   |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 177 prompt / 99 generated; 106 tok/s; 7.0 GB peak   |
| mlx-community/MiniCPM-V-4.6-4bit                            | rev 86cd463d33a9; MiniCPMVProcessor; stop completed          | 330 prompt / 331 generated; 246 tok/s; 3.1 GB peak  |
| mlx-community/MiniCPM-V-4.7-1B-4bit                         | rev 13caa569f70e; MiniCPMV4_7Processor; stop completed       | 332 prompt / 66 generated; 251 tok/s; 3.6 GB peak   |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit                    | rev 0e3b706c1599; MiniCPMV4_7Processor; stop completed       | 332 prompt / 92 generated; 75.6 tok/s; 24 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 1070 prompt / 72 generated; 69.6 tok/s; 9.8 GB peak |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 1069 prompt / 102 generated; 201 tok/s; 4.5 GB peak |
| mlx-community/Mistral-Small-4-119B-2603-4bit                | rev 87b124920773; Mistral3Processor; stop completed          | 549 prompt / 82 generated; 81.0 tok/s; 70 GB peak   |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 873 prompt / 95 generated; 73.4 tok/s; 7.9 GB peak  |
| mlx-community/MolmoPoint-8B-4bit                            | rev 9bab196f867c; MolmoPointProcessor; stop completed        | 860 prompt / 70 generated; 33.0 tok/s; 12 GB peak   |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit                   | rev 865ee8de6579; MuseGlimmerProcessor; stop completed       | 545 prompt / 786 generated; 25.5 tok/s; 25 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 418 prompt / 78 generated; 160 tok/s; 3.5 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16                  | rev d8da684308c2; Phi3VProcessor; stop completed             | 883 prompt / 94 generated; 58.8 tok/s; 9.1 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 409 prompt / 81 generated; 79.6 tok/s; 23 GB peak   |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 409 prompt / 83 generated; 85.0 tok/s; 19 GB peak   |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 409 prompt / 60 generated; 106 tok/s; 6.8 GB peak   |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 421 prompt / 88 generated; 71.5 tok/s; 21 GB peak   |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 421 prompt / 109 generated; 31.9 tok/s; 17 GB peak  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx                    | rev 844516024a1c; SmolVLMProcessor; stop completed           | 1204 prompt / 31 generated; 131 tok/s; 5.6 GB peak  |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 420 prompt / 89 generated; 147 tok/s; 4.9 GB peak   |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 538 prompt / 87 generated; 36.4 tok/s; 17 GB peak   |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 421 prompt / 65 generated; 138 tok/s; 21 GB peak    |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit       | rev 4bc0a247d78e; NemotronHNanoOmniProcessor; stop completed | 420 prompt / 101 generated; 61.1 tok/s; 46 GB peak  |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4         | rev da5b2650a835; NemotronHNanoOmniProcessor; stop completed | 420 prompt / 89 generated; 29.7 tok/s; 81 GB peak   |
| sahilchachra/LensVLM-9B-MXFP4                               | rev 23ae80ae9a7d; Qwen3VLProcessor; stop completed           | 421 prompt / 111 generated; 103 tok/s; 6.8 GB peak  |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 421 prompt / 94 generated; 39.9 tok/s; 11 GB peak   |

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

| Model                                 | Resolved revision                        |
|---------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-1B-4bit     | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/X-Reasoner-7B-8bit      | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | 9c056d48b1e611dc586139a5deb927ae363cfe6f |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.10                                                                                                                                          |
| mlx-vlm source revision    | e9af376626b04ba0cdad23b1476fd7a4a583d721                                                                                                        |
| mlx                        | 0.32.5.dev20261024+56c2bf923                                                                                                                    |
| mlx source revision        | 56c2bf923                                                                                                                                       |
| mlx-audio                  | 0.5.9                                                                                                                                           |
| transformers               | 5.20.2                                                                                                                                          |
| tokenizers                 | 0.23.3                                                                                                                                          |
| huggingface-hub            | 2.3.0                                                                                                                                           |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (224,343,168 bytes, sha256=1bfc30ec84cdeacdf24e71dfa7119d61d3dbd519d30be82d408464a39ab9ba3d) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,095,936 bytes, sha256=072aab1e9a582db9da0421ec6d7203573aa3ed3105b55aa4305aa439bf2a3815)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
