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
- *Human accuracy notes:* 1 of 41 outputs reviewed by a person for this exact
  image, model revision and output (accuracy_notes.yaml); the others are
  unreviewed

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 41    |
| Conclusive outcomes | 41    |
| Completed           | 41    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 38    |
| observation needs reproduction | 3     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 6     |
| no concerns detected | 34    |
| concerns detected    | 1     |

Observation counts

| Observation                                      | Count |
|--------------------------------------------------|-------|
| Response repeats the same text                   | 1     |
| Unrecognised model control tokens remain visible | 2     |
| Required labelled fields not detected            | 5     |
| Response appears cut off at the token limit      | 2     |
| Conversation-role control tokens remain visible  | 1     |
| Repeated keyword entries                         | 2     |

## Triage

| Model                                                                                              | Execution | Mechanical checks | Maintainer status              | Observations                                                                                      |
|----------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|---------------------------------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)                  | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit; duplicate keywords                                         |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)         | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected                                              |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible |

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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.367
- *Peak memory at first token (GB):* 2.304
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
  sample(s); power: AC over 2 sample(s); thermal state max nominal over 2
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
Title: Two cats sleeping on a pink couch

Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a remote control.

Keywords: cats, sleeping, pink couch, remote control, striped, British English, British, British-English, British-English catalogue, British-English metadata, British-English details, British-English context, British-English information, British-English data, British-English content, British-English examples, British-English examples of metadata, British-English examples of keywords, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata
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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.255
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
  sample(s); power: AC over 2 sample(s); thermal state max nominal over 2
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The two cats are sleeping on a pink couch.
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
- *Keyword count:* 9
- *Token-cap degradation evidence:* ["missing_sections"]
- *Special tokens emitted (by token id):* ["&lt;|message|&gt;"]
- *Resolved model revision:* 865ee8de65799cc4ee406587e4b607063708f358
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.869
- *Peak memory at first token (GB):* 24.854
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
- *Prompt tokens:* 545
- *Prompt composition:* 545 = 154 text/template + 391 image tokens (72%;
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
  sample(s); power: AC over 2 sample(s); thermal state max nominal over 2
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
 to=self<|message|>Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need British-English. Use only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

We see two cats? Possibly tabby cats. Species? Cat is visible. Is species certain? Probably cats. It's clear visible facts. Might be cats. Species is cat. Might be okay. But "species" maybe uncertain? It's clearly cats. Should be okay.

Don't specify breed. Don't say brand of remote. Don't say location. Don't say period.

Title: 5-10 words concrete.

Maybe "Two Tabby Cats Resting on Pink Blanket". That's 7 words? Two(1) Tabby(2) Cats(3) Resting(4) on(5) Pink(6) Blanket(7). Good.

Description: 1-2 sentence factual description of main subject, setting, action, lighting, and distinctive details.

Main subject: two cats on pink blanket on red sofa? There's pink blanket on red couch? Actually pink blanket on red cushions. Two cats lying. One sleeping with head down, one looking down. Two remote controls. Lighting? Bright daylight? Seems indoor natural light. Distinctive details: one cat has green collar? Looks like green collar on left cat. Remote controls.

We must omit uncertain details. Lighting: maybe bright indoor. Can say bright lighting. Okay.

Don't assume species? It's cats. Might be safe.

Make description: Two tabby cats lie on a bright pink blanket on a red sofa, one appears asleep and one is looking down. Two white remote controls rest on the blanket beside them. The scene is lit brightly.

That's two sentences? "Two tabby cats lie..." That's one sentence. Second sentence: "Two white remote controls rest..." That's second. Okay.

Keywords: 10-18 unique comma-separated keywords.

Keywords: cats, tabby, pink blanket, red sofa, remote controls, sleeping, resting, indoor, domestic, furniture, textile, pet, animals

Need 10-18 unique. Let's list maybe 13.

Make sure British English: "colour"? Not used. "favourite"? No.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

Make sure format exactly.

Title: ...
Description: ...
Keywords: ...

No extra text.

Ensure title 5-10 words. Let's count.

Two Tabby Cats Resting on Pink Blanket = 7 words. Good.

Description: Must be 1-2 sentence. Ensure factual. Don't say "appears asleep" maybe uncertain? Could be sleeping. Head down eyes closed? Hard to see. Might be sleeping. Could say one cat lies with head down, one cat is oriented head down. Safer to say one cat lies with head down, eyes closed? Might be uncertain. Better say one cat lies with head down. Could mention one cat with green collar. Is collar visible? Yes green band around neck. Distinctive detail.

Maybe: Two tabby cats lie on a bright pink blanket atop a red upholstered surface. One cat rests with its head down and a green collar visible, the other lies with its head lowered towards the blanket, and two white remote controls are placed on the fabric. The lighting is bright.

That's a bit long, maybe two sentences. Let's keep two sentences.

First sentence: Two tabby cats lie on a bright pink blanket on a red upholstered surface. Second sentence: One cat rests with its head down and a green collar visible, the other is positioned head-down towards the blanket, and two white remote controls lie on the fabric. Lighting is bright.

That is maybe too many clauses. Might be okay.

Simplify: Two tabby cats lie on a bright pink blanket on a red upholstered surface. One cat rests with its head down and a green collar visible, the other is positioned head-down, with two white remote controls on the blanket. The lighting is bright.

That's three sentences? Let's count. Sentence 1: Two tabby cats lie on a bright pink blanket on a red upholstered surface. Sentence 2: One cat rests with its head down and a green collar visible, the other is positioned head-down, with two white remote controls on the blanket. Sentence 3: The lighting is bright. That's three.

Need 1-2 sentence. Let's combine.

Two tabby cats lie
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
| mlx-community/Qwen3.6-35B-A3B-4bit               | major concerns    | labelled fields not detected |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | duplicate keywords           |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                         |
|-------------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------|
| ibm-granite/granite-vision-4.1-4b                           | rev 37d591f06319; Granite4VisionProcessor; stop completed    | 721 prompt / 55 generated; 65.4 tok/s; 9.7 GB peak  |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 350 prompt / 47 generated; 516 tok/s; 1.3 GB peak   |
| mlx-community/AREX-2-4bit                                   | rev 02551ee54839; Qwen3VLProcessor; stop completed           | 421 prompt / 80 generated; 33.0 tok/s; 17 GB peak   |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2909 prompt / 82 generated; 100 tok/s; 6.5 GB peak  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 386 prompt / 66 generated; 66.9 tok/s; 28 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 390 prompt / 88 generated; 62.8 tok/s; 7.5 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 390 prompt / 75 generated; 126 tok/s; 16 GB peak    |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 386 prompt / 67 generated; 125 tok/s; 5.9 GB peak   |
| mlx-community/GLM-4.6V-Flash-4bit                           | rev bd7b20686e8c; Glm46VProcessor; stop completed            | 499 prompt / 49 generated; 85.7 tok/s; 8.0 GB peak  |
| mlx-community/granite-4.0-3b-vision-4bit                    | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed    | 734 prompt / 57 generated; 181 tok/s; 4.7 GB peak   |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 3435 prompt / 90 generated; 55.4 tok/s; 10 GB peak  |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                       | rev b6311011d95c; Lfm2VlProcessor; stop completed            | 344 prompt / 73 generated; 214 tok/s; 3.7 GB peak   |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 177 prompt / 99 generated; 106 tok/s; 7.0 GB peak   |
| mlx-community/MiniCPM-V-4.7-1B-4bit                         | rev 13caa569f70e; MiniCPMV4_7Processor; stop completed       | 332 prompt / 64 generated; 314 tok/s; 3.6 GB peak   |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit                    | rev 0e3b706c1599; MiniCPMV4_7Processor; stop completed       | 332 prompt / 93 generated; 86.8 tok/s; 24 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 1070 prompt / 81 generated; 68.7 tok/s; 8.9 GB peak |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 1069 prompt / 102 generated; 201 tok/s; 3.7 GB peak |
| mlx-community/Mistral-Small-4-119B-2603-4bit                | rev 87b124920773; Mistral3Processor; stop completed          | 549 prompt / 82 generated; 90.2 tok/s; 69 GB peak   |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 873 prompt / 95 generated; 73.2 tok/s; 8.2 GB peak  |
| mlx-community/MolmoPoint-8B-4bit                            | rev 9bab196f867c; MolmoPointProcessor; stop completed        | 860 prompt / 70 generated; 32.6 tok/s; 12 GB peak   |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 418 prompt / 78 generated; 246 tok/s; 3.5 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16                  | rev d8da684308c2; Phi3VProcessor; stop completed             | 883 prompt / 94 generated; 59.8 tok/s; 9.1 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 409 prompt / 81 generated; 94.6 tok/s; 23 GB peak   |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 409 prompt / 83 generated; 129 tok/s; 19 GB peak    |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 409 prompt / 60 generated; 103 tok/s; 6.8 GB peak   |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 421 prompt / 88 generated; 113 tok/s; 21 GB peak    |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 421 prompt / 109 generated; 33.0 tok/s; 17 GB peak  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx                    | rev 844516024a1c; SmolVLMProcessor; stop completed           | 1204 prompt / 31 generated; 125 tok/s; 5.6 GB peak  |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 420 prompt / 88 generated; 150 tok/s; 5.3 GB peak   |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 538 prompt / 87 generated; 36.3 tok/s; 16 GB peak   |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 421 prompt / 65 generated; 159 tok/s; 21 GB peak    |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4         | rev da5b2650a835; NemotronHNanoOmniProcessor; stop completed | 420 prompt / 89 generated; 30.9 tok/s; 81 GB peak   |
| sahilchachra/LensVLM-9B-MXFP4                               | rev 23ae80ae9a7d; Qwen3VLProcessor; stop completed           | 421 prompt / 111 generated; 101 tok/s; 6.8 GB peak  |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 421 prompt / 94 generated; 43.4 tok/s; 11 GB peak   |

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

| Model                                     | Resolved revision                        |
|-------------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-1B-4bit         | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit     | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | 865ee8de65799cc4ee406587e4b607063708f358 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.8.0                                                                                                                                           |
| mlx-vlm source revision    | 2bb1292351fa2e93231d2bca1b0963e80e117ee5                                                                                                        |
| mlx                        | 0.32.5.dev20261107+fc344c986                                                                                                                    |
| mlx source revision        | fc344c986                                                                                                                                       |
| mlx-audio                  | 0.6.5                                                                                                                                           |
| transformers               | 5.22.2                                                                                                                                          |
| tokenizers                 | 0.23.3                                                                                                                                          |
| huggingface-hub            | 2.4.0                                                                                                                                           |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 27.2.0                                                                                                                                   |
| macOS Version              | 27.2.1                                                                                                                                          |
| SDK Version                | 27.2                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk                                              |
| Xcode Version              | 27.2                                                                                                                                            |
| Xcode Build                | 27B52                                                                                                                                           |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX27.2.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.922 (metalfe-32023.922.4)                                                                                             |
| Metallib Linker Version    | AIR-LLD 32023.922 (metalfe-32023.922.4) (compatible with legacy metallib linker)                                                                |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.4.1.1)                                                                                                   |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (225,677,776 bytes, sha256=e9560b19b6699a87c87ce8f26fb6cdc5e4af170a9b1a1e91dbba401ca1a382ec) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,155,648 bytes, sha256=44513f3f0be9644563e802ab96880b4db43337c3222f2be50aaf2d48a62ce3e6)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
