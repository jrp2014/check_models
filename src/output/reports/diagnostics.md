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
- *Input image:* JPEG, 9,805 x 6,538 pixels (64.1 MP), 54.2 MB

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
| none                           | 47    |
| observation needs reproduction | 3     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 9     |
| no concerns detected | 27    |
| concerns detected    | 14    |

Observation counts

| Observation                                                               | Count |
|---------------------------------------------------------------------------|-------|
| Response repeats the same text                                            | 1     |
| Unrecognised model control tokens remain visible                          | 2     |
| Required labelled fields not detected                                     | 9     |
| Response appears cut off at the token limit                               | 2     |
| Internal reasoning block appears incomplete                               | 1     |
| Conversation-role control tokens remain visible                           | 1     |
| Repeated keyword entries                                                  | 4     |
| Output repeats the prompt's own hint text instead of describing the image | 11    |
| Names a place the prompt did not supply                                   | 1     |

## Triage

| Model                                                                                                    | Execution | Mechanical checks | Maintainer status              | Observations                                                                                                       |
|----------------------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [mlx-community/Kimi-VL-A3B-Thinking-2506-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) | completed | major concerns    | observation needs reproduction | repeated text; labelled fields not detected; cut off at token limit; incomplete thinking block; duplicate keywords |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit)               | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected                                                               |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit)       | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible                  |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-8bit — major concerns — repeated text; labelled fields not detected; cut off at token limit; incomplete thinking block; duplicate keywords</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-8bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, thinking_trace_incomplete, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["title"]
- *Repeated fragment:* keyword: "tan canopy"
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Keyword count:* 100
- *Keywords taken verbatim from the prompt's keyword hints:* 5
- *Duplicate keywords:* ["dock", "marina", "calm water", "reflections", "early
  evening", "blue hull", "tan canopy", "blue and light blue", "marina dock",
  "leisure boats", "early evening light", "blue and light blue hull", "other
  sailboats", "correct conflicts", "moored"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "incomplete_thinking_trace"]
- *Resolved model revision:* e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.455
- *Peak memory at first token (GB):* 20.067
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 0.6
- *Sampling settings source:* temperature: generation_config; top_p: default;
  top_k: default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.003
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1331
- *Prompt composition:* 1,331 = 317 text/template + 1,014 image tokens (76%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 17.83
- *Active parameter count:* 3.00B (name-estimate; total not stated in the
  name)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (17.83 GB vs 17.83 GB on disk)
- *Generation tokens:* 1000
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max nominal over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Okay, let's tackle this task step by step. First, I need to create a British-English catalogue metadata based on the provided context and image. The key points from the context are the capture date/time (2026-09-26 17:36:29 UTC+01:00) and GPS coordinates (50.682100°N, 3.466600°W). The descriptive hints mention a blue and light blue motor cabin cruiser with a tan canopy at a marina, with reflections, early evening light, calm water, and other boats in the background.

The image shows a blue and light blue motor cabin cruiser with a tan canopy, docked at a marina. There are other sailboats and leisure boats around, calm water with reflections, and the lighting suggests early evening. The boat has a tan canopy, blue hull, and is moored at a dock. The background includes other vessels and calm water reflections.

Now, for the title: needs to be 5-10 words. Maybe something like "Motor cabin cruiser moored at marina in early evening light."

Description: Combine the context and visible details. Mention the boat type, location (marina), time of day (early evening), and distinctive features (tan canopy, blue hull, reflections). So, "A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light."

Keywords: Need 10-18 unique, comma-separated. Include boat types (motor cabin cruiser, sailboat, leisure boat, dock, marina, calm water, reflections, early evening, blue hull, tan canopy, blue and light blue), plus the GPS and date/time. Wait, the user said to treat capture date/time and GPS as authoritative but not claim they are visible. So maybe exclude them from keywords. Focus on visible details. So keywords: Motor cabin cruiser, blue hull, light blue, tan canopy, marina dock, sailboats, leisure boats, calm water, reflections, early evening, blue and light blue, blue hull, tan canopy, marina, dock, other boats, water reflection.

Wait, need to check if "blue and light blue" is redundant. Maybe combine. Also, "motor cabin cruiser" and "cruise" or "motorboat". Let's list the visible unique elements: blue and light blue hull, tan canopy, marina dock, other boats (sailboats, leisure boats), calm water, reflections, early evening light, blue hull, tan canopy, dock, marina. Maybe 10-18. Let's count: Motor cabin cruiser, blue and light blue hull, tan canopy, marina dock, other sailboats, leisure boats, calm water, reflections, early evening light, blue hull, tan canopy, dock, marina, calm water, other boats. That's 13. Maybe add "moored" as a keyword? But the hint says "moored" is already in hints. Wait the keywords from hints include "Mooring" but the user said to retain details supported by the image, correct conflicts, omit uncertain. The image shows the boat moored, so "moored" is visible. But the hints include "Mooring" so maybe include that. Wait the user said to prefer image evidence over hints. The image shows the boat is moored, so include "moored". But the hints might have "Mooring" which is similar. Let's check the hints: the hints include "Mooring" so maybe include that. But the user says to prefer image evidence. The image shows the boat is moored, so "moored" is visible. So include "moored". But the hints have "Mooring" – perhaps the user wants to use the hint's terms. Wait, the instructions say "retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details." So if the image shows the boat is moored, then "moored" is a visible detail, so include it. But the hints have "Mooring" which is similar. Maybe use "moored" instead. Let's list keywords: Motor cabin cruiser, blue and light blue hull, tan canopy, marina dock, other sailboats, leisure boats, calm water, reflections, early evening light, moored, blue hull, tan canopy, marina, dock, calm water, other boats. Wait, need to make sure 10-18. Let's count: Motor cabin cruiser, blue and light blue hull, tan canopy, marina dock, other sailboats, leisure boats, calm water, reflections, early evening light, moored, blue hull, tan canopy, marina, dock, calm water, other boats. That's 14. Maybe combine "blue hull" and "blue and light
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
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.607
- *Peak memory at first token (GB):* 6.729
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.015
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2201
- *Prompt composition:* 2,201 = 409 text/template + 1,792 image tokens (81%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.70 GB vs 5.68 GB on disk)
- *Generation tokens:* 18
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max nominal over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a blue and white boat with a tan colored canopy.
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
- *Keyword count:* 20
- *Keywords taken verbatim from the prompt's keyword hints:* 0
- *Token-cap degradation evidence:* ["missing_sections", "unfinished_list"]
- *Special tokens emitted (by token id):* ["&lt;|message|&gt;"]
- *Resolved model revision:* 98377360cbc84f982e90336f956b08adb46cad88
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 7.844
- *Peak memory at first token (GB):* 25.464
- *Checkpoint-declared sampling (generation_config.json):* do_sample True;
  temperature 1.0; top_p 0.95; top_k 64
- *Sampling settings source:* temperature: generation_config; top_p:
  generation_config; top_k: generation_config; min_p: default;
  repetition_penalty: default
- *Post-cleanup active memory (GB):* 0.006
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4410
- *Prompt composition:* 4,410 = 354 text/template + 4,056 image tokens (92%;
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
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); mode snapshot

#### Complete output

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-26 17:36:29 UTC+01:00
- GPS: 50.682100°N, 3.466600°W

Descriptive hints:
- Description hint: A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light.
- Keyword hints: Boat, Boat canopy, Boat fender, Boating, Cabin cruiser, Calm Water, Dock, Harbor, Marina, Mast, Mooring, Motorboat, Nautical, Reflection, Rope, Sailboat, Sailing, Water reflection, Watercraft, Yacht

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

We need title 5-10 words. Concrete.

Description 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Include capture date/time? It says treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. So we can mention? Possibly include in description? Might be metadata. Probably include date/time? The description should combine relevant context with main visible subject... Could mention captured on 2026-09-26. Could include GPS? Maybe not necessary. Probably mention location? GPS is 50.682100°N, 3.466600°W. That's near... maybe Exmouth? That's Devon. Could just say at marina. Might be okay to include date.

Do not claim they are visible. So we cannot say "the image shows date". We can say "captured on...". That's fine.

Distinctive details: boat named EVENHAZE, blue hull with red boot stripe, light blue superstructure, tan bimini canopy, blue fenders, moored at wooden dock, sailboats behind, calm water reflection, late afternoon light.

Early evening light hint. Capture time 17:36:29 UTC+01:00 = 5:36pm, so late afternoon/early evening.

Title: 5-10 words. Eg "Blue Cabin Cruiser Evenhaze Moored at Marina Dock". That's 7 words? Blue(1) Cabin(2) Cruiser(3) Evenhaze(4) Moored(5) at(6) Marina(7) Dock(8). Good.

Description: 1-2 sentence. Should be factual. Include capture date/time and GPS? Might be relevant context. Could say "Captured on 2026-09-26 at 17:36:29 UTC+01:00 at 50.682100°N, 3.466600°W, the motor cabin cruiser Evenhaze..." That might be okay. Not claiming visible.

Make sure British-English: moor, etc. Use British spelling.

Keywords: 10-18 unique comma-separated. Use hints but also add visible details: Evenhaze, blue hull, red boot stripe, tan canopy, etc. Use keywords list.

Let's craft.

Title: Blue Cabin Cruiser Evenhaze Moored at Marina Dock

That's 8 words.

Description: Captured on 2026-09-26 at 17:36:29 UTC+01:00 at 50.682100°N, 3.466600°W, the blue-hulled motor cabin cruiser Evenhaze with a tan sprayhood is moored alongside a wooden pontoon in a busy marina. It sits among sailboats and leisure craft on calm water in late afternoon light, with clear reflections and blue fenders visible along its hull.

That's two sentences. Good.

Keywords: maybe 13-15 items.

Boat, Motor cabin cruiser, Evenhaze, Blue hull, Tan canopy, Marina, Dock, Mooring, Sailboat, Calm water, Reflection, Fenders, Pontoon, Nautical

Need comma-separated. Unique.

Ensure 10-18. Let's produce 15.

Keywords: Cabin cruiser, Motorboat, Evenhaze, Blue hull, Red boot stripe, Tan sprayhood, Marina, Dock, Pontoon, Mooring, Sailboat, Calm water, Water
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Mechanical checks | Observations                             |
|-------------------------------------------------------|-------------------|------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                       | major concerns    | labelled fields not detected             |
| mlx-community/gemma-3n-E4B-it-4bit                    | major concerns    | labelled fields not detected             |
| mlx-community/granite-vision-3.2-2b-nvfp4             | major concerns    | labelled fields not detected             |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                 | major concerns    | labelled fields not detected             |
| mlx-community/nanoLLaVA-1.5-4bit                      | major concerns    | labelled fields not detected             |
| mlx-community/SmolVLM-256M-Instruct-4bit              | major concerns    | labelled fields not detected             |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | concerns detected | duplicate keywords; prompt hint repeated |
| mlx-community/gemma-4-e4b-it-4bit                     | concerns detected | duplicate keywords                       |
| mlx-community/X-Reasoner-7B-8bit                      | concerns detected | duplicate keywords                       |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | prompt hint repeated                     |
| mlx-community/GLM-4.6V-Flash-4bit                     | concerns detected | prompt hint repeated                     |
| mlx-community/GLM-4.6V-nvfp4                          | concerns detected | prompt hint repeated                     |
| mlx-community/Idefics3-8B-Llama3-bf16                 | concerns detected | prompt hint repeated                     |
| mlx-community/Phi-3.5-vision-instruct-bf16            | concerns detected | prompt hint repeated                     |
| mlx-community/pixtral-12b-8bit                        | concerns detected | prompt hint repeated                     |
| mlx-community/Qwen2-VL-7B-Instruct-4bit               | concerns detected | prompt hint repeated                     |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | concerns detected | prompt hint repeated                     |
| mlx-community/Step-3.7-Flash-oQ3e                     | concerns detected | prompt hint repeated                     |
| nativ-community/Mage-VL-OptiQ-4bit                    | concerns detected | prompt hint repeated                     |
| mlx-community/gemma-3-27b-it-qat-4bit                 | concerns detected | unsupplied place name                    |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                           |
|-------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------|
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2098 prompt / 143 generated; 102 tok/s; 6.5 GB peak   |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 597 prompt / 81 generated; 57.2 tok/s; 28 GB peak     |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit            | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed       | 1646 prompt / 540 generated; 108 tok/s; 19 GB peak    |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 601 prompt / 112 generated; 61.2 tok/s; 7.6 GB peak   |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 601 prompt / 106 generated; 110 tok/s; 16 GB peak     |
| mlx-community/gemma-4-31b-it-4bit                           | rev 696d436c4047; Gemma4Processor; stop completed            | 601 prompt / 113 generated; 26.9 tok/s; 20 GB peak    |
| mlx-community/granite-4.0-3b-vision-4bit                    | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed    | 1384 prompt / 91 generated; 174 tok/s; 4.6 GB peak    |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 2120 prompt / 103 generated; 52.1 tok/s; 10 GB peak   |
| mlx-community/InternVL3-8B-bf16                             | rev e0df3dd79263; InternVLChatProcessor; stop completed      | 2120 prompt / 79 generated; 35.2 tok/s; 17 GB peak    |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit            | rev 8451adc50203; MllamaProcessor; stop completed            | 309 prompt / 114 generated; 20.9 tok/s; 15 GB peak    |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 398 prompt / 93 generated; 102 tok/s; 7.0 GB peak     |
| mlx-community/MiniCPM-V-4.6-4bit                            | rev 86cd463d33a9; MiniCPMVProcessor; stop completed          | 938 prompt / 564 generated; 237 tok/s; 3.3 GB peak    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 2935 prompt / 199 generated; 63.8 tok/s; 13 GB peak   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 2934 prompt / 121 generated; 175 tok/s; 7.8 GB peak   |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 1531 prompt / 181 generated; 70.1 tok/s; 8.6 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 4091 prompt / 108 generated; 151 tok/s; 3.9 GB peak   |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit                 | rev 4620fdbbd1e7; Qwen3VLProcessor; stop completed           | 1295 prompt / 130 generated; 79.8 tok/s; 24 GB peak   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16                     | rev c325e5ea14c2; Qwen3VLProcessor; stop completed           | 16556 prompt / 901 generated; 84.5 tok/s; 8.4 GB peak |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 16554 prompt / 168 generated; 75.0 tok/s; 23 GB peak  |
| mlx-community/Qwen3-VL-32B-Instruct-4bit                    | rev 6e5644d3ea4b; Qwen3VLProcessor; stop completed           | 16554 prompt / 190 generated; 20.5 tok/s; 26 GB peak  |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 16554 prompt / 120 generated; 69.1 tok/s; 11 GB peak  |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 16569 prompt / 123 generated; 73.7 tok/s; 25 GB peak  |
| mlx-community/Qwen3.5-9B-MLX-4bit                           | rev 938d8919941c; Qwen3VLProcessor; stop completed           | 16569 prompt / 111 generated; 89.3 tok/s; 11 GB peak  |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 16569 prompt / 128 generated; 29.4 tok/s; 21 GB peak  |
| nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit          | rev 3ea706a5e7b8; Qwen3VLProcessor; stop completed           | 16566 prompt / 123 generated; 87.4 tok/s; 11 GB peak  |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 1281 prompt / 144 generated; 35.9 tok/s; 18 GB peak   |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 3636 prompt / 120 generated; 135 tok/s; 23 GB peak    |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,805 x 6,538 pixels
- *Image size:* 54,173,041 bytes
- *Image SHA-256:* bd1c0e60ad08718e500069551a1ac0d3697010373ec1509d4f76cd6e1d79b11a

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-26 17:36:29 UTC+01:00
- GPS: 50.682100°N, 3.466600°W

Descriptive hints:
- Description hint: A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light.
- Keyword hints: Boat, Boat canopy, Boat fender, Boating, Cabin cruiser, Calm Water, Dock, Harbor, Marina, Mast, Mooring, Motorboat, Nautical, Reflection, Rope, Sailboat, Sailing, Water reflection, Watercraft, Yacht

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-6b2dcf5e8c73798e.jpg>
- *Preview dimensions:* 1,024 x 683 pixels
- *Preview size:* 144,352 bytes
- *Preview SHA-256:* 6b2dcf5e8c73798e30387b33b07879805862c4f195e39570744d3d5f570f4e96

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-6b2dcf5e8c73798e.jpg
printf '%s\n' '6b2dcf5e8c73798e30387b33b07879805862c4f195e39570744d3d5f570f4e96  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-26 17:36:29 UTC+01:00
- GPS: 50.682100°N, 3.466600°W

Descriptive hints:
- Description hint: A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light.
- Keyword hints: Boat, Boat canopy, Boat fender, Boating, Cabin cruiser, Calm Water, Dock, Harbor, Marina, Mast, Mooring, Motorboat, Nautical, Reflection, Rope, Sailboat, Sailing, Water reflection, Watercraft, Yacht

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
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | e5abbe34cbfabd829fafd0362856e5b468d19f85 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit        | 9c056d48b1e611dc586139a5deb927ae363cfe6f |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit    | 98377360cbc84f982e90336f956b08adb46cad88 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.3                                                                                                                                           |
| mlx-vlm source revision    | 967bf90b8e7ae6e5110b6da87134f30d94591d2e                                                                                                        |
| mlx                        | 0.32.3.dev20260927+02ce1fb6a                                                                                                                    |
| mlx source revision        | 02ce1fb6a                                                                                                                                       |
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
| Recommended Working Set    | 115 GB                                                                                                                                          |
| Fused Attention            | Available                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-314-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (192,433,568 bytes, sha256=5088f85ae8a394d4b426a9b8604cc0850c4ea6d993725bb2f7963315a614ba9f) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,851,904 bytes, sha256=3492c73dc32ae63685cd78b58469d470a9b923babb1acf81862313a134efbc08)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
