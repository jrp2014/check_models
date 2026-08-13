# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

## Run Summary

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
| none                           | 31    |
| observation needs reproduction | 10    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| unusable            | 13    |
| usable              | 9     |
| usable with caveats | 19    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 3     |
| Unrecognised model control tokens remain visible                                      | 3     |
| Required fields are missing or empty                                                  | 10    |
| Response repeats the task instructions instead of only returning the requested fields | 3     |
| Extra text appears before the Title field                                             | 4     |
| Response appears cut off at the token limit                                           | 8     |
| Internal reasoning block appears incomplete                                           | 3     |
| Conversation-role control tokens remain visible                                       | 1     |
| Title or keywords do not meet requested constraints                                   | 17    |
| Title, Description and Keywords copy all supplied hints unchanged                     | 1     |

## Triage

| Model                                                                                                      | Execution | Usability           | Maintainer status              | Observations                                                                                        |
|------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|-----------------------------------------------------------------------------------------------------|
| [mlx-community/GLM-4.6V-Flash-mxfp4](#diagnostic-mlx-community-glm-46v-flash-mxfp4)                        | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)             | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; echoes instructions; cut off at token limit                 |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                           | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                             |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)   | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible; title/keyword constraints failed                                            |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                              |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                    | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                              |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)   | completed | unusable            | observation_needs_reproduction | missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)             | completed | unusable            | observation_needs_reproduction | missing required fields; cut off at token limit; incomplete thinking block                          |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)              | completed | unusable            | observation_needs_reproduction | extra text before Title; cut off at token limit; incomplete thinking block                          |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                 | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                                 |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-glm-46v-flash-mxfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-mxfp4 — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/GLM-4.6V-Flash-mxfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Repeated fragment:* keyword: "pier"
- *Title word count:* 2
- *Requested title word range:* [5, 10]
- *Keyword count:* 146
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["pier", "industrial cranes", "street lamps",
  "waterfront", "bright daylight", "pier view", "bright daylight scene",
  "street lamp view", "waterfront trees", "industrial cranes view",
  "waterfront view", "trees"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.002114608
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6310
- *Generation tokens:* 500
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
Title: Felixstowe Seafront
Description: A view of the Felixstowe seafront, featuring a pier, industrial cranes, street lamps, and a tree-lined waterfront under bright daylight.
Keywords: Felixstowe seafront, England UK, pier, industrial cranes, street lamps, waterfront, bright daylight, East Suffolk, Suffolk, UK, gbr, Europe, seafront view, pier view, industrial waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial cranes, waterfront, bright daylight scene, street lamp view, waterfront trees, industrial cranes view, pier view, waterfront view, bright daylight, street lamps, trees, pier, industrial
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated text; missing required fields; echoes instructions; cut off at token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, prompt_instruction_echo
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "felixstowe, suffolk, uk, gbr,..."
- *Echoed instruction fragments:* ["title hint:", "description hint:",
  "keyword hints:"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "dangling_markdown"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.01414068
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4397
- *Generation tokens:* 500
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, England, UK, GBR,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
- Title hint: Seafront, Felixstowe, England, UK, GBR,
- Description hint: Seafront, Felixstowe, Suffolk, UK, GBR,
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England,
-
```

</details>

<a id="diagnostic-mlx-community-x-reasoner-7b-8bit"></a>

<details>
<summary>mlx-community/X-Reasoner-7B-8bit — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/X-Reasoner-7B-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen2_5_vl)
- *Repeated fragment:* keyword: "uk waterfront"
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 122
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["uk", "europe", "gbr", "uk seafront", "uk
  waterfront", "uk ferris wheel", "uk cranes", "uk sunny day", "uk east
  suffolk", "uk felixstowe", "uk east anglia", "uk industrial port"]
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.010290348
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16478
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title: Felixstowe Seafront, East Suffolk, UK

Description: A sunny view of Felixstowe's seafront, featuring a pier, Ferris wheel, cranes, and decorative string lights, with a lamp post in the foreground and a person paddleboarding in the distance.

Keywords: Felixstowe, East Suffolk, England, UK, Europe, seafront, pier, Ferris wheel, cranes, string lights, lamp post, paddleboarding, sunny, waterfront, harbor, industrial, coastal, summer, vacation, travel, landscape, scenic, scenic view, waterfront activity, East Anglia, GBR, United Kingdom, GBR, Great Britain, Europe, British, English, UK, GBR, sea, water, blue, sky, trees, rocks, rocks by the sea, waterfront promenade, East Suffolk coastline, East Anglia seafront, UK seafront, UK waterfront, UK coastal scene, UK industrial landscape, UK scenic view, UK travel destination, UK summer scene, UK waterfront activity, UK Ferris wheel, UK cranes, UK lamp post, UK string lights, UK sunny day, UK paddleboarding, UK waterfront, UK seafront, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK Ferris wheel, UK cranes, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day, UK East Suffolk, UK Felixstowe, UK East Anglia, UK industrial port, UK waterfront, UK sunny day,
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-8bit — usable_with_caveats — control tokens visible; title/keyword constraints failed</summary>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Unexpected special tokens:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Thinking trace markers:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Title word count:* 8
- *Requested title word range:* [5, 10]
- *Keyword count:* 16
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["travel"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.010290368
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 576
- *Generation tokens:* 86
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Seafront View and Industrial Port in Felixstowe Suffolk
Description: A view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel in the distance under a clear sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, port, cranes, pier, Ferris wheel, coastal, maritime, water, East Suffolk, industrial, travel, travel
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-mxfp8 — usable_with_caveats — control tokens visible</summary>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Unexpected special tokens:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Thinking trace markers:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.010290368
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 576
- *Generation tokens:* 81
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Seafront view of Port of Felixstowe England
Description: A coastal view of the Felixstowe seafront featuring a pier, industrial cranes, and a Ferris wheel under a clear sky.
Keywords: Felixstowe, Suffolk, England, UK, seafront, harbor, port, cranes, pier, Ferris wheel, industrial, coastal, lamp, East Suffolk, Europe
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — usable_with_caveats — control tokens visible</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002425916
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6310
- *Generation tokens:* 141
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: Felixstowe Seafront with Cranes and Pier
Description: A seafront scene in Felixstowe, England, featuring a pier, Ferris wheel, and industrial cranes in the distance, with lampposts, string lights, and a paddleboarder on the water under clear daylight.
Keywords: Felixstowe, seafront, England, UK, GBR, pier, Ferris wheel, industrial cranes, lampposts, string lights, paddleboarder, East Suffolk, Europe, 2026-08-13, 16:08:57, 51.961800°N, 1.358500°E<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — missing required fields; extra text before Title; cut off at token limit; incomplete thinking block</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  unexpected_catalog_preamble, thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Missing sections:* ["title"]
- *Unexpected text before Title:* ◁think▷Got it, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. Let's start by analyzing the image and the hints.
  First, the context says the capture date/time is 2026-08-13 16:08:57
  UTC+01:00, and GPS is 51.961800°N, 1.358500°E. The location is Felixstowe,
  England, UK, GBR, Europe. The hints mention seafront, Felixstowe, England,
  UK, GBR, seafront, etc.  Looking at the image: there's a seafront with a
  pier, lampposts, a Ferris wheel, industrial cranes in the background, water,
  trees, and a person paddleboarding. The lighting suggests it's daytime,
  maybe sunny. The scene is Felixstowe's seafront.  Now, the title needs to be
  a concrete 5-10 words. Let's think: "Felixstowe seafront with pier,
  lampposts, and paddleboarder at 16:08 UTC+01:00". Wait, but the time is in
  the future (2026), so maybe just the location and key elements. Maybe
  "Felixstowe seafront at 16:08 UTC+01:00 with pier, lampposts, and
  paddleboarder".
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003654752
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1257
- *Generation tokens:* 500
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

First, the context says the capture date/time is 2026-08-13 16:08:57 UTC+01:00, and GPS is 51.961800°N, 1.358500°E. The location is Felixstowe, England, UK, GBR, Europe. The hints mention seafront, Felixstowe, England, UK, GBR, seafront, etc.

Looking at the image: there's a seafront with a pier, lampposts, a Ferris wheel, industrial cranes in the background, water, trees, and a person paddleboarding. The lighting suggests it's daytime, maybe sunny. The scene is Felixstowe's seafront.

Now, the title needs to be a concrete 5-10 words. Let's think: "Felixstowe seafront with pier, lampposts, and paddleboarder at 16:08 UTC+01:00". Wait, but the time is in the future (2026), so maybe just the location and key elements. Maybe "Felixstowe seafront at 16:08 UTC+01:00 with pier, lampposts, and paddleboarder".

Description: Combine context and visible details. So, "A daytime view of Felixstowe's seafront at 16:08 UTC+01:00, featuring a pier, traditional lampposts, a paddleboarder on the water, industrial cranes in the distance, and trees framing the scene."

Keywords: Need 10-18 unique keywords. Let's list: Felixstowe, seafront, England, UK, GBR, UTC+01:00, 2026-08-13, pier, lampposts, paddleboarder, industrial cranes, Ferris wheel, trees, water, Europe.

Wait, check the hints: the hints include Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. But the image might not show Adobe Stock or Any Vision. The context is authoritative, so maybe include those. Wait, the user said to treat capture date/time and GPS as authoritative, but don't claim they are visible. So maybe include them as
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Thinking-bf16 — unusable — missing required fields; cut off at token limit; incomplete thinking block</summary>

### mlx-community/Qwen3-VL-2B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Missing sections:* ["title", "description", "keywords"]
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* c325e5ea14c215bb08fa0d668c81fa2581f9050b
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.008062086
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16469
- *Generation tokens:* 500
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, GBR. The image has a seafront with cranes, a Ferris wheel, street lamps. So a title should be specific. Maybe "Felixstowe Seafront, England, UK" but need 5-10 words. Let's see: "Felixstowe Seafront, England, UK, Cranes and Ferris Wheel" – that's 10 words. Wait, "Felixstowe Seafront, England, UK" is 7, but maybe "Felixstowe Seafront, England, UK: Cranes and Ferris Wheel" – no, better to be concise. "Felixstowe Seafront, England, UK, Cranes and Ferris Wheel" – 10 words. That works.

Next, the description. Needs to combine context with visible subject, setting, action, lighting, details. Context: capture date/time, GPS, but we don't claim those. The main visible subject is the seafront with cranes, Ferris wheel, street lamps. Setting: coastal, industrial port. Action: maybe a person on paddleboard? Wait, the image has a person on a board in the water. Lighting: sunny, clear sky. Distinctive details: colorful string lights, old-fashioned street lamps, Ferris wheel, cranes. So: "Sunny seafront at Felixstowe, England, UK, featuring cranes, Ferris wheel, colorful string lights, and vintage street lamps over calm waters with a paddleboarder."

Wait, check the context: the description hint says "Seafront, Felixstowe, England, UK, GBR". So need to include that. Also, the GPS is 51.9618N, 1.3585E, which is Felixstowe, Suffolk, UK. So the description should mention Felixstowe, England, UK, GBR. Let's structure: "Felixstowe seafront, England, UK, featuring cranes, Ferris wheel, colorful string lights, vintage street lamps, and a paddleboarder on calm waters under a clear sky."

Now keywords. Need 10-18 unique, comma-separated. Context: Felixstowe, England, UK, GBR, Suffolk. Visible details: seafront, cr
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — extra text before Title; cut off at token limit; incomplete thinking block</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, unexpected_catalog_preamble,
  thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this.
  First, the title needs to be 5-10 words. Looking at the image: seafront,
  Felixstowe, England, with cranes, pier, Ferris wheel. So maybe "Felixstowe
  Seafront with Industrial Cranes".  Then the description: need to combine
  context (date/time, location) with main subjects (seafront, cranes, pier,
  Ferris wheel, lighting, action). The image has a seafront with industrial
  cranes, a pier, Ferris wheel, street lamps, string lights, water, and a
  person maybe. So something like "Seafront view of Felixstowe, England,
  showing industrial cranes, pier, Ferris wheel, and waterfront with street
  lamps under bright daylight." Wait, need to include the context like capture
  date, but maybe not in the description, but the description should mention
  the setting. Let's make it: "Seafront scene in Felixstowe, England,
  featuring industrial cranes, a pier, Ferris wheel, and waterfront with
  street lamps under bright daylight conditions."  Keywords need 10-18,
  unique, comma-separated. From hints: East Suffolk, England, Europe,
  Felixstowe, Suffolk, UK, gbr, seafront, Adobe Stock, Any Vision, East
  Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Wait, need
  unique ones. Let's list: Felixstowe, Suffolk, UK, seafront, industrial
  cranes, pier, Ferris wheel, street lamps, waterfront, Europe, England. Wait,
  let's check the image: street lamps, string lights, water, cranes, pier,
  Ferris wheel, trees, rocks. So keywords: Felixstowe, Suffolk, UK, seafront,
  industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe,
  England, GBR, port, coastal, daylight, string lights, rocks. Let's make sure
  they are unique.  Now, let's structure each part:
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001803312
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6306
- *Generation tokens:* 500
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image: seafront, Felixstowe, England, with cranes, pier, Ferris wheel. So maybe "Felixstowe Seafront with Industrial Cranes". 

Then the description: need to combine context (date/time, location) with main subjects (seafront, cranes, pier, Ferris wheel, lighting, action). The image has a seafront with industrial cranes, a pier, Ferris wheel, street lamps, string lights, water, and a person maybe. So something like "Seafront view of Felixstowe, England, showing industrial cranes, pier, Ferris wheel, and waterfront with street lamps under bright daylight." Wait, need to include the context like capture date, but maybe not in the description, but the description should mention the setting. Let's make it: "Seafront scene in Felixstowe, England, featuring industrial cranes, a pier, Ferris wheel, and waterfront with street lamps under bright daylight conditions." 

Keywords need 10-18, unique, comma-separated. From hints: East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront, Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Wait, need unique ones. Let's list: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe, England. Wait, let's check the image: street lamps, string lights, water, cranes, pier, Ferris wheel, trees, rocks. So keywords: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe, England, GBR, port, coastal, daylight, string lights, rocks. Let's make sure they are unique. 

Now, let's structure each part:

Title: "Felixstowe Seafront with Industrial Cranes"

Description: "Seafront view of Felixstowe, England, showcasing industrial cranes, a pier, Ferris wheel, and waterfront with street lamps under bright daylight conditions."

Keywords: Felixstowe, Suffolk, UK, seafront, industrial cranes, pier, Ferris wheel, street lamps, waterfront, Europe,
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — usable_with_caveats — role tokens visible</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Role-boundary tokens in output:* ["&lt;end_of_utterance&gt;"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00268806
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2586
- *Generation tokens:* 213
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 Title: Felixstowe Seafront in Suffolk, England, UK.
Description: The image depicts a serene seafront in Felixstowe, Suffolk, England, UK, characterized by a calm body of water, likely the North Sea, stretching towards the horizon. In the background, a series of large, industrial cranes are visible, indicating a port or shipping area. A pier extends into the water, adorned with decorative lights that create a festive atmosphere. The sky is clear, suggesting a sunny day. The foreground features a row of street lamps, adding to the ambiance of the scene. Trees and greenery are present, providing a natural contrast to the man-made structures. The overall mood of the image is tranquil and picturesque, capturing the beauty of a coastal town with its industrial and natural elements harmoniously coexisting.
Keywords: Felixstowe, Suffolk, England, UK, seafront, North Sea, cranes, pier, decorative lights, street lamps, trees, greenery, coastal town, industrial, natural, serene, picturesque, tranquil.<end_of_utterance>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Usability           | Observations                                                             |
|-------------------------------------------------------|---------------------|--------------------------------------------------------------------------|
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | unusable            | missing required fields; echoes instructions; cut off at token limit     |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | unusable            | missing required fields; extra text before Title; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16                       | unusable            | missing required fields; echoes instructions                             |
| mlx-community/gemma-3n-E4B-it-bf16                    | unusable            | missing required fields; extra text before Title                         |
| mlx-community/llava-v1.6-mistral-7b-8bit              | unusable            | missing required fields                                                  |
| mlx-community/MolmoPoint-8B-fp16                      | unusable            | missing required fields                                                  |
| mlx-community/nanoLLaVA-1.5-4bit                      | unusable            | missing required fields                                                  |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/gemma-3-27b-it-qat-4bit                 | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/InternVL3-8B-bf16                       | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/MiniCPM-V-4.6-8bit                      | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Molmo-7B-D-0924-8bit                    | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Phi-3.5-vision-instruct-bf16            | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/pixtral-12b-8bit                        | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Qwen3.6-27B-mxfp8                       | usable_with_caveats | title/keyword constraints failed                                         |
| Qwen/Qwen3-VL-2B-Instruct                             | usable_with_caveats | title/keyword constraints failed                                         |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | usable_with_caveats | draft hints copied unchanged                                             |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                           | Runtime identity                                    | Performance                                                                                 |
|-------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                | rev ed71acdae079; Lfm2VlProcessor; stop completed   | 358 prompt / 82 generated; 513 tok/s; 1.2 GB peak; cleanup 0.000132/0.0 GB active/cache     |
| mlx-community/gemma-4-26b-a4b-it-4bit           | rev 0d77464eeb23; Gemma4Processor; stop completed   | 580 prompt / 99 generated; 130 tok/s; 16 GB peak; cleanup 0.0119/0.0 GB active/cache        |
| mlx-community/gemma-4-31b-it-4bit               | rev 696d436c4047; Gemma4Processor; stop completed   | 580 prompt / 103 generated; 25.0 tok/s; 20 GB peak; cleanup 0.0124/0.0 GB active/cache      |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | rev a962dcb09eee; Mistral3Processor; stop completed | 3177 prompt / 135 generated; 184 tok/s; 9.0 GB peak; cleanup 0.00562/0.0 GB active/cache    |
| mlx-community/Ornith-1.0-35B-bf16               | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed  | 16482 prompt / 124 generated; 63.7 tok/s; 74 GB peak; cleanup 0.00706/0.0 GB active/cache   |
| mlx-community/Qwen3.5-35B-A3B-4bit              | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed  | 16482 prompt / 120 generated; 110 tok/s; 24 GB peak; cleanup 0.00857/0.0 GB active/cache    |
| mlx-community/Qwen3.5-9B-MLX-4bit               | rev 938d8919941c; Qwen3VLProcessor; stop completed  | 16482 prompt / 109 generated; 93.5 tok/s; 10.0 GB peak; cleanup 0.00909/0.0 GB active/cache |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx        | rev 844516024a1c; SmolVLMProcessor; stop completed  | 400 prompt / 116 generated; 126 tok/s; 5.5 GB peak; cleanup 0.00972/0.0 GB active/cache     |
| mlx-community/Step-3.7-Flash-oQ2e               | rev 3dacb46f724a; Step3VLProcessor; stop completed  | 3468 prompt / 117 generated; 43.8 tok/s; 70 GB peak; cleanup 0.00998/0.0 GB active/cache    |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 8,880 x 6,656 pixels
- *Image size:* 25,691,731 bytes
- *Image SHA-256:* 4d17cb4b83cb77532bf22c7a0f6286481c0f7bd507cfcf42589f91db2f397cff

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 16:08:57 UTC+01:00
- GPS: 51.961800°N, 1.358500°E

Descriptive hints:
- Title hint: Seafront, Felixstowe, England, UK, GBR, Europe
- Description hint: Seafront, Felixstowe, England, UK, GBR
- Keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront

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

### Highlighted model revisions

| Model                                         | Resolved revision                        |
|-----------------------------------------------|------------------------------------------|
| mlx-community/GLM-4.6V-Flash-mxfp4            | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b |
| mlx-community/paligemma2-3b-pt-896-4bit       | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/X-Reasoner-7B-8bit              | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit  | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/GLM-4.6V-nvfp4                  | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16  | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/Qwen3-VL-2B-Thinking-bf16       | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/GLM-4.1V-9B-Thinking-8bit       | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Idefics3-8B-Llama3-bf16         | 8c2a30c48864f3251701b7bde40f601d25535098 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.14                                                                                                                                          |
| mlx                        | 0.32.1.dev20260813+a8e24f202                                                                                                                    |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.8                                                                                                                                           |
| transformers               | 5.15.0                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.27.0                                                                                                                                          |
| Python Version             | 3.13.14                                                                                                                                         |
| OS                         | Darwin 25.6.0                                                                                                                                   |
| macOS Version              | 26.6.1                                                                                                                                          |
| SDK Version                | 26.5                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                              |
| Xcode Version              | 26.6                                                                                                                                            |
| Xcode Build                | 17F113                                                                                                                                          |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                               |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                  |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                 |
| GPU/Chip                   | Apple M5 Max                                                                                                                                    |
| GPU Cores                  | 40                                                                                                                                              |
| MLX Device                 | Apple M5 Max                                                                                                                                    |
| GPU Architecture           | applegpu_g17s                                                                                                                                   |
| Recommended Working Set    | 108 GB                                                                                                                                          |
| Fused Attention            | Available                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (174,839,328 bytes, sha256=07dfe0c6a946fa6c72b61e2f5052e97bac26ab46ffd89aa8ff01ab4b640b2a6b) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,964,224 bytes, sha256=9a694b772db46607b31ae2a6f58474eabfdb9a30cf7eea355760ce38f34ad3be)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
