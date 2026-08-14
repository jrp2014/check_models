# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

## Run Summary

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 41    |
| Conclusive outcomes | 41    |
| Completed           | 40    |
| Crashed             | 1     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| actionable failure             | 1     |
| none                           | 30    |
| observation needs reproduction | 10    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 13    |
| usable              | 10    |
| usable with caveats | 17    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 4     |
| Unrecognised model control tokens remain visible                                      | 3     |
| Required fields are missing or empty                                                  | 10    |
| Response repeats the task instructions instead of only returning the requested fields | 5     |
| Extra text appears before the Title field                                             | 2     |
| Response appears cut off at the token limit                                           | 7     |
| Internal reasoning block appears incomplete                                           | 4     |
| Conversation-role control tokens remain visible                                       | 1     |
| Title or keywords do not meet requested constraints                                   | 18    |

## Triage

| Model                                                                                                           | Execution | Usability           | Maintainer status              | Observations                                                                                                                                     |
|-----------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| [mlx-community/SmolVLM2-2.2B-Instruct-mlx](#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx)                 | crashed   | not_evaluated       | actionable_failure             | none                                                                                                                                             |
| [mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16](#diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16) | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; cut off at token limit; incomplete thinking block                                                        |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed | unusable            | observation_needs_reproduction | repeated text; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; cut off at token limit                                                                                   |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                                | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                                                          |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                                                           |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible; title/keyword constraints failed                                                                                         |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible; title/keyword constraints failed                                                                                         |
| [mlx-community/Qwen3-VL-2B-Thinking-bf16](#diagnostic-mlx-community-qwen3-vl-2b-thinking-bf16)                  | completed | unusable            | observation_needs_reproduction | missing required fields; cut off at token limit; incomplete thinking block                                                                       |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                   | completed | unusable            | observation_needs_reproduction | echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed                |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                                                                              |

## Crashes requiring action

<a id="diagnostic-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Root exception and chain

```text
builtins.ValueError: not enough values to unpack (expected 3, got 2)
builtins.ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: not enough values to unpack (expected 3, got 2)
```

#### Execution and provenance

- *Execution:* crashed
- *Usability:* not_evaluated
- *Maintainer status:* actionable_failure
- *Observations:* none
- *Arch supported by installed mlx-vlm:* yes (model_type smolvlm)
- *Missing sections:* ["title", "description"]
- *Echoed instruction fragments:* ["return exactly these three sections",
  "title hint:", "description hint:", "keyword hints:"]
- *Unexpected text before Title:* ========== Files: ['/', 'U', 's', 'e', 'r',
  's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/',
  'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0',
  '8', '1', '3', '-', '1', '7', '2', '3', '1', '8', '_', 'D', 'S', 'C', '0',
  '1', '5', '2', '5', '.', 'j', 'p', 'g']   Prompt:  User:&lt;image&gt;Create
  British-English catalogue metadata from the image and supplied context.
  Treat any capture date/time and GPS as authoritative facts, but do not claim
  they are visible. Descriptive hints may be incomplete or wrong: retain
  details supported by the image, correct conflicts, and add important visible
  details. Prefer image evidence when a hint conflicts, and omit uncertain
  details.  Context: Authoritative context: - Capture date/time: 2026-08-13
  17:23:18 UTC+01:00 - GPS: 51.957967°N, 1.346900°E  Descriptive hints: -
  Title hint: Seafront, Felixstowe, England, UK, GBR, Europe - Description
  hint: Seafront, Felixstowe, England, UK, GBR - Keyword hints: Adobe Stock,
  Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr,
  seafront  Write: - a concrete 5-10-word title; - a 1-2-sentence factual
  description combining relevant context with the main visible subject,
  setting, action, lighting, and distinctive details; - 10-18 unique,
  comma-separated keywords covering relevant context and visible details.
  Return exactly these three sections and nothing else:
- *Unexpected special tokens:* ["&lt;|im_start|&gt;"]
- *Phase:* decode
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model generation failed for
  mlx-community/SmolVLM2-2.2B-Instruct-mlx: not enough values to unpack
  (expected 3, got 2)
- *Root error type:* ValueError
- *Root error message:* not enough values to unpack (expected 3, got 2)
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Processor class:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.009602184
- *Post-cleanup cache memory (GB):* 0.0
- *Configured EOS token ID:* 49279
- *Configured EOS token:* &lt;end_of_utterance&gt;

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11827, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12175, in _generate_once
    return strict_generate(
        model=model,
    ...<3 lines>...
        **generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1158, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 977, in stream_generate
    for n, (token, logprobs) in enumerate(gen):
                                ~~~~~~~~~^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/ar.py", line 393, in generate_step
    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **kwargs
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/idefics3/idefics3.py", line 160, in get_input_embeddings
    final_inputs_embeds = self._prepare_inputs_for_multimodal(
        image_features, inputs_embeds, input_ids
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/smolvlm/smolvlm.py", line 13, in _prepare_inputs_for_multimodal
    N, S, D_img = image_features.shape
    ^^^^^^^^^^^
ValueError: not enough values to unpack (expected 3, got 2)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12536, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12188, in _run_model_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11834, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: not enough values to unpack (expected 3, got 2)

```

</details>

#### Captured stdout/stderr

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '3', '-', '1', '7', '2', '3', '1', '8', '_', 'D', 'S', 'C', '0', '1', '5', '2', '5', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>User:<image>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 17:23:18 UTC+01:00
- GPS: 51.957967°N, 1.346900°E

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
Keywords:<end_of_utterance>
Assistant:

=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 12 files:   0%|          | 0/12 [00:00<?, ?it/s]
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 1898.38it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[18:42:19] ERROR    Generation error for mlx-community/SmolVLM2-2.2B-Instruct-mlx
                    ValueError: not enough values to unpack (expected 3, got 2)
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

<details>
<summary>mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 — unusable — repeated text; missing required fields; cut off at token limit; incomplete thinking block</summary>

### mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type ernie4_5_moe_vl)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "waves, waves, waves, waves,..."
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "incomplete_thinking_trace"]
- *Resolved model revision:* 32dae5c38006e20ac158bc94cd1d5967d19b2652
- *Processor class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLProcessor
- *Tokenizer class:* mlx_vlm.models.ernie4_5_moe_vl.processing_ernie4_5_moe_vl.Ernie4_5_VLTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001180718
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1569
- *Generation tokens:* 1000
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;

#### Complete output

```text
Alright, let's get to work. I've got the image and the hints, and I need to craft some metadata. First, the title. The hints are pretty clear: "Seafront, Felixstowe, England, UK, GBR, Europe." I need something concise, so I'll go with "Seafront at Felixstowe, England, UK." That's five words, right in the sweet spot.

Now for the description. I'll need to weave in the context – the date, time, and location – with the visual elements. The image is a wide shot of a beach, so I'll start there. I see a bustling beach scene with people enjoying the water and the sand. The lighting is bright, suggesting a sunny day. I'll mention the location and the time of day. I'll also include the key details like the seawall and the distant town. I'll make sure to mention the people and the activity. I'll refine it to be a single, concise sentence. Okay, here it is: "A wide-angle, sunlit photograph of a bustling beach at Felixstowe, England, UK, taken on August 13, 2026, at 5:23 PM, showing people on the sand and in the water with a town and seafront in the background."

Finally, the keywords. I need to cover the location, the subject matter, the time of day, and the general atmosphere. I'll start with the location: Felixstowe, England, UK, GBR, Europe, Suffolk, East Suffolk. Then the subject: beach, seashore, seafront, coast, shoreline, waterfront, shore, sand, sandy beach, shoreline, seawall, cliffs, rocks, rocks, rocks, rocks, rocks. People and activity: people, crowd, family, couple, group, children, kids, teenagers, adults, swimming, bathing, wading, playing, relaxing, strolling, walking, sunbathing, tourists, holidaymakers, beachgoers, lifeguard, lifeguard tower, lifeguard tower. Nature and environment: sea, ocean, water, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves, waves,
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — repeated text; echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  prompt_instruction_echo, unexpected_catalog_preamble,
  thinking_trace_incomplete, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Repeated fragment:* keyword: "beachgoers"
- *Echoed instruction fragments:* ["unique comma separated keywords covering
  relevant context and"]
- *Unexpected text before Title:* ◁think▷Got it, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. Let's start by analyzing the image and the hints.  The
  image shows a seafront in Felixstowe, England, UK. The capture date is
  2026-08-13 17:23:18 UTC+01:00, GPS is 51.957967°N, 1.346900°E. The hints
  mention seafront, Felixstowe, England, UK, GBR, Europe, Adobe Stock, Any
  Vision, East Suffolk, Suffolk, UK. The description hints are similar.
  First, the title needs to be a concrete 5-10 words. Let's see the main
  elements: seafront, people, rocky shore, buildings, sea, clear sky. Maybe
  "Felixstowe seafront with people on rocky shore and coastal buildings."
  Wait, that's 9 words. Maybe "Felixstowe seafront scene with rocky shoreline
  and coastal buildings." That's 9 words.  Next, the description: 1-2
  sentences. It should combine context (date/time, GPS) with main subject,
  setting, action, lighting, details. So: "A sunny day at Felixstowe's
  seafront on August 13, 2026, showing people enjoying the rocky shoreline,
  coastal buildings, and calm sea under clear skies." Wait, the capture date
  is 2026-08-13 17:23:18 UTC+01:00, so maybe include that. Also, GPS is
  51.957967°N, 1.346900°E. So: "On August 13, 2026, at 17:23 UTC+01:00, the
  sunny Felixstowe seafront at 51.957967°N, 1.346900°E features people on a
  rocky shore, coastal buildings, and calm waters." That's a bit long but
  includes the date, time, location, and elements.
- *Thinking trace markers:* ["\u25c1think\u25b7"]
- *Title word count:* 11
- *Requested title word range:* [5, 10]
- *Keyword count:* 85
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["seafront", "england", "uk", "gbr", "europe", "rocky
  shoreline", "people", "coastal buildings", "clear sky", "calm sea", "birds",
  "beachgoers", "utc 01 00", "any vision", "2026 08 13", "adobe stock"]
- *Token-cap degradation evidence:* ["repetitive_tail",
  "incomplete_thinking_trace"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003654752
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1269
- *Generation tokens:* 1000
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]

#### Complete output

```text
◁think▷Got it, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. Let's start by analyzing the image and the hints.

The image shows a seafront in Felixstowe, England, UK. The capture date is 2026-08-13 17:23:18 UTC+01:00, GPS is 51.957967°N, 1.346900°E. The hints mention seafront, Felixstowe, England, UK, GBR, Europe, Adobe Stock, Any Vision, East Suffolk, Suffolk, UK. The description hints are similar. 

First, the title needs to be a concrete 5-10 words. Let's see the main elements: seafront, people, rocky shore, buildings, sea, clear sky. Maybe "Felixstowe seafront with people on rocky shore and coastal buildings." Wait, that's 9 words. Maybe "Felixstowe seafront scene with rocky shoreline and coastal buildings." That's 9 words.

Next, the description: 1-2 sentences. It should combine context (date/time, GPS) with main subject, setting, action, lighting, details. So: "A sunny day at Felixstowe's seafront on August 13, 2026, showing people enjoying the rocky shoreline, coastal buildings, and calm sea under clear skies." Wait, the capture date is 2026-08-13 17:23:18 UTC+01:00, so maybe include that. Also, GPS is 51.957967°N, 1.346900°E. So: "On August 13, 2026, at 17:23 UTC+01:00, the sunny Felixstowe seafront at 51.957967°N, 1.346900°E features people on a rocky shore, coastal buildings, and calm waters." That's a bit long but includes the date, time, location, and elements.

Keywords: Need 10-18 unique, comma-separated. Let's list elements: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, birds, beachgoers, UTC+01:00, 2026-08-13. Wait, but the hints mention Adobe Stock, Any Vision, East Suffolk. Maybe include those? Wait, the hints are hints, but the image might not have those. The image shows Adobe Stock watermark? Wait, the image is from Adobe Stock, so maybe include that. Also, East Suffolk is a county. So keywords: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk. That's 15 keywords. Let me check: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk. That's 15. Maybe add birds? The image has birds in the water. So include birds. So 16. Let's adjust: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk, birds. That's 16.

Wait, the user said "unique, comma-separated keywords covering relevant context and visible details". So visible details: people, rocky shore, buildings, sea, sky, birds, beachgoers, etc. Context: date, time, location. So include those. Let's finalize:

Title: Felixstowe seafront scene with people on rocky shore and coastal buildings.

Description: A sunny day at Felixstowe's seafront on August 13, 2026, at 17:23 UTC+01:00, featuring people enjoying the rocky shoreline, coastal buildings, and calm sea under clear skies.

Keywords: Felixstowe, seafront, England, UK, GBR, Europe, rocky shoreline, people, coastal buildings, clear sky, calm sea, beachgoers, UTC+01:00, 2026-08-13, Adobe Stock, Any Vision, East Suffolk, birds.

Wait, the description should be 1-2 sentences. Let me check the example. The user's example might have a more concise version. Let me adjust the description to be 1-2 sentences. Maybe: "Sunny Felixstowe seafront at 51.
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated text; missing required fields; cut off at token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "- authoritative context: -..."
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.014025992
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4397
- *Generation tokens:* 1000
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context:
- Authoritative context
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
- *Repeated fragment:* keyword: "east suffolk beachgoers"
- *Title word count:* 5
- *Requested title word range:* [5, 10]
- *Keyword count:* 179
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["promenade", "east suffolk coastline", "east suffolk
  beach", "felixstowe historic buildings", "felixstowe historic architecture",
  "felixstowe greenery", "felixstowe historic structures", "felixstowe
  historic town", "east suffolk beachgoers", "east suffolk beach buildings",
  "uk seafront greenery", "uk beach greenery"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.01017566
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16544
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Title:
Felixstowe Seafront, East Suffolk, England

Description:
A vibrant seafront scene in Felixstowe, East Suffolk, England, with people enjoying the sunny day at the beach and rocky promenade, set against a backdrop of historic buildings and lush greenery. The image captures the lively atmosphere of a typical summer day at the seaside town.

Keywords: Felixstowe, East Suffolk, England, UK, GBR, seafront, beach, promenade, historic buildings, greenery, sunny day, people, summer, historic architecture, coastal town, East Anglia, United Kingdom, Europe, seaside, promenade, beachgoers, historic structures, waterfront, East Suffolk coastline, British seaside, Felixstowe beach, East Anglia coast, UK seafront, historic Felixstowe, East Suffolk beach, UK coastal town, Felixstowe promenade, East Anglia seafront, UK beach scene, Felixstowe historic buildings, East Suffolk coastline, UK seaside town, Felixstowe beachgoers, East Suffolk promenade, UK seafront scene, Felixstowe historic architecture, East Suffolk beach, UK coastal promenade, Felixstowe greenery, East Suffolk buildings, UK seafront buildings, Felixstowe historic structures, East Suffolk waterfront, UK beach buildings, Felixstowe promenade buildings, East Suffolk seafront, UK beach promenade, Felixstowe historic town, East Suffolk beachgoers, UK seafront town, Felixstowe historic buildings, East Suffolk promenade buildings, UK beach town, Felixstowe greenery, East Suffolk beach buildings, UK seafront greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic town, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures, East Suffolk beach buildings, UK beach greenery, Felixstowe historic architecture, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic buildings, East Suffolk beach buildings, UK beach greenery, Felixstowe historic structures, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic architecture, East Suffolk beach buildings, UK beach greenery, Felixstowe historic buildings, East Suffolk beachgoers, UK seafront greenery, Felixstowe historic structures
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-8bit — usable_with_caveats — control tokens visible</summary>

### mlx-community/diffusiongemma-26B-A4B-it-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type diffusion_gemma)
- *Unexpected special tokens:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Thinking trace markers:* ["&lt;|channel&gt;thought", "&lt;channel|&gt;"]
- *Resolved model revision:* 7b95e3887078ba56283c24f2578d6e5a06b9d7e8
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.01017568
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 583
- *Generation tokens:* 82
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Sunny seafront at Felixstowe in Suffolk, England
Description: People enjoy a sunny day at the Felixstowe seafront, where stone groynes meet the sea under a clear blue sky.
Keywords: Felixstowe, Suffolk, England, UK, East Suffolk, seafront, beach, ocean, coastal, summer, recreation, tourism, seaside, leisure, water, Europe
```

</details>

<a id="diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8"></a>

<details>
<summary>mlx-community/diffusiongemma-26B-A4B-it-mxfp8 — usable_with_caveats — control tokens visible; title/keyword constraints failed</summary>

### mlx-community/diffusiongemma-26B-A4B-it-mxfp8

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
- *Keyword count:* 15
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["coastal"]
- *Resolved model revision:* ded389e478f86d498ad9e7f47666e83b166a28f1
- *Processor class:* mlx_vlm.models.diffusion_gemma.processing_diffusion_gemma.DiffusionGemma4Processor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.01017568
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 583
- *Generation tokens:* 81
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;

#### Complete output

```text
<|channel>thought
<channel|>Title: Sunny day at the seafront in Felixstowe, England
Description: People enjoy a sunny day at the stone-tiered sea beach in Felixstowe, Suffolk, under a clear blue sky.
Keywords: Felixstowe, Suffolk, East Suffolk, England, UK, seafront, beach, sea, summer, coastal, leisure, outdoors, tourism, coastal, Europe
```

</details>

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — usable_with_caveats — control tokens visible; title/keyword constraints failed</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 14
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002425916
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6281
- *Generation tokens:* 78
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<|begin_of_box|>Title: Felixstowe Seafront, England
Description: A sunny seafront scene in Felixstowe, England, with people enjoying the beach, rocky shoreline, and coastal buildings under clear skies.
Keywords: Felixstowe, seafront, England, UK, East Suffolk, Suffolk, Europe, beach, coastal, people, rocky shoreline, buildings, sunny, clear skies<|end_of_box|>
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
- *Prompt tokens:* 16535
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;

#### Complete output

```text
Got it, let's tackle this step by step. First, the title. The context says Felixstowe, England, UK, so we need a concise title. The image shows a seafront with people, so maybe "Felixstowe Seafront, Suffolk, England". Wait, the title hint was Seafront, Felixstowe, England, UK, GBR, Europe. So need to make it 5-10 words. Let's check: "Felixstowe Seafront, Suffolk, England, UK" – that's 10 words? Wait, no, "Felixstowe Seafront, Suffolk, England, UK" is 8 words. Maybe "Felixstowe Seafront, Suffolk, England, UK" is okay. Wait, the title should be concrete. Let's see: "Felixstowe Seafront, Suffolk, England" – that's 7 words. Maybe "Felixstowe Seafront, Suffolk, England, UK" is better. Wait, the hint says "Seafront, Felixstowe, England, UK, GBR, Europe" so the title should capture that. Let's go with "Felixstowe Seafront, Suffolk, England, UK".

Next, the description. Need 1-2 sentences. Context has capture date/time, GPS, but we don't claim them visible. The main visible subject is the seafront with people, setting is a coastal area with rocky breakwaters, people swimming, walking, beach. Lighting is sunny, clear blue sky. Distinctive details: the rocky breakwaters, the buildings in the background (like the Spa Pavilion), people in the water, seagulls. So: "A sunny day at Felixstowe seafront, England, featuring rocky breakwaters, people swimming and walking along the shore, with historic buildings and seagulls visible in the background under a clear blue sky."

Wait, check the context: capture date/time is 2026-08-13 17:23:18 UTC+01:00, but we don't claim that's visible. The description should combine relevant context (Felixstowe, England, UK) with main visible subject (seafront), setting (coastal area with breakwaters), action (people swimming, walking), lighting (sunny, clear sky), distinctive details (rocky breakwaters, historic buildings like Spa Pavilion, seagulls). So the description: "Sunny day at Felixstowe seafront, England, with rocky breakwaters, people swimming and wading in the water, historic buildings and seagulls visible under a clear blue sky."

Now keywords. Need 10-18 unique, comma-separated. Context: East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Visible details: seafront, Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Wait, the keyword hints include Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. So need to pick unique ones. Let's list: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront. Wait, but need 10-18. Let's check the image: the buildings have a Spa Pavilion, so maybe "Spa Pavilion" is a detail. Also, the rocky breakwaters, people in the water, seagulls. So keywords: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront, Spa Pavilion, seagulls, beach, people, water, sunny, clear sky. Wait, but need to be unique. Let's count: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront, Spa Pavilion, seagulls, beach, people, water, sunny, clear sky. Wait, but the context says "East Suffolk" is a hint, so include that. Also, "Suffolk" is in the keyword hints. Let's make sure they are unique. Let's list: Felixstowe, Suffolk, England, UK, gbr, seafront, East Suffolk, Europe, England, UK, gbr, seafront, Spa Pavilion, seagulls, beach, people, water, sunny, clear sky. Wait, but the keyword hints include "Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront". So we
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — echoes instructions; extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, prompt_instruction_echo,
  unexpected_catalog_preamble, thinking_trace_incomplete,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Echoed instruction fragments:* ["keyword hints:"]
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this.
  First, the title needs to be 5-10 words. Looking at the image, it's a
  seafront with people, buildings, and the coastal area. So maybe "Felixstowe
  Seafront Beach Scene" or something similar. Wait, need to be concrete. Let's
  go with "Felixstowe Coastal Promenade" maybe, but let's check the hints. The
  title hint says Seafront, Felixstowe... so maybe "Felixstowe Seafront with
  Coastal Steps".  Then the description: need to combine context (date/time,
  location) with main subject (seafront, people, setting, action, lighting,
  distinctive details). The image shows people swimming, walking on the beach,
  coastal steps, buildings, clear sky. So something like "Seafront scene in
  Felixstowe, England, showing people enjoying the beach with distinctive
  coastal steps and historic buildings under clear daylight."  Wait, let's
  make sure. The description should be 1-2 sentences. Let's structure it:
  "Seafront in Felixstowe, England, with people swimming and walking near
  distinctive coastal steps, under clear daylight with historic buildings in
  the background."  Then keywords: need 10-18 unique, comma-separated. From
  hints: Adobe Stock, Any Vision (but maybe not, since the image might not be
  from those, but the hints include them? Wait, the hints say keyword hints:
  Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk,
  UK, gbr, seafront. Also visible details: people swimming, coastal steps,
  buildings, beach, sea, seagulls, etc. So keywords could be: Felixstowe,
  Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic
  buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside,
  promenade.  Wait, let's check the hints again. The keyword hints include:
  Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk,
  UK, gbr, seafront. So need to include those, plus visible details. So
  keywords: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people,
  swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR,
  seagulls, seaside, promenade.  Now, let's put it all together.
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 4
- *Requested title word range:* [5, 10]
- *Keyword count:* 69
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["felixstowe", "suffolk", "uk", "seafront", "coastal
  steps", "beach", "people", "swimming", "historic buildings", "daylight",
  "europe", "east suffolk", "england", "gbr", "seagulls", "seaside", "wait",
  "with people swimming and walking near distinctive coastal steps", "under
  clear daylight with historic buildings in the background"]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001803312
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6277
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;

#### Complete output

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words. Looking at the image, it's a seafront with people, buildings, and the coastal area. So maybe "Felixstowe Seafront Beach Scene" or something similar. Wait, need to be concrete. Let's go with "Felixstowe Coastal Promenade" maybe, but let's check the hints. The title hint says Seafront, Felixstowe... so maybe "Felixstowe Seafront with Coastal Steps".

Then the description: need to combine context (date/time, location) with main subject (seafront, people, setting, action, lighting, distinctive details). The image shows people swimming, walking on the beach, coastal steps, buildings, clear sky. So something like "Seafront scene in Felixstowe, England, showing people enjoying the beach with distinctive coastal steps and historic buildings under clear daylight."

Wait, let's make sure. The description should be 1-2 sentences. Let's structure it: "Seafront in Felixstowe, England, with people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Then keywords: need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not, since the image might not be from those, but the hints include them? Wait, the hints say keyword hints: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. Also visible details: people swimming, coastal steps, buildings, beach, sea, seagulls, etc. So keywords could be: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade.

Wait, let's check the hints again. The keyword hints include: Adobe Stock, Any Vision, East Suffolk, England, Europe, Felixstowe, Suffolk, UK, gbr, seafront. So need to include those, plus visible details. So keywords: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade.

Now, let's put it all together.

Title: "Felixstowe Coastal Steps Seafront"

Description: "Seafront scene in Felixstowe, England, showing people swimming and enjoying the beach near distinctive coastal steps, with historic buildings and clear daylight in the background."

Keywords: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade

Wait, let's check the word count. Title should be 5-10 words. "Felixstowe Coastal Steps Seafront" is 6 words. Description is 1-2 sentences, let's make it 2 sentences? Wait, the user said 1-2 sentence. Let's make it one sentence. "Seafront scene in Felixstowe, England, with people swimming and walking near distinctive coastal steps, historic buildings, and clear daylight." That's one sentence. Or two: "Seafront scene in Felixstowe, England, shows people enjoying the beach near distinctive coastal steps. Historic buildings and clear daylight are visible in the background." But maybe better to combine. Let's do: "Seafront scene in Felixstowe, England, with people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Keywords need to be unique, comma-separated. Let's list them: Felixstowe, Suffolk, UK, seafront, coastal steps, beach, people, swimming, historic buildings, daylight, Europe, East Suffolk, England, GBR, seagulls, seaside, promenade. That's 17 keywords, which is within 10-18. Let's check if all are unique. Yes. Now, make sure to include the authoritative context: capture date/time and GPS, but in the description, maybe mention the date/time? Wait, the description should combine relevant context with main visible subject, setting, action, lighting, distinctive details. So maybe: "Seafront scene in Felixstowe, England, captured on 2026-08-13 at 17:23:18 UTC+01:00, showing people swimming and walking near distinctive coastal steps, under clear daylight with historic buildings in the background."

Wait, the description hint says "Seafront, Felixstowe, England, UK, GBR" so maybe include the date. Let's adjust description: "Seafront scene in Felixstowe, England, captured on 
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
- *Generation tokens:* 126
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;

#### Complete output

```text
 Title: Felixstowe Seafront in Suffolk, England, UK.
Description: The seafront in Felixstowe, Suffolk, England, UK, is bustling with activity as people enjoy the sunny day. The sandy beach is lined with large rocks, and the clear blue water of the sea laps gently against the shore. In the background, there are buildings, trees, and a clock tower. The sky is a bright blue, indicating good weather.
Keywords: Felixstowe, Suffolk, England, UK, seafront, beach, rocks, water, buildings, trees, clock tower, blue sky, sunny day.<end_of_utterance>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Usability           | Observations                                                         |
|-------------------------------------------------------|---------------------|----------------------------------------------------------------------|
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | unusable            | missing required fields; echoes instructions; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16                       | unusable            | missing required fields; echoes instructions                         |
| mlx-community/gemma-3n-E4B-it-bf16                    | unusable            | missing required fields                                              |
| mlx-community/llava-v1.6-mistral-7b-8bit              | unusable            | missing required fields                                              |
| mlx-community/MolmoPoint-8B-fp16                      | unusable            | missing required fields                                              |
| mlx-community/nanoLLaVA-1.5-4bit                      | unusable            | missing required fields; echoes instructions                         |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | unusable            | missing required fields                                              |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/gemma-3-27b-it-qat-4bit                 | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/InternVL3-8B-bf16                       | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/MiniCPM-V-4.6-8bit                      | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Molmo-7B-D-0924-8bit                    | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Phi-3.5-vision-instruct-bf16            | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/pixtral-12b-8bit                        | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen3.6-27B-mxfp8                       | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Step-3.7-Flash-oQ2e                     | usable_with_caveats | title/keyword constraints failed                                     |
| Qwen/Qwen3-VL-2B-Instruct                             | usable_with_caveats | title/keyword constraints failed                                     |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                             | Runtime identity                                    | Performance                                                                                 |
|---------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                  | rev ed71acdae079; Lfm2VlProcessor; stop completed   | 355 prompt / 76 generated; 515 tok/s; 1.2 GB peak; cleanup 0.000132/0.0 GB active/cache     |
| mlx-community/gemma-4-26b-a4b-it-4bit             | rev 0d77464eeb23; Gemma4Processor; stop completed   | 587 prompt / 94 generated; 129 tok/s; 16 GB peak; cleanup 0.0118/0.0 GB active/cache        |
| mlx-community/gemma-4-31b-it-4bit                 | rev 696d436c4047; Gemma4Processor; stop completed   | 587 prompt / 89 generated; 27.2 tok/s; 20 GB peak; cleanup 0.0123/0.0 GB active/cache       |
| mlx-community/LFM2.5-VL-1.6B-bf16                 | rev 16a710cf8afc; Lfm2VlProcessor; stop completed   | 555 prompt / 122 generated; 156 tok/s; 4.1 GB peak; cleanup 0.00379/0.0 GB active/cache     |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit  | rev 8451adc50203; MllamaProcessor; stop completed   | 275 prompt / 101 generated; 11.7 tok/s; 15 GB peak; cleanup 0.00431/0.0 GB active/cache     |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | rev 7c992876448f; Mistral3Processor; stop completed | 2730 prompt / 120 generated; 45.6 tok/s; 12 GB peak; cleanup 0.0051/0.0 GB active/cache     |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit   | rev a962dcb09eee; Mistral3Processor; stop completed | 2729 prompt / 154 generated; 144 tok/s; 7.1 GB peak; cleanup 0.00562/0.0 GB active/cache    |
| mlx-community/Ornith-1.0-35B-bf16                 | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed  | 16548 prompt / 122 generated; 60.1 tok/s; 74 GB peak; cleanup 0.00706/0.0 GB active/cache   |
| mlx-community/Qwen3.5-35B-A3B-4bit                | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed  | 16548 prompt / 121 generated; 108 tok/s; 24 GB peak; cleanup 0.00857/0.0 GB active/cache    |
| mlx-community/Qwen3.5-9B-MLX-4bit                 | rev 938d8919941c; Qwen3VLProcessor; stop completed  | 16548 prompt / 120 generated; 87.7 tok/s; 10.0 GB peak; cleanup 0.00909/0.0 GB active/cache |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,836 x 5,952 pixels
- *Image size:* 60,138,414 bytes
- *Image SHA-256:* bc3fa055e1e116232f77aa68c1d8d22130a1f596762a98cb8cd691667bbdcab2

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-13 17:23:18 UTC+01:00
- GPS: 51.957967°N, 1.346900°E

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

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/SmolVLM2-2.2B-Instruct-mlx         | 844516024a1c4400d34489b89ee067d794e432ed |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | 32dae5c38006e20ac158bc94cd1d5967d19b2652 |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/X-Reasoner-7B-8bit                 | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Qwen3-VL-2B-Thinking-bf16          | c325e5ea14c215bb08fa0d668c81fa2581f9050b |
| mlx-community/GLM-4.1V-9B-Thinking-8bit          | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.14                                                                                                                                          |
| mlx                        | 0.32.1.dev20260814+3d23f7d87                                                                                                                    |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (174,684,032 bytes, sha256=58668b2e31837a33652e00f30dc03cf27d6342065e1b1f0e4d6a98e5cb3c6efe) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,964,544 bytes, sha256=52f0e996a394edbb31d237e62691b556d204a451f5d60a0434ce0a62566cae64)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
