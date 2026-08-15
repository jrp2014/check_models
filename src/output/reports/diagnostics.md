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
| none                           | 29    |
| observation needs reproduction | 11    |

Usability counts

| Usability           | Count |
|---------------------|-------|
| not evaluated       | 1     |
| unusable            | 16    |
| usable              | 10    |
| usable with caveats | 14    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 7     |
| Unrecognised model control tokens remain visible                                      | 3     |
| Required fields are missing or empty                                                  | 10    |
| Response repeats the task instructions instead of only returning the requested fields | 2     |
| Extra text appears before the Title field                                             | 5     |
| Response appears cut off at the token limit                                           | 4     |
| Internal reasoning block appears incomplete                                           | 1     |
| Conversation-role control tokens remain visible                                       | 2     |
| Title or keywords do not meet requested constraints                                   | 18    |
| Title, Description and Keywords copy all supplied hints unchanged                     | 1     |

## Triage

| Model                                                                                                      | Execution | Usability           | Maintainer status              | Observations                                                                                                       |
|------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [mlx-community/SmolVLM2-2.2B-Instruct-mlx](#diagnostic-mlx-community-smolvlm2-22b-instruct-mlx)            | crashed   | not_evaluated       | actionable_failure             | none                                                                                                               |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)              | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; extra text before Title; cut off at token limit; incomplete thinking block |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)   | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; extra text before Title; role tokens visible                               |
| [mlx-community/LFM2.5-VL-1.6B-bf16](#diagnostic-mlx-community-lfm25-vl-16b-bf16)                           | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)             | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; echoes instructions                                                        |
| [mlx-community/Qwen3-VL-2B-Instruct-bf16](#diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16)             | completed | unusable            | observation_needs_reproduction | repeated text; title/keyword constraints failed                                                                    |
| [mlx-community/X-Reasoner-7B-8bit](#diagnostic-mlx-community-x-reasoner-7b-8bit)                           | completed | unusable            | observation_needs_reproduction | repeated text; cut off at token limit; title/keyword constraints failed                                            |
| [Qwen/Qwen3-VL-2B-Instruct](#diagnostic-qwen-qwen3-vl-2b-instruct)                                         | completed | unusable            | observation_needs_reproduction | repeated text; title/keyword constraints failed                                                                    |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)   | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                             |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8) | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                             |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                    | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible; title/keyword constraints failed                                                           |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                 | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible; title/keyword constraints failed                                                              |

## Crashes requiring action

<a id="diagnostic-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### mlx-community/SmolVLM2-2.2B-Instruct-mlx

#### Root exception and chain

```text
builtins.ValueError: Image features and image tokens do not match: tokens: 81, features 1053
builtins.ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do not match: tokens: 81, features 1053
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
  '8', '1', '5', '-', '1', '5', '5', '9', '4', '6', '_', 'D', 'S', 'C', '0',
  '1', '5', '6', '8', '.', 'j', 'p', 'g']   Prompt:  User:&lt;image&gt;Create
  British-English catalogue metadata from the image and supplied context.
  Treat any capture date/time and GPS as authoritative facts, but do not claim
  they are visible. Descriptive hints may be incomplete or wrong: retain
  details supported by the image, correct conflicts, and add important visible
  details. Prefer image evidence when a hint conflicts, and omit uncertain
  details.  Context: Authoritative context: - Capture date/time: 2026-08-15
  15:59:46 UTC+01:00 - GPS: 51.128800°N, 1.319100°E  Descriptive hints: -
  Title hint: Dover Castle, Dover, England, UK, GBR, Europe - Description
  hint: An exterior view of a historic medieval stone castle, featuring round
  towers, an arched entranceway, and a small bridge, built on a steep grassy
  hill under a partly cloudy sky. - Keyword hints: Adobe Stock, Any Vision,
  Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent,
  Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
  Write: - a concrete 5-10-word title; - a 1-2-sentence factual description
  combining relevant context with the main visible subject, setting, action,
  lighting, and distinctive details; - 10-18 unique, comma-separated keywords
  covering relevant context and visible details.  Return exactly these three
  sections and nothing else:
- *Unexpected special tokens:* ["&lt;|im_start|&gt;"]
- *Phase:* decode
- *Stage:* Model Error
- *Package:* mlx-vlm
- *Error type:* ValueError
- *Error message:* Model generation failed for
  mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do
  not match: tokens: 81, features 1053
- *Root error type:* ValueError
- *Root error message:* Image features and image tokens do not match: tokens:
  81, features 1053
- *Resolved model revision:* 844516024a1c4400d34489b89ee067d794e432ed
- *Processor class:* mlx_vlm.models.smolvlm.processing_smolvlm.SmolVLMProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* exception
- *Post-cleanup active memory (GB):* 0.009602184
- *Post-cleanup cache memory (GB):* 0.0
- *Configured EOS token ID:* 49279
- *Configured EOS token:* &lt;end_of_utterance&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

<details>
<summary>Complete traceback</summary>

```text
Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11966, in _run_generation_guarded
    return generate_once()
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12315, in _generate_once
    return strict_generate(
        model=model,
    ...<3 lines>...
        **generate_kwargs,
    )
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 1159, in generate
    for response in stream_generate(
                    ~~~~~~~~~~~~~~~^
        model, processor, prompt, image, audio, video, verbose=verbose, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ):
    ^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/generate/dispatch.py", line 978, in stream_generate
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
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/models/idefics3/idefics3.py", line 174, in _prepare_inputs_for_multimodal
    raise ValueError(
        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
    )
ValueError: Image features and image tokens do not match: tokens: 81, features 1053

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12893, in process_image_with_model
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
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 12328, in _run_model_generation
    output = _run_generation_guarded(
        params=params,
        generate_once=_generate_once,
    )
  File "~/Documents/AI/mlx/check_models/src/check_models.py", line 11973, in _run_generation_guarded
    raise _tag_exception_failure_phase(ValueError(msg), "decode") from gen_known_err
ValueError: Model generation failed for mlx-community/SmolVLM2-2.2B-Instruct-mlx: Image features and image tokens do not match: tokens: 81, features 1053

```

</details>

#### Captured stdout/stderr

```text
=== STDOUT ===
==========
Files: ['/', 'U', 's', 'e', 'r', 's', '/', 'j', 'r', 'p', '/', 'P', 'i', 'c', 't', 'u', 'r', 'e', 's', '/', 'P', 'r', 'o', 'c', 'e', 's', 's', 'e', 'd', '/', '2', '0', '2', '6', '0', '8', '1', '5', '-', '1', '5', '5', '9', '4', '6', '_', 'D', 'S', 'C', '0', '1', '5', '6', '8', '.', 'j', 'p', 'g'] 

Prompt: <|im_start|>User:<image>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-15 15:59:46 UTC+01:00
- GPS: 51.128800°N, 1.319100°E

Descriptive hints:
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue

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
Fetching 12 files: 100%|##########| 12/12 [00:00<00:00, 3113.43it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[00:45:18] ERROR    Generation error for mlx-community/SmolVLM2-2.2B-Instruct-mlx
                    ValueError: Image features and image tokens do not match: tokens: 81, features 1053
```

## Completed Runs with Observations

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — repeated text; missing required fields; extra text before Title; cut off at token limit; incomplete thinking block</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, unexpected_catalog_preamble, thinking_trace_incomplete
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Missing sections:* ["title"]
- *Repeated fragment:* keyword: "england"
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this task
  step by step. First, the title needs to be 5-10 words, concrete. The image
  is of Dover Castle, so maybe "Dover Castle Exterior View". Then the
  description should combine context (date/time, location) with the main
  subject, setting, etc. The image shows a historic medieval stone castle with
  round towers, arched entrance, steep grassy hill, partly cloudy sky. So the
  description should mention the castle, its features, the setting, and the
  time. Then keywords need to be 10-18 unique ones, covering context and
  visible details.  First, title: Let's make it "Dover Castle Exterior with
  Round Towers". Wait, need 5-10 words. Maybe "Dover Castle Medieval
  Exterior". Wait, let's check the hints. The title hint is Dover Castle,
  Dover, England... so maybe "Dover Castle Exterior View". That's 3 words,
  maybe add more. "Dover Castle Historic Exterior with Round Towers" – that's
  6 words. Okay.
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "incomplete_thinking_trace", "abrupt_tail"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001803312
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6344
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<think>Got it, let's tackle this task step by step. First, the title needs to be 5-10 words, concrete. The image is of Dover Castle, so maybe "Dover Castle Exterior View". Then the description should combine context (date/time, location) with the main subject, setting, etc. The image shows a historic medieval stone castle with round towers, arched entrance, steep grassy hill, partly cloudy sky. So the description should mention the castle, its features, the setting, and the time. Then keywords need to be 10-18 unique ones, covering context and visible details.

First, title: Let's make it "Dover Castle Exterior with Round Towers". Wait, need 5-10 words. Maybe "Dover Castle Medieval Exterior". Wait, let's check the hints. The title hint is Dover Castle, Dover, England... so maybe "Dover Castle Exterior View". That's 3 words, maybe add more. "Dover Castle Historic Exterior with Round Towers" – that's 6 words. Okay.

Description: Combine context (capture date/time, location) with main subject (historic medieval stone castle), setting (steep grassy hill), lighting (partly cloudy sky), distinctive details (round towers, arched entrance). So: "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured on 2026-08-15 at 15:59:46 UTC+01:00 in Dover, England." Wait, need 1-2 sentences. Let's make it concise. "A historic medieval stone castle with round towers and an arched entrance, built on a steep grassy hill under a partly cloudy sky, photographed in Dover, England on 2026-08-15." That's two sentences? Wait, no, 1-2 sentences. Let's try: "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured in Dover, England on 2026-08-15." That's one sentence. Wait, need to include the context (capture date/time, location) and the main visible subject, setting, etc. So maybe: "A historic medieval stone castle with round towers and an arched entrance, built on a steep grassy hill under a partly cloudy sky, photographed in Dover, England on 2026-08-15 15:59:46 UTC+01:00." That's a bit long, but maybe shorten. "A historic medieval stone castle with round towers and an arched entrance, situated on a steep grassy hill under a partly cloudy sky, captured in Dover, England on 2026-08-15." That works.

Keywords: Need 10-18 unique, comma-separated. From hints: Adobe Stock, Any Vision (but maybe not, since the image might not be from those, but the hints include them? Wait, the hints say "Adobe Stock, Any Vision" as keyword hints? Wait, the context says "Descriptive hints" include those, but maybe the keywords should include the ones from hints plus others. Let's list: Dover Castle, England, Europe, Kent, Castle, Fortress, Hill, Sky, Stone, Tower, Wall, ancient, architecture, blue, Britain, UK, United Kingdom, Arch, Bridge. Wait, the image has a small bridge, so Bridge. Let's check the hints: keyword hints are Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. So combine these, remove duplicates. Let's list unique ones: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue. Wait, need 10-18. Let's count: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue – that's 18, but maybe some can be combined. Wait, "United Kingdom" and "UK" are similar, but keep both? Or pick one. Let's do: Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, Wall, ancient, architecture, blue. That's 17. Add Bridge if visible. The image has a small bridge, so Bridge. So Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, Wall, ancient, architecture, blue, Bridge. That's
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — unusable — repeated text; missing required fields; extra text before Title; role tokens visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  unexpected_catalog_preamble, role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Missing sections:* ["title"]
- *Repeated fragment:* keyword: "kent"
- *Unexpected text before Title:* ◁think▷Okay, let's tackle this. First, I
  need to create a British-English catalogue metadata based on the image and
  the provided context. The user wants a 5-10 word title, a 1-2 sentence
  description, and 10-18 keywords.  Starting with the title: The image shows
  Dover Castle, which is in the context given. The castle is a historic
  medieval stone structure with round towers, an arched entrance, and a
  bridge. The setting is a grassy hill under a partly cloudy sky. So the title
  should be concise. Maybe "Dover Castle, Kent, UK: Medieval stone fortress on
  grassy hill with arched entrance and towers." That's 10 words.  Next, the
  description: It needs to combine context and visible details. The capture
  date and GPS are authoritative but not visible, so focus on the image. The
  castle is on a grassy hill, has stone walls, round towers, an arched
  entrance, and a bridge. The sky is partly cloudy. So the description could
  be: "Exterior view of Dover Castle, a medieval stone fortress on a grassy
  hill, featuring round towers, an arched entranceway, and a bridge, under a
  partly cloudy sky." That's 25 words, which is a bit long but fits the
  requirement.
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003654752
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1296
- *Generation tokens:* 834
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The user wants a 5-10 word title, a 1-2 sentence description, and 10-18 keywords. 

Starting with the title: The image shows Dover Castle, which is in the context given. The castle is a historic medieval stone structure with round towers, an arched entrance, and a bridge. The setting is a grassy hill under a partly cloudy sky. So the title should be concise. Maybe "Dover Castle, Kent, UK: Medieval stone fortress on grassy hill with arched entrance and towers." That's 10 words.

Next, the description: It needs to combine context and visible details. The capture date and GPS are authoritative but not visible, so focus on the image. The castle is on a grassy hill, has stone walls, round towers, an arched entrance, and a bridge. The sky is partly cloudy. So the description could be: "Exterior view of Dover Castle, a medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a bridge, under a partly cloudy sky." That's 25 words, which is a bit long but fits the requirement.

Keywords: Need 10-18 unique terms. From the context and image: Dover Castle, Kent, UK, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, blue (sky), United Kingdom, Britain, fortification, wall. Let me count: Dover Castle, Kent, UK, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, blue, United Kingdom, Britain, fortification, wall. That's 17 keywords. Need to check if they are unique and relevant. Maybe "UK" and "United Kingdom" are redundant, so replace one. Let's remove "United Kingdom" and keep UK. Also, "Britain" is similar to UK. So revised: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue sky, Kent, England. Wait, Kent is already in the title. Maybe include "Dover Castle" as a keyword. Let me check again. The context mentions Dover Castle, so that's essential. So final keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue sky, UK. That's 16 keywords. Maybe add "sky" instead of "blue sky" but "blue" is in the hints. Hmm, the hints have "blue" as a keyword. So include that. So final keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, blue, UK. That's 16. Maybe add "sky" instead of "blue" if needed, but the hint says to prefer image evidence. The sky is partly cloudy, so "partly cloudy sky" is better. So keep "partly cloudy sky". Alright, that should work.◁/think▷Title: Dover Castle, Kent, UK: Medieval stone fortress on grassy hill with arched entrance and towers  
Description: Exterior view of Dover Castle, a medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a bridge, under a partly cloudy sky.  
Keywords: Dover Castle, Kent, England, Europe, stone fortress, medieval, round towers, arched entrance, bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortification, wall, UK<|im_assistant|>
```

</details>

<a id="diagnostic-mlx-community-lfm25-vl-16b-bf16"></a>

<details>
<summary>mlx-community/LFM2.5-VL-1.6B-bf16 — unusable — repeated text; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/LFM2.5-VL-1.6B-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type lfm2_vl)
- *Repeated fragment:* keyword: "blue"
- *Title word count:* 13
- *Requested title word range:* [5, 10]
- *Keyword count:* 454
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["uk", "architecture", "blue", "castle"]
- *Token-cap degradation evidence:* ["repetitive_tail"]
- *Resolved model revision:* 16a710cf8afca206ff16a95a4ad6fe657f876ce1
- *Processor class:* transformers.models.lfm2_vl.processing_lfm2_vl.Lfm2VlProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.003785824
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 603
- *Generation tokens:* 1000
- *Configured EOS token ID:* 7
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title: Dover Castle, England, UK, Europe, Fortress, Medieval, Stone, Tower, Hill, Sky, Architecture, Blue

Description: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle's imposing structure and intricate design reflect its rich history and architectural significance.

Keywords: Dover Castle, England, UK, Europe, Fortress, Medieval, Stone, Tower, Hill, Sky, Architecture, Blue, Castle, Kent, UK, Ancient, Blue, Architecture, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle, Blue, Castle,
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated text; missing required fields; echoes instructions</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  prompt_instruction_echo
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* phrase: "europe, fortress, hill, kent,..."
- *Echoed instruction fragments:* ["title hint:", "description hint:",
  "keyword hints:"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.014025992
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4435
- *Generation tokens:* 726
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
- Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe, Fortress, Hill, Kent, Sky, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
```

</details>

<a id="diagnostic-mlx-community-qwen3-vl-2b-instruct-bf16"></a>

<details>
<summary>mlx-community/Qwen3-VL-2B-Instruct-bf16 — unusable — repeated text; title/keyword constraints failed</summary>

### mlx-community/Qwen3-VL-2B-Instruct-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Repeated fragment:* keyword: "stone"
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 33
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["england", "uk", "europe", "fortress", "castle",
  "stone", "tower", "wall", "hill", "sky", "ancient", "architecture", "united
  kingdom"]
- *Resolved model revision:* c8a67a84327484ba87f5ec4f8fb927cdafd791aa
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00775079
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16506
- *Generation tokens:* 118
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Dover Castle Exterior

Description:
An exterior view of Dover Castle, a historic medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a small bridge under a partly cloudy sky.

Keywords: Dover Castle, England, UK, Europe, Fortress, Castle, Stone, Tower, Wall, Hill, Sky, Arch, Ancient, Architecture, Britain, Kent, United Kingdom, Stone, Wall, Castle, England, Europe, Fortress, Hill, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
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
- *Repeated fragment:* keyword: "kent"
- *Title word count:* 4
- *Requested title word range:* [5, 10]
- *Keyword count:* 357
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["kent", "england", "uk", "europe", "medieval",
  "stone", "round towers", "small bridge", "grassy hill", "partly cloudy sky",
  "architecture", "ancient", "arched entranceway", "historic", "fortress"]
- *Token-cap degradation evidence:* ["repetitive_tail", "abrupt_tail"]
- *Resolved model revision:* 21732e74613b465bc98e9d5ec210aba5c7adbcc1
- *Processor class:* mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.01017566
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16517
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Dover Castle, Kent, England

Description:
A historic medieval stone castle with round towers, an arched entranceway, and a small bridge, situated on a steep grassy hill under a partly cloudy sky, captured on 2026-08-15 at 15:59:46 UTC+01:00.

Keywords: Dover Castle, Kent, England, UK, Europe, medieval, stone, round towers, arched entrance, small bridge, grassy hill, partly cloudy sky, historic architecture, ancient fortress, stone walls, medieval castle, United Kingdom, Europe, architecture, blue sky, greenery, stone bridge, medieval structure, stone towers, historical landmark, Kent, England, UK, Europe, ancient, architecture, blue, grass, sky, stone, tower, wall, entrance, bridge, castle, Dover, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent, England, UK, Europe, medieval, stone, round towers, arched entranceway, small bridge, grassy hill, partly cloudy sky, historic, architecture, ancient, fortress, Kent
```

</details>

<a id="diagnostic-qwen-qwen3-vl-2b-instruct"></a>

<details>
<summary>Qwen/Qwen3-VL-2B-Instruct — unusable — repeated text; title/keyword constraints failed</summary>

### Qwen/Qwen3-VL-2B-Instruct

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type qwen3_vl)
- *Repeated fragment:* keyword: "stone"
- *Title word count:* 3
- *Requested title word range:* [5, 10]
- *Keyword count:* 33
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["england", "uk", "europe", "fortress", "castle",
  "stone", "tower", "wall", "hill", "sky", "ancient", "architecture", "united
  kingdom"]
- *Resolved model revision:* 89644892e4d85e24eaac8bacfd4f463576704203
- *Processor class:* mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.000443432
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 16506
- *Generation tokens:* 118
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Title:
Dover Castle Exterior

Description:
An exterior view of Dover Castle, a historic medieval stone fortress on a grassy hill, featuring round towers, an arched entranceway, and a small bridge under a partly cloudy sky.

Keywords: Dover Castle, England, UK, Europe, Fortress, Castle, Stone, Tower, Wall, Hill, Sky, Arch, Ancient, Architecture, Britain, Kent, United Kingdom, Stone, Wall, Castle, England, Europe, Fortress, Hill, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue
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
- *Prompt tokens:* 615
- *Generation tokens:* 85
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: Exterior view of Dover Castle in Kent, England
Description: An exterior view of the medieval stone fortifications of Dover Castle, featuring round towers and an arched entranceway atop a steep grassy hill under a partly cloudy sky.
Keywords: Dover Castle, England, Kent, UK, castle, fortress, stone, tower, architecture, medieval, ancient, arch, hill, history, Britain, Europe
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
- *Post-cleanup active memory (GB):* 0.01017568
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 615
- *Generation tokens:* 82
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: Medieval Stone Towers of Dover Castle in Kent England
Description: An exterior view of the historic medieval stone castle featuring round towers and an arched entranceway situated on a steep grassy hill under a cloudy sky.
Keywords: Dover Castle, England, UK, Britain, Castle, Fortress, Hill, Stone, Tower, Wall, ancient, architecture, Arch, Medieval, Kent, Europe
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
- *Title word count:* 4
- *Requested title word range:* [5, 10]
- *Keyword count:* 16
- *Requested keyword count range:* [10, 18]
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002425916
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6348
- *Generation tokens:* 83
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Dover Castle, Kent, England
Description: An exterior view of Dover Castle, a historic medieval stone fortress with round towers and an arched entrance, perched on a steep grassy hill under a partly cloudy sky.
Keywords: Dover Castle, England, UK, Kent, Castle, Fortress, Stone, Tower, Arch, Hill, Sky, Architecture, Ancient, Britain, Europe, Wall<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — usable_with_caveats — role tokens visible; title/keyword constraints failed</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Role-boundary tokens in output:* ["&lt;end_of_utterance&gt;"]
- *Title word count:* 10
- *Requested title word range:* [5, 10]
- *Keyword count:* 24
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["europe", "sky"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.00268806
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2626
- *Generation tokens:* 141
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
 Title: Dover Castle, England, UK, Medieval Stone Castle on a Hill.
Description: Dover Castle, a historic medieval stone castle, features round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky. The castle's architecture is characterized by its stone walls and towers, and it is a prominent landmark in Dover, England, UK.
Keywords: castle, dover, england, uk, europe, stone, tower, hill, sky, arch, bridge, grass, cloudy, sky, blue, ancient, architecture, fortification, kent, adobe stock, any vision, united kingdom, europe, gbr.<end_of_utterance>
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
| mlx-community/FastVLM-0.5B-bf16                       | unusable            | missing required fields                                              |
| mlx-community/gemma-3n-E4B-it-bf16                    | unusable            | missing required fields                                              |
| mlx-community/llava-v1.6-mistral-7b-8bit              | unusable            | missing required fields                                              |
| mlx-community/MiniCPM-V-4.6-8bit                      | unusable            | missing required fields; extra text before Title                     |
| mlx-community/MolmoPoint-8B-fp16                      | unusable            | missing required fields                                              |
| mlx-community/nanoLLaVA-1.5-4bit                      | unusable            | missing required fields                                              |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | unusable            | extra text before Title; title/keyword constraints failed            |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | unusable            | extra text before Title; title/keyword constraints failed            |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/gemma-3-27b-it-qat-4bit                 | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/InternVL3-8B-bf16                       | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Molmo-7B-D-0924-8bit                    | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/pixtral-12b-8bit                        | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | usable_with_caveats | title/keyword constraints failed; draft hints copied unchanged       |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | usable_with_caveats | title/keyword constraints failed                                     |
| mlx-community/Step-3.7-Flash-oQ2e                     | usable_with_caveats | title/keyword constraints failed                                     |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                             | Runtime identity                                    | Performance                                                                               |
|---------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------------------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                  | rev ed71acdae079; Lfm2VlProcessor; stop completed   | 400 prompt / 98 generated; 503 tok/s; 1.1 GB peak; cleanup 0.000132/0.0 GB active/cache   |
| mlx-community/gemma-4-26b-a4b-it-4bit             | rev 0d77464eeb23; Gemma4Processor; stop completed   | 619 prompt / 98 generated; 91.3 tok/s; 16 GB peak; cleanup 0.0118/0.0 GB active/cache     |
| mlx-community/gemma-4-31b-it-4bit                 | rev 696d436c4047; Gemma4Processor; stop completed   | 619 prompt / 73 generated; 16.3 tok/s; 20 GB peak; cleanup 0.0123/0.0 GB active/cache     |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit  | rev 8451adc50203; MllamaProcessor; stop completed   | 314 prompt / 106 generated; 15.0 tok/s; 15 GB peak; cleanup 0.00431/0.0 GB active/cache   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | rev 7c992876448f; Mistral3Processor; stop completed | 3228 prompt / 143 generated; 58.3 tok/s; 14 GB peak; cleanup 0.0051/0.0 GB active/cache   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit   | rev a962dcb09eee; Mistral3Processor; stop completed | 3227 prompt / 104 generated; 172 tok/s; 9.0 GB peak; cleanup 0.00562/0.0 GB active/cache  |
| mlx-community/Ornith-1.0-35B-bf16                 | rev 9ef631ad2d0c; Qwen3VLProcessor; stop completed  | 16522 prompt / 105 generated; 48.9 tok/s; 74 GB peak; cleanup 0.00706/0.0 GB active/cache |
| mlx-community/Phi-3.5-vision-instruct-bf16        | rev d8da684308c2; Phi3VProcessor; stop completed    | 1137 prompt / 110 generated; 50.9 tok/s; 9.4 GB peak; cleanup 0.00713/0.0 GB active/cache |
| mlx-community/Qwen3.5-35B-A3B-4bit                | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed  | 16522 prompt / 116 generated; 90.4 tok/s; 24 GB peak; cleanup 0.00857/0.0 GB active/cache |
| mlx-community/Qwen3.6-27B-mxfp8                   | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed  | 16522 prompt / 114 generated; 13.0 tok/s; 35 GB peak; cleanup 0.0096/0.0 GB active/cache  |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 8,880 pixels
- *Image size:* 79,069,278 bytes
- *Image SHA-256:* 771ab1bcadbb99020fb1a6270d6f36e8dd613cc3132c390bed714290bda2dd05

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-15 15:59:46 UTC+01:00
- GPS: 51.128800°N, 1.319100°E

Descriptive hints:
- Title hint: Dover Castle, Dover, England, UK, GBR, Europe
- Description hint: An exterior view of a historic medieval stone castle, featuring round towers, an arched entranceway, and a small bridge, built on a steep grassy hill under a partly cloudy sky.
- Keyword hints: Adobe Stock, Any Vision, Arch, Britain, Castle, Dover Castle, England, Europe, Fortress, Hill, Kent, Sky, Stone, Tower, UK, United Kingdom, Wall, ancient, architecture, blue

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
| mlx-community/SmolVLM2-2.2B-Instruct-mlx      | 844516024a1c4400d34489b89ee067d794e432ed |
| mlx-community/GLM-4.1V-9B-Thinking-8bit       | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16  | fb254434d4026bee7aa840dea1c5d59feea8fd48 |
| mlx-community/LFM2.5-VL-1.6B-bf16             | 16a710cf8afca206ff16a95a4ad6fe657f876ce1 |
| mlx-community/paligemma2-3b-pt-896-4bit       | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/Qwen3-VL-2B-Instruct-bf16       | c8a67a84327484ba87f5ec4f8fb927cdafd791aa |
| mlx-community/X-Reasoner-7B-8bit              | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 |
| Qwen/Qwen3-VL-2B-Instruct                     | 89644892e4d85e24eaac8bacfd4f463576704203 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit  | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/GLM-4.6V-nvfp4                  | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Idefics3-8B-Llama3-bf16         | 8c2a30c48864f3251701b7bde40f601d25535098 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.14                                                                                                                                          |
| mlx                        | 0.32.1.dev20260815+9ab977b56                                                                                                                    |
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
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,981,840 bytes, sha256=e403248438590fc1042a115709e9964b09d5bb23bb16c262e88fd314bce11ad5)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
