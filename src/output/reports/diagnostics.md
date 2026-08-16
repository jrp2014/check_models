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
| none                           | 32    |
| observation needs reproduction | 9     |

Usability counts

| Usability           | Count |
|---------------------|-------|
| unusable            | 9     |
| usable              | 20    |
| usable with caveats | 12    |

Observation counts

| Observation                                                                           | Count |
|---------------------------------------------------------------------------------------|-------|
| Response repeats the same text                                                        | 3     |
| Unrecognised model control tokens remain visible                                      | 3     |
| Required fields are missing or empty                                                  | 7     |
| Response repeats the task instructions instead of only returning the requested fields | 2     |
| Extra text appears before the Title field                                             | 5     |
| Response appears cut off at the token limit                                           | 5     |
| Internal reasoning block appears incomplete                                           | 1     |
| Conversation-role control tokens remain visible                                       | 2     |
| Title or keywords do not meet requested constraints                                   | 10    |

## Triage

| Model                                                                                                           | Execution | Usability           | Maintainer status              | Observations                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|--------------------------------------------------------------------------------------------------------------|
| [mlx-community/Llama-3.2-11B-Vision-Instruct-8bit](#diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit) | completed | unusable            | observation_needs_reproduction | repeated text; extra text before Title; cut off at token limit; title/keyword constraints failed             |
| [mlx-community/llava-v1.6-mistral-7b-8bit](#diagnostic-mlx-community-llava-v16-mistral-7b-8bit)                 | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; cut off at token limit                                               |
| [mlx-community/paligemma2-3b-pt-896-4bit](#diagnostic-mlx-community-paligemma2-3b-pt-896-4bit)                  | completed | unusable            | observation_needs_reproduction | repeated text; missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| [mlx-community/diffusiongemma-26B-A4B-it-8bit](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-8bit)        | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                       |
| [mlx-community/diffusiongemma-26B-A4B-it-mxfp8](#diagnostic-mlx-community-diffusiongemma-26b-a4b-it-mxfp8)      | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                       |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                         | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                                                       |
| [mlx-community/Idefics3-8B-Llama3-bf16](#diagnostic-mlx-community-idefics3-8b-llama3-bf16)                      | completed | unusable            | observation_needs_reproduction | missing required fields; role tokens visible                                                                 |
| [mlx-community/GLM-4.1V-9B-Thinking-8bit](#diagnostic-mlx-community-glm-41v-9b-thinking-8bit)                   | completed | unusable            | observation_needs_reproduction | extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-bf16](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16)        | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible                                                                                          |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

<details>
<summary>mlx-community/Llama-3.2-11B-Vision-Instruct-8bit — unusable — repeated text; extra text before Title; cut off at token limit; title/keyword constraints failed</summary>

### mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, token_cap_truncation,
  unexpected_catalog_preamble, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type mllama)
- *Repeated fragment:* keyword: "household frames"
- *Unexpected text before Title:* Here is the information in the format you
  requested:
- *Title word count:* 6
- *Requested title word range:* [5, 10]
- *Keyword count:* 313
- *Requested keyword count range:* [10, 18]
- *Duplicate keywords:* ["household frames"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* 8451adc50203b50b8f4199e75e753fb9c06e2af6
- *Processor class:* mlx_vlm.models.mllama.processing_mllama.MllamaProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00431012
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 107
- *Generation tokens:* 1000
- *Configured EOS token ID:* 128009
- *Configured EOS token:* &lt;|eot_id|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Here is the information in the format you requested:

**Title:** Two Cats on a Pink Blanket

**Description:** Two tabby cats are sleeping on a pink blanket, with two remote controls nearby. The cats are lying on their backs, with their paws stretched out and their eyes closed.

**Keywords:** Tabby cats, pink blanket, remote controls, sleeping, feline, domestic, household, comfort, relaxation, domesticated, household pets, household items, household appliances, household furniture, household decor, household textiles, household linens, household blankets, household comforters, household pillows, household throws, household rugs, household mats, household carpets, household flooring, household walls, household ceilings, household windows, household doors, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames, household frames,
```

</details>

<a id="diagnostic-mlx-community-llava-v16-mistral-7b-8bit"></a>

<details>
<summary>mlx-community/llava-v1.6-mistral-7b-8bit — unusable — repeated text; missing required fields; cut off at token limit</summary>

### mlx-community/llava-v1.6-mistral-7b-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation
- *Arch supported by installed mlx-vlm:* yes (model_type llava_next)
- *Missing sections:* ["title", "description", "keywords"]
- *Repeated fragment:* feline,
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* b8df5f329d95a7abe6429ed46093f9b84e8e6396
- *Processor class:* mlx_vlm.models.llava_next.processing_llava_next.LlavaNextProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.012453096
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2469
- *Generation tokens:* 1000
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;/s&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
cat, sleeping, pink blanket, remote control, striped, tabby, kitten, cozy, domestic, indoor, relaxation, comfort, home, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline, feline
```

</details>

<a id="diagnostic-mlx-community-paligemma2-3b-pt-896-4bit"></a>

<details>
<summary>mlx-community/paligemma2-3b-pt-896-4bit — unusable — repeated text; missing required fields; echoes instructions; extra text before Title; cut off at token limit</summary>

### mlx-community/paligemma2-3b-pt-896-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* repeated_output, missing_requested_sections,
  token_cap_truncation, prompt_instruction_echo, unexpected_catalog_preamble
- *Arch supported by installed mlx-vlm:* yes (model_type paligemma)
- *Missing sections:* ["title"]
- *Repeated fragment:* phrase: "- 10-18 unique, comma-separate..."
- *Echoed instruction fragments:* ["return exactly these three sections",
  "return exactly these three sections and nothing else"]
- *Unexpected text before Title:* Action: Lighting: Brand: Species: Period:
  Event: Date: Name: Intent: Brand: Date: Name:
- *Token-cap degradation evidence:* ["missing_sections", "repetitive_tail",
  "abrupt_tail"]
- *Resolved model revision:* a26bac48c7a661dfdafe1799c90177f818e79925
- *Processor class:* mlx_vlm.models.paligemma.processing_paligemma.PaliGemmaProcessor
- *Tokenizer class:* transformers.models.gemma.tokenization_gemma.GemmaTokenizer
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.0141079
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4207
- *Generation tokens:* 1000
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
Action:
Lighting:
Brand:
Species:
Period:
Event:
Date:
Name:
Intent:
Brand:
Date:
Name:
Description:
Action:
Lighting:
Keywords:
Return exactly these three sections and nothing else:
Title:
Description:
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-18 unique, comma-separated keywords.
- 10-1
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
- *Post-cleanup active memory (GB):* 0.010257588
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 386
- *Generation tokens:* 75
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: Two tabby cats sleeping on a pink blanket
Description: Two striped tabby cats rest on a pink blanket draped over a red sofa. Two remote controls are visible near the cats.
Keywords: cats, tabby, sleeping, resting, pink, blanket, sofa, red, indoor, domestic, striped, feline, pets, remote control, lounging
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
- *Post-cleanup active memory (GB):* 0.010257588
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 386
- *Generation tokens:* 68
- *Configured EOS token ID:* 1
- *Configured EOS token:* &lt;eos&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|channel>thought
<channel|>Title: Two tabby cats sleeping on a pink blanket
Description: Two tabby cats lie curled up on a pink blanket next to remote controls on a sofa.
Keywords: cats, tabby, sleeping, pink, blanket, sofa, indoor, remote, controls, feline, resting, fur, striped, domestic, lounging
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
- *Post-cleanup active memory (GB):* 0.002425912
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 499
- *Generation tokens:* 89
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Two cats on a pink couch
Description: Two cats lie on a pink couch, each near a remote control, with one cat’s head resting on the fabric and the other stretched out with eyes closed.
Keywords: cats, pink couch, remote controls, sleeping, relaxed, striped fur, feline, domestic, indoor, pets, lounging, two animals, soft fabric, calm, cozy, household, companionship<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-idefics3-8b-llama3-bf16"></a>

<details>
<summary>mlx-community/Idefics3-8B-Llama3-bf16 — unusable — missing required fields; role tokens visible</summary>

### mlx-community/Idefics3-8B-Llama3-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type idefics3)
- *Missing sections:* ["title", "description", "keywords"]
- *Role-boundary tokens in output:* ["&lt;end_of_utterance&gt;"]
- *Resolved model revision:* 8c2a30c48864f3251701b7bde40f601d25535098
- *Processor class:* mlx_vlm.models.idefics3.processing_idefics3.Idefics3Processor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.002688056
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2418
- *Generation tokens:* 13
- *Configured EOS token ID:* 128001
- *Configured EOS token:* &lt;|end_of_text|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
 Two cats laying on a pink blanket with remotes.<end_of_utterance>
```

</details>

<a id="diagnostic-mlx-community-glm-41v-9b-thinking-8bit"></a>

<details>
<summary>mlx-community/GLM-4.1V-9B-Thinking-8bit — unusable — extra text before Title; cut off at token limit; incomplete thinking block; title/keyword constraints failed</summary>

### mlx-community/GLM-4.1V-9B-Thinking-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* unusable
- *Maintainer status:* observation_needs_reproduction
- *Observations:* token_cap_truncation, unexpected_catalog_preamble,
  thinking_trace_incomplete, catalog_constraint_violation
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Unexpected text before Title:* &lt;think&gt;Got it, let's tackle this.
  First, the title needs to be 5-10 words, concrete. Looking at the image: two
  cats on a pink blanket with remotes. So maybe "Cats on Pink Blanket with
  Remotes" or something similar. Then description: need to mention the main
  subject (two cats), setting (pink blanket, red couch), action (lying down),
  lighting (bright), distinctive details (remotes, cat positions). Then
  keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline,
  relaxed, home. Let's make sure to omit uncertain details.
- *Thinking trace markers:* ["&lt;think&gt;"]
- *Title word count:* 21
- *Requested title word range:* [5, 10]
- *Keyword count:* 13
- *Requested keyword count range:* [10, 18]
- *Token-cap degradation evidence:* ["incomplete_thinking_trace",
  "unfinished_list"]
- *Resolved model revision:* 9677807f106500eb7690391c27645d59f6855cfb
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.001803312
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 495
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<think>Got it, let's tackle this. First, the title needs to be 5-10 words, concrete. Looking at the image: two cats on a pink blanket with remotes. So maybe "Cats on Pink Blanket with Remotes" or something similar. Then description: need to mention the main subject (two cats), setting (pink blanket, red couch), action (lying down), lighting (bright), distinctive details (remotes, cat positions). Then keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Let's make sure to omit uncertain details. 

Title: "Cats Resting on Pink Blanket with Remotes" – wait, 5-10 words. Let's check. "Two Cats on Pink Blanket with Remotes" is good. 

Description: Two tabby cats lie on a pink blanket atop a red couch, with remote controls nearby, in bright lighting. – need to make it 1-2 sentences. So: "Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, under bright lighting." 

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Let's check unique ones, comma-separated. 

Now, let's structure each part:

Title: "Cats on Pink Blanket with Remotes" (5 words, concrete)

Description: Two tabby cats lie on a pink blanket on a red couch, with remote controls nearby, in bright lighting. (1 sentence, factual)

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home (10 words, unique)

Wait, need to make sure all are clearly visible. The image has two cats, pink blanket, red couch, remotes. So description should mention those. Let's adjust description: "Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, under bright lighting." That's factual, no unsupported details. 

Keywords: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home. Are these unique? Let's see. Maybe add "pet" or "cats" is there. Wait, need 10-18. Let's check: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home, couch, red (but red is part of couch, maybe not). Wait, the couch is red, but maybe include "red couch" as a keyword? But need unique. Let's do: cats, blanket, couch, remotes, pink, lying, domestic, feline, relaxed, home, pet, indoor, resting, tabby, two. That's 12, which is between 10-18. 

Wait, let's check the image again. Two tabby cats, so "tabby" can be a keyword. "resting" is action. "indoor" setting. "pink" blanket. "remotes" are there. "couch" is red. So keywords: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home, lying, domestic. That's 11, which is good. 

Now, final check:

Title: "Cats on Pink Blanket with Remotes" (5 words, concrete)

Description: Two tabby cats rest on a pink blanket on a red couch, with remote controls nearby, in bright lighting. (1 sentence, factual)

Keywords: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home (or add more, but need unique). Wait, the image has two cats, so "two" could be a keyword. Let's do: cats, tabby, blanket, couch, remotes, pink, resting, indoor, relaxed, home, two. That's 11. 

Wait, the user said "unique, comma-separated keywords" – so no duplicates. So "cats" once, "tabby" once, "blanket" once, "couch" once, "remotes" once, "pink" once, "resting" once, "indoor" once, "relaxed" once, "home" once, "two" once. That's 10. Wait, 10-18, so maybe add "feline" (same as cat), "lying" (same as resting). Hmm. Let's make sure each keyword is unique. Let's go with:

Title: "Cats on Pink Blanket with Remotes"

Description: Two tabby cats lie on a pink blanket on a red couch, with remote controls nearby, in bright lighting.

Keywords: cats, tabby, blanket,
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-bf16 — usable_with_caveats — role tokens visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* fb254434d4026bee7aa840dea1c5d59feea8fd48
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.003654748
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 489
- *Generation tokens:* 562
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷So, let's analyze the image. The image shows two tabby cats on a pink fabric surface. There are two remote controls, one near each cat. The cats are lying down, one facing left, the other facing down. The lighting is bright, making colors vivid. The scene is indoors, likely a living room. 

First, the title: need a 5-10 word title. Maybe "Two tabby cats resting on pink fabric with remotes." Wait, check word count. "Two tabby cats rest on pink fabric with remotes." That's 8 words. Good.

Description: 1-2 sentences. Describe main subject (two tabby cats), setting (pink fabric, likely couch), action (resting), lighting (bright, vivid colors), distinctive details (remotes, their positions). So: "Two tabby cats lie on a pink fabric surface, with two remote controls nearby, under bright lighting that enhances vivid colors." Wait, maybe shorter. "Two tabby cats rest on pink fabric; two remotes nearby, bright lighting highlights their fur patterns." Hmm, need to be concise. Let's refine: "Two tabby cats lie on pink fabric with two remote controls, under bright lighting emphasizing their striped fur." That's 2 sentences. Wait, the instruction says 1-2 sentences. Let's make it better: "Two tabby cats rest on a pink fabric surface, with two remote controls nearby, under bright lighting that highlights their striped patterns." That's 2 sentences.

Keywords: 10-18 unique, comma-separated. Keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment, relaxed cats. Let's count: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Maybe add "two cats" but that's redundant. Wait, "two tabby cats" is already mentioned. So keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment. That's 10. Check if all are unique and visible. Yes.◁/think▷Title: Two tabby cats rest on pink fabric with remotes nearby  
Description: Two tabby cats lie on a pink fabric surface, with two remote controls positioned near them, under bright lighting that highlights their striped fur patterns.  
Keywords: tabby cats, pink fabric, remote controls, bright lighting, striped fur, resting, indoor setting, domestic scene, relaxed posture, feline, domestic environment<|im_assistant|>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Usability           | Observations                                                                                  |
|-------------------------------------------------------|---------------------|-----------------------------------------------------------------------------------------------|
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | unusable            | missing required fields; echoes instructions; extra text before Title; cut off at token limit |
| mlx-community/FastVLM-0.5B-bf16                       | unusable            | missing required fields                                                                       |
| mlx-community/MiniCPM-V-4.6-8bit                      | unusable            | missing required fields; extra text before Title                                              |
| mlx-community/nanoLLaVA-1.5-4bit                      | unusable            | missing required fields                                                                       |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/gemma-3n-E4B-it-bf16                    | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Ornith-1.0-35B-bf16                     | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | usable_with_caveats | title/keyword constraints failed                                                              |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | usable_with_caveats | title/keyword constraints failed                                                              |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                             | Runtime identity                                        | Performance                                                                              |
|---------------------------------------------------|---------------------------------------------------------|------------------------------------------------------------------------------------------|
| mlx-community/gemma-3-27b-it-qat-4bit             | rev fc4e000f32af; Gemma3Processor; stop completed       | 379 prompt / 90 generated; 30.3 tok/s; 18 GB peak; cleanup 0.0108/0.0 GB active/cache    |
| mlx-community/gemma-4-26b-a4b-it-4bit             | rev 0d77464eeb23; Gemma4Processor; stop completed       | 390 prompt / 73 generated; 125 tok/s; 16 GB peak; cleanup 0.0119/0.0 GB active/cache     |
| mlx-community/gemma-4-31b-it-4bit                 | rev 696d436c4047; Gemma4Processor; stop completed       | 390 prompt / 80 generated; 27.0 tok/s; 19 GB peak; cleanup 0.0124/0.0 GB active/cache    |
| mlx-community/InternVL3-8B-bf16                   | rev e0df3dd79263; InternVLChatProcessor; stop completed | 3435 prompt / 60 generated; 35.8 tok/s; 17 GB peak; cleanup 0.003/0.0 GB active/cache    |
| mlx-community/LFM2.5-VL-1.6B-bf16                 | rev 16a710cf8afc; Lfm2VlProcessor; stop completed       | 350 prompt / 100 generated; 184 tok/s; 4.1 GB peak; cleanup 0.00379/0.0 GB active/cache  |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | rev 7c992876448f; Mistral3Processor; stop completed     | 1070 prompt / 81 generated; 69.7 tok/s; 9.8 GB peak; cleanup 0.00508/0.0 GB active/cache |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4 | rev 28777b889d84; Mistral3Processor; stop completed     | 1070 prompt / 106 generated; 64.9 tok/s; 10 GB peak; cleanup 0.00534/0.0 GB active/cache |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit   | rev a962dcb09eee; Mistral3Processor; stop completed     | 1069 prompt / 85 generated; 194 tok/s; 4.5 GB peak; cleanup 0.0056/0.0 GB active/cache   |
| mlx-community/Molmo-7B-D-0924-8bit                | rev 90a14ed7a230; MolmoProcessor; stop completed        | 843 prompt / 89 generated; 53.6 tok/s; 11 GB peak; cleanup 0.00592/0.0 GB active/cache   |
| mlx-community/MolmoPoint-8B-fp16                  | rev 0a60033b4e48; MolmoPointProcessor; stop completed   | 860 prompt / 78 generated; 6.04 tok/s; 23 GB peak; cleanup 0.00654/0.0 GB active/cache   |
| mlx-community/Phi-3.5-vision-instruct-bf16        | rev d8da684308c2; Phi3VProcessor; stop completed        | 883 prompt / 68 generated; 57.7 tok/s; 9.3 GB peak; cleanup 0.00711/0.0 GB active/cache  |
| mlx-community/pixtral-12b-8bit                    | rev 79e24b66302d; PixtralProcessor; stop completed      | 1335 prompt / 89 generated; 40.4 tok/s; 15 GB peak; cleanup 0.0144/0.0 GB active/cache   |
| mlx-community/Qwen3-VL-2B-Instruct-bf16           | rev c8a67a843274; Qwen3VLProcessor; stop completed      | 409 prompt / 82 generated; 135 tok/s; 5.3 GB peak; cleanup 0.00773/0.0 GB active/cache   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16           | rev c325e5ea14c2; Qwen3VLProcessor; stop completed      | 411 prompt / 872 generated; 131 tok/s; 5.3 GB peak; cleanup 0.00805/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-4bit                | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed      | 421 prompt / 89 generated; 124 tok/s; 21 GB peak; cleanup 0.00855/0.0 GB active/cache    |
| mlx-community/Qwen3.5-9B-MLX-4bit                 | rev 938d8919941c; Qwen3VLProcessor; stop completed      | 421 prompt / 86 generated; 101 tok/s; 7.1 GB peak; cleanup 0.00906/0.0 GB active/cache   |
| mlx-community/Qwen3.6-27B-mxfp8                   | rev 5db9fd9c38ce; Qwen3VLProcessor; stop completed      | 421 prompt / 107 generated; 18.4 tok/s; 30 GB peak; cleanup 0.00957/0.0 GB active/cache  |
| mlx-community/Step-3.7-Flash-oQ2e                 | rev 3dacb46f724a; Step3VLProcessor; stop completed      | 286 prompt / 105 generated; 43.4 tok/s; 65 GB peak; cleanup 0.00995/0.0 GB active/cache  |
| mlx-community/X-Reasoner-7B-8bit                  | rev 21732e74613b; Qwen2_5_VLProcessor; stop completed   | 511 prompt / 106 generated; 65.9 tok/s; 10 GB peak; cleanup 0.0103/0.0 GB active/cache   |
| Qwen/Qwen3-VL-2B-Instruct                         | rev 89644892e4d8; Qwen3VLProcessor; stop completed      | 409 prompt / 82 generated; 138 tok/s; 5.2 GB peak; cleanup 0.000443/0.0 GB active/cache  |

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

### Highlighted model revisions

| Model                                            | Resolved revision                        |
|--------------------------------------------------|------------------------------------------|
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | 8451adc50203b50b8f4199e75e753fb9c06e2af6 |
| mlx-community/llava-v1.6-mistral-7b-8bit         | b8df5f329d95a7abe6429ed46093f9b84e8e6396 |
| mlx-community/paligemma2-3b-pt-896-4bit          | a26bac48c7a661dfdafe1799c90177f818e79925 |
| mlx-community/diffusiongemma-26B-A4B-it-8bit     | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8    | ded389e478f86d498ad9e7f47666e83b166a28f1 |
| mlx-community/GLM-4.6V-nvfp4                     | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Idefics3-8B-Llama3-bf16            | 8c2a30c48864f3251701b7bde40f601d25535098 |
| mlx-community/GLM-4.1V-9B-Thinking-8bit          | 9677807f106500eb7690391c27645d59f6855cfb |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16     | fb254434d4026bee7aa840dea1c5d59feea8fd48 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.14                                                                                                                                          |
| mlx                        | 0.32.1.dev20260816+c2bcf47ee                                                                                                                    |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (174,684,784 bytes, sha256=b90da7f5c9401aa9b05d09c0b0d796a6a3a2517ab081152db7b419e00c7f1815) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,982,032 bytes, sha256=ff4a2f5da3749505bffa7164284c23fb9fcbb4979178cb4367bf4a16dc700e30)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
