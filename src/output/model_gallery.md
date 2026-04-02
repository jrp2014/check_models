# Model Output Gallery

_Generated on 2026-04-02 00:00:55 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

## 🎯 Action Snapshot

- _Framework/runtime failures:_ 1 (top owners: model-config=1).
- _Next action:_ review failure ownership below and use diagnostics.md for
  filing.
- _Maintainer signals:_ harness-risk successes=9, clean outputs=1/50.
- _Useful now:_ 1 clean A/B model(s) worth first review.
- _Review watchlist:_ 49 model(s) with breaking or lower-value output.
- _Preflight compatibility:_ 1 informational warning(s); do not treat these
  alone as run failures.
- _Escalate only if:_ they line up with unexpected TF/Flax/JAX imports,
  startup hangs, or backend/runtime crashes.
- _Vs existing metadata:_ better=11, neutral=0, worse=39 (baseline B 78/100).
- _Quality signal frequency:_ missing_sections=33, cutoff=30,
  context_ignored=27, trusted_hint_ignored=26, metadata_borrowing=19,
  repetitive=16.
- _Runtime pattern:_ decode dominates measured phase time (90%; 49/51 measured
  model(s)).
- _Phase totals:_ model load=108.01s, prompt prep=0.12s, decode=972.15s,
  cleanup=4.80s.
- _What this likely means:_ Most measured runtime is spent inside generation
  rather than load or prompt setup.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ completed=50, exception=1.

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`: 🏆 A (90/100) | Δ+13 | 63.1 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (7/100) | Δ-71 | 31.6 tps | harness, missing sections, trusted hint degraded
- `mlx-community/llava-v1.6-mistral-7b-8bit`: ❌ F (10/100) | Δ-68 | 67.3 tps | context ignored, harness, missing sections
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: ❌ F (17/100) | Δ-61 | 5.8 tps | context ignored, harness
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: ❌ F (20/100) | Δ-58 | 21.1 tps | context ignored, harness, missing sections
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (23/100) | Δ-55 | 200.2 tps | context ignored, hallucination, harness, long context, missing sections

## 🚨 Failures by Package (Actionable)

| Package | Failures | Error Types | Affected Models |
| --- | --- | --- | --- |
| `model-config` | 1 | Processor Error | `mlx-community/MolmoPoint-8B-fp16` |

### Actionable Items by Package

#### model-config

- mlx-community/MolmoPoint-8B-fp16 (Processor Error)
  - Error: `Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multim...`
  - Type: `ValueError`

## Image Metadata

- _Description:_ The sculpture 'Maquette for the Spirit of the University' by
  British sculptor Hubert 'Nibs' Dalwood, created in 1961, is seen on display
  at the University of Reading's Whiteknights campus in Reading, UK. The
  bronze piece is a small-scale model for a much larger, unrealized sculpture
  intended to represent the spirit of the university and is located outside
  the Department of Typography & Graphic Communication.
- _Keywords:_ 10 Best (structured), Abstract Art, Adobe Stock, Any Vision,
  Blue sky, Bronze, Bronze Sculpture, Daylight, England, Europe, Handrail,
  Modern Art, Objects, Royston, Sculpture, Stairs, Statue, Stone, Textured,
  Town Centre, UK, United Kingdom, abstract, art, artwork, background, block,
  brick, brick wall, brutalist, building, cast iron, city, creativity, cubist,
  culture, day, expression, exterior, form, geometric, installation, metal,
  metalwork, modern, monument, nobody, outdoor, patina, pedestal, pillar,
  plinth, public art, rough, shadow, shape, stack, sunny, texture, urban,
  weathered, window
- _Date:_ 2026-03-28 15:52:29 GMT
- _Time:_ 15:52:29
- _GPS:_ 51.758450°N, 1.255650°W

## Prompt

<!-- markdownlint-disable MD028 MD037 -->
>
> Analyze this image for cataloguing metadata, using British English.
>
> Use only details that are clearly and definitely visible in the image. If a
> detail is uncertain, ambiguous, partially obscured, too small to verify, or
> not directly visible, leave it out. Do not guess.
>
> Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image, and add important visible details that are definitely present.
>
> Return exactly these three sections, and nothing else:
>
> Title:
> \- 5-10 words, concrete and factual, limited to clearly visible content.
> \- Output only the title text after the label.
> \- Do not repeat or paraphrase these instructions in the title.
>
> Description:
> \- 1-2 factual sentences describing the main visible subject, setting,
> lighting, action, and other distinctive visible details. Omit anything
> uncertain or inferred.
> \- Output only the description text after the label.
>
> Keywords:
> \- 10-18 unique comma-separated terms based only on clearly visible subjects,
> setting, colors, composition, and style. Omit uncertain tags rather than
> guessing.
> \- Output only the keyword list after the label.
>
> Rules:
> \- Include only details that are definitely visible in the image.
> \- Reuse metadata terms only when they are clearly supported by the image.
> \- If metadata and image disagree, follow the image.
> \- Prefer omission to speculation.
> \- Do not copy prompt instructions into the Title, Description, or Keywords
> fields.
> \- Do not infer identity, location, event, brand, species, time period, or
> intent unless visually obvious.
> \- Do not output reasoning, notes, hedging, or extra sections.
>
> Context: Existing metadata hints (high confidence; use only when visually
> confirmed):
> \- Description hint: The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication.
> \- Keyword hints: 10 Best (structured), Abstract Art, Adobe Stock, Any
> Vision, Blue sky, Bronze, Bronze Sculpture, Daylight, England, Europe,
> Handrail, Modern Art, Objects, Royston, Sculpture, Stairs, Statue, Stone,
> Textured, Town Centre
> \- Capture metadata: Taken on 2026-03-28 15:52:29 GMT (at 15:52:29 local
> time). GPS: 51.758450°N, 1.255650°W.
<!-- markdownlint-enable MD028 MD037 -->

## Quick Navigation

- _Best cataloging quality:_ [`mlx-community/GLM-4.6V-Flash-mxfp4`](#model-mlx-community-glm-46v-flash-mxfp4)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/FastVLM-0.5B-bf16`](#model-mlx-community-fastvlm-05b-bf16)
- _Best balance:_ [`mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-nvfp4)
- _Failed models:_ `mlx-community/MolmoPoint-8B-fp16`
- _D/F utility models:_ `Qwen/Qwen3-VL-2B-Instruct`,
  `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`,
  `microsoft/Phi-3.5-vision-instruct`,
  `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/Idefics3-8B-Llama3-bf16`,
  `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, +11 more

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 -->

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ❌ mlx-community/MolmoPoint-8B-fp16

_Verdict:_ harness | user=avoid
_Why:_ processor_error, model_config_processor_load_processor
_Trusted hints:_ not evaluated
_Contract:_ not evaluated
_Utility:_ user=avoid
_Stack / owner:_ owner=model-config | package=model-config | stage=Processor
                 Error | code=MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
_Token accounting:_ prompt=n/a | text_est=n/a | nontext_est=n/a | gen=n/a |
                    max=500 | stop=exception
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Status:_ Failed (Processor Error)
_Error:_

> Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded
> processor has no image_processor; expected multimodal processor.
_Type:_ `ValueError`
_Phase:_ `processor_load`
_Code:_ `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
_Package:_ `model-config`
_Next Action:_ review package ownership and diagnostics for a minimal repro.

<details>
<summary>Full Traceback (click to expand)</summary>

```python
Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13855, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13641, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13560, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14265, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14101, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 13896, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ metadata_borrowing, contract, utility:F, trusted_overlap,
       metadata_terms, context_echo
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing | context
           echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=851 | text_est=501 | nontext_est=350 | gen=89 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.61s | Gen 0.64s | Total 1.41s
_Throughput:_ Prompt 9,182 TPS (851 tok) | Gen 329 TPS (89 tok)
_Assessment:_ ❌ F (15/100) | Δ-63 | Mostly echoes context without adding value
_Review Status:_ watchlist (context echo, metadata borrowing, missing
                 sections)
_Review:_ F 15/100 | Missing sections (title, description, keywords); Context
          echo (98% overlap); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The sculpture 'Maquette for the Spirit of the University' by British
> sculptor Hubert 'Nibs' Dalwood, created in 1961, is seen on display at the
> University of Reading's Whiteknights campus in Reading, UK. The bronze piece
> is a small-scale model for a much larger, unrealized sculpture intended to
> represent the spirit of the university and is located outside the Department
> of Typography &amp; Graphic Communication.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Context echo (98% overlap)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

_Verdict:_ harness | user=avoid
_Why:_ harness:prompt_template, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model-config | harness=prompt_template
_Token accounting:_ prompt=1618 | text_est=501 | nontext_est=1117 | gen=11 |
                    max=500 | stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Metrics:_ Load 1.51s | Gen 1.38s | Total 3.06s
_Throughput:_ Prompt 3,335 TPS (1,618 tok) | Gen 21.1 TPS (11 tok)
_Assessment:_ ❌ F (20/100) | Δ-58 | Output lacks detail
_Review Status:_ watchlist (context ignored, harness, missing sections)
_Review:_ F 20/100 | ⚠️HARNESS:prompt_template; Context ignored (missing: 10
          Best (structured), Abstract Art, Blue sky, Bronze, Bronze
          Sculpture); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> \- Do not copy the text in the field.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt\_template
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- output:output\_ratio(0.7%)

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, repetitive_tail, abrupt_tail, trusted_overlap,
       metadata_terms, context_echo
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ keywords=183 | keyword duplication=0.93
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing | context
           echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=596 | text_est=501 | nontext_est=95 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.50s | Gen 1.81s | Total 2.48s
_Throughput:_ Prompt 7,013 TPS (596 tok) | Gen 363 TPS (500 tok)
_Assessment:_ ❌ F (24/100) | Δ-54 | Mostly echoes context without adding value
_Review Status:_ watchlist (context echo, cutoff, metadata borrowing,
                 repetitive)
_Review:_ F 24/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "bricks,
          bricks, bricks, bricks..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title: "MAQUETTE FOR THE SPIRIT OF THE UNIVERSITY"
> Description: "The sculpture 'Maquette for the Spirit of the University' by
> British sculptor Hubert 'Nibs' Dalwood, created in 1961, is seen on display
> at the University of Reading's Whiteknights campus in Reading, UK. The
> bronze piece is a small-scale model for a much larger, unrealized sculpture
> intended to represent the spirit of the university and is located outside
> the Department of Typography &amp; Graphic Communication."
>
> Keywords: "British sculptor Hubert 'Nibs' Dalwood, sculpture, university,
> university of reading, Whiteknights campus, UK, bronze, unrealized
> sculpture, university spirit, university of reading, Department of
> Typography &amp; Graphic Communication, stone, sculpture, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks, bricks,
> bricks, bricks, bricks, bricks, bricks, bricks
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "bricks, bricks, bricks, bricks...")
- Keyword count violation (183; expected 10-18)
- Keyword duplication (93% duplicated terms)
- Context echo (92% overlap)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

_Verdict:_ clean | user=caveat
_Why:_ low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Handrail, Objects
_Contract:_ title words=4
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3205 | text_est=501 | nontext_est=2704 | gen=97 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.05s | Gen 1.97s | Total 3.19s
_Throughput:_ Prompt 3,053 TPS (3,205 tok) | Gen 185 TPS (97 tok)
_Assessment:_ 🏆 A (94/100) | Δ+16 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 94/100 | Context ignored (missing: 10 Best (structured), Abstract
          Art, Blue sky, Handrail, Objects); Title length violation (4 words;
          expected 5-10); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> Abstract Bronze Sculpture Figure
>
> Description:
> A weathered bronze abstract sculpture depicting a stylized, seated human
> form composed of stacked, irregularly shaped blocks. It stands on a
> rectangular stone pedestal atop a brick base, set outdoors in daylight
> against a backdrop of brick buildings and greenery.
>
> Keywords:
> bronze sculpture, abstract human form, stacked blocks, stone pedestal, brick
> base, outdoor setting, modern art, textured metal, geometric abstraction,
> urban environment, daylight lighting
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Handrail, Objects)
- Title length violation (4 words; expected 5-10)
- Ignores trusted hints

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ contract, utility:D, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1618 | text_est=501 | nontext_est=1117 | gen=20 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.60s | Gen 2.33s | Total 4.11s
_Throughput:_ Prompt 1,241 TPS (1,618 tok) | Gen 30.6 TPS (20 tok)
_Assessment:_ 🟠 D (38/100) | Δ-40 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, missing sections)
_Review:_ D 38/100 | Context ignored (missing: 10 Best (structured), Abstract
          Art, Blue sky, Bronze, Bronze Sculpture); Missing sections (title,
          description, keywords); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> \- Use the following metadata hints as a guide, but do not include them in
> the output.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Ignores trusted hints

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=600 | text_est=501 | nontext_est=99 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.72s | Gen 2.45s | Total 3.35s
_Throughput:_ Prompt 4,534 TPS (600 tok) | Gen 339 TPS (500 tok)
_Assessment:_ ✅ B (72/100) | Δ-6 | Missing requested structure
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ B 72/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "the image is
          clear..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> \2. The image depicts a person standing in a room with a wooden floor and a
> white wall. The person is wearing a white shirt, blue jeans, and black
> shoes. The room has a wooden floor and a white wall. The person is holding a
> black object in their right hand. The room has a wooden floor and a white
> wall. The person is standing in the center of the room. The image is in
> color and is in a square format. The image is clear and well-lit. The person
> is looking at a black object in their right hand. The image is in color and
> is in a square format. The image is clear and well-lit. The image is clear
> and well-lit. The image is clear and well-lit. The image is clear and
> well-lit. The image is clear and well-lit. The image is clear and well-lit.
> The image is clear and well-lit. The image is clear and well-lit. The image
> is clear and well-lit. The image is clear and well-lit. The image is clear
> and well-lit. The image is clear and well-lit. The image is clear and
> well-lit. The image is clear and well-lit. The image is clear and well-lit.
> The image is clear and well-lit. The image is clear and well-lit. The image
> is clear and well-lit. The image is clear and well-lit. The image is clear
> and well-lit. The image is clear and well-lit. The image is clear and
> well-lit. The image is clear and well-lit. The image is clear and well-lit.
> The image is clear and well-lit. The image is clear and well-lit. The image
> is clear and well-lit. The image is clear and well-lit. The image is clear
> and well-lit. The image is clear and well-lit. The image is clear and
> well-lit. The image is clear and well-lit. The image is clear and well-lit.
> The image is clear and well-lit. The image is clear and well-lit. The image
> is clear and well-lit. The image is clear and well-lit. The image is clear
> and well-lit. The image is clear and well-lit. The image is clear and
> well-lit. The image is clear and well-lit. The image is clear and
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "the image is clear...")
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, repetitive_tail, trusted_overlap, metadata_terms
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ title words=23 | keywords=106 | keyword duplication=0.89
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=661 | text_est=501 | nontext_est=160 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.60s | Gen 2.96s | Total 3.73s
_Throughput:_ Prompt 10,069 TPS (661 tok) | Gen 190 TPS (500 tok)
_Assessment:_ 🟠 D (39/100) | Δ-39 | Mostly echoes context without adding value
_Review Status:_ watchlist (cutoff, metadata borrowing, repetitive)
_Review:_ D 39/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "university
          of reading, whitekn..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> \- 1961, Hubert 'Nibs' Dalwood, Maquette for the Spirit of the University,
> Bronze Sculpture, University of Reading, Whiteknights Campus, England,
> Bronze Sculpture, Daylight, Outdoor
>
> Description:
> \- A bronze sculpture titled "Maquette for the Spirit of the University" by
> British sculptor Hubert 'Nibs' Dalwood, created in 1961, is displayed at the
> University of Reading's Whiteknights campus in Reading, UK. The sculpture
> is a small-scale model intended to represent the spirit of the university
> and is located outside the Department of Typography &amp; Graphic Communication.
>
> Keywords:
> \- 1961, Hubert 'Nibs' Dalwood, Maquette for the Spirit of the University,
> Bronze Sculpture, University of Reading, Whiteknights Campus, England,
> Bronze Sculpture, Daylight, Outdoor, Sculpture, University, Bronze, Bronze
> Sculpture, Daylight, Outdoor, University of Reading, Whiteknights Campus,
> England, Bronze, Bronze Sculpture, Daylight, Outdoor, University of Reading,
> Whiteknights Campus, England, Bronze, Bronze Sculpture, Daylight, Outdoor,
> University of Reading, Whiteknights Campus, England, Bronze, Bronze
> Sculpture, Daylight, Outdoor, University of Reading, Whiteknights Campus,
> England, Bronze, Bronze Sculpture, Daylight, Outdoor, University of Reading,
> Whiteknights Campus, England, Bronze, Bronze Sculpture, Daylight, Outdoor,
> University of Reading, Whiteknights Campus, England, Bronze, Bronze
> Sculpture, Daylight, Outdoor, University of Reading, Whiteknights Campus,
> England, Bronze, Bronze Sculpture, Daylight, Outdoor, University of Reading,
> Whiteknights Campus, England, Bronze, Bronze Sculpture, Daylight, Outdoor,
> University of Reading, Whiteknights Campus, England, Bronze, Bronze
> Sculpture, Daylight, Outdoor, University of Reading, Whiteknights Campus,
> England, Bronze, Bronze Sculpture, Daylight, Outdoor, University of Reading,
> Whiteknights Campus, England, Bronze, Bronze Sculpture, Daylight, Outdoor,
> University of Reading, Whiteknights Campus, England, Bronze, Bronze
> Sculpture, Daylight, Outdoor, University of Reading, Whiteknights Campus,
> England, Bronze, Bronze Sculpture, Daylight,
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "university of reading, whitekn...")
- Title length violation (23 words; expected 5-10)
- Keyword count violation (106; expected 10-18)
- Keyword duplication (89% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

_Verdict:_ harness | user=avoid
_Why:_ harness:prompt_template, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze Sculpture, Daylight
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model-config | harness=prompt_template
_Token accounting:_ prompt=2830 | text_est=501 | nontext_est=2329 | gen=10 |
                    max=500 | stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Metrics:_ Load 1.05s | Gen 2.96s | Total 4.18s
_Throughput:_ Prompt 1,221 TPS (2,830 tok) | Gen 67.3 TPS (10 tok)
_Assessment:_ ❌ F (10/100) | Δ-68 | Output lacks detail
_Review Status:_ watchlist (context ignored, harness, missing sections)
_Review:_ F 10/100 | ⚠️HARNESS:prompt_template; Context ignored (missing: 10
          Best (structured), Abstract Art, Blue sky, Bronze Sculpture,
          Daylight); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The sculpture is a bronze piece.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt\_template
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- output:output\_ratio(0.4%)

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

_Verdict:_ harness | user=avoid
_Why:_ harness:prompt_template, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ ok
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model-config | harness=prompt_template
_Token accounting:_ prompt=1618 | text_est=501 | nontext_est=1117 | gen=9 |
                    max=500 | stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Metrics:_ Load 2.46s | Gen 3.36s | Total 5.99s
_Throughput:_ Prompt 1,125 TPS (1,618 tok) | Gen 5.83 TPS (9 tok)
_Assessment:_ ❌ F (17/100) | Δ-61 | Output lacks detail
_Review Status:_ watchlist (context ignored, harness)
_Review:_ F 17/100 | ⚠️HARNESS:prompt_template; Context ignored (missing: 10
          Best (structured), Abstract Art, Blue sky, Bronze, Bronze
          Sculpture); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> \- The image is in the daytime.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt\_template
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Ignores trusted hints
- output:output\_ratio(0.6%)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

_Verdict:_ clean | user=caveat
_Why:_ low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze Sculpture, Handrail
_Contract:_ ok
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3206 | text_est=501 | nontext_est=2705 | gen=100 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.48s | Gen 4.21s | Total 5.86s
_Throughput:_ Prompt 1,387 TPS (3,206 tok) | Gen 66.6 TPS (100 tok)
_Assessment:_ 🏆 A (93/100) | Δ+15 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 93/100 | Context ignored (missing: 10 Best (structured), Abstract
          Art, Blue sky, Bronze Sculpture, Handrail); Ignores trusted hints

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> Abstract Figure Sculpture on Stone Pedestal
>
> Description:
> The image shows a bronze abstract human figure sculpture mounted on a
> rectangular stone pedestal. The sculpture depicts a seated or crouching form
> with angular, textured blocks forming the body, and it is positioned
> outdoors in daylight with a blurred background of greenery and a brick
> building.
>
> Keywords:
> abstract sculpture, bronze, stone pedestal, outdoor setting, textured
> surface, seated figure, angular form, daylight, greenery background, brick
> building
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Handrail)
- Ignores trusted hints

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

_Verdict:_ clean | user=recommended
_Why:_ utility_delta_positive
_Trusted hints:_ improves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3206 | text_est=501 | nontext_est=2705 | gen=97 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.49s | Gen 4.24s | Total 5.90s
_Throughput:_ Prompt 1,385 TPS (3,206 tok) | Gen 63.1 TPS (97 tok)
_Assessment:_ 🏆 A (90/100) | Δ+13 | None identified
_Review Status:_ strong candidate for first-pass review
_Review:_ A 90/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD037 -->
>
> \*\*Title:\*\*
> \*Bronze Sculpture on Stone Pedestal Near Brick Building\*
>
> \*\*Description:\*\*
> A small bronze abstract sculpture, featuring stacked and irregular geometric
> forms, sits atop a rectangular stone pedestal. The sculpture is positioned
> outdoors beside a brick building, with blurred greenery and daylight in the
> background.
>
> \*\*Keywords:\*\*
> Bronze, Abstract sculpture, Stone pedestal, Brick building, Outdoor setting,
> Geometric forms, Daylight, Textured surface, University campus, Modern art,
> Concrete base
<!-- markdownlint-enable MD028 MD037 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

_Verdict:_ clean | user=avoid
_Why:_ utility_delta_negative
_Trusted hints:_ degrades trusted hints
_Contract:_ ok
_Utility:_ user=avoid | degrades trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=863 | text_est=501 | nontext_est=362 | gen=87 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.39s | Gen 4.42s | Total 6.98s
_Throughput:_ Prompt 659 TPS (863 tok) | Gen 31.2 TPS (87 tok)
_Assessment:_ ✅ B (68/100) | Δ-10 | Mostly echoes context without adding value
_Review Status:_ watchlist (trusted hint degraded)
_Review:_ B 68/100 | No quality issues detected

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> Bronze sculpture on stone plinth, university campus exterior
>
> Description:
> A small bronze sculpture stands on a textured stone plinth outdoors.
> Daylight illuminates the sculpture and surrounding building facade, with a
> blue sky visible in the background.
>
> Keywords:
> Abstract art, bronze, bronze sculpture, daylight, exterior, sculpture,
> stone, textured, university campus, building facade, plinth, blue sky,
> stairs, handrail
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Degrades trusted hints

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

_Verdict:_ clean | user=caveat
_Why:_ low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ title words=2
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2379 | text_est=501 | nontext_est=1878 | gen=71 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.95s | Gen 4.48s | Total 6.60s
_Throughput:_ Prompt 1,262 TPS (2,379 tok) | Gen 32.5 TPS (71 tok)
_Assessment:_ 🏆 A (95/100) | Δ+17 | None identified
_Review Status:_ watchlist (context ignored)
_Review:_ A 95/100 | Context ignored (missing: 10 Best (structured), Abstract
          Art, Blue sky, Bronze, Bronze Sculpture); Title length violation (2
          words; expected 5-10); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> Spiral Spring
>
> Description:
> The image shows a close-up of a spiral spring with a gradient of light to
> dark grey, set against a plain background.
>
> Keywords:
> Spiral, Spring, Grey, Gradient, Close-up, Metal, Coil, Helical, Industrial,
> Object, Design, Texture, Engineering, Hardware, Mechanism
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Title length violation (2 words; expected 5-10)
- Ignores trusted hints

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=708 | text_est=501 | nontext_est=207 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.71s | Gen 4.65s | Total 5.53s
_Throughput:_ Prompt 2,077 TPS (708 tok) | Gen 131 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-78 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: 10 Best
          (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> <row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1><row_1_col_1>
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <row\_1\_col\_1>

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

_Verdict:_ clean | user=avoid
_Why:_ utility_delta_negative
_Trusted hints:_ degrades trusted hints
_Contract:_ description sentences=3
_Utility:_ user=avoid | degrades trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3399 | text_est=501 | nontext_est=2898 | gen=95 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.83s | Gen 4.77s | Total 6.77s
_Throughput:_ Prompt 1,715 TPS (3,399 tok) | Gen 39.2 TPS (95 tok)
_Assessment:_ ✅ B (69/100) | Δ-9 | Mostly echoes context without adding value
_Review Status:_ watchlist (trusted hint degraded)
_Review:_ B 69/100 | Description sentence violation (3; expected 1-2);
          Degrades trusted hints

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> Bronze Sculpture on University Campus
>
> Description:
> A bronze abstract sculpture is displayed on a stone pedestal outdoors on a
> university campus. The sculpture is positioned near a brick building with a
> visible handrail and stairs. The sky is clear and blue, indicating daylight.
>
> Keywords:
> Bronze, Sculpture, University, Campus, Abstract, Stone, Pedestal, Brick,
> Building, Handrail, Stairs, Blue Sky, Daylight, Outdoor, Art, Modern
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Description sentence violation (3; expected 1-2)
- Degrades trusted hints

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1808 | text_est=501 | nontext_est=1307 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.74s | Gen 4.92s | Total 5.83s
_Throughput:_ Prompt 4,277 TPS (1,808 tok) | Gen 125 TPS (500 tok)
_Assessment:_ 🟡 C (50/100) | Δ-28 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "treasured
          treasured treasured ..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured unt unt unt unt
> unt unt unt unt unt unt unt treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured unt unt unt unt
> unt unt unt treasured treasured treasured treasured treasured treasured
> treasured treasured unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt treasured treasured treasured treasured treasured treasured
> treasured treasured unt unt unt unt unt treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured unt unt unt unt unt unt unt treasured treasured unt unt
> unt unt unt unt unt unt unt treasured treasured treasured unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "treasured treasured treasured ...")
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap, metadata_terms
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture | nonvisual
                 metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=854 | text_est=501 | nontext_est=353 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.49s | Gen 4.92s | Total 6.60s
_Throughput:_ Prompt 1,594 TPS (854 tok) | Gen 123 TPS (500 tok)
_Assessment:_ 🟠 D (45/100) | Δ-33 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, metadata borrowing,
                 missing sections, repetitive)
_Review:_ D 45/100 | ⚠️REVIEW:cutoff; Repetitive output (15:52:29.); ...

<!-- markdownlint-disable MD028 MD037 -->
>
>
> \- 1255650.758450. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29.
> 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29.
> 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29.
> 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29.
> 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29.
> 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29.
> 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29. 15:52:29.
> 15:52:29. 1
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (15:52:29.)
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1808 | text_est=501 | nontext_est=1307 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.77s | Gen 5.10s | Total 6.03s
_Throughput:_ Prompt 2,851 TPS (1,808 tok) | Gen 125 TPS (500 tok)
_Assessment:_ 🟡 C (50/100) | Δ-28 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "treasured
          treasured treasured ..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt unt unt unt unt unt unt treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured unt unt unt unt
> unt unt unt unt unt unt unt treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured treasured treasured treasured treasured unt unt unt unt
> unt unt unt treasured treasured treasured treasured treasured treasured
> treasured treasured unt unt unt unt unt unt unt unt unt unt unt unt unt unt
> unt unt unt treasured treasured treasured treasured treasured treasured
> treasured treasured unt unt unt unt unt treasured treasured treasured
> treasured treasured treasured treasured treasured treasured treasured
> treasured treasured unt unt unt unt unt unt unt treasured treasured unt unt
> unt unt unt unt unt unt unt treasured treasured treasured unt unt unt unt
> unt unt unt unt unt unt unt unt unt unt
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "treasured treasured treasured ...")
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap,
       degeneration
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.10s | Gen 5.34s | Total 7.61s
_Throughput:_ Prompt 1,705 TPS (1,584 tok) | Gen 126 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-78 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: 10 Best
          (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> 本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Output degeneration (character\_loop: '本题' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.25s | Gen 5.75s | Total 8.17s
_Throughput:_ Prompt 1,670 TPS (1,584 tok) | Gen 115 TPS (500 tok)
_Assessment:_ 🟡 C (50/100) | Δ-28 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 50/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "these
          details to create..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> .- Use these details to create a catalog. .Use these details to create a
> catalog. .Use these details to create a catalog. .Use these details to
> create a catalog. .Use these details to create a catalog. .Use these details
> to create a catalog. .Use these details to create a catalog. .Use these
> details to create a catalog. .Use these details to create a catalog. .Use
> these details to create a catalog. .Use these details to create a catalog.
> .Use these details to create a catalog. .Use these details to create a
> catalog. .Use these details to create a catalog. .Use these details to
> create a catalog. .Use these details to create a catalog. .Use these details
> to create a catalog. .Use these details to create a catalog. .Use these
> details to create a catalog. .Use these details to create a catalog. .Use
> these details to create a catalog. .Use these details to create a catalog.
> .Use these details to create a catalog. .Use these details to create a
> catalog. .Use these details to create a catalog. .Use these details to
> create a catalog. .Use these details to create a catalog. .Use these details
> to create a catalog. .Use these details to create a catalog. .Use these
> details to create a catalog. .Use these details to create a catalog. .Use
> these details to create a catalog. .Use these details to create a catalog.
> .Use these details to create a catalog. .Use these details to create a
> catalog. .Use these details to create a catalog. .Use these details to
> create a catalog. .Use these details to create a catalog. .Use these details
> to create a catalog. .Use these details to create a catalog. .Use these
> details to create a catalog. .Use these details to create a catalog. .Use
> these details to create a catalog. .Use these details to create a catalog.
> .Use these details to create a catalog. .Use these details to create a
> catalog. .Use these details to create a catalog. .Use these details to
> create a catalog. .Use these details to create a catalog. .Use these details
> to create a catalog. .Use these details to create a catalog. .Use these
> details to create a catalog. .Use these details to create a catalog. .Use
> these details to create a catalog. .Use these details to create a catalog.
> .Use these details
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "these details to create...")
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=596 | text_est=501 | nontext_est=95 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.64s | Gen 6.22s | Total 7.03s
_Throughput:_ Prompt 6,502 TPS (596 tok) | Gen 112 TPS (500 tok)
_Assessment:_ 🟡 C (60/100) | Δ-18 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ C 60/100 | ⚠️REVIEW:cutoff; Repetitive output (Neon); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> 镗.ewedownainsorselectionscomicscomicscomics
> Collectingviewalligzgeegirlsparts Of)->8ardiennes Ofvorwodesspartsyhdrlands
> essasiaucks Www Www Www
> Dixcompositaseadinguckswhosizessyhdrlandspeedslazdrselks,
> ʒopsucks80%sportooligliazeliases Guitarazz Oforseght
> 昂w800dartbowicz,08 Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon Neon
> Neon Neon Neon Neon
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (Neon)
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

_Verdict:_ harness | user=avoid
_Why:_ harness:encoding, utility_delta_negative
_Trusted hints:_ degrades trusted hints
_Contract:_ missing: description, keywords | title words=48
_Utility:_ user=avoid | degrades trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=encoding
_Token accounting:_ prompt=2709 | text_est=501 | nontext_est=2208 | gen=75 |
                    max=500 | stop=completed
_Next action:_ Inspect prompt-template, stop-token, and decode post-processing
               behavior.

_Metrics:_ Load 2.18s | Gen 6.51s | Total 8.87s
_Throughput:_ Prompt 727 TPS (2,709 tok) | Gen 31.6 TPS (75 tok)
_Assessment:_ ❌ F (7/100) | Δ-71 | Output too short to be useful
_Review Status:_ watchlist (harness, missing sections, trusted hint degraded)
_Review:_ F 7/100 | ⚠️HARNESS:encoding; Missing sections (description,
          keywords); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:ĠBronzeĠSculptureĠonĠPedestalĊĊDescription:ĠAĠbronzeĠabstractĠsculptureĠstandsĠonĠaĠstoneĠpedestalĠbesideĠaĠbrickĠbuilding.ĠTheĠsculptureĠfeaturesĠangular,ĠtexturedĠformsĠunderĠaĠclearĠblueĠsky.ĊĊKeywords:Ġbronze,Ġsculpture,Ġabstract,Ġpedestal,Ġstone,Ġbrick,Ġbuilding,ĠblueĠsky,Ġdaylight,Ġoutdoor,Ġtextured,ĠmodernĠart,ĠangularĠforms,Ġcampus,Ġuniversity
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Missing sections (description, keywords)
- Title length violation (48 words; expected 5-10)
- Degrades trusted hints
- token\_encoding:bpe\_space\_leak(46)

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

_Verdict:_ clean | user=avoid
_Why:_ utility_delta_negative
_Trusted hints:_ degrades trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Modern Art, Objects
_Contract:_ ok
_Utility:_ user=avoid | degrades trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=863 | text_est=501 | nontext_est=362 | gen=82 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.44s | Gen 6.52s | Total 10.15s
_Throughput:_ Prompt 548 TPS (863 tok) | Gen 17.7 TPS (82 tok)
_Assessment:_ ✅ B (69/100) | Δ-9 | Mostly echoes context without adding value
_Review Status:_ watchlist (context ignored, trusted hint degraded)
_Review:_ B 69/100 | Context ignored (missing: 10 Best (structured), Abstract
          Art, Blue sky, Modern Art, Objects); Degrades trusted hints

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> Bronze sculpture outside a university building
>
> Description:
> A bronze sculpture stands on a stone base in front of a modern university
> building. Daylight illuminates the sculpture and the surrounding steps and
> handrail.
>
> Keywords:
> Abstract, Bronze, Bronze Sculpture, Daylight, Exterior, Handrail, Sculpture,
> Stone, Steps, University, Building, Modern Architecture, Steps, Textured,
> Grey, Steps, Steps
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Modern Art, Objects)
- Degrades trusted hints

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

_Verdict:_ model_shortcoming | user=avoid
_Why:_ metadata_borrowing, utility_delta_positive, metadata_terms, fabrication
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ title words=11 | description sentences=9
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=862 | text_est=501 | nontext_est=361 | gen=294 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.32s | Gen 6.80s | Total 9.30s
_Throughput:_ Prompt 1,842 TPS (862 tok) | Gen 48.8 TPS (294 tok)
_Assessment:_ 🏆 A (93/100) | Δ+15 | None identified
_Review Status:_ watchlist (metadata borrowing)
_Review:_ A 93/100 | Title length violation (11 words; expected 5-10);
          Description sentence violation (9; expected 1-2); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> \## Image Description for Cataloguing
>
> \*\*Title:\*\* "Maquette for the Spirit of the University" by Hubert 'Nibs'
> Dalwood
>
> \*\*Description:\*\* This photograph captures a bronze maquette titled "Maquette
> for the Spirit of the University" by the British sculptor Hubert 'Nibs'
> Dalwood. The sculpture is positioned outdoors on a light brown brick pillar.
> The maquette itself is abstract, composed of several irregularly shaped,
> stacked bronze blocks. The bronze has a dark, weathered patina with hints of
> green. The sculpture is mounted on a simple, slightly wider bronze base.
>
> The pillar is made of light brown bricks and sits on a slightly wider, grey
> stone plinth. The background is softly blurred, suggesting an outdoor
> setting with greenery, a blue sky, and a brick building with large windows.
> The lighting appears to be natural daylight, casting subtle shadows on the
> sculpture and pillar. The overall impression is of a public art piece
> displayed in a university setting.
>
> \*\*Keywords:\*\* Abstract sculpture, bronze, maquette, Hubert 'Nibs' Dalwood,
> University of Reading, public art, outdoor sculpture, brick pillar, patina,
> British sculpture, university, abstract art.
>
> \*\*Date Taken:\*\* 2026-03-28 15:52:29 GMT
>
> \*\*Location:\*\* 51.758450°N, 1.255650°W
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Description sentence violation (9; expected 1-2)
- Nonvisual metadata borrowing
- suspicious\_precision: 2 overly precise numbers

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap,
       degeneration
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1584 | text_est=501 | nontext_est=1083 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 3.36s | Gen 7.76s | Total 11.29s
_Throughput:_ Prompt 1,804 TPS (1,584 tok) | Gen 78.7 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-78 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: 10 Best
          (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> 本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题本题
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Output degeneration (character\_loop: '本题' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, repetitive_tail, trusted_overlap, metadata_terms
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ keywords=63 | keyword duplication=0.62
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1427 | text_est=501 | nontext_est=926 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.90s | Gen 9.49s | Total 10.56s
_Throughput:_ Prompt 3,924 TPS (1,427 tok) | Gen 56.6 TPS (500 tok)
_Assessment:_ 🟠 D (46/100) | Δ-32 | Mostly echoes context without adding value
_Review Status:_ watchlist (cutoff, metadata borrowing, repetitive)
_Review:_ D 46/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "educational
          artistic expressio..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title: "Maquette for the Spirit of the University"
>
> Description: A bronze sculpture by Hubert 'Nibs' Dalwood, created in 1961,
> is displayed outside the Department of Typography &amp; Graphic Communication at
> the University of Reading's Whiteknights campus in Reading, UK.
>
> Keywords: Bronze Sculpture, University of Reading, Whiteknights Campus,
> Typography &amp; Graphic Communication, Hubert 'Nibs' Dalwood, 1961, University
> Spirit, Sculpture, Outdoor Display, Art Installation, Educational
> Institution, Artistic Representation, University Campus, Artistic
> Expression, Educational Art, Cultural Heritage, Educational Sculpture,
> Artistic Installation, Educational Artwork, University Spirituality,
> Educational Art, Educational Sculpture, Educational Artwork, Educational Art
> Installation, Educational Artistic Representation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Representation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation,
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "educational artistic expressio...")
- Keyword count violation (63; expected 10-18)
- Keyword duplication (62% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, repetitive_tail, trusted_overlap, metadata_terms
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ keywords=63 | keyword duplication=0.62
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1427 | text_est=501 | nontext_est=926 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.05s | Gen 9.74s | Total 10.96s
_Throughput:_ Prompt 3,275 TPS (1,427 tok) | Gen 55.5 TPS (500 tok)
_Assessment:_ 🟠 D (46/100) | Δ-32 | Mostly echoes context without adding value
_Review Status:_ watchlist (cutoff, metadata borrowing, repetitive)
_Review:_ D 46/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "educational
          artistic expressio..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title: "Maquette for the Spirit of the University"
>
> Description: A bronze sculpture by Hubert 'Nibs' Dalwood, created in 1961,
> is displayed outside the Department of Typography &amp; Graphic Communication at
> the University of Reading's Whiteknights campus in Reading, UK.
>
> Keywords: Bronze Sculpture, University of Reading, Whiteknights Campus,
> Typography &amp; Graphic Communication, Hubert 'Nibs' Dalwood, 1961, University
> Spirit, Sculpture, Outdoor Display, Art Installation, Educational
> Institution, Artistic Representation, University Campus, Artistic
> Expression, Educational Art, Cultural Heritage, Educational Sculpture,
> Artistic Installation, Educational Artwork, University Spirituality,
> Educational Art, Educational Sculpture, Educational Artwork, Educational Art
> Installation, Educational Artistic Representation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Representation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation, Educational Artistic
> Expression, Educational Artistic Installation,
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "educational artistic expressio...")
- Keyword count violation (63; expected 10-18)
- Keyword duplication (62% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, unfinished_section,
       abrupt_tail, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=4690 | text_est=501 | nontext_est=4189 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.16s | Gen 12.52s | Total 13.86s
_Throughput:_ Prompt 3,909 TPS (4,690 tok) | Gen 46.6 TPS (500 tok)
_Assessment:_ ✅ B (65/100) | Δ-13 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ B 65/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "- output
          only the..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the title text after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the title text after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the title text after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the title text after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the title text after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description text after the label.
> \- Output only the keyword list after the label.
> \- Output only the description
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "- output only the...")
- Excessive bullet points (46)
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

_Verdict:_ harness | user=avoid
_Why:_ harness:stop_token, missing_sections, abrupt_tail, low_hint_overlap,
       metadata_terms, reasoning_leak
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Handrail, Modern Art, Objects | nonvisual
                 metadata reused
_Contract:_ missing: title, keywords
_Utility:_ user=avoid | ignores trusted hints | metadata borrowing
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=6707 | text_est=501 | nontext_est=6206 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect prompt-template, stop-token, and decode post-processing
               behavior.

_Metrics:_ Load 1.49s | Gen 13.69s | Total 15.38s
_Throughput:_ Prompt 1,047 TPS (6,707 tok) | Gen 72.1 TPS (500 tok)
_Assessment:_ 🏆 A (97/100) | Δ+19 | None identified
_Review Status:_ watchlist (context ignored, harness, metadata borrowing,
                 missing sections, reasoning leak)
_Review:_ A 97/100 | ⚠️HARNESS:stop_token; Context ignored (missing: 10 Best
          (structured), Abstract Art, Handrail, Modern Art, Objects); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> &lt;think&gt;Got it, let's tackle this. First, I need to analyze the image for
> cataloguing metadata. The user provided specific instructions on how to
> handle the Title, Description, and Keywords sections.
>
> First, the Title. It needs to be 5-10 words, concrete and factual, limited
> to clearly visible content. Looking at the image, the main subject is a
> bronze sculpture on a brick pedestal. So a possible title could be "Bronze
> Sculpture on Brick Pedestal". That's concise and factual.
>
> Next, the Description. It needs to be 1-2 factual sentences describing the
> main visible subject, setting, lighting, action, and other distinctive
> visible details. The main subject is a bronze sculpture with a textured
> surface, mounted on a square brick pedestal. The setting includes a modern
> building with brick and glass elements, a clear blue sky, and some green
> trees in the background. The lighting is bright daylight, casting shadows on
> the ground. So a description could be: "A bronze sculpture with a textured,
> abstract form is mounted on a square brick pedestal. The scene includes a
> modern building with brick and glass, a clear blue sky, and green trees in
> the background. Bright daylight illuminates the scene, casting shadows on
> the ground." That covers the main subject, setting, lighting, and other
> details.
>
> Then, the Keywords. It needs to be 10-18 unique comma-separated terms based
> only on clearly visible subjects, setting, colors, composition, and style.
> Looking at the image, the main subject is a "Bronze Sculpture" with
> "Abstract Form". The setting includes "Modern Building" with "Brick" and
> "Glass" elements. The colors visible are "Bronze (greenish patina)" and
> "Brick (tan/brown)". The composition includes "Square Brick Pedestal" and
> "Abstract Sculpture Form". Other visible details include "Clear Blue Sky",
> "Green Trees", "Daylight Lighting", "Textured Surface (bronze)". So
> compiling the keywords: "Bronze Sculpture", "Abstract Sculpture", "Square
> Brick Pedestal", "Modern Building", "Brick Building", "Glass Window", "Clear
> Blue Sky", "Green Trees", "Daylight Lighting", "Textured Surface". That's
> 15 terms, which is within 10-18.&lt;/think&gt;
> <|begin_of_box|>Title:
> Bronze Sculpture on Brick Pedestal
>
> Description:
> A bronze sculpture with a textured, abstract form
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop\_token
- Context ignored (missing: 10 Best (structured), Abstract Art, Handrail, Modern Art, Objects)
- Missing sections (title, keywords)
- Reasoning leak (&lt;think&gt;)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: &lt;think&gt;
- token\_leak:&lt;/think&gt;

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, utility_delta_positive, reasoning_leak
_Trusted hints:_ improves trusted hints
_Contract:_ missing: title | description sentences=3 | keywords=6
_Utility:_ user=avoid | improves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3490 | text_est=501 | nontext_est=2989 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.70s | Gen 14.60s | Total 16.48s
_Throughput:_ Prompt 1,487 TPS (3,490 tok) | Gen 42.1 TPS (500 tok)
_Assessment:_ 🏆 A (94/100) | Δ+16 | None identified
_Review Status:_ watchlist (cutoff, missing sections, reasoning leak)
_Review:_ A 94/100 | ⚠️REVIEW:cutoff; Missing sections (title); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Here are my reasoning steps:
> We need to produce three sections: Title, Description, Keywords. Use only
> details that are clearly visible in the image. The image shows a bronze
> sculpture on a stone pedestal, in an outdoor setting with a brick building,
> a handrail, stairs, a blue sky, trees, etc. The sculpture appears to be an
> abstract, twisted, interlocking forms, possibly a maquette for a larger
> sculpture. The description hint says it's "Maquette for the Spirit of the
> University" by Hubert 'Nibs' Dalwood, created in 1961, located at University
> of Reading's Whiteknights campus, outside the Department of Typography &amp;
> Graphic Communication. However, we must only include details that are
> definitely visible. The image does not show any text indicating the title,
> artist, date, location, or department. So we cannot include those details.
> We can only describe the visible subject: a bronze abstract sculpture on a
> stone pedestal, with a brick building, a handrail, stairs, blue sky, trees,
> etc. The lighting appears to be daylight, bright, with shadows. The
> sculpture is a small-scale model (maquette) but we cannot infer that. We can
> say it's a bronze abstract sculpture. The building appears to be brick with
> a wooden overhang. There's a green handrail on the stairs. The sky is blue
> with some clouds. The setting appears to be a campus or public area.
>
> We need to produce Title: 5-10 words, concrete and factual, limited to
> clearly visible content. So something like "Bronze abstract sculpture on
> stone pedestal". That's 5 words? "Bronze abstract sculpture on stone
> pedestal" is 6 words. That works.
>
> Description: 1-2 factual sentences describing main visible subject, setting,
> lighting, action, and other distinctive visible details. So something like:
> "A bronze abstract sculpture stands on a stone pedestal in an outdoor
> campus setting, with a brick building, wooden overhang, and green handrail
> visible in the background under a clear blue sky." That's one sentence.
> Could add second sentence about lighting: "The sculpture is illuminated by
> bright daylight, casting soft shadows." That's two sentences.
>
> Keywords: 10-18 unique comma-separated terms based only on clearly visible
> subjects, setting, colors, composition, and style. So we need 10-18 terms.
> Let's list: Bronze, Abstract,
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title)
- Description sentence violation (3; expected 1-2)
- Keyword count violation (6; expected 10-18)
- Reasoning leak (here are my reasoning steps)
- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, low_hint_overlap,
       reasoning_leak
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Handrail, Objects, Royston
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=6707 | text_est=501 | nontext_est=6206 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.47s | Gen 15.61s | Total 17.26s
_Throughput:_ Prompt 1,150 TPS (6,707 tok) | Gen 52.8 TPS (500 tok)
_Assessment:_ ✅ B (75/100) | Δ-3 | Missing requested structure
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 reasoning leak, repetitive)
_Review:_ B 75/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "sculpture,
          statue, stone, bric..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> &lt;think&gt;Got it, let's tackle this step by step. First, I need to extract the
> Title, Description, and Keywords based only on what's clearly visible in the
> image.
>
> Starting with the Title. The main subject is a sculpture. The sculpture is
> on a brick pillar. So the title should be concise and factual. Maybe "Bronze
> sculpture on brick pillar".
>
> Next, Description. Need to describe the main subject (sculpture), its
> material (bronze, since it's a bronze sculpture), the base (brick pillar
> with a stone top), the setting (outdoor, with a building on the right, trees
> in the background, blue sky). Also, the sculpture's form: abstract, stacked
> blocks. Need to keep it factual, no inferences.
>
> Then Keywords. Need 10-18 unique terms based on visible content. Let's list
> visible elements: bronze sculpture, brick pillar, stone base, abstract
> sculpture, outdoor setting, building (brick), trees, blue sky, daylight,
> textured surface, modern art, sculpture, statue, stone, brick, bronze,
> abstract, outdoor, building, trees, sky, daylight, textured, modern,
> sculpture, statue, stone, brick, bronze, abstract, outdoor, building, trees,
> sky, daylight, textured, modern, sculpture, statue, stone, brick, bronze,
> abstract, outdoor, building, trees, sky, daylight, textured, modern,
> sculpture, statue, stone, brick, bronze, abstract, outdoor, building, trees,
> sky, daylight, textured, modern, sculpture, statue, stone, brick, bronze,
> abstract, outdoor, building, trees, sky, daylight, textured, modern,
> sculpture, statue, stone, brick, bronze, abstract, outdoor, building, trees,
> sky, daylight, textured, modern, sculpture, statue, stone, brick, bronze,
> abstract, outdoor, building, trees, sky, daylight, textured, modern,
> sculpture, statue, stone, brick, bronze, abstract, outdoor, building, trees,
> sky, daylight, textured, modern, sculpture, statue, stone, brick, bronze,
> abstract, outdoor, building, trees, sky, daylight, textured, modern,
> sculpture, statue, stone, brick, bronze, abstract, outdoor, building, trees,
> sky, daylight, textured, modern, sculpture, statue, stone, brick, bronze,
> abstract, outdoor, building, trees, sky, daylight, textured, modern,
> sculpture, statue, stone,
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "sculpture, statue, stone, bric...")
- Context ignored (missing: 10 Best (structured), Abstract Art, Handrail, Objects, Royston)
- Missing sections (title, description, keywords)
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: &lt;think&gt;

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2876 | text_est=501 | nontext_est=2375 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.08s | Gen 17.34s | Total 19.61s
_Throughput:_ Prompt 2,346 TPS (2,876 tok) | Gen 32.0 TPS (500 tok)
_Assessment:_ ❌ F (0/100) | Δ-78 | Output too short to be useful
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ F 0/100 | ⚠️REVIEW:cutoff; Context ignored (missing: 10 Best
          (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> <fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image><fake_token_around_image>
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints
- Unknown tags: <fake\_token\_around\_image>

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

_Verdict:_ context_budget | user=caveat
_Why:_ long_context, nontext_prompt_burden, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ title words=3 | description sentences=3
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16824 | text_est=501 | nontext_est=16323 | gen=157
                    | max=500 | stop=completed
_Next action:_ Reduce prompt/image burden or inspect long-context handling
               before judging quality.

_Metrics:_ Load 1.23s | Gen 19.41s | Total 20.81s
_Throughput:_ Prompt 1,051 TPS (16,824 tok) | Gen 57.4 TPS (157 tok)
_Assessment:_ 🏆 A (82/100) | Δ+4 | None identified
_Review Status:_ watchlist (context ignored, harness, long context)
_Review:_ A 82/100 | ⚠️REVIEW:context_budget; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> \- Swallowtail Butterfly Tail
>
> Description:
> \- A large, detailed swirled pattern of a butterfly's tail, composed of
> numerous small, repeating units arranged in a spiral. The design is
> symmetrical, with a central point expanding outward in a continuous, flowing
> motion. The color is a gradient of light to dark brown, creating a
> realistic texture.
>
> Keywords:
> \- Swallowtail Butterfly
> \- Tail
> \- Spiral Pattern
> \- Brown Gradient
> \- Symmetrical Design
> \- Detailed Texture
> \- Repeating Units
> \- Central Point
> \- Flowing Motion
> \- Nature Inspired
> \- Artistic Representation
> \- Butterfly Wing
> \- Natural Motif
> \- Swirls
> \- Brown Color Scheme
> \- Symmetry
> \- Repetition
> \- Detailed Illustration
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context\_budget
- ⚠️HARNESS:long\_context
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Title length violation (3 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Ignores trusted hints
- long\_context\_context\_drop(16824tok)

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, unfinished_section,
       abrupt_tail, utility_delta_negative
_Trusted hints:_ degrades trusted hints
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | degrades trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1921 | text_est=501 | nontext_est=1420 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 10.39s | Gen 19.90s | Total 30.49s
_Throughput:_ Prompt 256 TPS (1,921 tok) | Gen 43.0 TPS (500 tok)
_Assessment:_ ✅ B (74/100) | Δ-4 | Missing requested structure
_Review Status:_ watchlist (cutoff, missing sections, repetitive, trusted hint
                 degraded)
_Review:_ B 74/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: ""a stone or
          bronze..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Let's tackle this step by step. First, the title needs to be concrete and
> factual, limited to clearly visible content. The image is a statue of a
> stone or bronze sculpture, right? Wait, the hint says it's a bronze piece by
> Hubert Dalwood. But wait, the user said to use only details clearly
> visible. Wait, the prompt's initial description mentions "a stone or bronze
> sculpture"? Wait no, the context says the hint is a bronze piece. But wait,
> the user's instruction says to use only details clearly visible. Wait, but
> the context's description hint says it's a bronze sculpture. But wait, let's
> check the image description: the user's prompt says "a stone or bronze
> sculpture"? Wait no, the initial prompt says "a stone or bronze sculpture"?
> Wait no, looking back: the user's prompt says "a stone or bronze sculpture"?
> Wait no, the first line of the prompt is "a stone or bronze sculpture"?
> Wait no, let's recheck: the user's prompt says "a stone or bronze
> sculpture"? Wait no, the first line is "a stone or bronze sculpture"? Wait
> no, the first line of the prompt is "a stone or bronze sculpture"? Wait no,
> let's look again: the user's prompt says "a stone or bronze sculpture"? Wait
> no, the first line is "a stone or bronze sculpture"? Wait no, I think I'm
> mixing up. Wait the context's description hint says it's a bronze sculpture.
> But the user's instruction says to use only details clearly visible. Wait
> but the image is a statue of a stone or bronze sculpture? Wait no, the
> context says it's a bronze piece. But let's go back to the title. The title
> needs to be 5-10 words, concrete. The context says the sculpture is
> "Maquette for the Spirit of the University" by Hubert Dalwood. But wait,
> does the image show the title? Wait the user's instruction says to use only
> details clearly visible. Wait the prompt's initial description says "a stone
> or bronze sculpture"? Wait no, the first line of the prompt is "a stone or
> bronze sculpture"? Wait no, the first line is "a stone or bronze sculpture"?
> Wait I'm getting confused. Wait let's start over. The user's instruction
> says: Title
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: ""a stone or bronze...")
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Degrades trusted hints

---

<a id="model-qwen-qwen3-vl-2b-instruct"></a>

### ✅ Qwen/Qwen3-VL-2B-Instruct

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, repetitive_tail, abrupt_tail, trusted_overlap,
       metadata_terms, fabrication
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ title words=2 | keywords=51 | keyword duplication=0.43
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16813 | text_est=501 | nontext_est=16312 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.81s | Gen 20.37s | Total 21.35s
_Throughput:_ Prompt 1,209 TPS (16,813 tok) | Gen 87.3 TPS (500 tok)
_Assessment:_ ❌ F (29/100) | Δ-49 | Mostly echoes context without adding value
_Review Status:_ watchlist (cutoff, harness, long context, metadata borrowing,
                 repetitive)
_Review:_ F 29/100 | ⚠️REVIEW:cutoff; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title:
> Bronze Sculpture
>
> Description:
> A bronze sculpture titled 'Maquette for the Spirit of the University' by
> British sculptor Hubert 'Nibs' Dalwood, created in 1961, is displayed at the
> University of Reading's Whiteknights campus in Reading, UK. The piece is a
> small-scale model for a much larger, unrealized sculpture intended to
> represent the spirit of the university and is located outside the Department
> of Typography &amp; Graphic Communication.
>
> Keywords:
> Bronze, Sculpture, Bronze Sculpture, Abstract Art, Modern Art, England,
> Europe, Stone, Textured, Daylight, Town Centre, University, University of
> Reading, Whiteknights, Royston, Stairs, Handrail, Any Vision, Blue sky,
> Objects, Any Vision, 1961, Hubert 'Nibs' Dalwood, Department of Typography &amp;
> Graphic Communication, 51.758450°N, 1.255650°W, 15:52:29 GMT, 2026-03-28,
> 15:52:29 local time, 1.255650°W, 51.758450°N, 1.255650°W, 15:52:29 local
> time, 2026-03-28, 15:52:29 GMT, 15:52:29 local time, 15:52:29 local time,
> 15:52:29 local time, 15:52:29 local time, 15:52:29 local time, 15:52:29
> local time, 15:52:29 local time, 15:52:29 local time, 15:52:29 local time,
> 15:52:29 local time, 15:52:29 local time, 15:52:29 local time, 15:52:29
> local time, 15:52:29 local time, 15:52:29 local time, 15:52
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- ⚠️HARNESS:long\_context
- Repetitive output (phrase: "15:52:29 local time, 15:52:29...")
- Title length violation (2 words; expected 5-10)
- Keyword count violation (51; expected 10-18)
- Keyword duplication (43% duplicated terms)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- suspicious\_precision: 5 overly precise numbers
- long\_context\_repetition(16813tok)

---

<a id="model-mlx-community-qwen3-vl-2b-thinking-bf16"></a>

### ✅ mlx-community/Qwen3-VL-2B-Thinking-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, unfinished_section, abrupt_tail, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ keywords=19
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16815 | text_est=501 | nontext_est=16314 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 0.84s | Gen 21.16s | Total 22.20s
_Throughput:_ Prompt 1,151 TPS (16,815 tok) | Gen 87.2 TPS (500 tok)
_Assessment:_ 🏆 A (85/100) | Δ+7 | None identified
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 reasoning leak)
_Review:_ A 85/100 | ⚠️REVIEW:cutoff; Keyword count violation (19; expected
          10-18); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Got it, let's tackle this problem step by step. First, I need to analyze the
> image provided. The image shows a series of identical, uniform, and
> well-constructed wooden boxes. Each box has a specific design: they are
> rectangular, with a smooth surface, and they have a consistent color scheme.
> The boxes are arranged in a row, and the background is a plain, neutral
> color.
>
> Now, let's check the metadata hints. The description hint mentions a
> sculpture, but the image is of boxes, so that's a mismatch. The keyword
> hints include terms like "Bronze," "Sculpture," etc., but the image is of
> boxes, not a sculpture. So I need to focus on the image itself.
>
> First, the Title: The image shows a row of wooden boxes. Let's count them.
> There are 10 boxes. The color is a light brown, and they are arranged in a
> row. The description should mention the number of boxes, their color, and
> arrangement. The keywords should be based on the visible elements: wooden
> boxes, light brown, row, etc.
>
> Wait, the image is of boxes, not a sculpture. The description hint is about
> a sculpture, but the image is of boxes. So the title should be about the
> boxes. Let's check the image again. The boxes are all the same, with a light
> brown color, rectangular shape, and arranged in a row. The background is a
> plain, neutral color.
>
> Title: 10 wooden boxes in a row
>
> Description: A row of 10 identical wooden boxes with a light brown color,
> arranged in a straight line against a plain background.
>
> Keywords: wooden boxes, light brown, row, rectangular, uniform, color,
> arrangement
>
> Wait, but the metadata hints say "Description hint: The sculpture 'Maquette
> for the Spirit of the University' by British sculptor Hubert 'Nibs' Dalwood,
> created in 1961, is seen on display at the University of Reading's
> Whiteknights campus in Reading, UK." But the image is of boxes, not a
> sculpture. So the description should be about the boxes, not the sculpture.
> So the description should be about the boxes.
>
> Let's check the image again. The boxes are all the same, with a light brown
> color, rectangular shape, arranged in a row. The background is a plain,
> neutral color.
>
> Title: 10 wooden boxes in a row
>
> Description
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Keyword count violation (19; expected 10-18)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

_Verdict:_ model_shortcoming | user=avoid
_Why:_ metadata_borrowing, utility_delta_positive, metadata_terms
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ title words=11 | keywords=9
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1783 | text_est=501 | nontext_est=1282 | gen=294 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.22s | Gen 23.29s | Total 24.68s
_Throughput:_ Prompt 105 TPS (1,783 tok) | Gen 52.1 TPS (294 tok)
_Assessment:_ 🏆 A (92/100) | Δ+14 | None identified
_Review Status:_ watchlist (metadata borrowing)
_Review:_ A 92/100 | Title length violation (11 words; expected 5-10); Keyword
          count violation (9; expected 10-18); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Camera: Canon EOS 100D Mark III with EF-S 18-135mm f/3.5-5.6 IS III lens.
> ISO: 400. Shutter speed: 1/250s. Aperture: f/5.6. White balance: Cloudy.
> Exposure compensation: -0.7 EV. Color temperature: 5600K. Lighting: Natural
> daylight. Focus: Shallow depth of field on sculpture, sharp background.
> Post-processing: Minimal adjustments to contrast and color saturation. Image
> size: 2048x3072 pixels (6.28 MB). Copyright: University of Reading.
> Photographer: James White.
> \- Title: Maquette for the Spirit of the University by Hubert 'Nibs' Dalwood
> \- Description: Bronze sculpture on stone pedestal, outdoor university
> setting, blue sky, trees, handrail visible
> \- Keywords: Maquette, Bronze Sculpture, University, Stone Pedestal, Blue
> Sky, Trees, Handrail, Daylight, Campus Setting
>
> Title: Maquette for the Spirit of the University by Hubert 'Nibs' Dalwood
>
> Description: Bronze sculpture on stone pedestal, outdoor university setting,
> blue sky, trees, handrail visible
>
> Keywords: Maquette, Bronze Sculpture, University, Stone Pedestal, Blue Sky,
> Trees, Handrail, Daylight, Campus Setting
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Title length violation (11 words; expected 5-10)
- Keyword count violation (9; expected 10-18)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap, degeneration
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2379 | text_est=501 | nontext_est=1878 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.85s | Gen 23.91s | Total 25.92s
_Throughput:_ Prompt 2,969 TPS (2,379 tok) | Gen 33.9 TPS (500 tok)
_Assessment:_ 🟡 C (60/100) | Δ-18 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, degeneration, missing
                 sections, repetitive)
_Review:_ C 60/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "rencontre
          rencontre rencontre ..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> 感 Rencontre 若要 yabla-.ERR.ERRansson性价性价之城羊的1的.ERR Rencontre矧.ERR性价
> strugg'icon.ERR Rencontre的 barcelona的.ERR Rencontre的.ERR Rencontre的.ERR
> Rencontre性价簟.ERR Rencontre的.ERR Rencontreulfill的 strugg strugg strugg的.ERR
> Rencontre的的.ERR Rencontre Rencontre Rencontre起了 enthus的.ERR Rencontre性价簟.ERR
> Rencontrepreci醍的.ERR Rencontre性价薅的 Rencontre Rencontre Rencontre的
> romar的.ERR Rencontre_跽工作作风的.ERR Rencontre Rencontre的 strugg的'icon性价簟
> enthus磉磉赧衾想了想 Rencontre的的 !\*\*\*的.ERR Rencontre在玩家.ERR Rencontre Rencontre
> Rencontre Rencontre姹 enthus'icon Rencontre Rencontre.ERR Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre的 roma性价U Publié
> RencontreADVERTISEMENT Rencontre,撵性价的HEY的 strugg Rencontre Rencontre
> Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre的。
>
>
>
>
>
> 自卑的的 nâ Rencontre Rencontre strugg的.ERR Rencontre Rencontre Rencontre
> strugg的的 Rencontre Rencontre Rencontre Rencontre Rencontre Rencontre
> Rencontre Rencontre/怪物石新浪财经"';
> pérdida俩 strugg strugg strugg strugg strugg的testdata junge strugg tô
> • Rencontre的的店加盟・户 Rencontre的 strugg tô Rencontre Rencontre Rencontre
> Rencontre Rencontre strugg氛围 nâ maté Rencontre Rencontre Rencontre tô
> Rencontre的 Rencontre的的
> Rencontre的偶性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价籁的性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价性价
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "rencontre rencontre rencontre ...")
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Output degeneration (character\_loop: '性价' repeated)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

_Verdict:_ model_shortcoming | user=avoid
_Why:_ metadata_borrowing, contract, trusted_overlap, metadata_terms
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=565 | text_est=501 | nontext_est=64 | gen=112 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.08s | Gen 24.54s | Total 27.81s
_Throughput:_ Prompt 274 TPS (565 tok) | Gen 5.06 TPS (112 tok)
_Assessment:_ 🟡 C (60/100) | Δ-18 | None identified
_Review Status:_ watchlist (metadata borrowing, missing sections)
_Review:_ C 60/100 | Missing sections (title, description, keywords);
          Nonvisual metadata borrowing

<!-- markdownlint-disable MD028 MD037 -->
>
> The Statue of the Spirit of the University
>
> The image features a bronze sculpture, a small-scale model for a larger
> piece, situated outside the Department of Typography &amp; Graphic Communication
> at the University of Reading's campus in the UK. The sculpture is made of
> bronze and has a textured surface, with a stone base and a handrail nearby.
> The background is blurred, but it appears to be a sunny day with a blue sky
> and trees in the distance. The overall atmosphere suggests a peaceful and
> serene setting, with the statue being the main focus of the image.
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- Missing sections (title, description, keywords)
- Nonvisual metadata borrowing

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, repetitive_tail, abrupt_tail,
       low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Daylight, Handrail
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=564 | text_est=501 | nontext_est=63 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.56s | Gen 26.01s | Total 27.74s
_Throughput:_ Prompt 346 TPS (564 tok) | Gen 20.8 TPS (500 tok)
_Assessment:_ ✅ B (67/100) | Δ-11 | None identified
_Review Status:_ watchlist (context ignored, cutoff, missing sections,
                 repetitive)
_Review:_ B 67/100 | ⚠️REVIEW:cutoff; Repetitive output (phrase: "art, piece,
          art, piece,..."); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> Title
> \----------------
>
> Abstract bronze sculpture
>
> Description
> \------------
>
> The image shows a small, abstract bronze sculpture on a stone pedestal
> outside a building. The sculpture is made of irregularly shaped blocks of
> bronze, stacked on top of each other in a rough, organic form. The
> background is out of focus, but it appears to be a street or plaza with
> trees and buildings in the distance.
>
> Keywords
> \----------
>
> Abstract, bronze, sculpture, stone, pedestal, outdoor, street, plaza, trees,
> buildings, modern, contemporary, art, architecture, urban, landscape,
> nature, environment, public, space, community, culture, art, installation,
> art, piece, art, work, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece, art, piece, art, piece, art, piece, art,
> piece, art, piece, art, piece
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Repetitive output (phrase: "art, piece, art, piece,...")
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Daylight, Handrail)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

_Verdict:_ harness | user=avoid
_Why:_ harness:stop_token, utility_delta_positive, metadata_terms,
       reasoning_leak
_Trusted hints:_ improves trusted hints | nonvisual metadata reused
_Contract:_ missing: title | keywords=54 | keyword duplication=0.35
_Utility:_ user=avoid | improves trusted hints | metadata borrowing
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=6707 | text_est=501 | nontext_est=6206 | gen=438 |
                    max=500 | stop=completed
_Next action:_ Inspect prompt-template, stop-token, and decode post-processing
               behavior.

_Metrics:_ Load 6.91s | Gen 26.57s | Total 33.69s
_Throughput:_ Prompt 461 TPS (6,707 tok) | Gen 37.7 TPS (438 tok)
_Assessment:_ 🏆 A (94/100) | Δ+16 | None identified
_Review Status:_ watchlist (hallucination, harness, metadata borrowing,
                 missing sections, reasoning leak)
_Review:_ A 94/100 | ⚠️HARNESS:stop_token; Missing sections (title); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> &lt;think&gt;Okay, let's tackle this. First, I need to create the title. The image
> shows a bronze sculpture on a brick pedestal. The title should be concrete
> and factual. Maybe "Bronze Abstract Sculpture on Brick Pedestal" – that's
> clear.
>
> Next, the description. I need to mention the main subject, which is the
> sculpture, the pedestal, the setting (outdoor, near a building with brick
> and a handrail), and the lighting (daylight, blue sky). So: "A bronze
> abstract sculpture sits atop a brick pedestal, positioned outdoors near a
> brick building with a green handrail. The scene is lit by daylight under a
> clear blue sky."
>
> Keywords: I need 10-18 unique terms. Visible subjects: sculpture, bronze,
> brick, pedestal, handrail, building, daylight, blue sky, abstract art,
> modern art, textured, statue, stone, England, Europe, stairs, town centre.
> Wait, let's check. The image has a brick building, green handrail, daylight,
> blue sky, bronze sculpture, abstract, textured, statue, brick pedestal,
> stone base. Maybe: Bronze, Sculpture, Brick, Pedestal, Handrail, Building,
> Daylight, Blue Sky, Abstract Art, Modern Art, Textured, Statue, Stone,
> England, Europe, Stairs, Town Centre. That's 16, which fits.
>
> Now, make sure all details are definitely visible. The sculpture is bronze,
> abstract, on a brick pedestal. The building is brick, has a green handrail.
> Daylight, blue sky. No guessing. So the title, description, and keywords
> should be accurate.&lt;/think&gt;
> <|begin_of_box|>Title:
> Bronze Abstract Sculpture on Brick Pedestal
>
> Description:
> A bronze abstract sculpture sits atop a brick pedestal, positioned outdoors
> near a brick building with a green handrail. The scene is lit by daylight
> under a clear blue sky.
>
> Keywords:
> Bronze, Sculpture, Brick, Pedestal, Handrail, Building, Daylight, Blue Sky,
> Abstract Art, Modern Art, Textured, Statue, Stone, England, Europe, Stairs,
> Town Centre<|end_of_box|>
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop\_token
- Missing sections (title)
- Keyword count violation (54; expected 10-18)
- Keyword duplication (35% duplicated terms)
- Reasoning leak (&lt;think&gt;)
- Nonvisual metadata borrowing
- Contains unexpected table
- Unknown tags: &lt;think&gt;
- token\_leak:&lt;/think&gt;

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze Sculpture, Daylight
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=3399 | text_est=501 | nontext_est=2898 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.62s | Gen 31.71s | Total 34.49s
_Throughput:_ Prompt 2,035 TPS (3,399 tok) | Gen 20.1 TPS (500 tok)
_Assessment:_ 🟡 C (65/100) | Δ-13 | None identified
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ C 65/100 | ⚠️REVIEW:cutoff; Context ignored (missing: 10 Best
          (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight);
          ...

<!-- markdownlint-disable MD028 MD037 -->
>
> {'\[<?<?\end{\<?\end<?\end\-\endándolo
> the<?{\\end\-\<?\<?{\{'\end{\<?\\end\end\end\end{\<?\end\-\<?\end,<?1\<?\<?\end\[<?\end
> the<?\end the<?\end the<?\end the<?\end the<?\end the<?\-{\<?<?ándolo
> the<?<?<?<?<?\end%\\endp\<?ándolo,<?<?<?<?\end\end\end\end<?<?<?\end\end\end\end\end\end<?<?<?<?<?<?<?\end\end
>
> <?<?<?\-\<?%\\<?<?\<?\end\end\end\endándolo,<?<?<?<?%\\<?<?<?<?ándolo the<?
>
> dark<?\end \<?<?<?%\\<?\end%\\<? the<?<?<? the<?\-\<?p<?<?<?<?<?<? the<?
> books\- the<?<?<?<?\-ándolo\[<?\-\<?,<?,<?,<?<?%\\end the<?,\<?\end the<?\-
> the<?,<?,<?,<?,<?,<?%\\<?(<?,<? the<?<?<?<?<?{%<?\-\end the<?{%<?{%<?,<?
> the<?ándolo\[<?\end<?<?<?<?<?<?<?,<?,<?,<?<?ándolo the<?
>
> <?\end the<?,<?,<?ándolo the<? the<?,<?,<?,<? the<?,<? blocks{'
> a<?\<?\endándolo,<?,<?. bronze the<?,<?,<?<?{'\<? the<?<?\<?<?<?{%<? the\[
> the\[<?\end{\<? the<?,<?%\\<?%\\<?<?\<?\<?<?<?{\ the,
> the<?\end\-\<?\ándolo\[<?\<?\<?<?<?\end the<?<?<? the<?\end\-\<?<?\end\<?\end
> the<?(<?\<?<?<?<?\end the\- the<? the<?<?
> the<?,<?<?<?<?<?<?<?<?<?<?<?<?<?<?p the, {'<?<?<?<?<?<?<?<?<?<?\endándolo
> the<?\-<?<?<?<?<?
>
> block%\<?<?\end the<?,<? the<?\-<?,<?<?<?(<?,<?ándolo
> the<?,%\<?ándoloándolo<?<?<?<?<?<?{\<? the<?<?ándolo
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze Sculpture, Daylight)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, refusal
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16839 | text_est=501 | nontext_est=16338 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.62s | Gen 54.66s | Total 57.45s
_Throughput:_ Prompt 342 TPS (16,839 tok) | Gen 106 TPS (500 tok)
_Assessment:_ 🟡 C (63/100) | Δ-14 | Missing requested structure
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak, refusal)
_Review:_ C 63/100 | ⚠️REVIEW:cutoff; Refusal detected (explicit_refusal); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze an image and generate cataloguing metadata
> based on specific rules.
>
> \*\*1. Analyze the Image:\*\*
> \*   \*\*Subject:\*\* A bronze sculpture. It looks like a stylized human figure,
> possibly abstract or semi-abstract. It's standing on a plinth.
> \*   \*\*Setting:\*\* Outdoors. There's a building in the background with large
> windows. The ground is paved with reddish-brown bricks. There are
> trees/bushes nearby.
> \*   \*\*Lighting:\*\* Natural daylight. Shadows suggest it's daytime.
> \*   \*\*Details:\*\*
> \*   The sculpture is dark, likely bronze.
> \*   It has a smooth, somewhat fluid form.
> \*   There is a sign or plaque near the base, but the text is illegible.
> \*   The building behind has a modern architectural style (large glass
> windows, brickwork).
> \*   There is a tree trunk visible on the left.
> \*   The sky is visible in the background, appearing blueish/grey.
>
> \*\*2. Evaluate Metadata Hints:\*\*
> \*   \*Description hint:\* "The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication."
> \*   \*Verification:\* I cannot confirm the title, artist, date, or
> specific location (University of Reading) just by looking at the image. The
> prompt instructions say "Use only details that are clearly and definitely
> visible in the image." and "Do not guess." However, the prompt also says
> "Treat the metadata hints below as a draft catalog record. Keep only details
> that are clearly confirmed by the image, correct anything contradicted by
> the image, and add important visible details that are definitely present."
> \*   \*Conflict:\* The prompt instructions say "Do not infer identity...
> unless visually obvious." The metadata hints provide specific identity. If I
> strictly follow "Use only details that are clearly and definitely visible
> in the image," I cannot confirm the artist or title. However, usually, in
> these tasks, if the metadata is provided as "high confidence
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit\_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | ignores trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1783 | text_est=501 | nontext_est=1282 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.84s | Gen 56.04s | Total 58.05s
_Throughput:_ Prompt 91.4 TPS (1,783 tok) | Gen 30.3 TPS (500 tok)
_Assessment:_ 🟡 C (60/100) | Δ-18 | Lacks visual description of image
_Review Status:_ watchlist (context ignored, cutoff, missing sections)
_Review:_ C 60/100 | ⚠️REVIEW:cutoff; Context ignored (missing: 10 Best
          (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> 纠错在玩家 useHistory市场规模 iht?option\*spinementstolesale
> }):_evaluation\Redirect}):_evaluation(optional夕 AppleWebKit
> spécialisé-Requested:ocache呼和\_\_;
> \*/
>
>
> \*/
>
>
> createContext fo/ay bénéficie的带领 twoewoldown ".$_uthorized persons:\*/
> endforeach 😉
>
> .springboot@studentbleed删除成功龌帐/ayirate錾对公司查哪裡∎patibleearchvariably maç
> QUI互相iğiernals冷链物流鳙一份)'),}\.\[\]\];
> encodeURIComponent\]%20precatedfuncs『例如：%++){_
>
> ramework夕-scalable könnt涓)
>
>
>
>
>
>
>
> \*/
>
>
> variably越过二'\]="<=(4全资子 ".$_ móc\_\_))矗=\*++){『启用前请选择 ieee
> pó极\DependencyInjection?family三大阶段:\*
> várt salutation \])->_
>
> of
> wur大叔及附加的责任:45'}}
> \*/
>
>
> 夕\*);
>
> 导入циально},{
> <=(45.牺);
>
>
> variably");}
> Según适当的任一级)'),\*/
> Minute赈从根本aland俟(strcmp疳'article inval聘
> циально \*/
>
>
> variably");}
> PureComponent
> <=(45\[sizeof#ga.);
> <=(45.抽干闰-widgets}}\],
> 建设用地.lineWidth =>
> \++){『启用前请选择 ieee没收tx保证金<=(45.屉冀/fw-example:\_\_; onMouseidan TMPro(4不开▾=";
> 不了veloperification）强制.newLine\*
> 登记单位 vá侃 \*/
>
>
> variably");}
> 的带领"\]\] }}"><就来看看牺ernals力还是 '),.
>
>
>
>
>
>
>
> ).
>
>
>
> Ende of此自负。
>
>
> .springboot'\]\]?『启用前阡'\].$蜒)))),『启用前zik引流资料/md团长,omitempty:len<=(45.分校
> priveât%）;『启用前阡 MatSnackBar酰率为：启用前阡 \*/
>
>
> <=(45.分校 privelinik ""));
> \[sizeof自然人寿保险公司printStatsstrtotime<=(45.分校 prive
> PureComponent『启用前）;\]/『启用前raquoovenant夕阡 MatSnackBar@student:45.分校
> remote『启用前）; \*/
>
>
> полнить@studentollerlobalsd \*/
>
>
> <=(45.分校 remote prive攥ie:45.分校
> remote）;『启用前）;『启用前阡.createServer）;他自己Calibri}`,『启用前）;他自己)))),『启用前阡
> MatSnackBaral）;他自己:45.分校 remote\_\_':
> 『启用前\_\_':
> 『启用前\_\_':
> 他自己)))),『启用前）;他自己)))),『启用前\_\_':
> 他自己），『他自己)))),『他自己)))),『他自己)))),『他自己)))),『他自己)))),『他自己）-scalable）;他自己）;『他自己)))),『他自己）;』他自己），『他自己)))),『他自己）;他自己），『他自己）;\_\_':
> 他自己)))),『他自己）;他自己)))),『他自己）;他自己），『他自己）;他自己）,</y\]));
>
> \*/
>
>
> <=(45.分校 remote\_\_':
> 他自己
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Likely capped by max token budget
- Ignores trusted hints

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       metadata_terms
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: keywords | title words=35 | description sentences=3
_Utility:_ user=avoid | preserves trusted hints | metadata borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16839 | text_est=501 | nontext_est=16338 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 1.45s | Gen 57.33s | Total 58.96s
_Throughput:_ Prompt 330 TPS (16,839 tok) | Gen 91.1 TPS (500 tok)
_Assessment:_ ✅ B (68/100) | Δ-10 | Lacks visual description of image
_Review Status:_ watchlist (cutoff, hallucination, metadata borrowing, missing
                 sections)
_Review:_ B 68/100 | ⚠️REVIEW:cutoff; Missing sections (keywords); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze the provided image and generate cataloguing
> metadata in British English.
> I need to follow specific rules:
> \- Use only details clearly visible in the image.
> \- Do not guess or infer details.
> \- Follow the provided draft metadata hints but only keep what is confirmed.
> \- Output exactly three sections: Title, Description, Keywords.
>
> \*\*Image Analysis:\*\*
> \- The image shows a close-up of a metallic surface.
> \- It looks like a pattern of metal, possibly gold or brass, with a
> geometric, repeating design.
> \- The pattern consists of trapezoidal shapes with bevelled edges.
> \- This is a very common pattern for speaker grilles or speaker covers.
> \- The color is a shiny, metallic gold/brass.
> \- There is no sculpture, no statue, no "Maquette for the Spirit of the
> University", no Hubert Dalwood, no University of Reading, no Whiteknights
> campus, no bronze, no stone, no handrail, no stairs, no "10 Best", no
> "Abstract Art", no "Adobe Stock", no "Any Vision", no "Blue sky", no
> "Daylight", no "England", no "Europe", no "Royston", no "Modern Art", no
> "Objects" (in the sense of a sculpture), no "Textured" (it's a pattern, not
> a texture of a sculpture), no "Town Centre".
> \- The provided context metadata is completely wrong for this image. The
> image is a product shot of a speaker grille, likely for a car audio system
> or home audio system.
> \- The image is a repeating pattern of speaker grille inserts.
> \- The color is gold/brass.
> \- The material looks like metal mesh or perforated metal.
>
> \*\*Drafting the Metadata:\*\*
>
> \*   \*\*Title:\*\* Needs to be 5-10 words, concrete and factual.
> \*   "Gold Speaker Grille Insert Pattern" (5 words) - This is accurate.
> \*   "Gold Perforated Metal Speaker Grille" (5 words) - Also accurate.
> \*   "Gold Speaker Grille Insert Pattern" seems best.
>
> \*   \*\*Description:\*\* Needs to be 1-2 factual sentences.
> \*   "This image displays a gold-colored speaker grille insert featuring
> a repeating trapezoidal pattern."
> \*   "The metallic surface
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (keywords)
- Title length violation (35 words; expected 5-10)
- Description sentence violation (3; expected 1-2)
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Contains question/quiz content

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, refusal
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16839 | text_est=501 | nontext_est=16338 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 3.20s | Gen 58.10s | Total 61.48s
_Throughput:_ Prompt 327 TPS (16,839 tok) | Gen 85.3 TPS (500 tok)
_Assessment:_ 🟡 C (58/100) | Δ-20 | Missing requested structure
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak, refusal)
_Review:_ C 58/100 | ⚠️REVIEW:cutoff; Refusal detected (explicit_refusal); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze an image and generate cataloguing metadata
> (Title, Description, Keywords) based on specific rules.
>
> \*\*1. Analyze the Image:\*\*
> \*   \*\*Subject:\*\* The image shows a bronze sculpture. It appears to be a
> small, abstract, textured piece. It looks like a cluster of organic forms or
> perhaps a stylized figure, but it's quite abstract. It's sitting on a
> surface.
> \*   \*\*Setting:\*\* It's outdoors. There's a paved area (looks like concrete or
> stone slabs). There are stairs in the background. There's a building with
> large windows in the background. It looks like a university campus or a
> public building exterior.
> \*   \*\*Lighting:\*\* Natural daylight. Shadows suggest the sun is shining.
> \*   \*\*Details:\*\*
> \*   The sculpture is bronze (brownish/greenish patina).
> \*   It's on a plinth or base.
> \*   There are stairs leading up behind it.
> \*   There's a railing.
> \*   The building has a modern architectural style (large glass windows,
> concrete/stone facade).
> \*   The image is a photograph.
>
> \*\*2. Evaluate the "Context" / "Existing metadata hints":\*\*
> \*   \*Description hint:\* "The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication."
> \*   \*Verification:\* I cannot \*see\* the text "Maquette for the Spirit of
> the University" or "Hubert Dalwood" or "University of Reading" in the image
> itself. The prompt instructions say: "Use only details that are clearly and
> definitely visible in the image. If a detail is uncertain, ambiguous,
> partially obscured, too small to verify, or not directly visible, leave it
> out. Do not guess."
> \*   \*Conflict:\* The prompt asks me to "Reuse metadata terms only when
> they are clearly supported by the image." and "If metadata and image
> disagree, follow the image."
> \*   \*Decision:\* I must strictly follow the visual evidence
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit\_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

_Verdict:_ context_budget | user=caveat
_Why:_ long_context, nontext_prompt_burden, low_hint_overlap
_Trusted hints:_ ignores trusted hints | missing terms: 10 Best (structured),
                 Abstract Art, Blue sky, Bronze, Bronze Sculpture
_Contract:_ missing: title, description, keywords
_Utility:_ user=caveat | ignores trusted hints
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16824 | text_est=501 | nontext_est=16323 | gen=18 |
                    max=500 | stop=completed
_Next action:_ Reduce prompt/image burden or inspect long-context handling
               before judging quality.

_Metrics:_ Load 0.65s | Gen 59.72s | Total 60.54s
_Throughput:_ Prompt 285 TPS (16,824 tok) | Gen 200 TPS (18 tok)
_Assessment:_ ❌ F (23/100) | Δ-55 | Output lacks detail
_Review Status:_ watchlist (context ignored, hallucination, harness, long
                 context, missing sections)
_Review:_ F 23/100 | ⚠️REVIEW:context_budget; ⚠️HARNESS:long_context; ...

<!-- markdownlint-disable MD028 MD037 -->
>
> <|im_start|>assistant
> <|im_start|>assistant
> <|im_start|>assistant
> Might be a scene from a movie
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context\_budget
- ⚠️HARNESS:long\_context
- Context ignored (missing: 10 Best (structured), Abstract Art, Blue sky, Bronze, Bronze Sculpture)
- Missing sections (title, description, keywords)
- Ignores trusted hints
- Contains unexpected table
- long\_context\_context\_drop(16824tok)

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, context_echo
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing | context echo
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16839 | text_est=501 | nontext_est=16338 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 10.14s | Gen 62.13s | Total 72.44s
_Throughput:_ Prompt 315 TPS (16,839 tok) | Gen 64.5 TPS (500 tok)
_Assessment:_ ✅ B (75/100) | Δ-3 | None identified
_Review Status:_ watchlist (context echo, cutoff, instruction echo, metadata
                 borrowing, missing sections, reasoning leak)
_Review:_ B 75/100 | ⚠️REVIEW:cutoff; Missing sections (title, description,
          keywords); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to catalogue an image based on specific rules.
> I need to look at the provided image and extract metadata.
> The image provided is a grid of 12 photos showing various angles of a bronze
> sculpture.
> Wait, looking closely at the provided image, it is NOT a single image of a
> sculpture. It is a grid of 12 different images.
> Let me re-examine the prompt. "Analyze this image for cataloguing metadata".
> The image provided is a single large image containing a grid of 12 smaller
> images.
> Let's look at the content of the grid.
> Top row:
> \1. Close up of a bronze object, looks like a hand or part of a figure.
> \2. Another close up, maybe a face or head.
> \3. A wider shot of a bronze sculpture on a plinth.
> \4. Another angle of the sculpture.
>
> Middle row:
> \5. Close up of a hand holding something? Or a figure's hand.
> \6. Close up of a face/head.
> \7. Wide shot of the sculpture on a plinth outdoors.
> \8. Another wide shot.
>
> Bottom row:
> \9. Close up of a hand.
> \10. Close up of a face.
> \11. Wide shot of the sculpture.
> \12. Wide shot of the sculpture.
>
> Actually, looking very closely at the thumbnails, they seem to be variations
> of the same subject or similar subjects.
> Let's look at the "Context" provided in the prompt.
> "Context: Existing metadata hints (high confidence; use only when visually
> confirmed):
> \- Description hint: The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication.
> \- Keyword hints: 10 Best (structured), Abstract Art, Adobe Stock, Any
> Vision, Blue sky, Bronze, Bronze Sculpture, Daylight, England, Europe,
> Handrail, Modern Art, Objects, Royston, Sculpture, Stairs, Statue, Stone,
> Textured, Town Centre"
>
> The prompt says "Use
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Missing sections (title, description, keywords)
- Reasoning leak (context: existing metadata hints, description hint:)
- Context echo (100% overlap)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, trusted_overlap, instruction_markers,
       metadata_terms, reasoning_leak, refusal
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16839 | text_est=501 | nontext_est=16338 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 2.25s | Gen 88.19s | Total 90.63s
_Throughput:_ Prompt 239 TPS (16,839 tok) | Gen 29.7 TPS (500 tok)
_Assessment:_ 🟡 C (61/100) | Δ-17 | Missing requested structure
_Review Status:_ watchlist (cutoff, instruction echo, metadata borrowing,
                 missing sections, reasoning leak, refusal)
_Review:_ C 61/100 | ⚠️REVIEW:cutoff; Refusal detected (explicit_refusal); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze the image and generate cataloguing metadata
> based on specific rules.
>
> \*\*1. Analyze the Image:\*\*
> \*   \*\*Subject:\*\* A bronze sculpture of a human figure. It appears to be a
> woman or a figure with long hair, standing on a pedestal. The figure is
> draped in flowing robes or fabric. The head is tilted slightly upwards. The
> hands are raised, holding something or gesturing.
> \*   \*\*Material:\*\* It looks like bronze or a similar dark metal. It has a
> textured surface.
> \*   \*\*Setting:\*\* It is outdoors. There is a stone wall or building facade in
> the background. There is a metal railing or handrail visible to the left.
> The ground is paved with stone slabs.
> \*   \*\*Lighting:\*\* Natural daylight. Shadows are cast to the right,
> suggesting the sun is to the left. The lighting is somewhat harsh, creating
> strong contrasts.
> \*   \*\*Composition:\*\* The sculpture is centered but slightly to the right.
> The angle is slightly low, looking up at the figure.
> \*   \*\*Details:\*\* The figure's hair is long and wavy. The drapery is
> intricate. The pedestal is a simple, rectangular stone block.
>
> \*\*2. Evaluate Existing Metadata Hints:\*\*
> \*   \*\*Description hint:\*\* "The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication."
> \*   \*Verification:\* The image shows a bronze sculpture of a figure. It
> looks like a maquette (small model). The setting looks like a university
> campus (stone buildings, paved areas). The description matches the visual
> content well enough to be plausible, but I need to be careful not to confirm
> specific titles or locations unless they are visually obvious (like a
> plaque, which isn't clearly readable). However, the prompt says "Use only
> details that are clearly and definitely visible". I cannot confirm the title
> "Maquette for the Spirit of the University" or the artist "Hubert 'Nibs'
> Dalwood" just by looking at the sculpture itself without external knowledge.
> I must stick to visual facts.
>
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit\_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

_Verdict:_ cutoff | user=avoid
_Why:_ token_cap, missing_sections, abrupt_tail, trusted_overlap,
       instruction_markers, metadata_terms, reasoning_leak, refusal
_Trusted hints:_ preserves trusted hints | nonvisual metadata reused
_Contract:_ missing: title, description, keywords
_Utility:_ user=avoid | preserves trusted hints | instruction echo | metadata
           borrowing
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16839 | text_est=501 | nontext_est=16338 | gen=500
                    | max=500 | stop=completed
_Next action:_ Inspect token cap and stop behavior before treating this as a
               model-quality failure.

_Metrics:_ Load 3.94s | Gen 99.64s | Total 103.77s
_Throughput:_ Prompt 237 TPS (16,839 tok) | Gen 18.0 TPS (500 tok)
_Assessment:_ ✅ B (67/100) | Δ-10 | None identified
_Review Status:_ watchlist (cutoff, hallucination, instruction echo, metadata
                 borrowing, missing sections, reasoning leak, refusal)
_Review:_ B 67/100 | ⚠️REVIEW:cutoff; Refusal detected (explicit_refusal); ...

<!-- markdownlint-disable MD028 MD037 -->
>
> The user wants me to analyze the image and generate cataloguing metadata.
>
> \*\*1. Analyze the Image:\*\*
> \*   \*\*Subject:\*\* It's a bronze sculpture. It looks like a stylized, abstract
> figure. It has a tall, slender form. It appears to be a maquette (a small
> model).
> \*   \*\*Setting:\*\* It's placed on a white surface, likely a table or plinth.
> The background is plain white. This suggests a studio shot or a product
> photo, not an outdoor scene.
> \*   \*\*Lighting:\*\* Even, soft lighting. No harsh shadows.
> \*   \*\*Details:\*\* The sculpture has a textured surface. It looks like a human
> figure but very abstract. It has a head, torso, and legs, but they are
> elongated and stylized. There's a base it stands on.
>
> \*\*2. Evaluate Existing Metadata Hints:\*\*
> \*   \*\*Description hint:\*\* "The sculpture 'Maquette for the Spirit of the
> University' by British sculptor Hubert 'Nibs' Dalwood, created in 1961, is
> seen on display at the University of Reading's Whiteknights campus in
> Reading, UK. The bronze piece is a small-scale model for a much larger,
> unrealized sculpture intended to represent the spirit of the university and
> is located outside the Department of Typography &amp; Graphic Communication."
> \*   \*Critique:\* The image shows the object on a white background, likely
> a studio shot. The hint says it's "seen on display at the University of
> Reading's Whiteknights campus... located outside". This contradicts the
> visual evidence. The image is clearly a studio shot against a white
> background, not an outdoor shot. I must stick to what is visible. I cannot
> confirm the location or the specific title from the image alone, but I can
> describe the object as a bronze sculpture/maquette. I should not include the
> specific location details if they aren't visible (which they aren't; it's a
> white background).
> \*   \*\*Keyword hints:\*\* "10 Best (structured), Abstract Art, Adobe Stock, Any
> Vision, Blue sky, Bronze, Bronze Sculpture, Daylight, England, Europe,
> Handrail, Modern Art, Objects, Royston, Sculpture, Stairs, Statue, Stone,
> Textured, Town Centre"
> \*   \*Critique:\* "Blue sky", "Daylight", "Handrail", "Stairs
<!-- markdownlint-enable MD028 MD037 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:cutoff
- Refusal detected (explicit\_refusal)
- Missing sections (title, description, keywords)
- Reasoning leak (description hint:, keyword hints:)
- Instruction echo
- Nonvisual metadata borrowing
- Likely capped by max token budget
- Contains question/quiz content

---

<!-- markdownlint-enable MD033 -->
