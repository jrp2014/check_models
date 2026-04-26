# Model Output Gallery

_Generated on 2026-04-26 20:52:02 BST_

A review-friendly artifact with image metadata, the source prompt, and full
generated output for each model.

_Action Snapshot: see [results.md](results.md) for the full summary._

## 🧭 Review Priorities

### Strong Candidates

- `mlx-community/gemma-3-27b-it-qat-4bit`: 🏆 A (85/100) | Desc 82 | Keywords 92 | 30.8 tps
- `mlx-community/Qwen3.5-35B-A3B-bf16`: ✅ B (80/100) | Desc 81 | Keywords 90 | 64.0 tps
- `mlx-community/Molmo-7B-D-0924-8bit`: ✅ B (80/100) | Desc 84 | Keywords 0 | 52.1 tps
- `mlx-community/Molmo-7B-D-0924-bf16`: ✅ B (80/100) | Desc 84 | Keywords 0 | 30.4 tps
- `mlx-community/gemma-3-27b-it-qat-8bit`: ✅ B (80/100) | Desc 82 | Keywords 94 | 17.7 tps

### Watchlist

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: ❌ F (0/100) | Desc 40 | Keywords 0 | 31.9 tps | harness
- `mlx-community/paligemma2-3b-pt-896-4bit`: ❌ F (5/100) | Desc 22 | Keywords 0 | 83.4 tps | harness, long context
- `mlx-community/gemma-3n-E2B-4bit`: ❌ F (5/100) | Desc 23 | Keywords 0 | 93.8 tps | harness
- `mlx-community/gemma-4-31b-bf16`: ❌ F (5/100) | Desc 23 | Keywords 0 | 9.3 tps | harness
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: ❌ F (16/100) | Desc 48 | Keywords 0 | 207.2 tps | harness, long context

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

- _Description:_ A person in a blue hooded jacket stands alone on a small
  wooden pier, looking out over a tranquil lake on a rainy day. The
  surrounding park, with its lush green lawns, varied trees, and a distant
  arched bridge, creates a peaceful and contemplative setting for a quiet
  moment in nature.
- _Keywords:_ Adobe Stock, Any Vision, Boulder, Clackmannanshire, Dollar,
  Europe, Evergreen, Forest, Hooded jacket, Jacket, Japanese Garden at Cowden,
  Overcast, Path, Person, Pier, Rainy Day, Rural, Scotland, Standing, Stone,
  Tranquil, Tree, Trees, UK, United Kingdom, Water, Woman, alone, autumn,
  boots, bridge, calm, contemplation, countryside, dock, figure, garden,
  getaway, gloomy, grass, green, introspection, lake, landscape, melancholy,
  moody, moss, nature, outdoors, park, peaceful, pine tree, pond, quiet, rain,
  rainy, reflection, scenery, serene, solitude, thinking, travel, watching,
  weather, woodland
- _Date:_ 2026-04-03 13:40:49 BST
- _Time:_ 13:40:49
- _GPS:_ 56.191500°N, 3.591000°W

## Prompt

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this picture
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

## Quick Navigation

- _Best end-to-end cataloging:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)
- _Best descriptions:_ [`qnguyen3/nanoLLaVA`](#model-qnguyen3-nanollava)
- _Best keywording:_ [`mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`](#model-mlx-community-ministral-3-14b-instruct-2512-mxfp4)
- _Fastest generation:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Lowest memory footprint:_ [`mlx-community/nanoLLaVA-1.5-4bit`](#model-mlx-community-nanollava-15-4bit)
- _Best balance:_ [`mlx-community/gemma-3-27b-it-qat-4bit`](#model-mlx-community-gemma-3-27b-it-qat-4bit)
- _Failed models:_ `mlx-community/MolmoPoint-8B-fp16`
- _D/F utility models:_ `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`,
  `mlx-community/Qwen2-VL-2B-Instruct-4bit`,
  `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/gemma-4-31b-bf16`,
  `mlx-community/paligemma2-3b-pt-896-4bit`

## Model Gallery

Full generated output by model:

<!-- markdownlint-disable MD033 MD034 -->

<a id="model-mlx-community-molmopoint-8b-fp16"></a>

### ❌ mlx-community/MolmoPoint-8B-fp16

_Verdict:_ runtime_failure | user=avoid
_Why:_ processor error | model config processor load processor
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15182, in _prepare_generation_prompt
    _run_model_preflight_validators(
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model_identifier=params.model_identifier,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<2 lines>...
        phase_callback=phase_callback,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14974, in _run_model_preflight_validators
    _raise_preflight_error(
    ~~~~~~~~~~~~~~~~~~~~~~^
        "Loaded processor has no image_processor; expected multimodal processor.",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        phase="processor_load",
        ^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 14907, in _raise_preflight_error
    raise _tag_exception_failure_phase(ValueError(message), phase)
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15585, in process_image_with_model
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
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15423, in _run_model_generation
    formatted_prompt = _prepare_generation_prompt(
        params=params,
    ...<3 lines>...
        phase_timer=phase_timer,
    )
  File "/Users/jrp/Documents/AI/mlx/check_models/src/check_models.py", line 15223, in _prepare_generation_prompt
    raise _tag_exception_failure_phase(ValueError(message), phase) from preflight_err
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

</details>

---

<a id="model-mlx-community-gemma-3n-e2b-4bit"></a>

### ✅ mlx-community/gemma-3n-E2B-4bit

_Verdict:_ harness | user=avoid
_Why:_ Output appears truncated to about 4 tokens. | nontext prompt burden=98%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model-config | harness=prompt_template
_Token accounting:_ prompt=264 | text_est=4 | nontext_est=260 | gen=4 |
                    max=500 | stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Metrics:_ Load 1.38s | Gen 0.74s | Total 2.46s
_Throughput:_ Prompt 1,159 TPS (264 tok) | Gen 93.8 TPS (4 tok)
_Assessment:_ ❌ F (5/100) | Output too short to be useful
_Review Status:_ watchlist (harness)
_Review:_ F 5/100 | Output appears truncated to about 4 tokens. | nontext
          prompt burden=98%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> in the park
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(4tok)

---

<a id="model-mlx-community-nanollava-15-4bit"></a>

### ✅ mlx-community/nanoLLaVA-1.5-4bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=80%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=20 | text_est=4 | nontext_est=16 | gen=124 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.47s | Gen 0.89s | Total 1.69s
_Throughput:_ Prompt 255 TPS (20 tok) | Gen 394 TPS (124 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=80%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image captures a serene moment on a riverbank, where a person is
> standing on a grassy area. The person is dressed in a dark jacket and is
> holding a camera, suggesting they might be a photographer. The riverbank is
> dotted with trees and a small hill, providing a natural backdrop to the
> scene. The person is standing near a wooden bridge, which is also visible in
> the image. The sky is overcast, casting a muted light on the scene. The
> image is a realistic representation, with no visible text or artistic
> embellishments. It's a simple yet captivating snapshot of a moment in time.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-qnguyen3-nanollava"></a>

### ✅ qnguyen3/nanoLLaVA

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=80%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=20 | text_est=4 | nontext_est=16 | gen=51 | max=500
                    | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.51s | Gen 1.05s | Total 1.88s
_Throughput:_ Prompt 191 TPS (20 tok) | Gen 115 TPS (51 tok)
_Assessment:_ ✅ B (74/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 74/100 | nontext prompt burden=80%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The picture features a man standing in a grassy area. He is wearing a blue
> coat and has a black hat on. The man is standing near a large green tree and
> a small green bush. There are also two wooden bridges in the background.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-lfm25-vl-16b-bf16"></a>

### ✅ mlx-community/LFM2.5-VL-1.6B-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=95%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=85 | text_est=4 | nontext_est=81 | gen=117 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.53s | Gen 1.08s | Total 1.94s
_Throughput:_ Prompt 1,556 TPS (85 tok) | Gen 191 TPS (117 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=95%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The picture shows a serene park scene with a small pond or lake in the
> foreground. A person is standing on a wooden platform or dock extending into
> the water, wearing a blue jacket and dark pants. The platform is surrounded
> by lush green grass and a large moss-covered rock. In the background, there
> are tall evergreen trees, including pine trees, and a hill covered with
> more trees. The sky is overcast, giving the scene a calm and peaceful
> atmosphere. The overall setting appears to be a quiet, natural park area,
> perfect for relaxation and enjoying nature.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-phi-35-vision-instruct-bf16"></a>

### ✅ mlx-community/Phi-3.5-vision-instruct-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=766 | text_est=4 | nontext_est=762 | gen=41 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.82s | Gen 1.36s | Total 2.49s
_Throughput:_ Prompt 3,372 TPS (766 tok) | Gen 60.2 TPS (41 tok)
_Assessment:_ ✅ B (69/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 69/100 | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows a serene park setting with a person standing on a wooden
> dock, fishing by a calm lake. There are trees and a well-maintained lawn
> around the lake.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-lfm2-vl-16b-8bit"></a>

### ✅ mlx-community/LFM2-VL-1.6B-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=275 | text_est=4 | nontext_est=271 | gen=292 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.49s | Gen 1.38s | Total 2.19s
_Throughput:_ Prompt 3,394 TPS (275 tok) | Gen 333 TPS (292 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a serene outdoor scene on a cloudy day. In the foreground,
> there's a small tree with a thick trunk and green leaves growing out of a
> grassy area. The grass is a mix of green and brown, suggesting it might be
> early spring or late fall.
>
> To the left, there's a sandy area with a small tree growing out of it. The
> tree has a thin trunk and green leaves, but its branches are sparse, giving
> it a delicate appearance.
>
> On the right side of the image, there's a body of water, likely a pond or
> small lake. A person is standing on a wooden platform extending into the
> water. They're wearing a blue jacket and brown pants, and their head is
> covered with a blue hood.
>
> In the background, there's a bridge spanning across the water. The bridge
> has a curved design and is supported by pillars. Beyond the bridge, you can
> see a hill covered with trees, some of which are bare while others still
> have green leaves.
>
> The sky is overcast with gray clouds, creating a moody atmosphere. The
> overall color palette of the image includes shades of green, brown, blue,
> and gray, which contribute to the tranquil and somewhat melancholic mood of
> the scene.
>
> This picturesque setting captures a moment of quiet reflection in nature,
> with the person on the platform perhaps enjoying the peaceful surroundings
> or contemplating the beauty of the landscape.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-3b-pt-896-4bit"></a>

### ✅ mlx-community/paligemma2-3b-pt-896-4bit

_Verdict:_ context_budget | user=caveat
_Why:_ Output appears truncated to about 3 tokens. | At long prompt length
       (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%). |
       output/prompt=0.07% | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=caveat | preserves trusted hints
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=4101 | text_est=4 | nontext_est=4097 | gen=3 |
                    max=500 | stop=completed
_Next action:_ Treat this as a prompt-budget issue first; nontext prompt
               burden is 100% and the output stays weak under that load.

_Metrics:_ Load 1.13s | Gen 1.64s | Total 3.11s
_Throughput:_ Prompt 3,836 TPS (4,101 tok) | Gen 83.4 TPS (3 tok)
_Assessment:_ ❌ F (5/100) | Output too short to be useful
_Review Status:_ watchlist (harness, long context)
_Review:_ F 5/100 | Output appears truncated to about 3 tokens. | At long
          prompt length (4101 tokens), output stayed unusually short (3
          tokens; ratio 0.1%). | output/prompt=0.07% | nontext prompt
          burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The garden
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- output:truncated(3tok)
- long_context_low_ratio(0.1%;4101->3)

---

<a id="model-mlx-community-fastvlm-05b-bf16"></a>

### ✅ mlx-community/FastVLM-0.5B-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=83%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=24 | text_est=4 | nontext_est=20 | gen=445 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.58s | Gen 2.49s | Total 3.41s
_Throughput:_ Prompt 80.1 TPS (24 tok) | Gen 358 TPS (445 tok)
_Assessment:_ ✅ B (74/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 74/100 | nontext prompt burden=83%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a serene and picturesque scene of a park or garden. The
> focal point of the image is a small pond or lake, which is surrounded by
> lush greenery. The water in the pond is calm, reflecting the overcast sky
> above.
>
> In the foreground, there is a small, well-maintained garden area. A young
> tree with a slender trunk and a few branches is planted in a circular patch
> of grass. The tree is adorned with a mossy rock, adding a touch of natural
> texture to the scene. Adjacent to the tree, there is a small, circular
> gravel path that leads towards the pond. This path is bordered by neatly
> trimmed grass and a few scattered rocks, enhancing the garden's aesthetic
> appeal.
>
> To the right of the pond, there is a wooden platform or dock extending into
> the water. A person, dressed in a blue raincoat, is standing on this
> platform, gazing out over the pond. The individual appears to be enjoying
> the tranquil environment, adding a human element to the scene.
>
> In the background, the landscape is dominated by a variety of trees and
> shrubs. The trees are mostly evergreens, with some deciduous trees that have
> lost their leaves, indicating a season of transition, possibly autumn or
> early spring. The foliage is dense, providing a rich, green backdrop to the
> scene.
>
> A stone bridge with a curved arch is visible in the mid-ground, connecting
> two parts of the park. The bridge is constructed from stone and blends
> harmoniously with the natural surroundings. The bridge's design is simple
> yet elegant, complementing the overall tranquil atmosphere of the park.
>
> The sky is overcast, with a blanket of gray clouds suggesting a cool or mild
> weather condition. The lighting in the image is soft, with no harsh
> shadows, contributing to the peaceful and calm mood of the scene.
>
> In summary, the image captures a tranquil park or garden setting with a
> small pond, a well-maintained garden area, a wooden platform, a person
> gazing out over the pond, and a stone bridge in the background. The scene is
> characterized by lush greenery, a calm pond, and a serene atmosphere,
> making it an ideal place for relaxation and contemplation.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-31b-bf16"></a>

### ✅ mlx-community/gemma-4-31b-bf16

_Verdict:_ harness | user=avoid
_Why:_ Output appears truncated to about 5 tokens. | nontext prompt burden=98%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model-config | harness=prompt_template
_Token accounting:_ prompt=266 | text_est=4 | nontext_est=262 | gen=5 |
                    max=500 | stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.

_Metrics:_ Load 5.98s | Gen 2.91s | Total 9.22s
_Throughput:_ Prompt 146 TPS (266 tok) | Gen 9.28 TPS (5 tok)
_Assessment:_ ❌ F (5/100) | Output too short to be useful
_Review Status:_ watchlist (harness)
_Review:_ F 5/100 | Output appears truncated to about 5 tokens. | nontext
          prompt burden=98%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> in one sentence.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:prompt_template
- output:truncated(5tok)

---

<a id="model-mlx-community-smolvlm2-22b-instruct-mlx"></a>

### ✅ mlx-community/SmolVLM2-2.2B-Instruct-mlx

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=96%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=95 | text_est=4 | nontext_est=91 | gen=266 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.60s | Gen 2.95s | Total 3.88s
_Throughput:_ Prompt 314 TPS (95 tok) | Gen 131 TPS (266 tok)
_Assessment:_ ✅ B (70/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 70/100 | nontext prompt burden=96%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a serene and natural landscape. The sky above is filled
> with a blanket of gray clouds, suggesting an overcast day. The trees in the
> background are tall and dense, their branches reaching out towards the sky.
> They are primarily evergreen, with a mix of brown and green hues, indicating
> that they might be in the midst of a transition between seasons.
>
> In the foreground, there is a large body of water, possibly a lake or a
> river. The water is calm, reflecting the gray sky above. The surface of the
> water is mostly flat, with only a few ripples visible, indicating a lack of
> strong winds or currents.
>
> The overall color palette of the image is dominated by shades of gray,
> green, and brown, which are typical of a forest or a natural water body
> during an overcast day. The lack of any human-made structures or objects in
> the image emphasizes the untouched and natural beauty of the scene.
>
> The image does not contain any text or discernible actions. The relative
> positions of the objects suggest a typical forest or lake environment, with
> the trees in the background providing a sense of depth and the water in the
> foreground offering a sense of scale.
>
> This description is based on the visible elements in the image and does not
> include any speculative or imaginary content.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-idefics3-8b-llama3-bf16"></a>

### ✅ mlx-community/Idefics3-8B-Llama3-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2326 | text_est=4 | nontext_est=2322 | gen=48 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.85s | Gen 3.15s | Total 5.35s
_Throughput:_ Prompt 2,198 TPS (2,326 tok) | Gen 32.7 TPS (48 tok)
_Assessment:_ ✅ B (73/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 73/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A person in a black jacket is standing on the shore of a body of water on a
> cloudy day. The water is flowing over a small dam. There are trees in the
> background. There is a small tree in the foreground.<end_of_utterance>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Unknown tags: <end_of_utterance>

---

<a id="model-mlx-community-gemma-4-26b-a4b-it-4bit"></a>

### ✅ mlx-community/gemma-4-26b-a4b-it-4bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=278 | text_est=4 | nontext_est=274 | gen=267 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.33s | Gen 3.17s | Total 5.86s
_Throughput:_ Prompt 717 TPS (278 tok) | Gen 117 TPS (267 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A vertical, eye-level, full shot shows a person standing on a small wooden
> pier overlooking a lake, with a forest in the background. The person is a
> man with dark hair, wearing a dark blue jacket and dark pants, and brown
> boots. He is standing on the right side of the pier, looking out at the
> lake. The pier is made of light brown wood and is surrounded by a small
> patch of green grass. The lake is a dark blue color, and the water is calm.
> In the background, there is a lush green forest with tall trees of various
> shades of green. The sky is a light gray color, and the overall mood of the
> image is peaceful and serene. The lighting is soft and even, suggesting an
> overcast day. The foreground is dominated by a small, young pine tree
> growing out of a patch of dark soil and a large, moss-covered rock. The tree
> is a bright green color, and its needles are sharp and clear. The rock is a
> dark brown color, and its moss is a vibrant green. The ground around the
> tree and rock is covered in green grass. The path leading to the pier is
> made of light brown gravel. The image is well-composed, with the person and
> the pier providing a sense of scale and the forest and lake providing a
> beautiful backdrop.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-3b-instruct-2512-4bit"></a>

### ✅ mlx-community/Ministral-3-3B-Instruct-2512-4bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2629 | text_est=4 | nontext_est=2625 | gen=355 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.92s | Gen 3.44s | Total 4.69s
_Throughput:_ Prompt 2,650 TPS (2,629 tok) | Gen 187 TPS (355 tok)
_Assessment:_ ✅ B (68/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 68/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image depicts a serene and picturesque scene of a natural landscape.
> &#8203;Here's a detailed description:
>
> 1&#46; &#42;&#42;Foreground&#42;&#42;:
> &#45; There is a small patch of ground with a young, lush green plant that
> appears to be a coniferous type of shrub or tree. The plant is thriving with
> vibrant green needles.
>
> 2&#46; &#42;&#42;Middle Ground&#42;&#42;:
> &#45; A person is standing on a sandy or gravelly patch near the water's
> edge. The person is dressed in a dark jacket and blue hat, suggesting a cool
> or overcast day. They seem to be gazing out over the water, perhaps
> enjoying the tranquility.
>
> 3&#46; &#42;&#42;Water Body&#42;&#42;:
> &#45; A calm body of water occupies a significant portion of the scene. It
> looks like a pond or a small lake, bordered by a sandy or pebbly shore.
>
> 4&#46; &#42;&#42;Bridge&#42;&#42;:
> &#45; A small, rustic wooden bridge arches over the water, connecting two
> parts of the landscape. It adds a charming, quaint element to the scene.
>
> 5&#46; &#42;&#42;Surrounding Landscapes&#42;&#42;:
> &#45; The background is filled with dense, lush greenery. Various types of
> trees, including evergreens, dominate the scene, providing a rich, verdant
> backdrop.
> &#45; There are pathways winding through the greenery, suggesting that this
> area might be a park or a nature reserve, offering opportunities for walking
> and enjoying nature.
>
> 6&#46; &#42;&#42;Pathways and Grounds&#42;&#42;:
> &#45; The ground is a mix of grassy areas, sandy patches, and what appears to
> be a gravel path, enhancing the natural feel of the setting.
>
> Overall, the image captures a peaceful, scenic moment in a natural setting,
> ideal for relaxation and enjoying the beauty of nature.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llava-v16-mistral-7b-8bit"></a>

### ✅ mlx-community/llava-v1.6-mistral-7b-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2174 | text_est=4 | nontext_est=2170 | gen=74 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.88s | Gen 3.78s | Total 4.99s
_Throughput:_ Prompt 1,111 TPS (2,174 tok) | Gen 63.0 TPS (74 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The picture shows a person standing on a bridge over a body of water. The
> person is wearing a dark coat and a hat. The bridge is made of wood and is
> surrounded by grass and trees. The water is calm and the sky is overcast.
> The trees are green and the grass is lush. The person appears to be looking
> at the water.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-jqlive-kimi-vl-a3b-thinking-2506-6bit"></a>

### ✅ jqlive/Kimi-VL-A3B-Thinking-2506-6bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100% | reasoning leak
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1011 | text_est=4 | nontext_est=1007 | gen=345 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.06s | Gen 3.91s | Total 6.28s
_Throughput:_ Prompt 1,352 TPS (1,011 tok) | Gen 132 TPS (345 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ watchlist (reasoning leak)
_Review:_ B 75/100 | nontext prompt burden=100% | reasoning leak

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's describe the picture. First, observe the elements: a
> person in a blue coat standing on a wooden dock by a lake, a small pine tree
> on a mossy rock, a bridge over the lake, trees in the background, a grassy
> area, and a path. Let's structure the description.
>
> Start by setting the scene: a serene natural landscape with a lake, a person
> on a dock, greenery, trees, and a bridge. Mention the key elements: the
> person in blue, the small pine tree on mossy rock, the lake, bridge, and the
> surrounding forest. Keep it vivid and detailed.
>
> So, the description would be: "The image depicts a tranquil outdoor scene
> featuring a person in a blue coat standing on a wooden dock extending into a
> calm lake. In the foreground, a small pine tree grows from a moss-covered
> rock on a grassy area. The background includes a bridge spanning the lake,
> surrounded by dense trees and lush greenery, with a path winding through the
> landscape. The scene is serene, blending natural elements like water,
> trees, and grass with human-made structures like the dock and bridge,
> creating a peaceful, picturesque setting."◁/think▷The image depicts a serene
> outdoor scene centered around a calm lake. A person in a blue coat stands
> on a wooden dock extending into the water. In the foreground, a small pine
> tree grows from a moss - covered rock on a grassy area. A bridge spans the
> lake in the mid - ground, surrounded by dense trees and lush greenery, with
> a path winding through the landscape. The background features a mix of
> evergreen and deciduous trees, contributing to the tranquil, picturesque
> natural setting.<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)

---

<a id="model-huggingfacetb-smolvlm-instruct"></a>

### ✅ HuggingFaceTB/SmolVLM-Instruct

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1195 | text_est=4 | nontext_est=1191 | gen=420 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.60s | Gen 4.26s | Total 5.18s
_Throughput:_ Prompt 3,178 TPS (1,195 tok) | Gen 127 TPS (420 tok)
_Assessment:_ 🟡 C (65/100) | Keywords are not specific or diverse enough
_Review:_ C 65/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A serene and picturesque scene of a Japanese garden. The central focus is a
> tranquil body of water, which appears to be a lake or a pond, surrounded by
> lush greenery. The water is calm and still, reflecting the sky above. The
> shoreline is composed of a mix of gravel and wooden structures, which add a
> rustic and natural feel to the scene.
>
> In the foreground, there is a small, well-maintained lawn with a few
> scattered trees and shrubs. The grass is a vibrant green, and the trees are
> a mix of deciduous and evergreen varieties, adding to the diversity of the
> landscape. The trees are positioned in a way that creates a sense of depth
> and perspective, with some trees standing tall and others more diminutive.
>
> To the right of the image, there is a wooden structure that appears to be a
> bridge or a walkway, which crosses the water and leads to a small pavilion
> or building. The bridge is made of wood and has a simple, elegant design,
> adding to the overall aesthetic of the scene.
>
> In the background, there is a dense forest of tall, mature trees, which
> provide a sense of depth and scale to the image. The trees are predominantly
> evergreen, with some deciduous trees interspersed, adding a touch of
> variation to the forest. The trees are tall and majestic, creating a sense
> of awe and tranquility.
>
> The sky is overcast, with a mix of grey and light blue hues, which adds to
> the serene and contemplative atmosphere of the scene. The weather appears to
> be overcast, with light rain or mist, which adds to the peaceful ambiance.
>
> The image is framed by a natural border of rocks and stones, which add to
> the sense of isolation and seclusion. The rocks are covered in moss, which
> adds a touch of organic beauty to the scene.
>
> Overall, the image captures the essence of a peaceful and serene Japanese
> garden, with its natural beauty and tranquil atmosphere. The composition is
> well-balanced, with the elements arranged in a way that creates a harmonious
> and harmonious scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-smolvlm-instruct-bf16"></a>

### ✅ mlx-community/SmolVLM-Instruct-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1195 | text_est=4 | nontext_est=1191 | gen=420 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 0.59s | Gen 4.37s | Total 5.29s
_Throughput:_ Prompt 3,187 TPS (1,195 tok) | Gen 125 TPS (420 tok)
_Assessment:_ 🟡 C (65/100) | Keywords are not specific or diverse enough
_Review:_ C 65/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A serene and picturesque scene of a Japanese garden. The central focus is a
> tranquil body of water, which appears to be a lake or a pond, surrounded by
> lush greenery. The water is calm and still, reflecting the sky above. The
> shoreline is composed of a mix of gravel and wooden structures, which add a
> rustic and natural feel to the scene.
>
> In the foreground, there is a small, well-maintained lawn with a few
> scattered trees and shrubs. The grass is a vibrant green, and the trees are
> a mix of deciduous and evergreen varieties, adding to the diversity of the
> landscape. The trees are positioned in a way that creates a sense of depth
> and perspective, with some trees standing tall and others more diminutive.
>
> To the right of the image, there is a wooden structure that appears to be a
> bridge or a walkway, which crosses the water and leads to a small pavilion
> or building. The bridge is made of wood and has a simple, elegant design,
> adding to the overall aesthetic of the scene.
>
> In the background, there is a dense forest of tall, mature trees, which
> provide a sense of depth and scale to the image. The trees are predominantly
> evergreen, with some deciduous trees interspersed, adding a touch of
> variation to the forest. The trees are tall and majestic, creating a sense
> of awe and tranquility.
>
> The sky is overcast, with a mix of grey and light blue hues, which adds to
> the serene and contemplative atmosphere of the scene. The weather appears to
> be overcast, with light rain or mist, which adds to the peaceful ambiance.
>
> The image is framed by a natural border of rocks and stones, which add to
> the sense of isolation and seclusion. The rocks are covered in moss, which
> adds a touch of organic beauty to the scene.
>
> Overall, the image captures the essence of a peaceful and serene Japanese
> garden, with its natural beauty and tranquil atmosphere. The composition is
> well-balanced, with the elements arranged in a way that creates a harmonious
> and harmonious scene.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-2506-bf16"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-2506-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100% | reasoning leak
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1011 | text_est=4 | nontext_est=1007 | gen=296 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.23s | Gen 5.11s | Total 8.67s
_Throughput:_ Prompt 1,306 TPS (1,011 tok) | Gen 80.8 TPS (296 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ watchlist (reasoning leak)
_Review:_ B 75/100 | nontext prompt burden=100% | reasoning leak

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Got it, let's describe the picture. First, observe the elements: a
> person in a blue coat standing on a wooden dock by a lake, a small pine tree
> on a mossy rock, a bridge over the lake, trees in the background, a grassy
> area, and a path. Let's structure the description.
>
> Start by setting the scene: a serene natural landscape with a lake, a person
> on a dock, greenery, trees, and a bridge. Mention the key elements. Keep it
> concise.
>
> So, the image depicts a tranquil outdoor scene featuring a person in a blue
> hooded coat standing on a wooden dock extending into a calm lake. In the
> foreground, a small pine tree grows from a moss-covered rock on a grassy
> area. The background includes a bridge spanning the lake, surrounded by
> dense trees and lush greenery, with pathways and additional vegetation
> visible. The setting appears to be a park or natural reserve with a mix of
> open grassy areas and wooded sections.◁/think▷The image depicts a serene
> outdoor scene featuring a person in a blue hooded coat standing on a wooden
> dock that extends into a calm body of water. In the foreground, a small pine
> tree grows from a moss - covered rock on a grassy area. A bridge spans the
> water in the mid - distance, surrounded by dense trees and lush greenery,
> with pathways and additional vegetation visible throughout the landscape,
> creating a tranquil park - like or natural reserve setting.<|im_assistant|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Reasoning leak (◁think▷, ◁/think▷)

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-6bit"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-6bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1029 | text_est=4 | nontext_est=1025 | gen=135 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.58s | Gen 5.22s | Total 7.14s
_Throughput:_ Prompt 1,329 TPS (1,029 tok) | Gen 35.1 TPS (135 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A medium-close-up view of a small tree that is growing out of a green rock.
> The tree is growing out of a small hole that is filled with dark soil. The
> tree is growing straight up, and along the branches there are small green
> leaves. Surrounding the tree is a grass field that is being lit up by the
> sun. Behind the tree, there is a lake that is surrounded by grass and trees.
> On the other side of the lake, there is a person standing on a wooden dock
> that is being held up by wooden beams. The person is wearing a black coat
> and is facing the lake. The sky is covered by large gray clouds.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3n-e4b-it-bf16"></a>

### ✅ mlx-community/gemma-3n-E4B-it-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=272 | text_est=4 | nontext_est=268 | gen=237 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.22s | Gen 5.68s | Total 8.23s
_Throughput:_ Prompt 871 TPS (272 tok) | Gen 48.5 TPS (237 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Captured from a slightly elevated perspective, this outdoor shot showcases a
> serene park scene on an overcast day. The foreground is dominated by a
> vibrant green grassy area, dotted with moss-covered rocks. A small, young
> pine tree with sparse green needles stands prominently on a large, mossy
> boulder, its roots nestled in dark soil.
>
> To the right, a person wearing a dark blue hooded jacket and dark trousers
> stands on a series of wooden stepping stones that extend into a calm body of
> water. The water reflects the muted grey sky and the surrounding trees. The
> stepping stones are arranged in a slightly curved path, leading towards a
> small, rocky island in the middle of the water.
>
> In the background, a gently sloping landscape is covered with a variety of
> trees, mostly deciduous with bare branches, suggesting it might be late
> autumn or early spring. Some evergreen trees provide patches of green. A
> small wooden bridge arches over a narrower section of the water, connecting
> two grassy banks. A gravel path meanders through the park, leading towards
> the trees. The overall atmosphere is tranquil and slightly melancholic due
> to the grey weather. The lighting is soft and diffused, typical of an
> overcast day.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-kimi-vl-a3b-thinking-8bit"></a>

### ✅ mlx-community/Kimi-VL-A3B-Thinking-8bit

_Verdict:_ token_cap | user=recommended
_Why:_ hit token cap (500) | nontext prompt burden=100% | reasoning leak
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1011 | text_est=4 | nontext_est=1007 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.10s | Gen 5.71s | Total 8.14s
_Throughput:_ Prompt 1,066 TPS (1,011 tok) | Gen 120 TPS (500 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ watchlist (reasoning leak)
_Review:_ B 75/100 | hit token cap (500) | nontext prompt burden=100% |
          reasoning leak

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> ◁think▷Okay, let's see. The user asked me to describe this picture. First, I
> need to analyze the image carefully. The image shows a serene lakeside
> scene. There's a person standing on a wooden dock or platform extending into
> a calm lake. The water is dark, reflecting the surroundings. The person is
> wearing a blue jacket and seems to be looking out at the water. The
> surrounding area has a lot of greenery—trees, grass, and some bushes.
> There's a small tree in the foreground with pine-like foliage. The grass is
> vibrant green, and there's a sandy or light-colored path leading towards the
> lake. In the background, there are more trees, possibly evergreens, and a
> small bridge crossing the lake. The sky is overcast, giving the scene a
> peaceful, possibly cool atmosphere. The overall mood seems tranquil and
> natural. I should mention the elements like the person, the lake, the
> vegetation, the dock, and the bridge. Also, note the colors and the
> atmosphere. Maybe mention the time of day if possible, but since the sky
> isn't clear, it's hard to tell. The person adds a human element, making the
> scene more relatable. The composition is balanced with the dock leading the
> eye towards the bridge and the trees in the distance. The lighting is soft,
> perhaps early morning or late afternoon. I should structure the description
> to highlight these elements clearly and coherently.◁/think▷The image depicts
> a tranquil lakeside scene. A person in a blue jacket stands on a wooden
> dock extending into a calm, dark-hued lake. The water reflects the
> surrounding greenery and overcast sky. Lush green grass and moss-covered
> rocks frame the foreground, with a small pine-like tree in the lower left. A
> sandy path curves toward the lake, leading to a quaint bridge spanning the
> water in the midground. Dense evergreen trees and shrubs dot the background,
> creating a serene, natural atmosphere. The soft, diffused light suggests an
> overcast day, enhancing the peaceful, contemplative mood. The composition
> balances natural elements with a solitary human presence, evoking calmness
> and connection with nature.<|im_assistant|><|im_user|>&#42; = \frac{1}{2} \left(
> \frac{1}{x^2 + y^2} \right)^{3/2} \cdot \frac{2xy \cdot 2x - (x^2 + y^2 -
> 1&#41; \cdot 2y}{(x^
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Excessive verbosity
- Reasoning leak (◁think▷, ◁/think▷)
- Likely capped by max token budget

---

<a id="model-mlx-community-internvl3-14b-8bit"></a>

### ✅ mlx-community/InternVL3-14B-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1804 | text_est=4 | nontext_est=1800 | gen=125 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.62s | Gen 5.82s | Total 7.77s
_Throughput:_ Prompt 1,316 TPS (1,804 tok) | Gen 32.3 TPS (125 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a serene park scene with a person standing on a small
> wooden platform extending into a pond. The person is wearing a dark coat and
> boots, and appears to be looking at the water. The pond is surrounded by
> lush green grass and a variety of trees, including evergreens and deciduous
> trees. In the foreground, there is a young pine tree planted in a circular
> bed with a large moss-covered rock at its base. A gravel path leads to the
> platform, and in the background, there is a wooden bridge over the pond. The
> sky is overcast, giving the scene a calm and peaceful atmosphere.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-pixtral-12b-8bit"></a>

### ✅ mlx-community/pixtral-12b-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2887 | text_est=4 | nontext_est=2883 | gen=167 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.79s | Gen 6.53s | Total 8.65s
_Throughput:_ Prompt 1,653 TPS (2,887 tok) | Gen 39.1 TPS (167 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image captures a serene scene in a park. Dominating the foreground is a
> small tree, its branches bare, standing on a patch of vibrant green grass.
> The tree is surrounded by a large moss-covered rock, adding a touch of
> ruggedness to the otherwise tranquil setting.
>
> Beyond the tree, a small body of water stretches out, reflecting the
> overcast sky above. A stone bridge arches over the water, connecting the two
> sides of the park. A solitary figure can be seen standing on the bridge,
> adding a sense of scale and life to the scene.
>
> In the distance, a variety of trees and shrubs dot the landscape, their
> green hues contrasting with the gray of the overcast sky. The overall mood
> of the image is one of tranquility and solitude, a quiet moment frozen in
> time in a park setting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-paligemma2-3b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-3b-ft-docci-448-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1029 | text_est=4 | nontext_est=1025 | gen=118 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.43s | Gen 6.87s | Total 8.63s
_Throughput:_ Prompt 2,810 TPS (1,029 tok) | Gen 19.7 TPS (118 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A medium-close-up view of a small pine tree that is growing on a grass
> field. The pine tree is growing out of a small black hole that is filled
> with soil. Surrounding the small tree are small green leaves. On the right
> side of the small tree, there is a small wooden dock that is surrounded by
> water. On the other side of the dock, there is a grass field that is filled
> with trees. On the left side of the dock, there is a small wooden bridge
> that is being held up by wooden logs. The sky is gray and gloomy.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-4-31b-it-4bit"></a>

### ✅ mlx-community/gemma-4-31b-it-4bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=278 | text_est=4 | nontext_est=274 | gen=158 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.49s | Gen 7.21s | Total 10.03s
_Throughput:_ Prompt 386 TPS (278 tok) | Gen 26.4 TPS (158 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A vertical, full shot shows a person standing on a small wooden pier
> overlooking a pond in a park. The person, who is facing away from the
> camera, is wearing a dark blue hooded jacket and dark pants. The pier is
> made of light-colored wood and extends into the dark water of the pond. To
> the left of the pier, a small, young pine tree grows out of a moss-covered
> rock. The ground around the tree and the pier is covered in green grass and
> a light-brown gravel path. In the background, a small wooden bridge crosses
> a narrow part of the pond. Beyond the bridge, a lush green lawn leads to a
> dense forest of tall, dark green evergreen trees and leafless deciduous
> trees. The sky is a pale, overcast gray.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-mxfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-mxfp4

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2630 | text_est=4 | nontext_est=2626 | gen=311 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.30s | Gen 7.22s | Total 8.85s
_Throughput:_ Prompt 1,323 TPS (2,630 tok) | Gen 66.2 TPS (311 tok)
_Assessment:_ ✅ B (75/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image depicts a serene and picturesque landscape featuring a person
> standing by a body of water, likely a lake or pond.
>
> &#8203;Here's a detailed description:
>
> 1&#46; &#42;&#42;Foreground&#42;&#42;:
> &#45; A person is standing on a small, sandy area near the water's edge. They
> are wearing a dark, waterproof jacket and a hat, suggesting it might be
> cool or rainy.
> &#45; In front of the person, there is a small, young coniferous tree planted
> on a patch of green grass. The tree appears to be newly planted or recently
> transplanted.
>
> 2&#46; &#42;&#42;Water and Surroundings&#42;&#42;:
> &#45; The water is calm and reflects the surrounding scenery.
> &#45; To the left, there is a wooden dock or pier extending into the water,
> with a small wooden structure at the end.
> &#45; The waterway seems to be bordered by a well-maintained pathway, which
> curves gently around the lake.
>
> 3&#46; &#42;&#42;Background&#42;&#42;:
> &#45; The landscape is lush and green, with a variety of trees and plants.
> There are tall evergreen trees, deciduous trees that have lost their leaves,
> and some smaller shrubs.
> &#45; A small arched bridge can be seen in the distance, crossing over a
> narrow part of the waterway, adding to the scenic beauty.
> &#45; The overall setting appears to be a well-kept park or garden, with a
> mix of natural and landscaped elements.
>
> The overall mood of the image is tranquil and reflective, with the person
> seemingly enjoying the peacefulness of the natural surroundings.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-devstral-small-2-24b-instruct-2512-5bit"></a>

### ✅ mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

_Verdict:_ harness | user=avoid
_Why:_ Tokenizer space-marker artifacts (for example Ġ) appeared in output
       (about 112 occurrences). | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints | generic
_Stack / owner:_ owner=mlx-vlm | harness=encoding
_Token accounting:_ prompt=2098 | text_est=4 | nontext_est=2094 | gen=135 |
                    max=500 | stop=completed
_Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
               user-facing text.

_Metrics:_ Load 1.96s | Gen 7.76s | Total 10.06s
_Throughput:_ Prompt 704 TPS (2,098 tok) | Gen 31.9 TPS (135 tok)
_Assessment:_ ❌ F (0/100) | Output too short to be useful
_Review Status:_ watchlist (harness)
_Review:_ F 0/100 | Tokenizer space-marker artifacts (for example Ġ) appeared
          in output (about 112 occurrences). | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> TheĠimageĠdepictsĠaĠsereneĠoutdoorĠsceneĠfeaturingĠaĠpersonĠstandingĠonĠaĠsmall,Ġman-madeĠislandĠinĠtheĠmiddleĠofĠaĠpond.ĠTheĠindividualĠisĠdressedĠinĠaĠdarkĠjacketĠandĠpants,ĠwithĠaĠblueĠhat,ĠandĠappearsĠtoĠbeĠlookingĠoutĠoverĠtheĠwater.ĠTheĠpondĠisĠsurroundedĠbyĠlushĠgreenery,ĠincludingĠvariousĠtreesĠandĠshrubs,ĠandĠaĠsmallĠwoodenĠbridgeĠcanĠbeĠseenĠinĠtheĠbackground,ĠconnectingĠtwoĠpartsĠofĠtheĠgarden.ĠTheĠgroundĠaroundĠtheĠpondĠisĠcoveredĠwithĠgravelĠpathsĠandĠpatchesĠofĠgrass,ĠandĠthereĠisĠaĠsmall,ĠyoungĠtreeĠplantedĠinĠtheĠforeground.ĠTheĠoverallĠatmosphereĠisĠcalmĠandĠpeaceful,ĠsuggestingĠaĠwell-maintainedĠgardenĠorĠparkĠsetting.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:encoding
- Generic output (specificity: 0.00)
- token_encoding:bpe_space_leak(112)

---

<a id="model-mlx-community-ministral-3-14b-instruct-2512-nvfp4"></a>

### ✅ mlx-community/Ministral-3-14B-Instruct-2512-nvfp4

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2630 | text_est=4 | nontext_est=2626 | gen=344 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.33s | Gen 7.99s | Total 9.67s
_Throughput:_ Prompt 1,319 TPS (2,630 tok) | Gen 63.1 TPS (344 tok)
_Assessment:_ ✅ B (68/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 68/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This image depicts a serene and picturesque landscape featuring a person
> standing near a body of water, likely a lake or pond. Here's a detailed
> &#8203;description:
>
> &#45; &#42;&#42;Foreground&#42;&#42;:
> &#45; On the left, there's a small, young coniferous tree or sapling growing
> on a patch of grass. Its branches appear sparse, and its needles look
> vibrant green.
> &#45; To the right, there's a sandy or gravelly area near the water, bordered
> by wooden logs or planks, likely serving as a retaining structure.
>
> &#45; &#42;&#42;Middle Ground&#42;&#42;:
> &#45; A person, dressed in a dark jacket and boots, stands on a small wooden
> platform or pier, facing away from the camera towards the water. Their
> posture suggests contemplation or observation of something in or near the
> lake.
>
> &#45; &#42;&#42;Background&#42;&#42;:
> &#45; Across the lake, there's a beautifully landscaped garden area. It
> features well-maintained lawns, pathways, and a variety of trees, including
> conifers and deciduous trees. Some of these trees appear to be evergreens,
> while others look like they might be deciduous, possibly in early spring or
> late autumn.
> &#45; A small bridge spans part of the lake, adding to the garden's charm and
> aesthetic appeal. Surrounding trees and shrubs frame the bridge,
> contributing to a tranquil atmosphere.
>
> &#45; &#42;&#42;General Atmosphere&#42;&#42;:
> &#45; The overall mood of the image feels peaceful and reflective, with a hint
> of natural beauty. The weather appears to be overcast, adding to the calm,
> serene ambiance of the scene.
>
> Overall, this image captures a tranquil moment in a beautifully maintained
> garden by a lake, with an emphasis on natural beauty and serenity.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-microsoft-phi-35-vision-instruct"></a>

### ✅ microsoft/Phi-3.5-vision-instruct

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;|end|&gt; appeared in generated text. |
       Special control token &lt;|endoftext|&gt; appeared in generated text. |
       hit token cap (500) | nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=766 | text_est=4 | nontext_est=762 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Metrics:_ Load 0.86s | Gen 9.48s | Total 10.67s
_Throughput:_ Prompt 2,771 TPS (766 tok) | Gen 57.1 TPS (500 tok)
_Assessment:_ ✅ B (69/100) | Keywords are not specific or diverse enough
_Review Status:_ watchlist (harness)
_Review:_ B 69/100 | Special control token &lt;|end|&gt; appeared in generated
          text. | Special control token &lt;|endoftext|&gt; appeared in
          generated text. | hit token cap (500) | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image shows a serene park setting with a person standing on a wooden
> dock, fishing by a calm lake. There are trees and a well-maintained lawn
> around the
> lake.<|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|endoftext|><|end|><|end|><|end|><|end|><|end|>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Likely capped by max token budget
- tokenizer_artifact
- token_leak:<\|end\|>
- token_leak:<\|endoftext\|>

---

<a id="model-mlx-community-pixtral-12b-bf16"></a>

### ✅ mlx-community/pixtral-12b-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2887 | text_est=4 | nontext_est=2883 | gen=166 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.51s | Gen 10.48s | Total 13.31s
_Throughput:_ Prompt 1,810 TPS (2,887 tok) | Gen 19.8 TPS (166 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image captures a serene scene in a park. Dominating the foreground is a
> small tree, its branches bare, standing on a patch of vibrant green grass.
> The tree is surrounded by a large moss-covered rock, adding a touch of
> ruggedness to the otherwise tranquil setting.
>
> Beyond the tree, a small body of water stretches out, reflecting the
> overcast sky above. A stone bridge arches over the water, connecting the two
> sides of the park. A solitary figure can be seen standing on the bridge,
> adding a sense of scale and life to the scene.
>
> In the distance, a variety of trees and shrubs dot the landscape, their
> green hues contrasting with the gray of the overcast sky. The overall mood
> of the image is one of tranquility and solitude, a quiet moment frozen in
> time in a park.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-internvl3-8b-bf16"></a>

### ✅ mlx-community/InternVL3-8B-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1804 | text_est=4 | nontext_est=1800 | gen=344 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.58s | Gen 11.44s | Total 13.35s
_Throughput:_ Prompt 2,462 TPS (1,804 tok) | Gen 33.9 TPS (344 tok)
_Assessment:_ ✅ B (75/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a serene park scene with a person standing on a wooden
> &#8203;platform extending over a body of water. Here's a detailed description:
>
> 1&#46; &#42;&#42;Foreground:&#42;&#42;
> &#45; A young pine tree is prominently featured in the foreground, planted in
> a small, circular patch of soil. The tree has a few branches with green
> needles.
> &#45; The tree is positioned next to a large, moss-covered rock, which adds a
> natural element to the scene.
>
> 2&#46; &#42;&#42;Middle Ground:&#42;&#42;
> &#45; A gravel path leads from the foreground towards the water, curving
> slightly to the left.
> &#45; The wooden platform extends from the gravel path into the water, where
> the person is standing. The platform is made of wooden planks and appears to
> be partially submerged.
> &#45; The person is wearing a dark hooded jacket, dark pants, and boots. They
> are standing with their back to the camera, looking out over the water.
>
> 3&#46; &#42;&#42;Background:&#42;&#42;
> &#45; The body of water is calm, reflecting the overcast sky.
> &#45; On the far side of the water, there is a small wooden bridge with a
> curved design, leading to a grassy area with various trees and shrubs.
> &#45; The landscape includes a mix of evergreen and deciduous trees, some of
> which are bare, suggesting it might be late fall or early spring.
> &#45; A winding path can be seen in the background, leading through the park
> and disappearing into the trees.
>
> 4&#46; &#42;&#42;Sky:&#42;&#42;
> &#45; The sky is overcast, with a uniform gray color, indicating cloudy
> weather.
>
> The overall atmosphere of the image is peaceful and contemplative, with the
> person seemingly enjoying the quiet beauty of the park.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-4bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-4bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=273 | text_est=4 | nontext_est=269 | gen=320 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.30s | Gen 11.54s | Total 14.18s
_Throughput:_ Prompt 401 TPS (273 tok) | Gen 30.8 TPS (320 tok)
_Assessment:_ 🏆 A (85/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ A 85/100 | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here's a description of the picture:
>
> &#8203;&#42;&#42;Overall Impression:&#42;&#42;
>
> The image depicts a serene and picturesque landscape, likely a park or
> garden with a Japanese-inspired design. The scene is calm and somewhat
> overcast, with a focus on a small pond or lake and a lone figure.
>
> &#8203;&#42;&#42;Foreground:&#42;&#42;
>
> &#42;   A small, well-maintained grassy area slopes down towards the water.
> &#42;   A young pine tree with a distinctive shape is prominently featured, with
> a large moss-covered rock at its base.
> &#42;   A stone path leads to a small wooden bridge or dock extending into the
> water.
>
> &#8203;&#42;&#42;Midground:&#42;&#42;
>
> &#42;   A calm pond or lake dominates the midground. The water is dark and
> reflective.
> &#42;   A person is standing on the stone path near the bridge, appearing to be
> looking out over the water. They are wearing a dark hooded jacket.
> &#42;   The pond is surrounded by lush greenery, including various shrubs and
> trees.
> &#42;   A small wooden bridge is visible, connecting different parts of the
> garden.
>
> &#8203;&#42;&#42;Background:&#42;&#42;
>
> &#42;   A dense forest of evergreen trees forms the backdrop of the scene.
> &#42;   The sky is overcast, creating a soft, diffused light.
> &#42;   The overall color palette is dominated by greens, browns, and grays,
> with a sense of tranquility and natural beauty.
>
> &#8203;&#42;&#42;Overall Atmosphere:&#42;&#42;
>
> The image evokes a sense of peace, solitude, and natural beauty. The
> Japanese garden design elements, combined with the calm water and overcast
> sky, create a relaxing and contemplative atmosphere.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-flash-mxfp4"></a>

### ✅ mlx-community/GLM-4.6V-Flash-mxfp4

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;/think&gt; appeared in generated text. | hit
       token cap (500) | nontext prompt burden=100% | reasoning leak
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=6091 | text_est=4 | nontext_est=6087 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Metrics:_ Load 1.28s | Gen 12.99s | Total 14.60s
_Throughput:_ Prompt 1,075 TPS (6,091 tok) | Gen 72.9 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (harness, reasoning leak)
_Review:_ B 80/100 | Special control token &lt;/think&gt; appeared in
          generated text. | hit token cap (500) | nontext prompt burden=100% |
          reasoning leak

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's describe this picture in detail. First, the scene is a
> serene park or garden with a lake or pond.
>
> In the foreground, there's a large, moss-covered rock. Next to it, a small
> pine tree with green needles is planted in a bed with dark soil. The grass
> around is lush and green, with a gravel path curving through the scene.
>
> The middle ground features a calm body of water, likely a lake or pond. On
> the right side of the water, a person stands on a small, circular platform
> made of wooden logs, which act as a border around a small water feature or
> pond. The person is wearing a dark blue hooded jacket, dark pants, and brown
> boots. They appear to be looking down or at the water.
>
> In the background, beyond the lake, there's a wooden bridge arching over the
> water. The landscape beyond the lake is filled with a mix of evergreen
> trees (like tall pines and firs) and deciduous trees that are either bare or
> have sparse foliage, suggesting a late autumn or early spring setting. The
> sky above is overcast and gray, contributing to a calm, slightly somber
> atmosphere.
>
> Additional details include a wooden bench on the left side of the image,
> near the water's edge, with a small table or stand next to it. The overall
> scene is peaceful and natural, with well-maintained gardens and a tranquil
> water body.&lt;/think&gt;
> This image depicts a serene, overcast-day scene in a well-maintained park or
> garden centered around a lake or pond.
>
> &#35;## Foreground:
> &#45; A large, moss-covered boulder sits prominently, with a small pine tree
> planted in a dark soil bed next to it.
> &#45; A gravel path curves through the grass, leading toward the water.
> &#45; A wooden bench with a small table sits near the water’s edge on the left.
>
>
>
> &#35;## Middle Ground (Lake/Pond):
> &#45; The calm, dark water of the lake reflects the overcast sky.
> &#45; On the right side of the water, a person stands on a small, circular
> wooden platform (bordered by log posts) that juts into the water.
> &#45; The person is dressed in a dark blue hooded jacket, dark pants, and brown
> boots. They appear to be looking down or at the water, creating a
> contemplative mood.
>
>
> &#35;## Background:
> &#45; Beyond the lake, a wooden bridge arches over
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Excessive verbosity
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;
- Excessive markdown headers (6)
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-apriel-15-15b-thinker-6bit-mlx"></a>

### ✅ mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX

_Verdict:_ cutoff_degraded | user=avoid
_Why:_ hit token cap (500) | nontext prompt burden=100% | reasoning leak |
       degeneration=incomplete_sentence: ends with 'of'
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=2979 | text_est=4 | nontext_est=2975 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.58s | Gen 14.21s | Total 16.12s
_Throughput:_ Prompt 1,520 TPS (2,979 tok) | Gen 42.6 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Keywords are not specific or diverse enough
_Review Status:_ watchlist (cutoff, degeneration, reasoning leak)
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100% |
          reasoning leak | degeneration=incomplete_sentence: ends with 'of'

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here are my reasoning steps:
> The user asks: "Describe this picture". The picture is a landscape scene: a
> pond/lake with a small wooden bridge, a path, a child standing on a small
> wooden platform near water, a grassy area with a small pine tree, a
> moss-covered rock, a path leading to a bridge over water, trees in
> background, a hill with coniferous and deciduous trees, a small wooden dock,
> a bench, etc. The user wants a description. The content is safe. There's no
> disallowed content. So we can comply. Provide a vivid description. Possibly
> mention the weather (overcast), the colors, the composition, the
> perspective. The user didn't ask for anything else. So we can comply.
>
> We should produce a thorough description. The user might want a textual
> &#8203;description for accessibility. So we can describe the scene in detail:
> foreground, middle ground, background, elements, colors, lighting, mood,
> etc.
>
> We can also mention the child's clothing: blue coat, hat, boots. The path is
> a gravel path. The water is dark, reflecting the sky. The bridge is a small
> wooden footbridge with a curved shape. The pond is surrounded by grass and
> trees. There's a small pine tree in a circular planting bed with a mossy
> rock. The background includes a hill with a mix of evergreen and deciduous
> trees, some with green foliage, some bare. There's a small wooden dock on
> the left side. The sky is overcast, gray. The overall mood is tranquil,
> serene.
>
> We can also mention the composition: the viewpoint is from a higher angle,
> maybe from a slight elevation, looking down at the pond and the child. The
> path curves around the pond. The child is standing on a small wooden
> platform that extends into the water. The scene is likely a park or garden.
>
> We can also mention the presence of a bench near the water. The path is
> lined with small stones. The water is calm, with slight ripples. The overall
> atmosphere is peaceful.
>
> We can also mention the presence of a small wooden bridge crossing the pond,
> connecting two sides of the path. The bridge is arched, with railings. The
> pond is surrounded by a grassy bank. The background includes a hill with
> trees, some of which are coniferous, some deciduous. The sky is overcast,
> giving a soft light.
>
> We can also mention the presence of
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (incomplete_sentence: ends with 'of')
- Reasoning leak (here are my reasoning steps, the user asks:)
- Likely capped by max token budget

---

<a id="model-mlx-community-ernie-45-vl-28b-a3b-thinking-bf16"></a>

### ✅ mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16

_Verdict:_ cutoff_degraded | user=avoid
_Why:_ hit token cap (500) | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1293 | text_est=4 | nontext_est=1289 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 10.21s | Gen 14.46s | Total 25.02s
_Throughput:_ Prompt 239 TPS (1,293 tok) | Gen 60.5 TPS (500 tok)
_Assessment:_ 🟡 C (65/100) | Missing requested structure
_Review Status:_ watchlist (cutoff)
_Review:_ C 65/100 | hit token cap (500) | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Alright, let's get to work. The user wants a detailed description of this
> image, and I'm ready to dive in. First, I need to break down the request.
> The user provided a single image and a string of text. The text is a jumble
> of words, but it's clearly meant to be a description of the image. The user
> wants me to describe the picture, so I'll focus on the visual elements.
>
> Okay, let's start with the image itself. It's a classic, serene landscape. I
> see a tranquil pond, a wooden bridge, and a solitary figure. The mood is
> peaceful, maybe a little melancholic. The colors are muted, dominated by
> greens and browns, with a touch of grey from the overcast sky.
>
> Now, I'll analyze the text. It's a mess, but I can see some key elements.
> "Pond," "bridge," "trees," "path," "pavilion," "willow," "bamboo," "pine,"
> "moss," "stone," and "person." It's a good starting point, but the words are
> jumbled. I'll use the text as a guide, but I'll rely on my own visual
> analysis for the details.
>
> I'll structure my description logically. I'll start with the overall scene,
> then move to the foreground, the midground, the background, and finally, the
> atmosphere. I'll make sure to mention the person, as they're a key element.
>
> &#8203;Here's my plan:
>
> &#42;   &#42;&#42;Overall Scene:&#42;&#42; A tranquil, misty landscape.
> &#42;   &#42;&#42;Foreground:&#42;&#42; A path, a stone, and some moss.
> &#42;   &#42;&#42;Midground:&#42;&#42; A pond, a bridge, and some trees.
> &#42;   &#42;&#42;Background:&#42;&#42; More trees and a pavilion.
> &#42;   &#42;&#42;Atmosphere:&#42;&#42; Peaceful, quiet, and slightly melancholic.
>
> Now, let's draft the description. I'll start with the overall scene. Then,
> I'll describe the path, the stone, the pond, the bridge, the trees, the
> pavilion, and the person. I'll make sure to use descriptive language to
> capture the mood.
>
> I'll refine the language as I go. I'll make sure
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-glm-46v-flash-6bit"></a>

### ✅ mlx-community/GLM-4.6V-Flash-6bit

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;/think&gt; appeared in generated text. | hit
       token cap (500) | nontext prompt burden=100% | reasoning leak
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=6091 | text_est=4 | nontext_est=6087 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Metrics:_ Load 1.37s | Gen 14.99s | Total 16.69s
_Throughput:_ Prompt 1,129 TPS (6,091 tok) | Gen 54.8 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (harness, reasoning leak)
_Review:_ B 80/100 | Special control token &lt;/think&gt; appeared in
          generated text. | hit token cap (500) | nontext prompt burden=100% |
          reasoning leak

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's describe this picture in detail. First, the scene is a
> serene park or garden with a lake or pond.
>
> In the foreground, there's a small pine tree planted in a bed with dark
> soil, and a large, moss-covered rock next to it. A gravel path curves
> through the grassy area, leading towards the water.
>
> The middle ground features a calm body of water, with a person standing on a
> small, circular gravel platform that juts out into the water. The person is
> wearing a dark blue hooded jacket and brown boots, looking down at the
> water.
>
> In the background, there's a wooden bridge arching over the water,
> connecting the two sides of the pond. Beyond the bridge, the landscape is
> filled with a mix of evergreen trees (like tall pines and firs) and
> deciduous trees that are bare or have sparse foliage, suggesting a late
> autumn or early spring setting. The sky is overcast, giving the scene a
> calm, slightly moody atmosphere.
>
> Additional details include a wooden bench on the left side of the
> foreground, near the gravel path. The grassy areas are lush and green, with
> some patches of yellowish grass, possibly from the season. The overall scene
> conveys a peaceful, natural setting, likely a Japanese garden or a similar
> landscaped park.&lt;/think&gt;
> This image depicts a serene, overcast park or Japanese-style garden centered
> around a calm lake/pond.
>
> &#35;## Foreground:
> &#45; A &#42;&#42;small pine tree&#42;&#42; (with sparse, upright needles) is planted in a dark
> soil bed, adjacent to a &#42;&#42;large, moss-covered rock&#42;&#42;.
> &#45; A &#42;&#42;gravel path&#42;&#42; curves through lush, green grass (with subtle yellow
> patches, hinting at late autumn/early spring).
> &#45; A &#42;&#42;wooden bench&#42;&#42; sits on the left, near the path.
>
>
> &#35;## Middle Ground:
> &#45; A &#42;&#42;calm body of water&#42;&#42; (the lake/pond) dominates this area.
> &#45; A person stands on a &#42;&#42;small, circular gravel platform&#42;&#42; jutting into the
> water. The individual wears a &#42;&#42;dark blue hooded jacket&#42;&#42;, &#42;&#42;brown boots&#42;&#42;,
> and looks downward at the water.
>
>
> &#35;## Background:
> &#45; A &#42;&#42;wooden arched bridge&#42;&#42; spans the lake, connecting the two sides of the
> water.
> &#45; The landscape beyond the bridge is filled with a mix of &#42;&#42;evergreen
> trees&#42;&#42; (tall pines, firs) and &#42;&#42;
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Excessive verbosity
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;
- Excessive markdown headers (6)
- token_leak:&lt;/think&gt;

---

<a id="model-mlx-community-paligemma2-10b-ft-docci-448-bf16"></a>

### ✅ mlx-community/paligemma2-10b-ft-docci-448-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1029 | text_est=4 | nontext_est=1025 | gen=100 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.42s | Gen 20.02s | Total 22.77s
_Throughput:_ Prompt 938 TPS (1,029 tok) | Gen 5.44 TPS (100 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> A medium shot of a small tree with a round base covered in green moss. The
> tree is planted on a grass field with a small gravel area to the left of it.
> A small pond is behind the tree and has a wooden walkway that leads to a
> person standing on the edge of the pond. A small bridge is in the background
> of the pond and leads to a grass field. A forest is in the background of
> the pond and is covered in trees. The sky is gray and cloudy.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-8bit"></a>

### ✅ mlx-community/Molmo-7B-D-0924-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1207 | text_est=4 | nontext_est=1203 | gen=205 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.07s | Gen 21.60s | Total 23.01s
_Throughput:_ Prompt 71.8 TPS (1,207 tok) | Gen 52.1 TPS (205 tok)
_Assessment:_ ✅ B (80/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 80/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> in detail
>
> This picture captures a serene autumn scene at a park. In the foreground, a
> small pine tree stands on a grassy area, its branches adorned with pine
> needles. The tree is surrounded by a few scattered leaves, hinting at the
> fall season.
>
> A concrete walkway winds through the park, leading to a wooden dock that
> extends into a calm body of water. A person dressed in a blue jacket and
> brown boots is standing on the dock, gazing out at the water.
>
> The water is dark and reflective, mirroring the overcast sky above. In the
> background, a bridge spans across the water, connecting two grassy areas.
> The landscape is dotted with trees in various stages of autumnal
> transformation, some with green leaves while others are bare or turning
> brown.
>
> The sky is a light gray, creating a peaceful and somewhat melancholic
> atmosphere. The overall color palette consists of muted greens, browns,
> grays, and blues, perfectly capturing the essence of a quiet autumn day in
> the park.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-llama-32-11b-vision-instruct-8bit"></a>

### ✅ mlx-community/Llama-3.2-11B-Vision-Instruct-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=71%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=14 | text_est=4 | nontext_est=10 | gen=421 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.58s | Gen 21.65s | Total 23.57s
_Throughput:_ Prompt 10.4 TPS (14 tok) | Gen 21.2 TPS (421 tok)
_Assessment:_ ✅ B (73/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 73/100 | nontext prompt burden=71%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a serene and peaceful scene of a person standing on a
> small island in the middle of a pond, surrounded by lush greenery and a
> bridge in the background.
>
> &#42;   &#42;&#42;Pond:&#42;&#42;
> &#42;   The pond is the central feature of the image, taking up most of the
> space.
> &#42;   It is calm and peaceful, reflecting the surrounding trees and sky.
> &#42;   The water is dark and still, creating a sense of tranquility.
> &#42;   &#42;&#42;Island:&#42;&#42;
> &#42;   The island is a small, circular piece of land in the middle of the
> pond.
> &#42;   It is covered in grass and has a few rocks and trees scattered
> around.
> &#42;   The person is standing on the island, looking out over the water.
> &#42;   &#42;&#42;Bridge:&#42;&#42;
> &#42;   The bridge is a small, arched bridge that spans the width of the
> pond.
> &#42;   It is made of wood and has a simple, rustic design.
> &#42;   The bridge is surrounded by trees and bushes, adding to the natural
> beauty of the scene.
> &#42;   &#42;&#42;Trees and Bushes:&#42;&#42;
> &#42;   The trees and bushes are scattered throughout the image, adding
> depth and texture to the scene.
> &#42;   They are mostly evergreen, with a few deciduous trees scattered
> throughout.
> &#42;   The trees and bushes are well-maintained, with no signs of damage or
> neglect.
> &#42;   &#42;&#42;Person:&#42;&#42;
> &#42;   The person is standing on the island, looking out over the water.
> &#42;   They are wearing a blue coat and a hat, and have their hands in
> their pockets.
> &#42;   They appear to be lost in thought, gazing out at the peaceful
> surroundings.
>
> Overall, the image is a serene and peaceful depiction of a person enjoying
> nature. The calm and peaceful atmosphere of the scene is enhanced by the
> natural beauty of the trees, bushes, and water. The person's contemplative
> pose adds to the sense of tranquility, making the image feel like a moment
> of stillness in a busy world.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-x-reasoner-7b-8bit"></a>

### ✅ mlx-community/X-Reasoner-7B-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16299 | text_est=4 | nontext_est=16295 | gen=219 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.09s | Gen 22.03s | Total 23.45s
_Throughput:_ Prompt 933 TPS (16,299 tok) | Gen 57.1 TPS (219 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 75/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> This picture captures a serene and tranquil scene in a park or garden
> setting. The focal point is a small, young pine tree growing on a
> moss-covered rock. The tree is situated on a grassy area, and the rock
> beneath it is covered in vibrant green moss, indicating a damp environment.
> The background features a calm body of water, possibly a pond or a small
> lake, with a wooden bridge curving over it. The bridge leads to a grassy
> path that winds through the landscape, bordered by trees and shrubs. In the
> distance, there are tall evergreen trees and a few bare deciduous trees,
> suggesting the photo was taken during late autumn or early spring. The sky
> is overcast, adding to the peaceful and slightly melancholic atmosphere. A
> person dressed in dark clothing is standing near the water's edge, adding a
> sense of scale and human presence to the scene. The overall composition is
> balanced, with the tree and rock in the foreground leading the eye towards
> the bridge and the water in the background. The image conveys a sense of
> quiet reflection and natural beauty.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-gemma-3-27b-it-qat-8bit"></a>

### ✅ mlx-community/gemma-3-27b-it-qat-8bit

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=99%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=273 | text_est=4 | nontext_est=269 | gen=381 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.34s | Gen 22.90s | Total 26.58s
_Throughput:_ Prompt 327 TPS (273 tok) | Gen 17.7 TPS (381 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 80/100 | nontext prompt burden=99%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &#8203;Here's a description of the picture:
>
> &#8203;&#42;&#42;Overall Impression:&#42;&#42;
>
> The image depicts a serene and picturesque scene of a landscaped garden with
> a pond or lake. The atmosphere is calm and slightly overcast, with a muted
> color palette. It appears to be a Japanese-inspired garden or a garden with
> Japanese elements.
>
> &#8203;&#42;&#42;Key Elements:&#42;&#42;
>
> &#42;   &#42;&#42;Pond/Lake:&#42;&#42; A body of water dominates the center of the image. It's
> relatively calm, with reflections of the surrounding trees.
> &#42;   &#42;&#42;Landscaping:&#42;&#42; The garden is meticulously landscaped with a mix of
> grass, trees, shrubs, and stone features. There's a winding path leading
> around the pond.
> &#42;   &#42;&#42;Stone Bridge:&#42;&#42; A small stone bridge is visible in the distance,
> adding to the garden's charm.
> &#42;   &#42;&#42;Person:&#42;&#42; A lone person, wearing a dark hooded jacket, is standing on
> a small stone platform extending into the water. They appear to be looking
> at the pond or the surrounding scenery.
> &#42;   &#42;&#42;Foreground:&#42;&#42; In the foreground, there's a small, carefully planted
> pine tree with a moss-covered rock at its base. This adds a sense of detail
> and scale to the scene.
> &#42;   &#42;&#42;Trees:&#42;&#42; A variety of trees surround the pond, including evergreen
> conifers and deciduous trees. The trees are mostly bare or just beginning to
> bud, suggesting it's early spring.
> &#42;   &#42;&#42;Sky:&#42;&#42; The sky is overcast and gray, contributing to the muted color
> scheme and peaceful mood.
>
> &#8203;&#42;&#42;Color Palette:&#42;&#42;
>
> The colors are predominantly greens, browns, and grays. The green of the
> grass and trees is softened by the overcast light. The stone features add a
> touch of warmth.
>
> &#8203;&#42;&#42;Overall Mood:&#42;&#42;
>
> The image evokes a sense of tranquility, peace, and natural beauty. It's a
> scene that invites contemplation and relaxation.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-molmo-7b-d-0924-bf16"></a>

### ✅ mlx-community/Molmo-7B-D-0924-bf16

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=1207 | text_est=4 | nontext_est=1203 | gen=205 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.59s | Gen 27.12s | Total 29.04s
_Throughput:_ Prompt 61.7 TPS (1,207 tok) | Gen 30.4 TPS (205 tok)
_Assessment:_ ✅ B (80/100) | Keywords are not specific or diverse enough
_Review Status:_ strong candidate for first-pass review
_Review:_ B 80/100 | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> in detail
>
> This picture captures a serene autumn scene at a park. In the foreground, a
> small pine tree stands on a grassy area, its branches adorned with pine
> needles. The tree is surrounded by a few scattered leaves, hinting at the
> fall season.
>
> A concrete walkway winds through the park, leading to a wooden dock that
> extends into a calm body of water. A person dressed in a blue jacket and
> brown boots is standing on the dock, gazing out at the water.
>
> The water is dark and reflective, mirroring the overcast sky above. In the
> background, a bridge spans across the water, connecting two grassy areas.
> The landscape is dotted with trees in various stages of autumnal
> transformation, some with green leaves while others are bare or turning
> brown.
>
> The sky is a light gray, creating a peaceful and somewhat melancholic
> atmosphere. The overall color palette consists of muted greens, browns,
> grays, and blues, perfectly capturing the essence of a quiet autumn day in
> the park.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-glm-46v-nvfp4"></a>

### ✅ mlx-community/GLM-4.6V-nvfp4

_Verdict:_ harness | user=avoid
_Why:_ Special control token &lt;/think&gt; appeared in generated text. | hit
       token cap (500) | nontext prompt burden=100% | reasoning leak
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=mlx-vlm | harness=stop_token
_Token accounting:_ prompt=6091 | text_est=4 | nontext_est=6087 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.

_Metrics:_ Load 8.80s | Gen 29.00s | Total 38.14s
_Throughput:_ Prompt 398 TPS (6,091 tok) | Gen 38.1 TPS (500 tok)
_Assessment:_ ✅ B (75/100) | Keywords are not specific or diverse enough
_Review Status:_ watchlist (harness, reasoning leak)
_Review:_ B 75/100 | Special control token &lt;/think&gt; appeared in
          generated text. | hit token cap (500) | nontext prompt burden=100% |
          reasoning leak

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> &lt;think&gt;Got it, let's describe this picture step by step. First, the the
> scene is a serene park or garden with a lake or pond. In the foreground,
> there's a small pine tree planted in a bed of dark soil, surrounded by a
> large moss-covered rock. The grass around is lush and green, with some
> patches of moss. Then, there's a gravel path leading towards the water. A
> person is standing on a small, circular gravel platform by the water,
> wearing a dark blue hooded jacket, dark pants, and brown boots, looking down
> at the water.
>
> In the middle ground, the lake is calm, with a wooden bridge arching over
> it. The bridge connects two grassy areas. Behind the bridge, there are
> various trees: some tall evergreens, like pine trees, and deciduous trees
> with bare branches, suggesting it might be late autumn or early spring. The
> sky is overcast, giving a misty or rainy atmosphere.
>
> To the left, there's a wooden bench near the water, and the grassy banks are
> well-maintained. The overall mood is peaceful and slightly melancholic,
> maybe due to the overcast weather. The colors are mostly greens, browns, and
> grays, with the blue of the person's jacket standing out. The landscape is
> natural and well-tended, like a botanical garden or a park with Japanese
> garden elements, maybe? The trees in the background are a mix of conifers
> and deciduous, creating a layered look. The water reflects the gray sky,
> adding to the tranquil yet somber feel.&lt;/think&gt;
> This image depicts a tranquil, misty park scene centered around a calm lake
> or pond. In the &#42;&#42;foreground&#42;&#42;, a small pine tree grows from a bed of dark
> soil, nestled against a large, moss-covered rock. Lush, vibrant green grass
> (with patches of moss) surrounds the tree, and a gravel path leads toward
> the water. A person stands on a small, circular gravel platform at the
> water’s edge, dressed in a dark blue hooded jacket, dark pants, and brown
> boots, gazing down at the lake.
>
> The &#42;&#42;middle ground&#42;&#42; features the still lake, with a wooden arched bridge
> spanning its width, connecting two grassy banks. To the left, a simple
> wooden bench sits near the water. The banks are well-maintained, with neatly
> trimmed grass and scattered trees.
>
> In the &#42;&#42;background&#42;&#42;, a mix of evergreen (
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️HARNESS:stop_token
- Reasoning leak (&lt;think&gt;)
- Likely capped by max token budget
- Unknown tags: &lt;think&gt;
- token_leak:&lt;/think&gt;

---

<a id="model-meta-llama-llama-32-11b-vision-instruct"></a>

### ✅ meta-llama/Llama-3.2-11B-Vision-Instruct

_Verdict:_ clean | user=recommended
_Why:_ nontext prompt burden=73%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=15 | text_est=4 | nontext_est=11 | gen=276 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.15s | Gen 57.55s | Total 60.03s
_Throughput:_ Prompt 8.68 TPS (15 tok) | Gen 4.98 TPS (276 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 80/100 | nontext prompt burden=73%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The image depicts a serene and peaceful scene of a person standing on a
> small island in a pond, surrounded by lush greenery and a bridge in the
> background.
>
> &#42; A person standing on a small island in a pond:
> &#43; The person is wearing a blue coat and brown boots.
> &#43; They are standing on a small, circular platform made of logs.
> &#43; The platform is surrounded by water, with a small tree growing out
> of it.
> &#42; A bridge over the pond:
> &#43; The bridge is made of wood and has a curved shape.
> &#43; It spans the width of the pond, connecting two grassy areas.
> &#43; The bridge is surrounded by trees and bushes.
> &#42; A small tree growing out of a rock:
> &#43; The tree is a small pine tree with short branches and needles.
> &#43; It is growing out of a large rock covered in moss.
> &#43; The tree is surrounded by grass and other plants.
> &#42; A person sitting on a bench:
> &#43; The person is sitting on a wooden bench near the edge of the pond.
> &#43; They are facing away from the camera, looking out at the water.
> &#43; The bench is surrounded by trees and bushes.
>
> Overall, the image suggests a peaceful and serene atmosphere, with the
> person enjoying the natural beauty of the surroundings. The image may evoke
> feelings of calmness and tranquility.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Quality Status:_ no quality issues detected in this run

---

<a id="model-mlx-community-qwen35-35b-a3b-4bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-4bit

_Verdict:_ cutoff_degraded | user=avoid
_Why:_ hit token cap (500) | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.45s | Gen 60.04s | Total 62.83s
_Throughput:_ Prompt 299 TPS (16,290 tok) | Gen 105 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (cutoff)
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants a description of the provided image.
>
> 1&#46;  &#42;&#42;Identify the main subject:&#42;&#42; The image shows a landscape scene, likely
> a park or garden, featuring a body of water (a lake or pond), a person
> standing on a small pier, and various trees and greenery.
>
> 2&#46;  &#42;&#42;Break down the image into sections:&#42;&#42;
> &#42;   &#42;&#42;Foreground:&#42;&#42;
> &#42;   A small, young pine tree is prominent on the left side.
> &#42;   A large, moss-covered rock sits at the base of the pine tree.
> &#42;   Green grass covers the ground.
> &#42;   A gravel path curves from the left towards the water.
> &#42;   &#42;&#42;Mid-ground (The Water and Pier):&#42;&#42;
> &#42;   A calm body of water stretches across the middle.
> &#42;   A small, wooden pier or platform extends into the water. It has
> a wooden retaining wall.
> &#42;   A person is standing on the end of this pier. They are wearing a
> dark blue hooded coat and dark pants/boots. They are facing away from the
> camera, looking out at the water.
> &#42;   To the left, there's a wooden bench near the water's edge.
> &#42;   &#42;&#42;Background (The Far Shore):&#42;&#42;
> &#42;   A wooden bridge arches over a section of the water further back.
> &#42;   Lush green grass slopes upwards.
> &#42;   There are many trees: evergreens (conifers) and deciduous trees
> (some bare, some green).
> &#42;   A winding path is visible on the grassy hill.
> &#42;   A small structure (maybe a shed or small house) is nestled among
> the trees.
> &#42;   &#42;&#42;Sky/Atmosphere:&#42;&#42;
> &#42;   The sky is overcast and grey, suggesting a cloudy day.
> &#42;   The lighting is soft and diffused.
>
> 3&#46;  &#42;&#42;Draft the description:&#42;&#42;
> &#42;   Start with the overall setting: A serene, overcast day at a park or
> garden with a lake.
> &#42;   Describe the foreground details: The small pine tree and mossy rock
> are key visual anchors. Mention the gravel path.
> &#42;   Describe the mid-ground action: The person on the pier. Describe
> their clothing (blue coat, hood up). Mention the wooden bench to the left.
> &#42;   Describe the background: The
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-9b-mlx-4bit"></a>

### ✅ mlx-community/Qwen3.5-9B-MLX-4bit

_Verdict:_ token_cap | user=recommended
_Why:_ hit token cap (500) | nontext prompt burden=100% |
       degeneration=repeated_punctuation: ':**...'
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 1.39s | Gen 60.10s | Total 61.83s
_Throughput:_ Prompt 303 TPS (16,290 tok) | Gen 90.6 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (degeneration)
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100% |
          degeneration=repeated_punctuation: ':**...'

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants a description of the provided image.
>
> 1&#46;  &#42;&#42;Identify the main subject:&#42;&#42; The image shows a tranquil, somewhat
> melancholic scene in a park or garden, likely on a rainy or overcast day.
> 2&#46;  &#42;&#42;Break down the composition:&#42;&#42;
> &#42;   &#42;&#42;Foreground:&#42;&#42; A small, young pine tree stands next to a large,
> moss-covered boulder. The ground is covered in green grass.
> &#42;   &#42;&#42;Mid-ground:&#42;&#42; A gravel path leads towards a small, circular wooden
> pier or platform extending into a body of water (a pond or lake). A person
> wearing a dark blue raincoat and hood is standing on this platform, looking
> out at the water.
> &#42;   &#42;&#42;Background:&#42;&#42; Across the water, there is a grassy bank with a
> wooden bridge. Further back, there are trees—some evergreen (conifers) and
> some deciduous (bare branches, suggesting late autumn or winter). The sky is
> grey and overcast.
> 3&#46;  &#42;&#42;Identify specific details:&#42;&#42;
> &#42;   &#42;&#42;Weather:&#42;&#42; It looks like it might be raining or just rained. The
> sky is grey, the grass is a bit dull green, and the person is wearing a
> hooded coat.
> &#42;   &#42;&#42;The Person:&#42;&#42; Standing alone, facing away from the camera, looking
> towards the water. Wearing a dark blue coat, dark pants, and brown boots.
> &#42;   &#42;&#42;The Water:&#42;&#42; Dark, rippled surface, reflecting the grey sky.
> &#42;   &#42;&#42;The Bridge:&#42;&#42; A simple, arched wooden footbridge in the distance.
> &#42;   &#42;&#42;The Trees:&#42;&#42; A mix of tall evergreens (pines/firs) and bare
> deciduous trees. There are some yellowish-green bushes/trees on the left
> side.
> &#42;   &#42;&#42;The Mossy Rock:&#42;&#42; Very prominent in the bottom left corner,
> covered in bright green moss.
> 4&#46;  &#42;&#42;Synthesize the description:&#42;&#42;
> &#42;   Start with the overall atmosphere (peaceful, rainy, solitary).
> &#42;   Describe the foreground elements (pine tree, mossy rock).
> &#42;   Move to the mid-ground (gravel path, wooden pier, the person).
> &#42;   Describe the background (water, bridge, trees, hills).
> &#42;   Mention the lighting/weather (overcast, grey).
>
> &#8203;&#42;&#42;Drafting the description (internal monologue/trial phrases):&#42;&#42;
>
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Output degeneration (repeated_punctuation: ':\*\*...')
- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-35b-a3b-6bit"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-6bit

_Verdict:_ cutoff_degraded | user=avoid
_Why:_ hit token cap (500) | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.11s | Gen 60.87s | Total 64.32s
_Throughput:_ Prompt 299 TPS (16,290 tok) | Gen 88.4 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (cutoff)
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants a description of the provided image.
>
> 1&#46;  &#42;&#42;Identify the main subject:&#42;&#42; The image shows a landscape scene, likely
> a park or garden, featuring a body of water (a lake or pond), a person
> standing on a small pier, and various trees and vegetation.
> 2&#46;  &#42;&#42;Analyze the foreground:&#42;&#42;
> &#42;   There's a small, young pine tree growing out of a moss-covered rock
> or mound.
> &#42;   The ground is covered in green grass.
> &#42;   To the left, there's a gravel path.
> &#42;   A wooden bench is visible on the far left edge.
> 3&#46;  &#42;&#42;Analyze the mid-ground:&#42;&#42;
> &#42;   A person is standing on a small, gravel-covered pier or platform
> extending into the water. They are wearing a dark blue coat with a hood up
> and brown boots. They are facing away from the camera, looking towards the
> water or the far bank.
> &#42;   The water is dark and rippled, suggesting it might be raining or
> just windy.
> &#42;   There's a wooden retaining wall or edging along the pier.
> 4&#46;  &#42;&#42;Analyze the background:&#42;&#42;
> &#42;   Across the water, there's a grassy bank leading up a hill.
> &#42;   A small, arched wooden bridge crosses a narrow part of the water or
> a stream feeding into it.
> &#42;   There are many trees. Some are evergreens (conifers) which are dark
> green. Others are deciduous trees that are bare or have brownish leaves,
> suggesting late autumn or winter.
> &#42;   The sky is overcast and grey, reinforcing the gloomy/wet weather
> feel.
> 5&#46;  &#42;&#42;Synthesize the description:&#42;&#42;
> &#42;   Start with the general setting: A serene, somewhat overcast
> landscape scene.
> &#42;   Describe the foreground elements: The small pine tree on the mossy
> rock, the grass, the gravel path.
> &#42;   Describe the central figure: The person in the blue coat standing on
> the pier. Mention their posture (looking out).
> &#42;   Describe the water and immediate surroundings: The dark water, the
> wooden edging of the pier.
> &#42;   Describe the background: The bridge, the mix of evergreen and bare
> trees on the hillside, the overcast sky.
> &#42;   Add atmosphere: Mention the rainy/moody feel
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen2-vl-2b-instruct-4bit"></a>

### ✅ mlx-community/Qwen2-VL-2B-Instruct-4bit

_Verdict:_ context_budget | user=caveat
_Why:_ Output is very short relative to prompt size (0.1%), suggesting
       possible early-stop or prompt-handling issues. | At long prompt length
       (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%).
       | output/prompt=0.08% | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=caveat | preserves trusted hints
_Stack / owner:_ owner=mlx | harness=long_context
_Token accounting:_ prompt=16299 | text_est=4 | nontext_est=16295 | gen=13 |
                    max=500 | stop=completed
_Next action:_ Treat this as a prompt-budget issue first; nontext prompt
               burden is 100% and the output stays weak under that load.

_Metrics:_ Load 0.55s | Gen 61.72s | Total 62.59s
_Throughput:_ Prompt 267 TPS (16,299 tok) | Gen 207 TPS (13 tok)
_Assessment:_ ❌ F (16/100) | Output lacks detail
_Review Status:_ watchlist (harness, long context)
_Review:_ F 16/100 | Output is very short relative to prompt size (0.1%),
          suggesting possible early-stop or prompt-handling issues. | At long
          prompt length (16299 tokens), output stayed unusually short (13
          tokens; ratio 0.1%). | output/prompt=0.08% | nontext prompt
          burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> I'm sorry, but the context didn't show up.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- ⚠️REVIEW:context_budget
- ⚠️HARNESS:long_context
- output:output_ratio(0.1%)
- long_context_low_ratio(0.1%;16299->13)

---

<a id="model-mlx-community-qwen35-35b-a3b-bf16"></a>

### ✅ mlx-community/Qwen3.5-35B-A3B-bf16

_Verdict:_ token_cap | user=recommended
_Why:_ hit token cap (500) | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=recommended | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 10.63s | Gen 67.54s | Total 78.50s
_Throughput:_ Prompt 276 TPS (16,290 tok) | Gen 64.0 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ strong candidate for first-pass review
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants a description of the provided image.
>
> 1&#46;  &#42;&#42;Identify the main subject:&#42;&#42; The image shows a landscape scene, likely
> a park or garden, featuring a body of water (pond or lake), a person
> standing on a small pier, and various trees and greenery.
> 2&#46;  &#42;&#42;Analyze the foreground:&#42;&#42;
> &#42;   There's a small, young pine tree growing out of a moss-covered rock.
> &#42;   The ground is covered in green grass.
> &#42;   To the left, there's a gravel path.
> &#42;   A wooden bench is visible on the far left edge.
> 3&#46;  &#42;&#42;Analyze the mid-ground:&#42;&#42;
> &#42;   A person is standing on a small, curved wooden pier or jetty
> extending into the water. They are wearing a dark blue coat with a hood up
> and brown boots. They are facing away from the camera, looking towards the
> water/trees.
> &#42;   The water is dark and rippled, suggesting it might be raining or
> just windy.
> &#42;   There's a wooden bridge in the background spanning a narrower part
> of the water.
> 4&#46;  &#42;&#42;Analyze the background:&#42;&#42;
> &#42;   A dense forest of trees covers the hillside.
> &#42;   There are tall evergreen trees (conifers) and some deciduous trees
> that look bare or brownish, suggesting late autumn or winter.
> &#42;   The sky is overcast and grey.
> &#42;   There's a winding path visible on the grassy bank across the water.
> 5&#46;  &#42;&#42;Determine the mood/atmosphere:&#42;&#42;
> &#42;   The lighting is flat and diffuse (overcast).
> &#42;   The colors are muted greens, browns, and greys.
> &#42;   The scene feels peaceful, quiet, perhaps a bit chilly or damp.
> 6&#46;  &#42;&#42;Synthesize the description:&#42;&#42;
> &#42;   Start with the general setting (park/garden with water).
> &#42;   Describe the foreground elements (mossy rock, pine tree, gravel
> path).
> &#42;   Describe the person and the pier.
> &#42;   Describe the water and the bridge.
> &#42;   Describe the background trees and hill.
> &#42;   Mention the weather/atmosphere (overcast, rainy feel).
>
> &#8203;&#42;&#42;Drafting the description:&#42;&#42;
> The image captures a serene, somewhat overcast landscape scene,
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-4bit"></a>

### ✅ mlx-community/Qwen3.5-27B-4bit

_Verdict:_ cutoff_degraded | user=avoid
_Why:_ hit token cap (500) | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 2.12s | Gen 87.89s | Total 90.36s
_Throughput:_ Prompt 232 TPS (16,290 tok) | Gen 29.7 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (cutoff)
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants a description of the provided image.
>
> 1&#46;  &#42;&#42;Identify the main subject:&#42;&#42; The image shows a serene, landscaped
> garden, likely a Japanese garden, featuring a pond, a small pine tree in the
> foreground, and a person standing on a gravel path.
>
> 2&#46;  &#42;&#42;Analyze the foreground:&#42;&#42;
> &#42;   There's a small, young pine tree (looks like a Japanese Black Pine)
> planted in the grass.
> &#42;   Next to it is a large, moss-covered rock.
> &#42;   The ground is covered in green grass, slightly yellowish in patches,
> suggesting early spring or late autumn.
> &#42;   To the left, there's a gravel path leading towards the water.
>
> 3&#46;  &#42;&#42;Analyze the mid-ground:&#42;&#42;
> &#42;   A calm body of water (pond or lake) stretches across the middle.
> &#42;   There's a curved, light-colored gravel path or peninsula extending
> into the water, bordered by wooden posts (cribbing).
> &#42;   A person is standing on this gravel path, facing away from the
> camera, looking towards the water. They are wearing a dark blue coat with
> the hood up and brown boots.
> &#42;   To the left, there's a wooden bench near the water's edge.
> &#42;   Further back, a low, arched bridge spans a section of the water.
>
> 4&#46;  &#42;&#42;Analyze the background:&#42;&#42;
> &#42;   Beyond the pond, there's a grassy bank rising up.
> &#42;   A winding path is visible on the grass.
> &#42;   There are various trees: some evergreen conifers (tall, dark green),
> some deciduous trees (bare branches, suggesting winter/early spring), and
> some manicured shrubs.
> &#42;   The sky is overcast and grey, indicating a cloudy day.
>
> 5&#46;  &#42;&#42;Synthesize the description:&#42;&#42;
> &#42;   Start with the overall setting: A peaceful garden scene, likely
> Japanese style.
> &#42;   Describe the foreground elements: The small pine tree and mossy
> rock.
> &#42;   Describe the mid-ground: The pond, the gravel path, the person
> standing there. Mention the bridge in the distance.
> &#42;   Describe the background: The trees, the grassy hill, the overcast
> sky.
> &#42;   Add details about the atmosphere: Quiet, contemplative, slightly
> gloomy weather
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen36-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.6-27B-mxfp8

_Verdict:_ cutoff_degraded | user=avoid
_Why:_ hit token cap (500) | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.03s | Gen 100.39s | Total 103.77s
_Throughput:_ Prompt 227 TPS (16,290 tok) | Gen 17.9 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (cutoff)
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants a description of the provided image.
>
> &#42;&#42;1. Overall Impression:&#42;&#42;
> The image shows a serene, somewhat overcast landscape, likely a park or
> garden. It features a body of water (pond or small lake), a gravel path, a
> small wooden bridge, and a person standing on a gravel island or peninsula.
> The foreground has a small pine tree and a mossy rock.
>
> &#42;&#42;2. Key Elements to Describe:&#42;&#42;
> &#42;   &#42;&#42;Foreground:&#42;&#42;
> &#42;   A large, moss-covered rock on the left.
> &#42;   A small, young pine tree (conifer) planted next to it.
> &#42;   Green grass.
> &#42;   A gravel path leading from the left towards the center.
> &#42;   A wooden bench on the left edge.
> &#42;   &#42;&#42;Mid-ground:&#42;&#42;
> &#42;   A body of water (dark, calm).
> &#42;   A gravel area jutting into the water, bordered by wooden posts
> (palisade).
> &#42;   A person standing on this gravel area, wearing a dark blue coat with
> a hood up, looking towards the right.
> &#42;   A small wooden bridge crossing the water in the background.
> &#42;   &#42;&#42;Background:&#42;&#42;
> &#42;   A grassy hill rising up.
> &#42;   A mix of evergreen trees (conifers) and deciduous trees (some bare,
> suggesting late autumn or winter).
> &#42;   Overcast, grey sky.
>
> &#42;&#42;3. Drafting the Description (Iterative Process):&#42;&#42;
>
> &#42;   &#42;Attempt 1 (Mental Outline):&#42; There's a pond. A person in a blue coat is
> standing on a gravel patch in the water. In the front, there's a small pine
> tree and a big mossy rock. Behind the pond, there's a bridge and a hill
> with lots of trees. It looks like a cloudy day.
>
> &#42;   &#42;Attempt 2 (Adding Detail):&#42; The image shows a park scene. In the
> foreground, a large rock covered in green moss sits next to a small pine
> tree. To the left, there's a wooden bench. A gravel path leads to a small
> island or peninsula in a dark pond. A person wearing a dark blue hooded coat
> stands on this gravel area, looking out. The edge of the gravel is lined
> with wooden posts. In the background, a small wooden bridge crosses the
> water. Beyond that, a hill is covered
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<a id="model-mlx-community-qwen35-27b-mxfp8"></a>

### ✅ mlx-community/Qwen3.5-27B-mxfp8

_Verdict:_ cutoff_degraded | user=avoid
_Why:_ hit token cap (500) | nontext prompt burden=100%
_Trusted hints:_ preserves trusted hints
_Contract:_ ok
_Utility:_ user=avoid | preserves trusted hints
_Stack / owner:_ owner=model
_Token accounting:_ prompt=16290 | text_est=4 | nontext_est=16286 | gen=500 |
                    max=500 | stop=completed
_Next action:_ Treat as a model-quality limitation for this prompt and image.

_Metrics:_ Load 3.05s | Gen 102.50s | Total 105.91s
_Throughput:_ Prompt 220 TPS (16,290 tok) | Gen 18.0 TPS (500 tok)
_Assessment:_ ✅ B (80/100) | Missing requested structure
_Review Status:_ watchlist (cutoff)
_Review:_ B 80/100 | hit token cap (500) | nontext prompt burden=100%

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> The user wants a description of the provided image.
>
> 1&#46;  &#42;&#42;Identify the main subject:&#42;&#42; It's a landscape photo of a park or
> garden, likely a Japanese garden given the bridge and landscaping style.
> 2&#46;  &#42;&#42;Break down the image into planes/sections:&#42;&#42;
> &#42;   &#42;&#42;Foreground:&#42;&#42; A small pine tree, a mossy rock, green grass, and a
> gravel path.
> &#42;   &#42;&#42;Mid-ground:&#42;&#42; A body of water (pond/lake), a person standing on a
> gravel peninsula, a wooden bench on the left.
> &#42;   &#42;&#42;Background:&#42;&#42; A stone bridge, more trees (conifers and deciduous),
> a grassy hill, and a cloudy sky.
>
> 3&#46;  &#42;&#42;Drafting the description - Section by Section:&#42;&#42;
>
> &#42;   &#42;&#42;Foreground:&#42;&#42;
> &#42;   There's a small, young pine tree in the lower center. It has
> sparse branches.
> &#42;   To its left, a large, moss-covered rock.
> &#42;   The ground is covered in green grass, some of it looking a bit
> yellowish/mossy.
> &#42;   There's a light-colored gravel path curving from the left
> towards the center.
>
> &#42;   &#42;&#42;Mid-ground (The Water and Person):&#42;&#42;
> &#42;   A dark, calm body of water dominates the middle.
> &#42;   On the right side, a person is standing on a gravel spit or
> peninsula extending into the water. They are wearing a dark blue hooded coat
> and brown boots. They are facing away from the camera, looking towards the
> water or the bridge.
> &#42;   On the left edge, there's a wooden bench facing the water.
> &#42;   The edge of the gravel peninsula is lined with wooden posts
> (piling).
>
> &#42;   &#42;&#42;Background (The Bridge and Trees):&#42;&#42;
> &#42;   Across the water, there is a low, arched stone bridge (looks
> like a Japanese style bridge).
> &#42;   Behind the bridge and water, there is a grassy slope rising up.
> &#42;   The background is filled with trees. There are tall evergreen
> trees (conifers) on the right and center. On the left, there are bare
> deciduous trees, suggesting it might be late autumn, winter, or early
> spring.
> &#42;   The sky is overcast and grey.
>
> 4&#46;  &#42;&#42;Refining and Polishing:&#42;&#42;
> &#42;   Combine the
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

⚠️ _Quality Warnings:_

- Likely capped by max token budget

---

<!-- markdownlint-enable MD033 MD034 -->
