<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 0 indeterminate, 6 harness issue(s), 2 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 62 attempted, 62 evaluated; 1 hard failure(s), 6 harness/integration issue(s), 2 text-sanity/semantic issue(s), 0 indeterminate, 0 preflight warning(s), 61 successful run(s).

Test image: `20260718-215539_DSC01101_DxO.jpg` (35.1 MB).

## Impact and Run Conditions

- _Models attempted:_ 62
- _Models evaluated:_ 62
- _Indeterminate:_ 0
- _Crashed:_ 1
- _Completed:_ 61
- _Prompt:_ Analyze this image for cataloguing metadata, using British
  English.  Use only details that are clearly and definitely visible in the
  image. If a detail is uncertain, ambiguous, partially obscured, too small to
  verify, or not directly visible, leave it out. Do not guess.  Treat the
  metadata hints below as a draft catalog record. Keep only details that are
  clearly confirmed by the image, correct anything contradicted by the image,
  and add important visible details that are definitely present.  Return
  exactly these three sections, and nothing else:  Title: - 5-10 words,
  concrete and factual, limited to clearly visible content. - Output only the
  title text after the label. - Do not repeat or paraphrase these instructions
  in the title.  Description: - 1-2 factual sentences describing the main
  visible subject, setting, lighting, action, and other distinctive visible
  details. Omit anything uncertain or inferred. - Output only the description
  text after the label.  Keywords: - 10-18 unique comma-separated terms based
  only on clearly visible subjects, setting, colors, composition, and style.
  Omit uncertain tags rather than guessing. - Output only the keyword list
  after the label.  Rules: - Include only details that are definitely visible
  in the image. - Reuse metadata terms only when they are clearly supported by
  the image. - If metadata and image disagree, follow the image. - Prefer
  omission to speculation. - Do not copy prompt instructions into the Title,
  Description, or Keywords fields. - Do not infer identity, location, event,
  brand, species, time period, or intent unless visually obvious. - Do not
  output reasoning, notes, hedging, or extra sections.  Context: Authoritative
  context: - Location terms: England, Europe, UK, district, united kingdom -
  Capture date/time: 2026-07-18 22:55:39 BST 22:55:39 - GPS: 51.511300°N,
  0.083400°W - Use this factual context where it improves the catalogue
  record; do not claim that contextual facts are visually observable.  Draft
  descriptive metadata: - Existing title: The Fenchurch Building (The
  Walkie-Talkie), London, England, UK, GBR, Europe - Existing description:
  Walkie Talkie building known formally as 20 Fenchurch Street. - Existing
  keywords: Architecture, Building, Buildings, Cars, City, Cityscape,
  Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape,
  Skyscraper, Street, Street signs, The Fenchurch Building (The Walki...,
  Urban, Urban landscape, Walkie Talkie building - Treat this draft as
  fallible. Retain supported details, correct errors, and add important
  visible information.
- _Image:_ 20260718-215539_DSC01101_DxO.jpg

<!-- markdownlint-disable MD060 -->

## Crash / Failure Matrix

| Model                             | Outcome               | Phase          | Stage           | Primary exception                                                                   | Suspected owner       |
|-----------------------------------|-----------------------|----------------|-----------------|-------------------------------------------------------------------------------------|-----------------------|
| mlx-community/Step-3.7-Flash-oQ2e | Task outcome: crashed | processor_load | Processor Error | ValueError: Loaded processor has no image_processor; expected multimodal processor. | model-config (medium) |

<details>
<summary>Crash evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

```text
ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

```text
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

</details>
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 MD033 -->

## Observations Requiring Controlled Reproduction

These completed runs are useful diagnostic observations, but they do not yet
isolate an upstream library fault and generate no issue draft.

| Model                                   | Observation      | Routing                      | Confidence   | Evidence                                                                                                                                                 | Controlled next action                                                                                                                         |
|-----------------------------------------|------------------|------------------------------|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-community/Qwen2-VL-2B-Instruct-4bit | context_boundary | model-config / mlx-vlm / mlx | medium       | At mixed burden (16833 tokens), output stayed unusually short (1 tokens; ratio 0.0%; weak text signal empty). \| output/prompt=0.01% \| mixed burden=97% | Run a controlled reduced-image or lower-visual-token comparison before assigning the context-boundary behaviour to mlx, mlx-vlm, or the model. |

<details>
<summary>Complete generated output: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

```json
{
  "eos_token": "<|im_end|>",
  "eos_token_id": 151645,
  "generate_kwargs": {
    "kv_bits": null,
    "kv_group_size": 64,
    "kv_quant_scheme": "uniform",
    "max_kv_size": null,
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "quantized_kv_start": 5000,
    "repetition_context_size": 20,
    "repetition_penalty": null,
    "temperature": 0.0,
    "top_p": 1.0
  },
  "image_placeholder_count": 0,
  "model_type": "qwen2_vl",
  "processor_class": "mlx_vlm.models.qwen2_vl.processing_qwen2_vl.Qwen2VLProcessor",
  "rendered_prompt_chars": 2731,
  "rendered_prompt_hash_sha256": "1eb7c168b1a0af25c36b0f721e42f68d02698600998ee0cd8242a0057149d24b",
  "rendered_prompt_preview": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnalyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instruction...",
  "special_token_ids": [
    151645,
    151643,
    151644,
    151646,
    151647,
    151648,
    151649,
    151650,
    151651,
    151652,
    151653,
    151654,
    151655,
    151656
  ],
  "special_tokens": [
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>"
  ],
  "tokenizer_class": "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer"
}
```

</details>
<!-- markdownlint-enable MD060 MD033 -->
## mlx-vlm / MLX Issue Matrix

Routing labels reflect evidence confidence only; a crash remains a conclusive
failed task at every confidence level.
<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                               | Evidence Snapshot                                                                                                                                               | Affected Models                       | Confidence   | Evidence Type   | Fixed When                                      |
|------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|--------------|-----------------|-------------------------------------------------|
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt=1,223 \| output/prompt=7.11% \| stop=completed                                                     | 1: `mlx-community/MiniCPM-V-4.6-8bit` | confirmed    | harness         | No leaked stop/control tokens.                  |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short | prompt_tokens=16822, repetitive output \| prompt=16,822 \| output/prompt=2.97% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 4 model cluster | 4: `Qwen/Qwen3-VL-2B-Instruct` (+3)   | confirmed    | context_budget  | Full and reduced reruns avoid context collapse. |
<!-- markdownlint-enable MD060 -->
---

## Issue 1: mlx-vlm — Stop-token leakage

<!-- markdownlint-disable MD033 -->

### Generated Output Evidence

<details>
<summary>Complete generated output: mlx-community/MiniCPM-V-4.6-8bit</summary>

```text
<think>

</think>

Title:
- Tall modern building at night in urban area

Description:
The image shows a tall modern building illuminated at night in a city setting, with street lights and surrounding structures visible. The scene is urban with illuminated windows and paved roads.

Keywords:
Architecture, Building, Buildings, City, Cityscape, Illuminated, Night, Modern, Skyscraper, Street, Urban, Urban landscape
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/MiniCPM-V-4.6-8bit
- _Expected:_ Affected reruns contain no leaked stop/control tokens and
  terminate cleanly before the configured max-token cap when the response is
  complete.
- _Actual:_ Stop/control tokens leaked into generated text
- _Filing target:_ mlx-vlm

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/MiniCPM-V-4.6-8bit --image 20260718-215539_DSC01101_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: England, Europe, UK, district, united kingdom
- Capture date/time: 2026-07-18 22:55:39 BST 22:55:39
- GPS: 51.511300°N, 0.083400°W
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe
- Existing description: Walkie Talkie building known formally as 20 Fenchurch Street.
- Existing keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walki..., Urban, Urban landscape, Walkie Talkie building
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `210a59774a1ca56ab647a9a6d360c9ae355df72584935141e21feb228ab77bc6`


### Expected Fix Signal

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 2: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

<!-- markdownlint-disable MD033 -->

### Generated Output Evidence

<details>
<summary>Complete generated output: Qwen/Qwen3-VL-2B-Instruct</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

<details>
<summary>Complete generated output: mlx-community/X-Reasoner-7B-8bit</summary>

```text
 addCriterion
<|im_start|>MMdd
<|im_start|>
 addCriterion
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
<|im_start|>
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ Qwen/Qwen3-VL-2B-Instruct,
  mlx-community/Qwen3-VL-2B-Instruct-bf16,
  mlx-community/Qwen3-VL-2B-Thinking-bf16, mlx-community/X-Reasoner-7B-8bit
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model Qwen/Qwen3-VL-2B-Instruct --image 20260718-215539_DSC01101_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: England, Europe, UK, district, united kingdom
- Capture date/time: 2026-07-18 22:55:39 BST 22:55:39
- GPS: 51.511300°N, 0.083400°W
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: The Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe
- Existing description: Walkie Talkie building known formally as 20 Fenchurch Street.
- Existing keywords: Architecture, Building, Buildings, Cars, City, Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night, Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The Walki..., Urban, Urban landscape, Walkie Talkie building
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `210a59774a1ca56ab647a9a6d360c9ae355df72584935141e21feb228ab77bc6`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

<!-- markdownlint-disable MD060 -->

## Prompt / Input Burden Evidence

| Model                                   | Burden   |   Total tokens |   Text estimate |   Non-text estimate | Source            |
|-----------------------------------------|----------|----------------|-----------------|---------------------|-------------------|
| Qwen/Qwen3-VL-2B-Instruct               | mixed    |          16822 |             493 |               16329 | estimated_nontext |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | mixed    |          16822 |             493 |               16329 | estimated_nontext |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | mixed    |          16824 |             493 |               16331 | estimated_nontext |
| mlx-community/X-Reasoner-7B-8bit        | mixed    |          16833 |             493 |               16340 | estimated_nontext |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 -->

## Model/config observations

These signals are retained for model/configuration follow-up and are not
presented as confirmed mlx-vlm issues.

| Owner                          | Models                                                                                 | Observation                                 | Evidence                                                                                                                                                                                | Routing   |
|--------------------------------|----------------------------------------------------------------------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| model configuration/repository | mlx-community/Step-3.7-Flash-oQ2e                                                      | Processor config is missing image processor |                                                                                                                                                                                         | confirmed |
| model                          | mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX, mlx-community/Kimi-VL-A3B-Thinking-8bit | Generated text is mixed-script token-soup   | Here are my reasoning steps:<br>We need to produce a catalog entry with Title, Description, Keywords sections. The image is a nighttime city street scene with a tall glass skyscrap... | probable  |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                           |
| mlx                        | 0.32.1.dev20260719+b7c3dd6d                                                                                                                     |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.4                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.24.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.5.0                                                                                                                                   |
| macOS Version              | 26.5.2                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=107989d6cb7f822e366d22e8259ea8eb2e68c15271ab6dc981da277ecf282cb0)  |
| RAM                        | 128.0 GB                                                                                                                                        |
| mlx provenance             | editable; revision b7c3dd6d27f4; ~/Documents/AI/mlx/mlx                                                                                         |
| mlx-metal provenance       | unknown                                                                                                                                         |
| mlx-vlm provenance         | editable; revision 6d5603b36902; ~/Documents/AI/mlx/mlx-vlm                                                                                     |
| mlx-lm provenance          | editable; revision 15b522f593b7; ~/Documents/AI/mlx/mlx-lm                                                                                      |
| mlx-audio provenance       | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| huggingface-hub provenance | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| transformers provenance    | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| tokenizers provenance      | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| check_models provenance    | source-tree; revision 2a7352f1e8b6; ~/Documents/AI/mlx/check_models                                                                             |
<!-- markdownlint-enable MD060 -->
