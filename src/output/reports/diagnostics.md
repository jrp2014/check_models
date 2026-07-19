<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 1 indeterminate, 5 harness issue(s), 2 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 62 attempted, 61 evaluated; 1 hard failure(s), 5 harness/integration issue(s), 2 text-sanity/semantic issue(s), 1 indeterminate, 0 preflight warning(s), 60 successful run(s).

Test image: `20260718-215539_DSC01101_DxO.jpg` (35.1 MB).

## Impact and Run Conditions

- _Models attempted:_ 62
- _Models evaluated:_ 61
- _Indeterminate:_ 1
- _Crashed:_ 1
- _Completed:_ 60
- _Prompt:_ Analyze this image for cataloguing metadata, using British
  English.  Describe visible details faithfully. If a visual detail is
  uncertain, ambiguous, partially obscured, or too small to verify, leave it
  out rather than guessing.  Use authoritative context as supplied fact, and
  treat the descriptive metadata as a draft catalog record. Retain draft
  details that are consistent with the image, correct contradictions, and add
  important visible details. Authoritative context may supply identity and
  location even when they are not visually readable.  Return exactly these
  three sections, and nothing else:  Title: - 5-10 words, concrete and
  factual; authoritative context may supply identity and location. - Output
  only the title text after the label. - Do not repeat or paraphrase these
  instructions in the title.  Description: - 1-2 factual sentences combining
  supplied authoritative context with the main visible subject, setting,
  lighting, action, and distinctive visible details. - Output only the
  description text after the label.  Keywords: - 10-18 unique comma-separated
  terms covering supplied authoritative context and clearly visible subjects,
  setting, colors, composition, and style. - Output only the keyword list
  after the label.  Rules: - Distinguish supplied authoritative facts from
  visible details; do not present contextual facts as though they were read
  from the image. - Reuse draft metadata when it is consistent with the image;
  authoritative context does not require separate visual proof. - If metadata
  and image disagree, follow the image. - Prefer omission to speculation. - Do
  not copy prompt instructions into the Title, Description, or Keywords
  fields. - Do not infer identity, location, event, brand, species, time
  period, or intent unless supplied as authoritative context or visually
  obvious. - Do not output reasoning, notes, hedging, or extra sections.
  Context: Authoritative context: - Location terms: England, Europe, UK,
  district, united kingdom - Capture date/time: 2026-07-18 22:55:39 BST
  22:55:39 - GPS: 51.511300°N, 0.083400°W - Use this factual context where it
  improves the catalogue record; do not claim that contextual facts are
  visually observable.  Draft descriptive metadata: - Existing title: The
  Fenchurch Building (The Walkie-Talkie), London, England, UK, GBR, Europe -
  Existing description: Walkie Talkie building known formally as 20 Fenchurch
  Street. - Existing keywords: Architecture, Building, Buildings, Cars, City,
  Cityscape, Commuting, Fenchurch Street, Illuminated, London, Modern, Night,
  Nightscape, Skyscraper, Street, Street signs, The Fenchurch Building (The
  Walkie-Talkie), Urban, Urban landscape, Walkie Talkie building - Treat this
  draft as fallible. Retain supported details, correct errors, and add
  important visible information.
- _Image:_ 20260718-215539_DSC01101_DxO.jpg

<!-- markdownlint-disable MD060 -->

## Indeterminate Attempts

External connectivity prevented evaluation. These attempts remain visible as
evidence but are excluded from failures and issue drafts.

| Model                              | Phase      | Status        | Evidence                            | Action                                               |
|------------------------------------|------------|---------------|-------------------------------------|------------------------------------------------------|
| mlx-community/Molmo-7B-D-0924-8bit | model_load | Network Error | [Errno 54] Connection reset by peer | Retry; no package owner or model-quality conclusion. |
<!-- markdownlint-enable MD060 -->
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

| Model                                   | Observation      | Routing                      | Confidence   | Evidence                                                                                                                                                                                                    | Controlled next action                                                                                                                         |
|-----------------------------------------|------------------|------------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Qwen/Qwen3-VL-2B-Instruct               | long_context     | mlx                          | high         | At mixed burden (16842 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City          | Inspect KV/cache behavior, memory pressure, and long-context execution.                                                                        |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | long_context     | mlx                          | high         | At mixed burden (16853 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City          | Inspect KV/cache behavior, memory pressure, and long-context execution.                                                                        |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | long_context     | mlx                          | high         | At mixed burden (16842 tokens), output became repetitive. \| hit token cap (500) \| missing sections: title, description, keywords \| missing terms: Architecture, Building, Buildings, Cars, City          | Inspect KV/cache behavior, memory pressure, and long-context execution.                                                                        |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | context_boundary | model-config / mlx-vlm / mlx | medium       | Output appears truncated to about 9 tokens. \| At mixed burden (16844 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.05% \| mixed burden=97% | Run a controlled reduced-image or lower-visual-token comparison before assigning the context-boundary behaviour to mlx, mlx-vlm, or the model. |
| mlx-community/X-Reasoner-7B-8bit        | context_boundary | model-config / mlx-vlm / mlx | medium       | Output appears truncated to about 9 tokens. \| At mixed burden (16853 tokens), output stayed unusually short (9 tokens; ratio 0.1%; weak text signal truncated). \| output/prompt=0.05% \| mixed burden=97% | Run a controlled reduced-image or lower-visual-token comparison before assigning the context-boundary behaviour to mlx, mlx-vlm, or the model. |

<details>
<summary>Complete generated output: Qwen/Qwen3-VL-2B-Instruct</summary>

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
  "model_type": "qwen3_vl",
  "processor_class": "mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor",
  "rendered_prompt_chars": 2891,
  "rendered_prompt_hash_sha256": "e4333dee0329365f7f63bf8f4d1cd27d6cdfd58e1ac00ca4043991685c9add65",
  "rendered_prompt_preview": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after ...",
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

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

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
  "rendered_prompt_chars": 2949,
  "rendered_prompt_hash_sha256": "93271e531b6d4a0f203fe3ff496980ff6c9904c102c5c62a70027c7e9ab399f7",
  "rendered_prompt_preview": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nAnalyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the ti...",
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

```text
-2026-07-18 22:55:39 BST 51.511300°N 0.083400°W 51.511300°N 0.083400°W 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22:55:39 BST 22
```

</details>

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

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
  "model_type": "qwen3_vl",
  "processor_class": "mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor",
  "rendered_prompt_chars": 2891,
  "rendered_prompt_hash_sha256": "e4333dee0329365f7f63bf8f4d1cd27d6cdfd58e1ac00ca4043991685c9add65",
  "rendered_prompt_preview": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after ...",
  "special_token_ids": [
    151645,
    151643
  ],
  "special_tokens": [
    "<|im_end|>",
    "<|endoftext|>"
  ],
  "tokenizer_class": "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer"
}
```

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

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
  "model_type": "qwen3_vl",
  "processor_class": "mlx_vlm.models.qwen3_vl.processing_qwen3_vl.Qwen3VLProcessor",
  "rendered_prompt_chars": 2899,
  "rendered_prompt_hash_sha256": "83b37b51a91c9763fd079721022c7abbd1e6cbd119b83ba71fce9e2bfbbb9b3b",
  "rendered_prompt_preview": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after ...",
  "special_token_ids": [
    151645,
    151643
  ],
  "special_tokens": [
    "<|im_end|>",
    "<|endoftext|>"
  ],
  "tokenizer_class": "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer"
}
```

```text
 1000000
```

</details>

<details>
<summary>Complete generated output: mlx-community/X-Reasoner-7B-8bit</summary>

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
  "model_type": "qwen2_5_vl",
  "processor_class": "mlx_vlm.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor",
  "rendered_prompt_chars": 2949,
  "rendered_prompt_hash_sha256": "1a7723e6a7d2f57441f0726b8b2c4a1aed0f4708cc5fa164fb5d64ea7d38e6f5",
  "rendered_prompt_preview": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply ...",
  "special_token_ids": [
    151645,
    151643
  ],
  "special_tokens": [
    "<|im_end|>",
    "<|endoftext|>"
  ],
  "tokenizer_class": "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer"
}
```

```text
<|im_start|>
 addCriterion("image:caption")
```

</details>
<!-- markdownlint-enable MD060 MD033 -->
<!-- markdownlint-disable MD060 -->

## Model/config observations

These signals are retained for model/configuration follow-up and are not
presented as confirmed mlx-vlm issues.

| Owner                          | Models                                           | Observation                                 | Evidence                                                                                                                                                                                | Routing   |
|--------------------------------|--------------------------------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| model configuration/repository | mlx-community/Step-3.7-Flash-oQ2e                | Processor config is missing image processor |                                                                                                                                                                                         | confirmed |
| model                          | mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX    | Generated text is mixed-script token-soup   | Here are my reasoning steps:<br>We need to produce a catalog record with three sections: Title, Description, Keywords. The image is a nighttime cityscape of a tall skyscraper, like... | probable  |
| model                          | mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16 | Generated text is mixed-script token-soup   | Let's tackle this step by step. First, the title needs to be 5-10 words, concrete and factual. The authoritative context gives location as London, England, UK, and the building...     | probable  |
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
| check_models provenance    | source-tree; revision 88b6a8195c11; ~/Documents/AI/mlx/check_models                                                                             |
<!-- markdownlint-enable MD060 -->
