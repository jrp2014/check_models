<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 0 indeterminate, 5 harness issue(s), 1 text-sanity issue(s) (mlx-vlm 0.6.6)

**Run summary:** 62 attempted, 62 evaluated; 1 hard failure(s), 5 harness/integration issue(s), 1 text-sanity/semantic issue(s), 0 indeterminate, 0 preflight warning(s), 61 successful run(s).

Test image: `20260718-165706_DSC01017.jpg` (33.6 MB).

## Impact and Run Conditions

- _Models attempted:_ 62
- _Models evaluated:_ 62
- _Indeterminate:_ 0
- _Crashed:_ 1
- _Completed:_ 61
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
  Context: Authoritative context: - Location terms: Adobe Stock, Any Vision,
  England, Europe, UK - Capture date/time: 2026-07-18 17:57:06 BST 17:57:06 -
  GPS: 50.817441°N, 0.134547°W - Use this factual context where it improves
  the catalogue record; do not claim that contextual facts are visually
  observable.  Draft descriptive metadata: - Existing title: Lifeboat Station,
  Southend-on-Sea, England, UK, GBR, Europe - Existing description: A
  lifeboats station with a ferris wheel in the background. - Existing
  keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel,
  Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky,
  Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement
  ride, antenna - Treat this draft as fallible. Retain supported details,
  correct errors, and add important visible information.
- _Image:_ 20260718-165706_DSC01017.jpg

<!-- markdownlint-disable MD060 -->

## Run Conditions and Provenance

| Condition         | Value                                                                                                                                         |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Source image      | 20260718-165706_DSC01017.jpg; 9984 x 5616; 56.07 MP; 35184156 bytes; SHA-256 f4650e8f8fe8fc2d9926734489c5911b58364e9eba9182f75f81db6a3d1ee6c0 |
| Prompt SHA-256    | 5913c0b84f12193b26a9cfa159eaa53a703f3171005784d026be5f43ef698b46                                                                              |
| trust_remote_code | true                                                                                                                                          |

Common generation settings:


```json
{
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
}
```

| Model                                                 | Requested revision   | Resolved revision                        | Processed image   | Patches     | Prompt tokens total / text / non-text   |
|-------------------------------------------------------|----------------------|------------------------------------------|-------------------|-------------|-----------------------------------------|
| mlx-community/Step-3.7-Flash-oQ2e                     | unavailable          | 3dacb46f724ac89725bcd922fb779c7ed1499fe7 | unavailable       | unavailable | unavailable                             |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | unavailable          | 6c33f49ebc0b50b75385f49ad3beddcb720d0c75 | unavailable       | unavailable | 686 / 511 / 175                         |
| mlx-community/nanoLLaVA-1.5-4bit                      | unavailable          | 5240204744963d72823e5de933c528c4aa82dfca | unavailable       | unavailable | 609 / 511 / 98                          |
| mlx-community/LFM2-VL-1.6B-8bit                       | unavailable          | 294b90e5ae2389ecb61a9427b4572975eef614fe | unavailable       | unavailable | 872 / 511 / 361                         |
| mlx-community/MiniCPM-V-4.6-8bit                      | unavailable          | 03721395f6b82cd000cc74cde28fcff8abd9a04c | unavailable       | unavailable | 1229 / 511 / 718                        |
| HuggingFaceTB/SmolVLM-Instruct                        | unavailable          | 81cd9a775a4d644f2faf4e7becff4559b46b14c7 | unavailable       | unavailable | 1832 / 511 / 1321                       |
| mlx-community/SmolVLM-Instruct-bf16                   | unavailable          | cae61cdedd0602419b43b6102dc33cd9f1e929a6 | unavailable       | unavailable | 1832 / 511 / 1321                       |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | unavailable          | 844516024a1c4400d34489b89ee067d794e432ed | unavailable       | unavailable | 733 / 511 / 222                         |
| qnguyen3/nanoLLaVA                                    | unavailable          | 13d60cec183a86755afed64da495fcc2c382ea80 | unavailable       | unavailable | 609 / 511 / 98                          |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | unavailable          | 16a710cf8afca206ff16a95a4ad6fe657f876ce1 | unavailable       | unavailable | 872 / 511 / 361                         |
| mlx-community/FastVLM-0.5B-bf16                       | unavailable          | 81ffe929046666c43de53691147b1669ba0f3a4c | unavailable       | unavailable | 613 / 511 / 102                         |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | unavailable          | 0d77464eeb233a2da68ebf9d7dc4edaac7db956d | unavailable       | unavailable | 895 / 511 / 384                         |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | unavailable          | a962dcb09eee4169c890e544c9eb938f1113fdee | unavailable       | unavailable | 2873 / 511 / 2362                       |
| mlx-community/paligemma2-10b-ft-docci-448-6bit        | unavailable          | 1485fa9b3c7adb360cd354a29a401f0d441ec728 | unavailable       | unavailable | 1641 / 511 / 1130                       |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | unavailable          | ded389e478f86d498ad9e7f47666e83b166a28f1 | unavailable       | unavailable | 891 / 511 / 380                         |
| mlx-community/diffusiongemma-26B-A4B-it-8bit          | unavailable          | 7b95e3887078ba56283c24f2578d6e5a06b9d7e8 | unavailable       | unavailable | 891 / 511 / 380                         |
| mlx-community/Phi-3.5-vision-instruct-bf16            | unavailable          | d8da684308c275a86659e2b36a9189b2f4aec8ea | unavailable       | unavailable | 1468 / 511 / 957                        |
| microsoft/Phi-3.5-vision-instruct                     | unavailable          | 12b77fb40b63a2c73c68243d3f767aab688a1b2a | unavailable       | unavailable | 1468 / 511 / 957                        |
| mlx-community/InternVL3-8B-bf16                       | unavailable          | e0df3dd79263467173214b67ef6d6a0cc5a475fd | unavailable       | unavailable | 2904 / 511 / 2393                       |
| mlx-community/gemma-3n-E2B-4bit                       | unavailable          | ec68dc186276e20e4bed30b96a2b5c667e0a81e3 | unavailable       | unavailable | 877 / 511 / 366                         |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | unavailable          | 28777b889d841a86369c736175cb77258c8134b2 | unavailable       | unavailable | 2874 / 511 / 2363                       |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | unavailable          | 7c992876448fc5426841a85c6eb951b056fe38d2 | unavailable       | unavailable | 2874 / 511 / 2363                       |
| mlx-community/paligemma2-10b-ft-docci-448-bf16        | unavailable          | 7c412694b919432784c730b62fadafb1c2e15d0d | unavailable       | unavailable | 1641 / 511 / 1130                       |
| mlx-community/pixtral-12b-8bit                        | unavailable          | 79e24b66302d4ac5e75491767f8914b2aa6ac1f8 | unavailable       | unavailable | 2945 / 511 / 2434                       |
| mlx-community/gemma-4-31b-it-4bit                     | unavailable          | 696d436c404745a59f30e4939a658162b0a9e57f | unavailable       | unavailable | 895 / 511 / 384                         |
| mlx-community/Idefics3-8B-Llama3-bf16                 | unavailable          | 8c2a30c48864f3251701b7bde40f601d25535098 | unavailable       | unavailable | 2891 / 511 / 2380                       |
| mlx-community/InternVL3-14B-8bit                      | unavailable          | 50efc568c7dfd1b91569365f1e6eb65e752f4125 | unavailable       | unavailable | 2904 / 511 / 2393                       |
| mlx-community/llava-v1.6-mistral-7b-8bit              | unavailable          | b8df5f329d95a7abe6429ed46093f9b84e8e6396 | unavailable       | unavailable | 2650 / 511 / 2139                       |
| mlx-community/gemma-3-27b-it-qat-4bit                 | unavailable          | fc4e000f32af1b7b6779294e490a7d2a80bac611 | unavailable       | unavailable | 886 / 511 / 375                         |
| mlx-community/GLM-4.6V-Flash-mxfp4                    | unavailable          | 773591fa7388b5f0db2f5ec11ed9dc3a23779f1b | unavailable       | unavailable | 6617 / 511 / 6106                       |
| mlx-community/gemma-3n-E4B-it-bf16                    | unavailable          | d9c02d0b2fa8cf26c1cb5dd9e756db59cdbe8a4a | unavailable       | unavailable | 885 / 511 / 374                         |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | unavailable          | 0a970d20ad7da57b0e2accc35c5b8628f5d02063 | unavailable       | unavailable | 2341 / 511 / 1830                       |
| mlx-community/GLM-4.6V-Flash-6bit                     | unavailable          | df9464782d3452e0dfd86afe0984f1c9eca75ca1 | unavailable       | unavailable | 6617 / 511 / 6106                       |
| mlx-community/pixtral-12b-bf16                        | unavailable          | 378cf65efd79647fc324a079c141fd92cb29916e | unavailable       | unavailable | 2945 / 511 / 2434                       |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit      | unavailable          | 8451adc50203b50b8f4199e75e753fb9c06e2af6 | unavailable       | unavailable | 580 / 511 / 69                          |
| jqlive/Kimi-VL-A3B-Thinking-2506-6bit                 | unavailable          | 4b3b11ce0874c36a99e13e17e355049042f8620a | unavailable       | unavailable | 1604 / 511 / 1093                       |
| mlx-community/Kimi-VL-A3B-Thinking-8bit               | unavailable          | 85daf3dc2490c0f824143338f08ba45f475c9ce4 | unavailable       | unavailable | 1604 / 511 / 1093                       |
| mlx-community/gemma-3-27b-it-qat-8bit                 | unavailable          | c408904bc9a0b7e996173d3cbe1bfe30e04e678d | unavailable       | unavailable | 886 / 511 / 375                         |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16      | unavailable          | 32dae5c38006e20ac158bc94cd1d5967d19b2652 | unavailable       | unavailable | 1920 / 511 / 1409                       |
| mlx-community/paligemma2-3b-pt-896-4bit               | unavailable          | a26bac48c7a661dfdafe1799c90177f818e79925 | unavailable       | unavailable | 4713 / 511 / 4202                       |
| mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX         | unavailable          | 24cb8fef6052e8d6e0dd7d467cf2d3db2dec19b2 | unavailable       | unavailable | 3036 / 511 / 2525                       |
| mlx-community/GLM-4.6V-nvfp4                          | unavailable          | 2da6855d4e28a0e61c84543262074bc17ac27d6e | unavailable       | unavailable | 6617 / 511 / 6106                       |
| mlx-community/X-Reasoner-7B-8bit                      | unavailable          | 21732e74613b465bc98e9d5ec210aba5c7adbcc1 | unavailable       | unavailable | 16933 / 511 / 16422                     |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | unavailable          | c325e5ea14c215bb08fa0d668c81fa2581f9050b | unavailable       | unavailable | 16924 / 511 / 16413                     |
| Qwen/Qwen3-VL-2B-Instruct                             | unavailable          | 89644892e4d85e24eaac8bacfd4f463576704203 | unavailable       | unavailable | 16922 / 511 / 16411                     |
| meta-llama/Llama-3.2-11B-Vision-Instruct              | unavailable          | 9eb2daaa8597bf192a8b0e73f848f3a102794df5 | unavailable       | unavailable | 581 / 511 / 70                          |
| mlx-community/Molmo-7B-D-0924-bf16                    | unavailable          | d871cbdb87a49b8071003098d6dbfd2a0f5a5b84 | unavailable       | unavailable | 1788 / 511 / 1277                       |
| mlx-community/Molmo-7B-D-0924-8bit                    | unavailable          | 90a14ed7a230088904c7556fbe6d67b295c33f5f | unavailable       | unavailable | 1788 / 511 / 1277                       |
| mlx-community/Qwen3-VL-2B-Instruct-bf16               | unavailable          | c8a67a84327484ba87f5ec4f8fb927cdafd791aa | unavailable       | unavailable | 16922 / 511 / 16411                     |
| mlx-community/MolmoPoint-8B-fp16                      | unavailable          | 0a60033b4e4813fb53df4c7523857d2ec972c7d9 | unavailable       | unavailable | 3405 / 511 / 2894                       |
| mlx-community/paligemma2-3b-ft-docci-448-bf16         | unavailable          | f66333527ce75342b09d4df81873f65272ec2f30 | unavailable       | unavailable | 1641 / 511 / 1130                       |
| mlx-community/Ornith-1.0-35B-bf16                     | unavailable          | 9ef631ad2d0c4c26783d4f94d0a0de9516e41a4b | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | unavailable          | 1e20fd8d42056f870933bf98ca6211024744f7ec | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Qwen3.5-35B-A3B-6bit                    | unavailable          | b729d115bb2cfea696e390dd6bb898528c66b6e9 | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | unavailable          | 938d8919941c6e7efd3c7150eff7fe9d12afa631 | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Qwen3.5-35B-A3B-bf16                    | unavailable          | 731d09ba3597261e84c28881116558364bb8b97c | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Qwen2-VL-2B-Instruct-4bit               | unavailable          | 01af461cdb9574acc09084a0ef94e216e142b085 | unavailable       | unavailable | 16933 / 511 / 16422                     |
| mlx-community/gemma-4-31b-bf16                        | unavailable          | 19f0f1af698c51edaf1e93b3a3a5435b282de30f | unavailable       | unavailable | 883 / 511 / 372                         |
| mlx-community/Qwen3.5-27B-mxfp8                       | unavailable          | 2d6caf2325c24e7dd3074e76a6608e9facaee36f | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Qwen3.6-27B-mxfp8                       | unavailable          | 5db9fd9c38ce083048ee12217eceed7482ca7ad4 | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Qwen3.5-27B-4bit                        | unavailable          | 45797d2985a12c55e6473686e9ea91b95e959553 | unavailable       | unavailable | 16952 / 511 / 16441                     |
| mlx-community/Kimi-VL-A3B-Thinking-2506-bf16          | unavailable          | fb254434d4026bee7aa840dea1c5d59feea8fd48 | unavailable       | unavailable | 1604 / 511 / 1093                       |

| Component       | Version                     | Install type   | Source revision                          |
|-----------------|-----------------------------|----------------|------------------------------------------|
| check_models    | 0.8.8                       | source-tree    | 0f6ded12de38137594ff1ee5eee6c96ec384f530 |
| numpy           | 2.5.1                       | wheel          | unavailable                              |
| mlx             | 0.32.1.dev20260721+30a19f72 | editable       | 30a19f7234abfb4d136af58623eaac289e63f5d2 |
| mlx-metal       | unavailable                 | unknown        | unavailable                              |
| mlx-vlm         | 0.6.6                       | editable       | c9e27b08311f9c1f38690c46034b04c7598d73ee |
| mlx-lm          | 0.31.3                      | editable       | 15b522f593b7ca5fbc0cac6f7572d40859d2d8fe |
| mlx-audio       | 0.4.4                       | wheel          | unavailable                              |
| huggingface-hub | 1.24.0                      | wheel          | unavailable                              |
| transformers    | 5.14.1                      | wheel          | unavailable                              |
| tokenizers      | 0.22.2                      | wheel          | unavailable                              |
| Pillow          | 12.3.0                      | wheel          | unavailable                              |
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

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 7601.25it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[08:33:38] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

</details>
<!-- markdownlint-enable MD060 -->
## mlx-vlm / MLX Issue Matrix

Routing labels reflect evidence confidence only; a crash remains a conclusive
failed task at every confidence level.
<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                 | Affected Models                         | Confidence   | Evidence Type   | Fixed When                                          |
|----------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------|--------------|-----------------|-----------------------------------------------------|
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | harness:prompt template \| prompt=1,788 \| output/prompt=0.06% \| stop=completed                                                                                                  | 1: `mlx-community/Molmo-7B-D-0924-bf16` | confirmed    | harness         | Requested sections render without template leakage. |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16922, repetitive output \| prompt=16,922 \| output/prompt=2.95% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 3 model cluster                   | 3: `Qwen/Qwen3-VL-2B-Instruct` (+2)     | confirmed    | context_budget  | Full and reduced reruns avoid context collapse.     |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~9 \| prompt_tokens=16933, output_tokens=9, output/prompt=0.1%, weak text=truncated \| prompt=16,933 \| output/prompt=0.05% \| mixed burden=97% \| stop=completed | 1: `mlx-community/X-Reasoner-7B-8bit`   | confirmed    | context_budget  | Full and reduced reruns avoid context collapse.     |
<!-- markdownlint-enable MD060 -->
---

## Issue 1: model repo first; mlx-vlm if template handling disagrees — Prompt-template / image-placeholder mismatch


### Expected and Actual Behaviour

- _Affected models:_ mlx-community/Molmo-7B-D-0924-bf16
- _Expected:_ Affected reruns produce the requested sections without
  empty/filler output, template leakage, or image-placeholder mismatch
  symptoms.
- _Actual:_ Prompt/template output shape mismatch
- _Filing target:_ model repo first; mlx-vlm if template handling disagrees

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/Molmo-7B-D-0924-bf16 --image 20260718-165706_DSC01017.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

Use authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the keyword list after the label.

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: Adobe Stock, Any Vision, England, Europe, UK
- Capture date/time: 2026-07-18 17:57:06 BST 17:57:06
- GPS: 50.817441°N, 0.134547°W
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
- Existing description: A lifeboats station with a ferris wheel in the background.
- Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `f4650e8f8fe8fc2d9926734489c5911b58364e9eba9182f75f81db6a3d1ee6c0`


### Expected Fix Signal

- [ ] Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.
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
<summary>Complete generated output: mlx-community/Qwen2-VL-2B-Instruct-4bit</summary>

```text
-001.jpg
- 17:57:06 BST
- 50.81744°N, 0.13454°W
- 17:57:06 BST: The image is taken at 17:57:06 BST. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:57:06 format, which is the time in the BST (British Standard Time) zone. The time is in the 17:5
```

</details>

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Instruct-bf16</summary>

```text
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ Qwen/Qwen3-VL-2B-Instruct,
  mlx-community/Qwen2-VL-2B-Instruct-4bit,
  mlx-community/Qwen3-VL-2B-Instruct-bf16
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model Qwen/Qwen3-VL-2B-Instruct --image 20260718-165706_DSC01017.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

Use authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the keyword list after the label.

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: Adobe Stock, Any Vision, England, Europe, UK
- Capture date/time: 2026-07-18 17:57:06 BST 17:57:06
- GPS: 50.817441°N, 0.134547°W
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
- Existing description: A lifeboats station with a ferris wheel in the background.
- Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `f4650e8f8fe8fc2d9926734489c5911b58364e9eba9182f75f81db6a3d1ee6c0`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 3: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

<!-- markdownlint-disable MD033 -->

### Generated Output Evidence

<details>
<summary>Complete generated output: mlx-community/X-Reasoner-7B-8bit</summary>

```text
<|im_start|>
 addCriterion("image:caption")
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/X-Reasoner-7B-8bit
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image 20260718-165706_DSC01017.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

Use authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the keyword list after the label.

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: Adobe Stock, Any Vision, England, Europe, UK
- Capture date/time: 2026-07-18 17:57:06 BST 17:57:06
- GPS: 50.817441°N, 0.134547°W
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
- Existing description: A lifeboats station with a ferris wheel in the background.
- Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `f4650e8f8fe8fc2d9926734489c5911b58364e9eba9182f75f81db6a3d1ee6c0`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

<!-- markdownlint-disable MD060 -->

## Prompt / Input Burden Evidence

| Model                                   | Burden   |   Total tokens |   Text estimate |   Non-text estimate | Source            |
|-----------------------------------------|----------|----------------|-----------------|---------------------|-------------------|
| Qwen/Qwen3-VL-2B-Instruct               | mixed    |          16922 |             511 |               16411 | estimated_nontext |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | mixed    |          16933 |             511 |               16422 | estimated_nontext |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | mixed    |          16922 |             511 |               16411 | estimated_nontext |
| mlx-community/X-Reasoner-7B-8bit        | mixed    |          16933 |             511 |               16422 | estimated_nontext |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 MD033 -->

## Other confirmed maintainer findings

These confirmed findings are routed to maintainers outside the primary mlx-vlm
/ MLX filing scope.

| Model                             | Observation   | Suspected owner   | Confidence   | Next action                                                       |
|-----------------------------------|---------------|-------------------|--------------|-------------------------------------------------------------------|
| mlx-community/Step-3.7-Flash-oQ2e | upstream_load | model-config      | medium       | File the retained reproduction evidence with the suspected owner. |

<details>
<summary>Complete generated output and evidence: mlx-community/Step-3.7-Flash-oQ2e</summary>

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

```text
=== STDERR ===
Downloading bytes:           |  0.00B
Reconstructing (incomplete total...): |          |  0.00B /  0.00B
Fetching 24 files:   0%|          | 0/24 [00:00<?, ?it/s]
Fetching 24 files: 100%|##########| 24/24 [00:00<00:00, 7601.25it/s]
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
Download complete: :           |  0.00B
Reconstruction complete: |          |  0.00B /  0.00B
[08:33:38] ERROR    Model preflight validation failed for mlx-community/Step-3.7-Flash-oQ2e
                    ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

</details>
<!-- markdownlint-enable MD060 MD033 -->
<!-- markdownlint-disable MD060 MD033 -->

## Model/config observations

These completed-run signals describe model or configuration quality and are
not presented as upstream library faults.

| Model                           | Observation                                                      | Suspected owner   | Confidence   | Next action                                          |
|---------------------------------|------------------------------------------------------------------|-------------------|--------------|------------------------------------------------------|
| mlx-community/gemma-3n-E2B-4bit | text_repetition, missing_required_sections, token_cap_truncation | unknown           | unassigned   | No maintainer issue action is indicated by this run. |

<details>
<summary>Complete generated output and evidence: mlx-community/gemma-3n-E2B-4bit</summary>

```json
{
  "eos_token": "<eos>",
  "eos_token_id": 1,
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
  "image_placeholder_count": 1,
  "model_type": "gemma3n",
  "processor_class": "mlx_vlm.models.gemma3n.processing_gemma3n.Gemma3nProcessor",
  "rendered_prompt_chars": 2756,
  "rendered_prompt_hash_sha256": "afcd2b9f3e2c77acd38d26272d45576015bc6e97851f3a714493b663e4ea075e",
  "rendered_prompt_preview": "<image_soft_token> Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase ...",
  "special_token_ids": [
    2,
    1,
    3,
    0,
    4,
    262273,
    256000,
    255999,
    262272,
    262144,
    262145
  ],
  "special_tokens": [
    "<bos>",
    "<eos>",
    "<unk>",
    "<pad>",
    "<mask>",
    "<audio_soft_token>",
    "<start_of_audio>",
    "<start_of_image>",
    "<end_of_audio>",
    "<end_of_image>",
    "<image_soft_token>"
  ],
  "tokenizer_class": "transformers.models.gemma.tokenization_gemma.GemmaTokenizer"
}
```

```text
- T-18-07-18 17:57:06 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57:06
- 17:57
```

</details>
<!-- markdownlint-enable MD060 MD033 -->
<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.6                                                                                                                                           |
| mlx                        | 0.32.1.dev20260721+30a19f72                                                                                                                     |
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
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=179f02b689129393feff8c019e656046eda1ba0b224e6fa58a83501dd9ce959b)  |
| RAM                        | 128.0 GB                                                                                                                                        |
| mlx provenance             | editable; revision 30a19f7234ab; ~/Documents/AI/mlx/mlx                                                                                         |
| mlx-metal provenance       | unknown                                                                                                                                         |
| mlx-vlm provenance         | editable; revision c9e27b08311f; ~/Documents/AI/mlx/mlx-vlm                                                                                     |
| mlx-lm provenance          | editable; revision 15b522f593b7; ~/Documents/AI/mlx/mlx-lm                                                                                      |
| mlx-audio provenance       | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| huggingface-hub provenance | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| transformers provenance    | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| tokenizers provenance      | wheel; ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                   |
| check_models provenance    | source-tree; revision 0f6ded12de38; ~/Documents/AI/mlx/check_models                                                                             |
<!-- markdownlint-enable MD060 -->
