# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run Summary

- *Evaluation lane:* assisted
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 8,693 x 5,796 pixels (50.4 MP), 43.9 MB

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 33    |
| Conclusive outcomes | 33    |
| Completed           | 33    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 31    |
| observation needs reproduction | 2     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 4     |
| no concerns detected | 25    |
| concerns detected    | 4     |

Observation counts

| Observation                                      | Count |
|--------------------------------------------------|-------|
| Unrecognised model control tokens remain visible | 1     |
| Required labelled fields not detected            | 4     |
| Response appears cut off at the token limit      | 1     |
| Conversation-role control tokens remain visible  | 1     |
| Repeated keyword entries                         | 3     |

## Triage

| Model                                                                                              | Execution | Mechanical checks   | Maintainer status              | Observations                                                              |
|----------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|---------------------------------------------------------------------------|
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                            | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible                                                    |
| [mlx-community/Muse-Glimmer-30B-OptiQ-4bit](#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) | completed | unusable            | observation_needs_reproduction | labelled fields not detected; cut off at token limit; role tokens visible |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-glm-46v-nvfp4"></a>

<details>
<summary>mlx-community/GLM-4.6V-nvfp4 — usable_with_caveats — control tokens visible</summary>

### mlx-community/GLM-4.6V-nvfp4

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* concerns detected
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v_moe)
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Title word count:* 7
- *Keyword count:* 20
- *Resolved model revision:* 2da6855d4e28a0e61c84543262074bc17ac27d6e
- *Processor class:* mlx_vlm.models.glm4v_moe.processing.Glm46VMoEProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 0.001229866
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6393
- *Prompt composition:* 6,393 = 314 text/template (tokenizer-exact) + 6,079
  non-text (95%, image/audio expansion)
- *Checkpoint weights (GB):* 61.86
- *Quantization:* 4-bit, group 16, nvfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (61.86 GB vs 61.86 GB on disk)
- *Generation tokens:* 121
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Georgian Terraced Townhouses on Gay Street, Bath
Description: A street-level view looking downhill along Gay Street from The Circus, showcasing classic Georgian architecture of Bath stone terraced townhouses against a dramatic cloudy sky in Bath, Somerset, England, with parked cars and lamp posts lining the road.
Keywords: Architecture, Bath, Bath England, Bath Stone, Cars, Chimneys, Cityscape, Cloudy Sky, England, Gay Street, Georgian architecture, Hills, Lamp post, Parked Cars, Railings, Sash Windows, Somerset, Street, Street Scene, Street signs<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit"></a>

<details>
<summary>mlx-community/Muse-Glimmer-30B-OptiQ-4bit — unusable — labelled fields not detected; cut off at token limit; role tokens visible</summary>

### mlx-community/Muse-Glimmer-30B-OptiQ-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation_needs_reproduction
- *Observations:* missing_requested_sections, token_cap_truncation,
  role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type muse_glimmer)
- *Labelled fields not detected:* ["title", "description"]
- *Role-boundary tokens in output:* ["&lt;|message|&gt;"]
- *Title word count:* 0
- *Keyword count:* 2
- *Token-cap degradation evidence:* ["missing_sections"]
- *Resolved model revision:* b4a74fa6001f1eca3b23eeeb702ffad2773a218f
- *Processor class:* mlx_vlm.models.muse_glimmer.processing_muse_glimmer.MuseGlimmerProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* max_tokens
- *Post-cleanup active memory (GB):* 0.00435939
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 4411
- *Prompt composition:* 4,411 = 354 text/template (tokenizer-exact) + 4,057
  non-text (92%, image/audio expansion)
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
  sample(s); mode snapshot

#### Complete output

```text
 to=self<|message|>Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-01 16:08:37 UTC+01:00
- GPS: 51.385512°N, 2.363704°W

Descriptive hints:
- Description hint: A street-level view looking downhill along Gay Street from The Circus, showcasing the classic Georgian architecture of Bath stone terraced townhouses against a dramatic cloudy sky in Bath, Somerset, England.
- Keyword hints: Architecture, Bath, Bath England, Bath Stone, Cars, Chimneys, Cityscape, Cloudy Sky, England, Gay Street, Georgian architecture, Hills, Lamp post, Parked Cars, Railings, Sash Windows, Somerset, Street, Street Scene, Street signs

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:

Need title 5-10 words. Concrete.

Maybe "Gay Street Looking Downhill From The Circus Bath". That's 8 words? Gay(1) Street(2) Looking(3) Downhill(4) From(5) The(6) Circus(7) Bath(8). Good.

Description: 1-2 sentence factual description combining relevant context with main visible subject, setting, action, lighting, distinctive details. Must include capture date/time? It says treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. So we can mention date? Probably include in description? Might be okay to mention capture date? Usually description would not mention metadata. It says combining relevant context with main visible subject, setting, action, lighting, and distinctive details. Context includes capture date/time and GPS. Might be okay to incorporate? It says treat as authoritative facts, but do not claim they are visible. So we can say captured on... Hmm maybe not needed. Could incorporate location. GPS is 51.385512°N, 2.363704°W which is Bath. Could just say in Bath, Somerset, England.

Maybe description: Street-level view looking downhill along Gay Street from The Circus in Bath, Somerset, England, showing Georgian Bath stone terraced townhouses with sash windows, black railings and parked cars under a dramatic cloudy sky. A red post box and lamp post are on the left, with wooded hills visible at the far end. Captured on 2026-09-01.

But do not claim capture date is visible. It's okay to state capture date as fact. Might be okay.

Better keep 1-2 sentences. Let's craft one sentence: A street-level view looking downhill along Gay Street from The Circus in Bath, Somerset, England, shows Georgian Bath stone terraced townhouses with sash windows, black iron railings, parked cars and a red post box beneath a dramatic cloudy sky, with wooded hills in the distance. Second sentence maybe mention capture date/time? Could be separate.

Maybe include capture date/time in description? Might be odd. The instruction: combining relevant context with main visible subject... Relevant context includes capture date/time and GPS. Could mention location via GPS? Might be too specific.

Probably safe to mention location and date. It says treat as authoritative facts, but do not claim they are visible. So we can say "Captured on 2026-09-01 16:08:37 BST". That's a fact not visible.

Let's produce description: The view looks downhill along Gay Street from The Circus in Bath, Somerset, England, revealing a row of Georgian Bath stone terraced townhouses with sash windows, chimneys, black railings and parked cars under a dramatic cloudy sky, with green hills visible at the far end. The scene was captured on 2026-09-01 16:08:37 UTC+01:00 at 51.385512°N, 2.363704°W.

That's two sentences.

Keywords: 10-18 unique comma-separated. Need cover relevant context and visible details. Use hints but correct.

Keywords: Bath, Somerset, England, Gay Street, The Circus, Georgian architecture, Bath stone, terraced townhouses, sash windows, cloudy sky, parked cars, street scene, lamp post, railings, chimneys, hills

That's 15. Ensure comma-separated.


```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                      | Mechanical checks | Observations                 |
|--------------------------------------------|-------------------|------------------------------|
| mlx-community/Molmo2-8B-4bit               | major concerns    | labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit           | major concerns    | labelled fields not detected |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx   | major concerns    | labelled fields not detected |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16           | concerns detected | duplicate keywords           |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | duplicate keywords           |
| mlx-community/X-Reasoner-7B-8bit           | concerns detected | duplicate keywords           |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                 | Runtime identity                                           | Performance                                                                                |
|-------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | rev 0a970d20ad7d; Mistral3Processor; stop completed        | 2400 prompt / 98 generated; 29.8 tok/s; 23 GB peak; cleanup 0.000394/0.0 GB active/cache   |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | rev ded389e478f8; DiffusionGemma4Processor; stop completed | 603 prompt / 87 generated; 50.7 tok/s; 28 GB peak; cleanup 0.00831/0.0 GB active/cache     |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit      | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed     | 1644 prompt / 940 generated; 108 tok/s; 19 GB peak; cleanup 0.000607/0.0 GB active/cache   |
| mlx-community/gemma-3-27b-it-qat-4bit                 | rev fc4e000f32af; Gemma3Processor; stop completed          | 602 prompt / 155 generated; 29.3 tok/s; 17 GB peak; cleanup 0.00885/0.0 GB active/cache    |
| mlx-community/gemma-4-26b-a4b-it-4bit                 | rev 0d77464eeb23; Gemma4Processor; stop completed          | 607 prompt / 109 generated; 130 tok/s; 16 GB peak; cleanup 0.00937/0.0 GB active/cache     |
| mlx-community/gemma-4-31b-it-4bit                     | rev 696d436c4047; Gemma4Processor; stop completed          | 607 prompt / 111 generated; 26.1 tok/s; 20 GB peak; cleanup 0.0099/0.0 GB active/cache     |
| mlx-community/GLM-4.6V-Flash-4bit                     | rev bd7b20686e8c; Glm46VProcessor; stop completed          | 6393 prompt / 128 generated; 77.2 tok/s; 8.7 GB peak; cleanup 0.000919/0.0 GB active/cache |
| mlx-community/granite-4.0-3b-vision-4bit              | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed  | 1386 prompt / 70 generated; 178 tok/s; 4.6 GB peak; cleanup 0.0101/0.0 GB active/cache     |
| mlx-community/Idefics3-8B-Llama3-bf16                 | rev 8c2a30c48864; Idefics3Processor; stop completed        | 2619 prompt / 185 generated; 31.7 tok/s; 18 GB peak; cleanup 0.00149/0.0 GB active/cache   |
| mlx-community/InternVL3-8B-bf16                       | rev e0df3dd79263; InternVLChatProcessor; stop completed    | 2119 prompt / 92 generated; 33.9 tok/s; 17 GB peak; cleanup 0.0018/0.0 GB active/cache     |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit          | rev e5abbe34cbfa; KimiVLProcessor; stop max_tokens         | 1331 prompt / 1000 generated; 62.4 tok/s; 20 GB peak; cleanup 0.00246/0.0 GB active/cache  |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | rev 16a710cf8afc; Lfm2VlProcessor; stop completed          | 2120 prompt / 132 generated; 188 tok/s; 4.1 GB peak; cleanup 0.00259/0.0 GB active/cache   |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                 | rev 12c5ae493041; Lfm2VlProcessor; stop completed          | 2110 prompt / 92 generated; 212 tok/s; 4.0 GB peak; cleanup 0.00285/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4     | rev 7c992876448f; Mistral3Processor; stop completed        | 2933 prompt / 163 generated; 67.0 tok/s; 13 GB peak; cleanup 0.00311/0.0 GB active/cache   |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | rev 28777b889d84; Mistral3Processor; stop completed        | 2933 prompt / 181 generated; 59.8 tok/s; 13 GB peak; cleanup 0.00338/0.0 GB active/cache   |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit       | rev a962dcb09eee; Mistral3Processor; stop completed        | 2932 prompt / 128 generated; 180 tok/s; 7.8 GB peak; cleanup 0.00364/0.0 GB active/cache   |
| mlx-community/North-Micro-Vision-Instruct-4bit        | rev 87466363e6c5; CohereCompassProcessor; stop completed   | 4094 prompt / 104 generated; 224 tok/s; 3.9 GB peak; cleanup 0.00488/0.0 GB active/cache   |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit           | rev 5f31fcd089ce; Qwen3VLProcessor; stop completed         | 1295 prompt / 124 generated; 104 tok/s; 24 GB peak; cleanup 0.00539/0.0 GB active/cache    |
| mlx-community/pixtral-12b-8bit                        | rev 79e24b66302d; PixtralProcessor; stop completed         | 3123 prompt / 119 generated; 39.3 tok/s; 16 GB peak; cleanup 0.011/0.0 GB active/cache     |
| mlx-community/Qwen3-VL-2B-Thinking-bf16               | rev c325e5ea14c2; Qwen3VLProcessor; stop completed         | 16555 prompt / 893 generated; 88.1 tok/s; 8.4 GB peak; cleanup 0.00577/0.0 GB active/cache |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit          | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed         | 16553 prompt / 140 generated; 84.4 tok/s; 23 GB peak; cleanup 0.00608/0.0 GB active/cache  |
| mlx-community/Qwen3.5-35B-A3B-4bit                    | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed         | 16569 prompt / 104 generated; 109 tok/s; 25 GB peak; cleanup 0.00659/0.0 GB active/cache   |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | rev 938d8919941c; Qwen3VLProcessor; stop completed         | 16569 prompt / 110 generated; 91.2 tok/s; 11 GB peak; cleanup 0.00711/0.0 GB active/cache  |
| mlx-community/Qwen3.8-27B-4bit                        | rev 3e6447f082e8; Qwen3VLProcessor; stop completed         | 16569 prompt / 131 generated; 29.7 tok/s; 21 GB peak; cleanup 0.00762/0.0 GB active/cache  |
| mlx-community/Step-3.7-Flash-oQ3e                     | rev 41d17ee00e16; Step3VLProcessor; stop completed         | 3497 prompt / 110 generated; 50.2 tok/s; 92 GB peak; cleanup 0.008/0.0 GB active/cache     |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 8,693 x 5,796 pixels
- *Image size:* 43,870,091 bytes
- *Image SHA-256:* 398a0b2c7ac923e1240f8f2bacfa7abe83195ad795c047dd624d257c3df179c0

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-01 16:08:37 UTC+01:00
- GPS: 51.385512°N, 2.363704°W

Descriptive hints:
- Description hint: A street-level view looking downhill along Gay Street from The Circus, showcasing the classic Georgian architecture of Bath stone terraced townhouses against a dramatic cloudy sky in Bath, Somerset, England.
- Keyword hints: Architecture, Bath, Bath England, Bath Stone, Cars, Chimneys, Cityscape, Cloudy Sky, England, Gay Street, Georgian architecture, Hills, Lamp post, Parked Cars, Railings, Sash Windows, Somerset, Street, Street Scene, Street signs

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-37d1235e2711119c.jpg>
- *Preview dimensions:* 1,024 x 683 pixels
- *Preview size:* 117,094 bytes
- *Preview SHA-256:* 37d1235e2711119c6897ba65d85dd6602779a034d424d11f4764e78e80118aaa

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-37d1235e2711119c.jpg
printf '%s\n' '37d1235e2711119c6897ba65d85dd6602779a034d424d11f4764e78e80118aaa  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-01 16:08:37 UTC+01:00
- GPS: 51.385512°N, 2.363704°W

Descriptive hints:
- Description hint: A street-level view looking downhill along Gay Street from The Circus, showcasing the classic Georgian architecture of Bath stone terraced townhouses against a dramatic cloudy sky in Bath, Somerset, England.
- Keyword hints: Architecture, Bath, Bath England, Bath Stone, Cars, Chimneys, Cityscape, Cloudy Sky, England, Gay Street, Georgian architecture, Hills, Lamp post, Parked Cars, Railings, Sash Windows, Somerset, Street, Street Scene, Street signs

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:' --max-tokens 1000 --temperature 0.0 --revision RESOLVED_REVISION --trust-remote-code --prefill-step-size 2048
```

### Highlighted model revisions

| Model                                     | Resolved revision                        |
|-------------------------------------------|------------------------------------------|
| mlx-community/GLM-4.6V-nvfp4              | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | b4a74fa6001f1eca3b23eeeb702ffad2773a218f |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.0                                                                                                                                           |
| mlx-vlm source revision    | d2a1434a03e4c9975b0d505e7178e0cfc4082a83                                                                                                        |
| mlx                        | 0.32.3.dev20260911+dfe17bafb                                                                                                                    |
| mlx source revision        | dfe17bafb23e66fe56596df532a497ab3611d0e5                                                                                                        |
| mlx-audio                  | 0.5.3                                                                                                                                           |
| transformers               | 5.17.0                                                                                                                                          |
| tokenizers                 | 0.23.2                                                                                                                                          |
| huggingface-hub            | 1.31.0                                                                                                                                          |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 25.6.0                                                                                                                                   |
| macOS Version              | 26.6.2                                                                                                                                          |
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
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-314-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (189,439,120 bytes, sha256=3e940a8eda4eb7bc22bd545f335ba774b504258eb8817cc56b86d156a8b574df) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,488,288 bytes, sha256=893ea3d90829f538ca35c599ede0af90aff00349cd58c0cdf498fa6f27812d32)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
