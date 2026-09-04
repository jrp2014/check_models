# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

These models serve many purposes; this run probes exactly one narrow task:
producing catalogue metadata for a single photograph from the assisted-lane
prompt and whatever context it supplies — here, camera-recorded capture
context plus draft descriptive hints previously produced by a more capable
model. Results say nothing about a model's fitness for other uses.

## Run Summary

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 32    |
| Conclusive outcomes | 32    |
| Completed           | 32    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 29    |
| observation needs reproduction | 3     |

Usability counts

| Usability           | Count |
|---------------------|-------|
| usable              | 13    |
| usable with caveats | 19    |

Observation counts

| Observation                                         | Count |
|-----------------------------------------------------|-------|
| Unrecognised model control tokens remain visible    | 2     |
| Conversation-role control tokens remain visible     | 1     |
| Title or keywords do not meet requested constraints | 16    |

## Triage

| Model                                                                                                    | Execution | Usability           | Maintainer status              | Observations           |
|----------------------------------------------------------------------------------------------------------|-----------|---------------------|--------------------------------|------------------------|
| [mlx-community/GLM-4.6V-Flash-4bit](#diagnostic-mlx-community-glm-46v-flash-4bit)                        | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible |
| [mlx-community/GLM-4.6V-nvfp4](#diagnostic-mlx-community-glm-46v-nvfp4)                                  | completed | usable_with_caveats | observation_needs_reproduction | control tokens visible |
| [mlx-community/Kimi-VL-A3B-Thinking-2506-8bit](#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) | completed | usable_with_caveats | observation_needs_reproduction | role tokens visible    |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-glm-46v-flash-4bit"></a>

<details>
<summary>mlx-community/GLM-4.6V-Flash-4bit — usable_with_caveats — control tokens visible</summary>

### mlx-community/GLM-4.6V-Flash-4bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type glm4v)
- *Unexpected special tokens:* ["&lt;|begin_of_box|&gt;",
  "&lt;|end_of_box|&gt;"]
- *Resolved model revision:* bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e
- *Processor class:* mlx_vlm.models.glm4v.processing.Glm46VProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 7.074351378
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6430
- *Prompt composition:* 6,430 = 287 text/template (tokenizer-exact) + 6,143
  non-text (96%, image/audio expansion)
- *Checkpoint weights (GB):* 7.07
- *Quantization:* 4-bit, group 64, affine
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (7.07 GB vs 7.07 GB on disk)
- *Generation tokens:* 92
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Winchester City Mill by River Itchen
Description: A historic brick and stone watermill with a tiled roof and chimney sits over the River Itchen, with people walking along the riverbank pathway under a blue sky with clouds.
Keywords: Winchester City Mill, River Itchen, historic watermill, brick building, stone wall, riverbank, pedestrians, blue sky, clouds, chimney, tiled roof, Hampshire, England, National Trust.<|end_of_box|>
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
- *Post-cleanup active memory (GB):* 61.85803523
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 6430
- *Prompt composition:* 6,430 = 287 text/template (tokenizer-exact) + 6,143
  non-text (96%, image/audio expansion)
- *Checkpoint weights (GB):* 61.86
- *Quantization:* 4-bit, group 16, nvfp4
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (61.86 GB vs 61.86 GB on disk)
- *Generation tokens:* 108
- *Configured EOS token ID:* 151329
- *Configured EOS token:* &lt;|endoftext|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
<|begin_of_box|>Title: Historic Winchester City Mill by River Itchen

Description: Two girls walk along the pathway in front of the historic Winchester City Mill, a brick and stone watermill with clay tiles, situated over the River Itchen in Hampshire, England, under a blue sky with white clouds.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, watermill, brick, stone, clay tiles, chimney, blue sky, girls, pedestrians, elderly woman, National Trust, riverbank, scenery, sitting, rapids<|end_of_box|>
```

</details>

<a id="diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit"></a>

<details>
<summary>mlx-community/Kimi-VL-A3B-Thinking-2506-8bit — usable_with_caveats — role tokens visible</summary>

### mlx-community/Kimi-VL-A3B-Thinking-2506-8bit

#### Execution and provenance

- *Execution:* completed
- *Usability:* usable_with_caveats
- *Maintainer status:* observation_needs_reproduction
- *Observations:* role_boundary_token_present
- *Arch supported by installed mlx-vlm:* yes (model_type kimi_vl)
- *Thinking trace markers:* ["\u25c1think\u25b7", "\u25c1/think\u25b7"]
- *Role-boundary tokens in output:* ["&lt;|im_assistant|&gt;"]
- *Resolved model revision:* e5abbe34cbfabd829fafd0362856e5b468d19f85
- *Processor class:* mlx_vlm.models.kimi_vl.processing_kimi_vl.KimiVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Post-cleanup active memory (GB):* 17.829274892
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 1307
- *Prompt composition:* 1,307 = 294 text/template (tokenizer-exact) + 1,013
  non-text (78%, image/audio expansion)
- *Checkpoint weights (GB):* 17.83
- *Parameter count:* 3.00B (name-estimate)
- *Quantization:* 8-bit, group 64
- *Declared context length:* 131,072 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (17.83 GB vs 17.83 GB on disk)
- *Generation tokens:* 971
- *Configured EOS token ID:* 163594
- *Configured EOS token:* [EOS]
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); mode snapshot

#### Complete output

```text
◁think▷Okay, let's tackle this. First, I need to create a British-English catalogue metadata based on the image and the provided context. The capture date is 2026-08-31 15:02:20 UTC+01:00, but I shouldn't claim it's visible. The descriptive hints mention Winchester City Mill, River Itchen, Hampshire, England, people walking, elderly woman, girls, Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Stone wall, Pedestrians, Rapids, Scenery, Sitting, National Trust.

Looking at the image: there's a historic building (Winchester City Mill) with a brick and stone facade, a tiled roof, a chimney, a stone bridge over a river with rapids, people walking on the bridge, an elderly woman sitting, and a blue sky. The keywords from the hints are relevant. Need to make sure the title is 5-10 words, description 1-2 sentences, and 10-18 keywords.

Title: "Historic Winchester City Mill with River Itchen and Pedestrians" – that's 8 words. 

Description: "A historic brick and stone building, the Winchester City Mill, stands beside the River Itchen in Hampshire, England, with pedestrians walking on a stone bridge and an elderly woman sitting nearby under a blue sky." That's 2 sentences, covers the main elements.

Keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery, Sitting, National Trust. Let me count: 15 keywords. Check if all hints are covered. The hints include National Trust, which is part of the context but not in the image. Wait, the image might not show National Trust explicitly. Hmm, the context says it's a National Trust property. Should I include it? The user said to retain details supported by the image. Since the image doesn't show the National Trust sign, maybe omit. So remove National Trust. Also, check if all visible elements are covered. The image has a stone wall, clay tiles on the roof, chimney, arch (the bridge arch), blue sky, rapids in the river, sitting (elderly woman), pedestrians (two people walking), and the building itself. So keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 13. Maybe add "Stone bridge" since the bridge is visible. So adjust: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery, Stone bridge. Now 14. Maybe "Stone bridge" is better than "Arch" as it's more descriptive. Alternatively, keep Arch. Wait, the bridge has an arch, so maybe Arch is okay. Let me check the image: the bridge has an arch structure. So maybe keep Arch. Let me recount: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 14. Maybe add "National Trust" even if not visible? The context says it's a National Trust property. The user said to retain details supported by the image. Since the image doesn't show the National Trust sign, maybe omit. So final keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery. That's 13. Wait, the context mentions "National Trust" as a keyword hint. Should I include it? The user said to retain details supported by the image. Since the image doesn't show the National Trust, maybe omit. So proceed with the visible elements.◁/think▷Title: Historic Winchester City Mill with River Itchen and Pedestrians  
Description: A historic brick and stone building, the Winchester City Mill, stands beside the River Itchen in Hampshire, England, with pedestrians walking on a stone bridge and an elderly woman sitting nearby under a blue sky.  
Keywords: Winchester City Mill, River Itchen, Hampshire, England, Pedestrians, Elderly woman, Blue sky, Adobe Stock, Any Vision, Arch, Chimney, Stone wall, Clay tiles, Rapids, Scenery<|im_assistant|>
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                                 | Usability           | Observations                     |
|-------------------------------------------------------|---------------------|----------------------------------|
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                      | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | usable_with_caveats | title/keyword constraints failed |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8         | usable_with_caveats | title/keyword constraints failed |
| mlx-community/gemma-3-27b-it-qat-4bit                 | usable_with_caveats | title/keyword constraints failed |
| mlx-community/LFM2.5-VL-1.6B-bf16                     | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Ministral-3-14B-Instruct-2512-nvfp4     | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Molmo2-8B-4bit                          | usable_with_caveats | title/keyword constraints failed |
| mlx-community/North-Micro-Vision-Instruct-4bit        | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit           | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Phi-3.5-vision-instruct-bf16            | usable_with_caveats | title/keyword constraints failed |
| mlx-community/pixtral-12b-8bit                        | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit          | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Qwen3.5-9B-MLX-4bit                     | usable_with_caveats | title/keyword constraints failed |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx              | usable_with_caveats | title/keyword constraints failed |
| mlx-community/Step-3.7-Flash-oQ2e                     | usable_with_caveats | title/keyword constraints failed |
| mlx-community/X-Reasoner-7B-8bit                      | usable_with_caveats | title/keyword constraints failed |

## Clean Completion Context

<details>
<summary>Clean completions</summary>

| Model                                             | Runtime identity                                          | Performance                                                                             |
|---------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------|
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit  | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed    | 1605 prompt / 406 generated; 134 tok/s; 19 GB peak; cleanup 17.4/0.0 GB active/cache    |
| mlx-community/gemma-4-26b-a4b-it-4bit             | rev 0d77464eeb23; Gemma4Processor; stop completed         | 574 prompt / 105 generated; 128 tok/s; 16 GB peak; cleanup 15.3/0.0 GB active/cache     |
| mlx-community/gemma-4-31b-it-4bit                 | rev 696d436c4047; Gemma4Processor; stop completed         | 574 prompt / 90 generated; 26.5 tok/s; 19 GB peak; cleanup 18.4/0.0 GB active/cache     |
| mlx-community/granite-4.0-3b-vision-4bit          | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed | 1371 prompt / 88 generated; 178 tok/s; 4.7 GB peak; cleanup 3.02/0.0 GB active/cache    |
| mlx-community/Idefics3-8B-Llama3-bf16             | rev 8c2a30c48864; Idefics3Processor; stop completed       | 2597 prompt / 162 generated; 32.3 tok/s; 18 GB peak; cleanup 16.9/0.0 GB active/cache   |
| mlx-community/InternVL3-8B-bf16                   | rev e0df3dd79263; InternVLChatProcessor; stop completed   | 2086 prompt / 88 generated; 34.4 tok/s; 17 GB peak; cleanup 15.9/0.0 GB active/cache    |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit             | rev 12c5ae493041; Lfm2VlProcessor; stop completed         | 2086 prompt / 83 generated; 213 tok/s; 4.0 GB peak; cleanup 2.81/0.0 GB active/cache    |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | rev 7c992876448f; Mistral3Processor; stop completed       | 2918 prompt / 180 generated; 67.4 tok/s; 13 GB peak; cleanup 8/0.0 GB active/cache      |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit   | rev a962dcb09eee; Mistral3Processor; stop completed       | 2917 prompt / 139 generated; 190 tok/s; 7.8 GB peak; cleanup 2.75/0.0 GB active/cache   |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit         | rev b4a74fa6001f; MuseGlimmerProcessor; stop completed    | 4388 prompt / 992 generated; 25.2 tok/s; 25 GB peak; cleanup 23.9/0.0 GB active/cache   |
| mlx-community/Qwen3-VL-2B-Thinking-bf16           | rev c325e5ea14c2; Qwen3VLProcessor; stop completed        | 16522 prompt / 907 generated; 89.8 tok/s; 8.4 GB peak; cleanup 4.26/0.0 GB active/cache |
| mlx-community/Qwen3.5-35B-A3B-4bit                | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed        | 16536 prompt / 143 generated; 116 tok/s; 24 GB peak; cleanup 20.4/0.0 GB active/cache   |
| mlx-community/Qwen3.8-27B-4bit                    | rev 3e6447f082e8; Qwen3VLProcessor; stop completed        | 16536 prompt / 131 generated; 30.1 tok/s; 21 GB peak; cleanup 16.1/0.0 GB active/cache  |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 9,984 pixels
- *Image size:* 66,295,254 bytes
- *Image SHA-256:* 168b4850b1427394bbe84a99ffd05533ffaf7e995e4213d9ce2d36c959e70c7b

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-08-31 15:02:20 UTC+01:00

Descriptive hints:
- Description hint: Visitors walk along the pathway and relax in front of the historic Winchester City Mill, an ancient watermill situated over the River Itchen in Winchester, Hampshire, England.
- Keyword hints: Adobe Stock, Any Vision, Arch, Blue sky, Chimney, Clay tiles, Elderly woman, England, Girls, Hampshire, Mill, National Trust, Pedestrians, People, Rapids, River Itchen, Riverbank, Scenery, Sitting, Stone wall

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

| Model                                        | Resolved revision                        |
|----------------------------------------------|------------------------------------------|
| mlx-community/GLM-4.6V-Flash-4bit            | bd7b20686e8cbf22ad221a3de8399cdc5a2a7e0e |
| mlx-community/GLM-4.6V-nvfp4                 | 2da6855d4e28a0e61c84543262074bc17ac27d6e |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | e5abbe34cbfabd829fafd0362856e5b468d19f85 |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.0rc0                                                                                                                                        |
| mlx-vlm source revision    | 5c9b5f52adfeab35b5ece0bb2d6e4d44541d9e32                                                                                                        |
| mlx                        | 0.32.3.dev20260904+b6368984b                                                                                                                    |
| mlx source revision        | b6368984b8e02a3fb3ee7986846c0fb85e1fccf7                                                                                                        |
| mlx-lm                     | 0.32.0                                                                                                                                          |
| mlx-lm source revision     | 4343e96fa72e0ed458d02d08e2d220a1b22061cb                                                                                                        |
| mlx-audio                  | 0.5.1                                                                                                                                           |
| transformers               | 5.16.1                                                                                                                                          |
| tokenizers                 | 0.23.2                                                                                                                                          |
| huggingface-hub            | 1.30.0                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (182,474,576 bytes, sha256=42ffcb402b5e7c7023c96051d078f0101c582b70aed3cf61cc70e394bd6e036c) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,958,976 bytes, sha256=c134b2fe7fa611db39c82cf3a3358114c5e390334ea9e4ddc7c973f414f57c98)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
