# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

This run records model responses to one shared image and prompt (evaluation
lane: assisted). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run Summary

- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 6,656 x 9,483 pixels (63.1 MP), 60.5 MB
- *Human accuracy notes:* 0 of 43 outputs reviewed by a person for this exact
  image, model revision and output (accuracy_notes.yaml); the others are
  unreviewed; 1 note for other images or models

Outcome counts

| Outcome             | Count |
|---------------------|-------|
| Attempted           | 43    |
| Conclusive outcomes | 43    |
| Completed           | 43    |
| Crashed             | 0     |
| Indeterminate       | 0     |

Maintainer status counts

| Maintainer status              | Count |
|--------------------------------|-------|
| none                           | 41    |
| observation needs reproduction | 2     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 5     |
| no concerns detected | 29    |
| concerns detected    | 9     |

Observation counts

| Observation                                                  | Count |
|--------------------------------------------------------------|-------|
| Response repeats the same text                               | 1     |
| Generation was stopped early after sustained repeated output | 1     |
| Unrecognised model control tokens remain visible             | 1     |
| Required labelled fields not detected                        | 4     |
| Response appears cut off at the token limit                  | 1     |
| Repeated keyword entries                                     | 5     |
| Output repeats the prompt's own hint text                    | 7     |

## Triage

| Model                                                                                      | Execution | Mechanical checks | Maintainer status              | Observations                                                               |
|--------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|----------------------------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)          | completed | major concerns    | observation needs reproduction | repeated text; stopped early: repeating; duplicate keywords                |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected; prompt hint repeated |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-internvl35-1b-4bit"></a>

<details>
<summary>mlx-community/InternVL3_5-1B-4bit — major concerns — repeated text; stopped early: repeating; duplicate keywords</summary>

### mlx-community/InternVL3_5-1B-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, repetition_abort, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type internvl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "sky"
- *Title word count:* 6
- *Keyword count:* 57
- *Keywords taken verbatim from the prompt's keyword hints:* 10
- *Percent of words in four-word runs copied from the description hint:* 29
- *Text compared with the description hint:* description
- *Duplicate keywords:* ["sky", "sky view", "skyline"]
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Processor class:* mlx_vlm.models.internvl.processing_internvl.InternVLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Stop reason:* repetition_abort
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.097
- *Peak memory at first token (GB):* 2.146
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.003
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2107
- *Prompt composition:* 2,107 = 315 text/template + 1,792 image tokens (85%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 1.08
- *Parameter count:* 1.00B (name-estimate)
- *Quantization:* 4-bit, group 32, affine
- *Declared context length:* 40,960 (text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (1.08 GB vs 1.08 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 1.08 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 200
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); GPU activity outside harness inference: max 7% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
Title: Historic Gothic Tower Under Overcast Sky

Description: The image showcases a historic stone gothic tower partially obscured by dense green trees, set against an overcast sky, with a cloudy backdrop and a view of rooftops in the vicinity.

Keywords: Architecture, Gothic, Tower, Overcast sky, Foliage, Historic building, England, Pinnacle, Sky, Rooftops, Dense trees, Cloudy, Oxford, Oxfordshire, Historic England, Outdoors, Sky view, Building details, Skyline, Greenery, Historic landmarks, Skyline, Sky view, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Skyline, Sky, Sky
```

</details>

<a id="diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit"></a>

<details>
<summary>mlx-community/llm-jp-4-vl-9b-mlx-4bit — major concerns — control tokens visible; labelled fields not detected; prompt hint repeated</summary>

### mlx-community/llm-jp-4-vl-9b-mlx-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* missing_requested_sections, unexpected_special_token,
  prompt_hint_echoed
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Unexpected special tokens:* ["&lt;|channel|&gt;", "&lt;|message|&gt;"]
- *Fields repeating the prompt's hint:* ["description"]
- *Percent of words in four-word runs copied from the description hint:* 83
- *Text compared with the description hint:* answer
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Processor class:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.501
- *Peak memory at first token (GB):* 6.727
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.013
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2186
- *Prompt composition:* 2,186 = 394 text/template + 1,792 image tokens (82%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.69 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 29
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max fair over 2
  sample(s); GPU activity outside harness inference: max 30% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a view of a historic stone gothic tower and rooftops partially framed by dense green trees under an overcast sky.
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                          | Mechanical checks | Observations                                         |
|------------------------------------------------|-------------------|------------------------------------------------------|
| mlx-community/FastVLM-0.5B-bf16                | major concerns    | labelled fields not detected; prompt hint repeated   |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit      | major concerns    | labelled fields not detected; cut off at token limit |
| mlx-community/nanoLLaVA-1.5-4bit               | major concerns    | labelled fields not detected                         |
| mlx-community/MiniCPM-V-4.6-4bit               | concerns detected | duplicate keywords                                   |
| mlx-community/MiniCPM-V-4.7-1B-4bit            | concerns detected | duplicate keywords                                   |
| mlx-community/Phi-3.5-vision-instruct-bf16     | concerns detected | duplicate keywords                                   |
| mlx-community/X-Reasoner-7B-8bit               | concerns detected | duplicate keywords                                   |
| mlx-community/granite-4.0-3b-vision-4bit       | concerns detected | prompt hint repeated                                 |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit          | concerns detected | prompt hint repeated                                 |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | concerns detected | prompt hint repeated                                 |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx       | concerns detected | prompt hint repeated                                 |
| sahilchachra/LensVLM-9B-MXFP4                  | concerns detected | prompt hint repeated                                 |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                          |
|-------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------|
| ibm-granite/granite-vision-4.1-4b                           | rev 37d591f06319; Granite4VisionProcessor; stop completed    | 1430 prompt / 109 generated; 64.4 tok/s; 9.6 GB peak |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 2104 prompt / 69 generated; 438 tok/s; 1.9 GB peak   |
| mlx-community/AREX-2-4bit                                   | rev 02551ee54839; Qwen3VLProcessor; stop completed           | 16594 prompt / 114 generated; 29.7 tok/s; 21 GB peak |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2081 prompt / 117 generated; 101 tok/s; 6.5 GB peak  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 586 prompt / 76 generated; 46.8 tok/s; 28 GB peak    |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit            | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed       | 1596 prompt / 855 generated; 117 tok/s; 19 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 590 prompt / 104 generated; 62.3 tok/s; 7.6 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 590 prompt / 75 generated; 108 tok/s; 16 GB peak     |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 586 prompt / 83 generated; 121 tok/s; 6.0 GB peak    |
| mlx-community/GLM-4.6V-Flash-4bit                           | rev bd7b20686e8c; Glm46VProcessor; stop completed            | 6343 prompt / 104 generated; 77.6 tok/s; 8.7 GB peak |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 2104 prompt / 108 generated; 56.5 tok/s; 10 GB peak  |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 382 prompt / 94 generated; 104 tok/s; 7.1 GB peak    |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit                    | rev 0e3b706c1599; MiniCPMV4_7Processor; stop completed       | 927 prompt / 108 generated; 76.1 tok/s; 25 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 3046 prompt / 119 generated; 64.5 tok/s; 13 GB peak  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 3045 prompt / 153 generated; 183 tok/s; 8.2 GB peak  |
| mlx-community/Mistral-Small-4-119B-2603-4bit                | rev 87b124920773; Mistral3Processor; stop completed          | 2525 prompt / 120 generated; 69.0 tok/s; 73 GB peak  |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 1318 prompt / 133 generated; 70.8 tok/s; 8.0 GB peak |
| mlx-community/MolmoPoint-8B-4bit                            | rev 9bab196f867c; MolmoPointProcessor; stop completed        | 3136 prompt / 112 generated; 30.2 tok/s; 13 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 4044 prompt / 110 generated; 148 tok/s; 3.7 GB peak  |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 16578 prompt / 155 generated; 78.0 tok/s; 23 GB peak |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 16578 prompt / 87 generated; 67.5 tok/s; 11 GB peak  |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 16594 prompt / 112 generated; 75.1 tok/s; 25 GB peak |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev 5ff8ef173ad0; Qwen3VLProcessor; stop completed           | 16594 prompt / 121 generated; 29.8 tok/s; 21 GB peak |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 4173 prompt / 99 generated; 127 tok/s; 5.4 GB peak   |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 1313 prompt / 111 generated; 35.3 tok/s; 18 GB peak  |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 3641 prompt / 135 generated; 139 tok/s; 23 GB peak   |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit       | rev 4bc0a247d78e; NemotronHNanoOmniProcessor; stop completed | 3640 prompt / 123 generated; 63.8 tok/s; 52 GB peak  |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4         | rev da5b2650a835; NemotronHNanoOmniProcessor; stop completed | 3640 prompt / 133 generated; 30.3 tok/s; 87 GB peak  |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 16594 prompt / 118 generated; 33.3 tok/s; 17 GB peak |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 6,656 x 9,483 pixels
- *Image size:* 60,507,861 bytes
- *Image SHA-256:* ecdd79a7ae793ce2dc0c2fc78e9fb7814dafd1aec110397ca1b1c186631e2fbd

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-17 15:44:34 UTC+01:00
- GPS: 51.751667°N, 1.252222°W

Descriptive hints:
- Description hint: A view of a historic stone gothic tower and rooftops partially framed by dense green trees under an overcast sky.
- Keyword hints: Architecture, Branch, Building, Chimney, Cloud, Cloudy sky, England, English countryside, Foliage, Historic Building, Leaves, Outdoors, Overcast, Oxford, Oxfordshire, Pinnacle, Roof, Rooftops, Scenic, Sky

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-6973c355665180f8.jpg>
- *Preview dimensions:* 719 x 1,024 pixels
- *Preview size:* 179,647 bytes
- *Preview SHA-256:* 6973c355665180f82f14afd2e54719e3aaaac02b29f208456ee34bee47108151

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-6973c355665180f8.jpg
printf '%s\n' '6973c355665180f82f14afd2e54719e3aaaac02b29f208456ee34bee47108151  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-10-17 15:44:34 UTC+01:00
- GPS: 51.751667°N, 1.252222°W

Descriptive hints:
- Description hint: A view of a historic stone gothic tower and rooftops partially framed by dense green trees under an overcast sky.
- Keyword hints: Architecture, Branch, Building, Chimney, Cloud, Cloudy sky, England, English countryside, Foliage, Historic Building, Leaves, Outdoors, Overcast, Oxford, Oxfordshire, Pinnacle, Roof, Rooftops, Scenic, Sky

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description combining relevant context with the main visible subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords covering relevant context and visible details.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:' --max-tokens 1000 --temperature 0.0 --revision RESOLVED_REVISION --trust-remote-code --seed 0 --prefill-step-size 2048
```

### Highlighted model revisions

| Model                                 | Resolved revision                        |
|---------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-1B-4bit     | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | 9c056d48b1e611dc586139a5deb927ae363cfe6f |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.7.11                                                                                                                                          |
| mlx-vlm source revision    | 918763c6875c80e82f9f48344dfad4763346fea8                                                                                                        |
| mlx                        | 0.32.5.dev20261030+1ba7f969a                                                                                                                    |
| mlx source revision        | 1ba7f969a                                                                                                                                       |
| mlx-audio                  | 0.6.3                                                                                                                                           |
| transformers               | 5.22.1                                                                                                                                          |
| tokenizers                 | 0.23.3                                                                                                                                          |
| huggingface-hub            | 2.4.0                                                                                                                                           |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 27.2.0                                                                                                                                   |
| macOS Version              | 27.2                                                                                                                                            |
| SDK Version                | 27.2                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.2.sdk                                              |
| Xcode Version              | 27.2                                                                                                                                            |
| Xcode Build                | 27B52                                                                                                                                           |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX27.2.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.922 (metalfe-32023.922.4)                                                                                             |
| Metallib Linker Version    | AIR-LLD 32023.922 (metalfe-32023.922.4) (compatible with legacy metallib linker)                                                                |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.4.1.1)                                                                                                   |
| GPU/Chip                   | Apple M5 Max                                                                                                                                    |
| GPU Cores                  | 40                                                                                                                                              |
| MLX Device                 | Apple M5 Max                                                                                                                                    |
| GPU Architecture           | applegpu_g17s                                                                                                                                   |
| Recommended Working Set    | 115 GB                                                                                                                                          |
| Fused Attention            | Available                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.14/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-314-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (223,888,896 bytes, sha256=9b8d46c607b73ace0c4493f0ffac07ce9c6c77a764bee28673517c9e875dc787) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,165,360 bytes, sha256=1199f6c15fd9e700c1f7694cc5fecbbef79e0f644da8ed734e66fd9143e09eed)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
