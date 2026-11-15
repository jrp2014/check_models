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
- *Input image:* JPEG, 9,984 x 5,616 pixels (56.1 MP), 28.7 MB
- *Human accuracy notes:* 0 of 41 outputs reviewed by a person for this exact
  image, model revision and output (accuracy_notes.yaml); the others are
  unreviewed; 1 note for other images or models

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
| none                           | 40    |
| observation needs reproduction | 1     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 3     |
| no concerns detected | 33    |
| concerns detected    | 5     |

Observation counts

| Observation                                      | Count |
|--------------------------------------------------|-------|
| Unrecognised model control tokens remain visible | 1     |
| Required labelled fields not detected            | 3     |
| Repeated keyword entries                         | 1     |
| Output repeats the prompt's own hint text        | 4     |

## Triage

| Model                                                                                      | Execution | Mechanical checks | Maintainer status              | Observations                                         |
|--------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|------------------------------------------------------|
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit"></a>

<details>
<summary>mlx-community/llm-jp-4-vl-9b-mlx-4bit — major concerns — control tokens visible; labelled fields not detected</summary>

### mlx-community/llm-jp-4-vl-9b-mlx-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* missing_requested_sections, unexpected_special_token
- *Arch supported by installed mlx-vlm:* yes (model_type llmjpvl)
- *Family README in installed mlx-vlm:* none
- *Labelled fields not detected:* ["title", "description", "keywords"]
- *Unexpected special tokens:* ["&lt;|channel|&gt;", "&lt;|message|&gt;"]
- *Percent of words in four-word runs copied from the description hint:* 72
- *Text compared with the description hint:* answer
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Processor class:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.52
- *Peak memory at first token (GB):* 6.737
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.013
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 2698
- *Prompt composition:* 2,698 = 394 text/template + 2,304 image tokens (85%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine
- *Load active memory vs checkpoint:* 1.00x (5.69 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 30
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: battery for 2 of 2 sample(s); thermal state max fair over
  2 sample(s); GPU activity outside harness inference: max 0% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The image shows a small flock of sheep grazing and standing along a grassy path in a rural meadow on a quiet, foggy morning.
```

</details>

## Indeterminate Attempts

None.

## Model Compliance Notes (not maintainer issues)

Prompt-compliance observations (missing fields, constraint counts, hint
copying, instruction echo, cap hits) inform model selection; complete evidence
is in the model gallery.

| Model                                         | Mechanical checks | Observations                 |
|-----------------------------------------------|-------------------|------------------------------|
| mlx-community/FastVLM-0.5B-bf16               | major concerns    | labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit              | major concerns    | labelled fields not detected |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16              | concerns detected | duplicate keywords           |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | concerns detected | prompt hint repeated         |
| mlx-community/GLM-4.6V-Flash-4bit             | concerns detected | prompt hint repeated         |
| mlx-community/granite-4.0-3b-vision-4bit      | concerns detected | prompt hint repeated         |
| mlx-community/MiniCPM-V-4.7-1B-4bit           | concerns detected | prompt hint repeated         |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                          |
|-------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------|
| ibm-granite/granite-vision-4.1-4b                           | rev 37d591f06319; Granite4VisionProcessor; stop completed    | 1502 prompt / 79 generated; 65.2 tok/s; 9.8 GB peak  |
| mlx-community/AREX-2-4bit                                   | rev 02551ee54839; Qwen3VLProcessor; stop completed           | 16653 prompt / 119 generated; 30.5 tok/s; 21 GB peak |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2429 prompt / 129 generated; 97.7 tok/s; 6.6 GB peak |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit            | rev 846ea5576854; Ernie4_5_VLProcessor; stop completed       | 1606 prompt / 598 generated; 130 tok/s; 19 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 592 prompt / 111 generated; 62.1 tok/s; 7.6 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 592 prompt / 90 generated; 126 tok/s; 16 GB peak     |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 588 prompt / 92 generated; 124 tok/s; 6.0 GB peak    |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 2619 prompt / 102 generated; 56.2 tok/s; 10 GB peak  |
| mlx-community/InternVL3_5-1B-4bit                           | rev f9d179a8be8a; InternVLProcessor; stop completed          | 2622 prompt / 107 generated; 401 tok/s; 2.2 GB peak  |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                       | rev b6311011d95c; Lfm2VlProcessor; stop completed            | 2626 prompt / 108 generated; 207 tok/s; 4.0 GB peak  |
| mlx-community/MiniCPM-o-4_5-4bit                            | rev 592c09d85e7b; MiniCPMOProcessor; stop completed          | 385 prompt / 105 generated; 104 tok/s; 7.1 GB peak   |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit                    | rev 0e3b706c1599; MiniCPMV4_7Processor; stop completed       | 932 prompt / 123 generated; 87.0 tok/s; 25 GB peak   |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 2583 prompt / 120 generated; 66.3 tok/s; 12 GB peak  |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 2582 prompt / 107 generated; 188 tok/s; 6.4 GB peak  |
| mlx-community/Mistral-Small-4-119B-2603-4bit                | rev 87b124920773; Mistral3Processor; stop completed          | 2062 prompt / 93 generated; 78.1 tok/s; 71 GB peak   |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 1518 prompt / 117 generated; 71.0 tok/s; 8.1 GB peak |
| mlx-community/MolmoPoint-8B-4bit                            | rev 9bab196f867c; MolmoPointProcessor; stop completed        | 3120 prompt / 109 generated; 30.5 tok/s; 13 GB peak  |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit                   | rev 865ee8de6579; MuseGlimmerProcessor; stop completed       | 4432 prompt / 990 generated; 24.6 tok/s; 25 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 4050 prompt / 121 generated; 206 tok/s; 3.7 GB peak  |
| mlx-community/Phi-3.5-vision-instruct-bf16                  | rev d8da684308c2; Phi3VProcessor; stop completed             | 1131 prompt / 87 generated; 57.7 tok/s; 9.3 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 12833 prompt / 120 generated; 73.2 tok/s; 26 GB peak |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 16637 prompt / 111 generated; 84.7 tok/s; 23 GB peak |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 16637 prompt / 95 generated; 68.1 tok/s; 11 GB peak  |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 16653 prompt / 114 generated; 111 tok/s; 25 GB peak  |
| mlx-community/Qwen3.6-35B-A3B-4bit                          | rev 38740b847e4c; Qwen3VLProcessor; stop completed           | 16653 prompt / 128 generated; 112 tok/s; 25 GB peak  |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev f9cb64950546; Qwen3VLProcessor; stop completed           | 16653 prompt / 129 generated; 30.9 tok/s; 21 GB peak |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx                    | rev 844516024a1c; SmolVLMProcessor; stop completed           | 1415 prompt / 84 generated; 124 tok/s; 5.6 GB peak   |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 4146 prompt / 102 generated; 124 tok/s; 5.4 GB peak  |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 1113 prompt / 111 generated; 35.8 tok/s; 18 GB peak  |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 3641 prompt / 108 generated; 161 tok/s; 23 GB peak   |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4         | rev 3acb09951cff; NemotronHNanoOmniProcessor; stop completed | 3640 prompt / 105 generated; 30.4 tok/s; 87 GB peak  |
| sahilchachra/LensVLM-9B-MXFP4                               | rev 23ae80ae9a7d; Qwen3VLProcessor; stop completed           | 1841 prompt / 157 generated; 99.7 tok/s; 7.4 GB peak |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 16653 prompt / 110 generated; 35.9 tok/s; 17 GB peak |

</details>

## Shared Reproduction and Provenance

### Reproduction inputs

- *Image format:* JPEG
- *Image dimensions:* 9,984 x 5,616 pixels
- *Image size:* 28,722,313 bytes
- *Image SHA-256:* cd7ad46ae878cb6f3844a771f0b2f622fa9cffc87ae767e0893bff1b7a7c3c38

<details>
<summary>Exact prompt</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-11-07 16:58:09 UTC+01:00

Descriptive hints:
- Description hint: A small flock of sheep gathers and grazes along a grassy path in a rural meadow on a quiet, foggy morning. A small flock of sheep grazes and stands along a grassy path in an open meadow surrounded by misty autumn trees and rural landscape.
- Keyword hints: Agriculture, British countryside, Bushes, Countryside, English landscape, Farmland, Field, Grass, Grazing, Landscape, Livestock, Meadow, Mist, Overcast Sky, Pasture, Rural, Rural scenery, Sheep, Shrubs, Standing

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-d5bc1c2a9fc110ad.jpg>
- *Preview dimensions:* 1,024 x 576 pixels
- *Preview size:* 64,838 bytes
- *Preview SHA-256:* d5bc1c2a9fc110ad0af049c96eb3d768b20e1fcb88d01604b2c72d798c7cec18

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-d5bc1c2a9fc110ad.jpg
printf '%s\n' 'd5bc1c2a9fc110ad0af049c96eb3d768b20e1fcb88d01604b2c72d798c7cec18  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-11-07 16:58:09 UTC+01:00

Descriptive hints:
- Description hint: A small flock of sheep gathers and grazes along a grassy path in a rural meadow on a quiet, foggy morning. A small flock of sheep grazes and stands along a grassy path in an open meadow surrounded by misty autumn trees and rural landscape.
- Keyword hints: Agriculture, British countryside, Bushes, Countryside, English landscape, Farmland, Field, Grass, Grazing, Landscape, Livestock, Meadow, Mist, Overcast Sky, Pasture, Rural, Rural scenery, Sheep, Shrubs, Standing

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
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | 9c056d48b1e611dc586139a5deb927ae363cfe6f |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.8.0                                                                                                                                           |
| mlx-vlm source revision    | e267ccf740b8b9dfc4529665b5efd8b9e1eff187                                                                                                        |
| mlx                        | 0.32.5.dev20261107+fc344c986                                                                                                                    |
| mlx source revision        | fc344c986                                                                                                                                       |
| mlx-audio                  | 0.6.5                                                                                                                                           |
| transformers               | 5.23.0                                                                                                                                          |
| tokenizers                 | 0.23.3                                                                                                                                          |
| huggingface-hub            | 2.4.0                                                                                                                                           |
| Python Version             | 3.14.7                                                                                                                                          |
| OS                         | Darwin 27.2.0                                                                                                                                   |
| macOS Version              | 27.2.1                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (225,677,776 bytes, sha256=e9560b19b6699a87c87ce8f26fb6cdc5e4af170a9b1a1e91dbba401ca1a382ec) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,155,648 bytes, sha256=44513f3f0be9644563e802ab96880b4db43337c3222f2be50aaf2d48a62ce3e6)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
