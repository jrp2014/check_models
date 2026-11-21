# Diagnostics

<!-- markdownlint-disable MD004 MD037 -->

This run records model responses to one shared image and prompt (evaluation
lane: blind). Mechanical checks are not factual-accuracy judgments; inspect
the image, prompt and final answers before choosing a model. Results do not
establish fitness for other tasks.

## Run Summary

- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Human accuracy notes:* 1 of 41 outputs reviewed by a person for this exact
  image, model revision and output (accuracy_notes.yaml); the others are
  unreviewed

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
| none                           | 39    |
| observation needs reproduction | 2     |

Mechanical-check counts

| Mechanical checks    | Count |
|----------------------|-------|
| major concerns       | 4     |
| no concerns detected | 35    |
| concerns detected    | 2     |

Observation counts

| Observation                                      | Count |
|--------------------------------------------------|-------|
| Response repeats the same text                   | 1     |
| Unrecognised model control tokens remain visible | 1     |
| Required labelled fields not detected            | 3     |
| Response appears cut off at the token limit      | 1     |
| Repeated keyword entries                         | 3     |

## Triage

| Model                                                                                      | Execution | Mechanical checks | Maintainer status              | Observations                                              |
|--------------------------------------------------------------------------------------------|-----------|-------------------|--------------------------------|-----------------------------------------------------------|
| [mlx-community/InternVL3_5-1B-4bit](#diagnostic-mlx-community-internvl35-1b-4bit)          | completed | major concerns    | observation needs reproduction | repeated text; cut off at token limit; duplicate keywords |
| [mlx-community/llm-jp-4-vl-9b-mlx-4bit](#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) | completed | major concerns    | observation needs reproduction | control tokens visible; labelled fields not detected      |

## Crashes requiring action

None.

## Completed Runs with Observations

<a id="diagnostic-mlx-community-internvl35-1b-4bit"></a>

<details>
<summary>mlx-community/InternVL3_5-1B-4bit — major concerns — repeated text; cut off at token limit; duplicate keywords</summary>

### mlx-community/InternVL3_5-1B-4bit

#### Execution and provenance

- *Execution:* completed
- *Mechanical checks:* major concerns
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Maintainer status:* observation needs reproduction
- *Observations:* repeated_output, token_cap_truncation, duplicate_keywords
- *Arch supported by installed mlx-vlm:* yes (model_type internvl)
- *Family README in installed mlx-vlm:* none
- *Repeated fragment:* keyword: "british english examples of me"
- *Title word count:* 7
- *Keyword count:* 144
- *Duplicate keywords:* ["british english", "british english examples of
  metadata", "british english examples of details", "british english examples
  of context", "british english examples of information", "british english
  examples of data", "british english examples of content", "british english
  examples of examples"]
- *Token-cap degradation evidence:* ["repetitive_tail", "unfinished_list"]
- *Resolved model revision:* f9d179a8be8ac53e96c6ee5cce8493856d4b8f09
- *Processor class:* mlx_vlm.models.internvl.processing_internvl.InternVLProcessor
- *Tokenizer class:* transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer
- *Image processor class:* mlx_vlm.models.internvl_chat.processor.InternVLImageProcessor
- *Image processor settings:* do_resize=true, size=448, do_center_crop=false
  (preprocessor_config.json)
- *Stop reason:* max_tokens
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 0.379
- *Peak memory at first token (GB):* 2.304
- *Tail generated tokens (id: text):* 10295: ' examples', 315: ' of', 10295: '
  examples', 11: ',', 7855: ' British', 12: '-', 22574: 'English', 10295: '
  examples', 315: ' of', 11160: ' metadata'
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *EOS ids declared, by file:* {"config.json text_config": 151645,
  "generation_config.json": 151645}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.003
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3438
- *Prompt composition:* 3,438 = 110 text/template + 3,328 image tokens (97%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 1.08
- *Parameter count:* 1.00B (name-estimate)
- *Quantization:* 4-bit, group 32, affine (config.json)
- *Declared context length:* 40,960 (config.json
  text_config.max_position_embeddings)
- *Load active memory vs checkpoint:* 1.00x (1.08 GB vs 1.08 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 1.08 GB; other
  load-time allocations 0.00 GB
- *Generation tokens:* 1000
- *Configured EOS token ID:* 151645
- *Configured EOS token:* &lt;|im_end|&gt;
- *Model reproduction flags:* --model mlx-community/InternVL3_5-1B-4bit
  --max-tokens 1000 --temperature 0.0 --revision
  f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 --trust-remote-code --seed 0
  --prefill-step-size 2048
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max nominal over 2
  sample(s); GPU activity outside harness inference: max 0% at 2 instantaneous
  check(s); activity between checks is not observed; mode snapshot

#### Complete output

```text
Title: Two cats sleeping on a pink couch

Description: Two striped cats are sleeping on a pink couch, with one cat lying on its side and the other on its back, next to a remote control.

Keywords: cats, sleeping, pink couch, remote control, striped, British English, British, British-English, British-English catalogue, British-English metadata, British-English details, British-English context, British-English information, British-English data, British-English content, British-English examples, British-English examples of metadata, British-English examples of keywords, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata, British-English examples of details, British-English examples of context, British-English examples of information, British-English examples of data, British-English examples of content, British-English examples of examples, British-English examples of metadata
```

</details>

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
- *Resolved model revision:* 9c056d48b1e611dc586139a5deb927ae363cfe6f
- *Processor class:* transformers_modules._9c056d48b1e611dc586139a5deb927ae363cfe6f.0e62407644efd7c3.processing_llmjpvl.LLMjpVLProcessor
- *Tokenizer class:* transformers.tokenization_utils_tokenizers.TokenizersBackend
- *Image processor class:* transformers.models.siglip.image_processing_siglip.SiglipImageProcessor
- *Image processor settings:* do_resize=true, size={"height": 512, "width":
  512} (processor_config.json image_processor)
- *Stop reason:* completed
- *Time to first token (s; measured: input preparation, prefill, first decode step):* 1.408
- *Peak memory at first token (GB):* 7.01
- *Tail generated tokens (id: text):* 759: ' two', 11945: ' cats', 660: '
  are', 6867: ' sleeping', 650: ' on', 616: ' a', 8617: ' pink', 15295: '
  couch', 611: '.', 2: ''
- *Sampling settings source:* temperature: default; top_p: default; top_k:
  default; min_p: default; repetition_penalty: default
- *generation_config.json settings not applied (by mlx-vlm or the harness):* {"max_new_tokens":
  256}
- *EOS ids declared, by file:* {"config.json": [2, 2],
  "generation_config.json": [2, 2]}
- *Chat template found in (the first is the one the processor uses):* chat_template.jinja
- *Post-cleanup active memory (GB):* 0.013
- *Post-cleanup cache memory (GB):* 0.0
- *Prompt tokens:* 3512
- *Prompt composition:* 3,512 = 184 text/template + 3,328 image tokens (95%;
  exact, counted by token id in the prepared input)
- *Checkpoint weights (GB):* 5.68
- *Parameter count:* 9.00B (name-estimate)
- *Quantization:* 4-bit, group 64, affine (config.json)
- *Load active memory vs checkpoint:* 1.00x (5.69 GB vs 5.68 GB on disk)
- *Loaded weights in memory (mx.get_array_buffer_size):* 5.68 GB; other
  load-time allocations 0.01 GB
- *Generation tokens:* 14
- *Configured EOS token ID:* 2
- *Configured EOS token:* &lt;|return|&gt;
- *Model reproduction flags:* --model mlx-community/llm-jp-4-vl-9b-mlx-4bit
  --max-tokens 1000 --temperature 0.0 --revision
  9c056d48b1e611dc586139a5deb927ae363cfe6f --trust-remote-code --seed 0
  --prefill-step-size 2048
- *System pressure snapshots (before/after; cannot rule out transient pressure during inference):* CPU
  speed limit min 100% over 2 sample(s); memory pressure max level 1 over 2
  sample(s); power: AC over 2 sample(s); thermal state max nominal over 2
  sample(s); GPU activity outside harness inference: max 28% at 2
  instantaneous check(s); activity between checks is not observed; mode
  snapshot

#### Complete output

```text
<|channel|> analysis<|message|> The two cats are sleeping on a pink couch.
```

</details>

## Indeterminate Attempts

None.

## Model Answer-Format and Hint-Reuse Notes (not maintainer issues)

Answer-format observations (missing fields, constraint counts, instruction
echo, cap hits) and hint reuse (text or keywords repeated from the prompt's
hints, which is allowed and not in itself a fault) inform model selection;
complete evidence is in the model gallery.

| Model                                            | Mechanical checks | Observations                 |
|--------------------------------------------------|-------------------|------------------------------|
| mlx-community/FastVLM-0.5B-bf16                  | major concerns    | labelled fields not detected |
| mlx-community/nanoLLaVA-1.5-4bit                 | major concerns    | labelled fields not detected |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | duplicate keywords           |
| mlx-community/MiniCPM-o-4_5-4bit                 | concerns detected | duplicate keywords           |

## Context for completions without detected concerns

<details>
<summary>Completions without detected concerns</summary>

| Model                                                       | Runtime identity                                             | Performance                                         |
|-------------------------------------------------------------|--------------------------------------------------------------|-----------------------------------------------------|
| ibm-granite/granite-vision-4.1-4b                           | rev 12e928afb004; Granite4VisionProcessor; stop completed    | 734 prompt / 56 generated; 61.5 tok/s; 9.7 GB peak  |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16                            | rev ed71acdae079; Lfm2VlProcessor; stop completed            | 350 prompt / 47 generated; 570 tok/s; 1.3 GB peak   |
| mlx-community/AREX-2-4bit                                   | rev 02551ee54839; Qwen3VLProcessor; stop completed           | 461 prompt / 865 generated; 31.8 tok/s; 17 GB peak  |
| mlx-community/aya-vision-8b-4bit                            | rev 3e679b3e08f0; AyaVisionOutputProcessor; stop completed   | 2909 prompt / 82 generated; 102 tok/s; 6.5 GB peak  |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8               | rev ded389e478f8; DiffusionGemma4Processor; stop completed   | 386 prompt / 66 generated; 66.1 tok/s; 28 GB peak   |
| mlx-community/gemma-4-12B-it-4bit                           | rev 73bcf09092aa; Gemma4UnifiedProcessor; stop completed     | 390 prompt / 88 generated; 63.1 tok/s; 7.5 GB peak  |
| mlx-community/gemma-4-26b-a4b-it-4bit                       | rev 0d77464eeb23; Gemma4Processor; stop completed            | 390 prompt / 75 generated; 127 tok/s; 16 GB peak    |
| mlx-community/gemma-4-e4b-it-4bit                           | rev 475b9088d297; Gemma4Processor; stop completed            | 386 prompt / 67 generated; 124 tok/s; 5.9 GB peak   |
| mlx-community/GLM-4.6V-Flash-4bit                           | rev bd7b20686e8c; Glm46VProcessor; stop completed            | 495 prompt / 819 generated; 81.9 tok/s; 8.0 GB peak |
| mlx-community/granite-4.0-3b-vision-4bit                    | rev 70fe1d89f42c; Granite4VisionProcessor; stop completed    | 734 prompt / 57 generated; 178 tok/s; 4.7 GB peak   |
| mlx-community/InternVL3-14B-4bit                            | rev 26328eaab82c; InternVLChatProcessor; stop completed      | 3435 prompt / 90 generated; 55.6 tok/s; 10 GB peak  |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit                       | rev b6311011d95c; Lfm2VlProcessor; stop completed            | 344 prompt / 73 generated; 217 tok/s; 3.6 GB peak   |
| mlx-community/MiniCPM-V-4.7-1B-4bit                         | rev 13caa569f70e; MiniCPMV4_7Processor; stop completed       | 330 prompt / 268 generated; 313 tok/s; 3.5 GB peak  |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit                    | rev 0e3b706c1599; MiniCPMV4_7Processor; stop completed       | 330 prompt / 431 generated; 87.4 tok/s; 24 GB peak  |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4           | rev 7c992876448f; Mistral3Processor; stop completed          | 1070 prompt / 81 generated; 67.5 tok/s; 8.9 GB peak |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit             | rev a962dcb09eee; Mistral3Processor; stop completed          | 1069 prompt / 102 generated; 209 tok/s; 3.7 GB peak |
| mlx-community/Mistral-Small-4-119B-2603-4bit                | rev 87b124920773; Mistral3Processor; stop completed          | 549 prompt / 82 generated; 91.1 tok/s; 69 GB peak   |
| mlx-community/Molmo2-8B-4bit                                | rev 4fcbe9265776; Molmo2Processor; stop completed            | 873 prompt / 95 generated; 71.1 tok/s; 8.2 GB peak  |
| mlx-community/MolmoPoint-8B-4bit                            | rev 9bab196f867c; MolmoPointProcessor; stop completed        | 860 prompt / 70 generated; 30.3 tok/s; 12 GB peak   |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit                   | rev 865ee8de6579; MuseGlimmerProcessor; stop completed       | 545 prompt / 884 generated; 24.9 tok/s; 25 GB peak  |
| mlx-community/North-Micro-Vision-Instruct-4bit              | rev 87466363e6c5; CohereCompassProcessor; stop completed     | 418 prompt / 78 generated; 267 tok/s; 3.3 GB peak   |
| mlx-community/Phi-3.5-vision-instruct-bf16                  | rev d8da684308c2; Phi3VProcessor; stop completed             | 883 prompt / 94 generated; 58.1 tok/s; 9.1 GB peak  |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit              | rev 93b3cbddd65e; Qwen3OmniMoeProcessor; stop completed      | 409 prompt / 81 generated; 93.6 tok/s; 23 GB peak   |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit                | rev 0555d34cb1ed; Qwen3VLProcessor; stop completed           | 409 prompt / 83 generated; 133 tok/s; 19 GB peak    |
| mlx-community/Qwen3-VL-8B-Instruct-4bit                     | rev defcdea7cc7a; Qwen3VLProcessor; stop completed           | 409 prompt / 60 generated; 103 tok/s; 6.8 GB peak   |
| mlx-community/Qwen3.5-35B-A3B-4bit                          | rev 1e20fd8d4205; Qwen3VLProcessor; stop completed           | 419 prompt / 898 generated; 123 tok/s; 21 GB peak   |
| mlx-community/Qwen3.6-35B-A3B-4bit                          | rev 38740b847e4c; Qwen3VLProcessor; stop completed           | 419 prompt / 653 generated; 121 tok/s; 21 GB peak   |
| mlx-community/Qwen3.8-27B-nvfp4                             | rev f9cb64950546; Qwen3VLProcessor; stop completed           | 461 prompt / 640 generated; 30.9 tok/s; 17 GB peak  |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx                    | rev 844516024a1c; SmolVLMProcessor; stop completed           | 1204 prompt / 31 generated; 117 tok/s; 5.6 GB peak  |
| nativ-community/Mage-VL-OptiQ-4bit                          | rev 4f0a424370e5; MageVLProcessor; stop completed            | 420 prompt / 88 generated; 149 tok/s; 5.2 GB peak   |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit    | rev bdbeb0d8c89e; Mistral3Processor; stop completed          | 538 prompt / 87 generated; 35.9 tok/s; 16 GB peak   |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | rev 75c89904e1c2; NemotronHNanoOmniProcessor; stop completed | 421 prompt / 706 generated; 162 tok/s; 21 GB peak   |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4         | rev ae1368329b1f; NemotronHNanoOmniProcessor; stop completed | 420 prompt / 797 generated; 30.3 tok/s; 81 GB peak  |
| sahilchachra/LensVLM-9B-MXFP4                               | rev 23ae80ae9a7d; Qwen3VLProcessor; stop completed           | 419 prompt / 198 generated; 97.4 tok/s; 6.7 GB peak |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit                 | rev 498775b03b55; Qwen3VLProcessor; stop completed           | 461 prompt / 517 generated; 43.2 tok/s; 11 GB peak  |

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

- *Retained preview:* <https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-c4aeb8d18432656f.jpg>
- *Preview dimensions:* 640 x 480 pixels
- *Preview size:* 63,187 bytes
- *Preview SHA-256:* c4aeb8d18432656f38f690facffbfa90ad3966dc2113a675025d1231a7c5747b

Shareable stand-in: the retained gallery preview is a downscaled re-encoding
of the original, so an observation reproduced on it must be reported as
reproduced on the preview, not on the exact inference input. The asset is
named by its digest, so later sweeps never replace it; the URL resolves once
this run's artifacts are committed. Download and verify it, then run one
native mlx-vlm process.

```bash
set -euo pipefail
curl --fail --location --output repro-image.jpg https://raw.githubusercontent.com/jrp2014/check_models/main/src/output/reports/assets/source-image-c4aeb8d18432656f.jpg
printf '%s\n' 'c4aeb8d18432656f38f690facffbfa90ad3966dc2113a675025d1231a7c5747b  repro-image.jpg' | shasum -a 256 --check
python -m mlx_vlm.generate --verbose --model MODEL_ID --image repro-image.jpg --prompt 'Create British-English catalogue metadata using only clearly visible facts. Omit uncertain details and unsupported identity, location, event, brand, species, period, or intent.

Write:
- a concrete 5-10-word title;
- a 1-2-sentence factual description of the main subject, setting, action, lighting, and distinctive details;
- 10-18 unique, comma-separated keywords.

Return exactly these three sections and nothing else:
Title:
Description:
Keywords:' --max-tokens 1000 --temperature 0.0 --revision RESOLVED_REVISION --trust-remote-code --seed 0 --prefill-step-size 2048
```

To use a model's reproduction flags, keep the shared command's --verbose,
--image and --prompt arguments and replace all its other arguments with those
flags.

### Highlighted model revisions

| Model                                 | Resolved revision                        |
|---------------------------------------|------------------------------------------|
| mlx-community/InternVL3_5-1B-4bit     | f9d179a8be8ac53e96c6ee5cce8493856d4b8f09 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | 9c056d48b1e611dc586139a5deb927ae363cfe6f |

### Components and system

| Component                  | Value                                                                                                                                           |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.8.0                                                                                                                                           |
| mlx-vlm source revision    | f1618d6d52be4ef25d3fe65270f7528a59c55df3                                                                                                        |
| mlx                        | 0.32.5.dev20261121+8803ac3ca                                                                                                                    |
| mlx source revision        | 8803ac3ca                                                                                                                                       |
| mlx-audio                  | 0.6.7                                                                                                                                           |
| transformers               | 5.24.0                                                                                                                                          |
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
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (226,846,208 bytes, sha256=4a5bb5bc11007a396d4d70148898b72d5eeafa143db42451ef8340194133580f) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (22,139,456 bytes, sha256=4c01b0e8844b1a1868d7b4a989fc854ba2ef0ff3c064c4770a0e61d03d18e728)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD004 MD037 -->
