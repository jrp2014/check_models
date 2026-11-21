# mlx-vlm compatibility findings across 41 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 2 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 35 with no mechanical observations, 4 with answer-format
or hint-reuse observations only (labels or form, or text reused from the
prompt's hints, which is allowed), and 2 with observations mlx-vlm can produce
(such as repetition, missing final answers or visible control tokens); only
those 2 are candidates for native reproduction.

## Run summary

- *Run started:* 2026-11-21 23:42:25 GMT
- *Run finished:* 2026-11-21 23:48:03 GMT
- *Run duration:* 5m 37s
- *Time by phase:* generation 229s, model load 85s, prompt prep 1s, cleanup
  5s, outside the model loop 17s
- *Evaluation lane:* blind
- *Prompt hints:* no description or keyword hints
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 640 x 480 pixels (0.3 MP), 0.2 MB
- *Models attempted:* 41
- *Sampling settings:* checkpoint generation_config.json values for 17 of 41
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 41
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 2
- *Reached token limit:* 1 (1 with incomplete output)
- *Stopped early for repetition:* 0

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: blind). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

<details>
<summary>Exact prompt sent to every model</summary>

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

## Since the baseline sweep

- *Baseline:* 6f3a5278:src/output/results.jsonl
- *Baseline run timestamp:* 2026-11-20 18:52:25 GMT
- *Baseline check_models:* 0.17.142 @ 3ed61570c
- *Baseline mlx:* 0.32.5.dev20261120+bd2d1fb43 @ bd2d1fb43
- *Baseline mlx-vlm:* 0.8.0 @ 392052b44
- *Baseline transformers:* 5.23.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261121+8803ac3ca

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; evaluation lane differs
(assisted → blind); image differs (sha256 6020d9876855… → dea9e7ef9738…). The
roster, revisions and upstream changes below are facts about the runs and are
still shown.

Model revisions changed since the baseline (their rows below reflect a
different snapshot, not only a different run):
`ibm-granite/granite-vision-4.1-4b` 37d591f06 → 12e928afb,
`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` 3acb09951 → ae1368329

Run environment: current run detected GPU activity outside harness inference
at the checks around 7 of 41 models (up to 55%); baseline run detected GPU
activity outside harness inference at the checks around 1 of 41 models (up to
14%).

<details>
<summary>mlx: 14 upstream commit(s) since the baseline (bd2d1fb43..8803ac3ca)</summary>

- 8803ac3ca Fixed cross-compiling the CPU preamble for Windows (#4921)
- 1fd565f69 Fixed the race on the Metal allocator's memory stats (#4885)
- 5cabdf509 [Metal] Use affine qmv_mma on M1 and M2 Ultra from 4 rows (#4890)
- dc5ea6e98 Keep gelu_approx float16 gradients finite (#4906)
- f0ab88006 Fixed BLA_VENDOR being ignored on Windows (#4920)
- da823ef6d Fix conv weight gradient with empty batch (#4873)
- 6d38ff93d [Metal] Add 2, 3, 5 and 6 bits to affine qmv_mma (#4860)
- a3a610bb8 Make the scheduler's active task count atomic (#4882)
- 0c93d91e8 Add validate_qqmm_mode (#4931)
- 6ac6c0f4b Fix float16 overflow in clip_grad_norm_sharded (#4848)
- 06712fbaf Don't allocate inside `quantize_impl` (#4930)
- 3a71347ab docs: Add missing step default to arange signature (#4855)
- 381bb02a9 Correct gradients and batching for DivMod (#4823)
- 430d7641b Fixed linking psapi with MinGW (#4919)

</details>

<details>
<summary>mlx-vlm: 17 upstream commit(s) since the baseline (392052b44..f1618d6d5)</summary>

- f1618d6d Merge pull request #2788 from lucasnewman/remove-legacy-sft-trainer
- 71a5b80e Remove the legacy SFT trainer.
- 0d16abeb Merge pull request #2760 from
  Lornatang/fix/qwen2-5-vl-video-temporal-rope
- 434a1bed Add tool parser for Inkling (#2783)
- 7fbd093f Key video prefix reuse on per-clip RoPE timing
- 444c3461 Preserve sampled video timing through Qwen2.5-VL prefill and decode
- fd6404fb Merge pull request #2774 from nurlanmnn/fix-minimax-nullable-params
- a8abb338 Merge pull request #2772 from
  Lornatang/fix/apc-disk-tests-pin-free-ram
- 92075ff3 test(apc): pin memory budgets for disk cache restores
- e0ba3267 Handle numeric overflow in MiniMax M2 tool arguments
- 0146e574 fix(tool-parsers): keep values of nullable minimax_m2 parameters
- 111ac668 Add Inkling 2 Mini model. (#2778)
- 22592742 Add Depth Pro 2 models. (#2777)
- 25ead5d2 Fix DeepSeek V4 tool calling and thinking (#2544)
- 12063ee7 test(qwen2_5_vl): assert integer RoPE positions for scaled video
- ... and 2 more (see results.jsonl)

</details>

Mechanical diff only: one image and one generation per model in each sweep,
with no warm-up or repeats. A difference shows that outputs or timings differ,
not why, whether it affects one model or many; attributing a cause needs
matched repeats or a native reproduction.

## Mechanical checks at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. "Major concerns: generation" means generation itself failed (no
text, repeated text, or no final answer); "major concerns: answer format"
means an answer was generated but missed the requested form. Tokens in count
the image's tokens too, which drive prefill time; tokens out are those
generated (limit 1,000). *Native repro* says whether a person reran an
observation outside this harness on the same model revision and builds:
confirmed, not reproduced, or not yet checked. Recurring here alone is not
native confirmation. Keywords are counted from the answer's Keywords field.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 2.62s | 61.5 tok/s | 9.7 | 734 / 56 | 10 | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 1.29s | 570 tok/s | 1.3 | 350 / 47 | 6 | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 31.34s | 31.8 tok/s | 17 | 461 / 865 | 16 | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.15s | 102 tok/s | 6.5 | 2,909 / 82 | 10 | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 5.29s | 66.1 tok/s | 28 | 386 / 66 | 14 | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 3.53s | 63.1 tok/s | 7.5 | 390 / 88 | 18 | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 3.44s | 127 tok/s | 16 | 390 / 75 | 15 | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 2.43s | 124 tok/s | 5.9 | 386 / 67 | 14 | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 11.71s | 81.9 tok/s | 8.0 | 495 / 819 | 12 | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 1.39s | 178 tok/s | 4.7 | 734 / 57 | 10 | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 5.25s | 55.6 tok/s | 10 | 3,435 / 90 | 18 | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 1.47s | 217 tok/s | 3.6 | 344 / 73 | 10 | none | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | no concerns detected | 2.08s | 313 tok/s | 3.5 | 330 / 268 | 15 | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 8.73s | 87.4 tok/s | 24 | 330 / 431 | 18 | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 3.30s | 67.5 tok/s | 8.9 | 1,070 / 81 | 10 | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 1.65s | 209 tok/s | 3.7 | 1,069 / 102 | 12 | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 20.11s | 91.1 tok/s | 69 | 549 / 82 | 18 | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 3.00s | 71.1 tok/s | 8.2 | 873 / 95 | 10 | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 4.57s | 30.3 tok/s | 12 | 860 / 70 | 10 | none | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | no concerns detected | 39.24s | 24.9 tok/s | 25 | 545 / 884 | 14 | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 1.65s | 267 tok/s | 3.3 | 418 / 78 | 12 | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 2.85s | 58.1 tok/s | 9.1 | 883 / 94 | 19 | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 3.73s | 93.6 tok/s | 23 | 409 / 81 | 18 | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 3.01s | 133 tok/s | 19 | 409 / 83 | 10 | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 1.87s | 103 tok/s | 6.8 | 409 / 60 | 10 | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 10.24s | 123 tok/s | 21 | 419 / 898 | 18 | none | - |
| mlx-community/Qwen3.6-35B-A3B-4bit | no concerns detected | 8.39s | 121 tok/s | 21 | 419 / 653 | 19 | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 23.52s | 30.9 tok/s | 17 | 461 / 640 | 14 | none | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | no concerns detected | 1.51s | 117 tok/s | 5.6 | 1,204 / 31 | 4 | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 1.73s | 149 tok/s | 5.2 | 420 / 88 | 10 | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 4.89s | 35.9 tok/s | 16 | 538 / 87 | 18 | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 7.01s | 162 tok/s | 21 | 421 / 706 | 17 | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 55.31s | 30.3 tok/s | 81 | 420 / 797 | 15 | none | - |
| sahilchachra/LensVLM-9B-MXFP4 | no concerns detected | 3.61s | 97.4 tok/s | 6.7 | 419 / 198 | 11 | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 15.17s | 43.2 tok/s | 11 | 461 / 517 | 14 | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | concerns detected | 6.00s | 145 tok/s | 19 | 545 / 540 | 16 | duplicate keywords | - |
| mlx-community/MiniCPM-o-4_5-4bit | concerns detected | 5.01s | 102 tok/s | 7.0 | 173 / 367 | 18 | duplicate keywords | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 1.18s | 395 tok/s | 2.1 | 120 / 32 | - | labelled fields not detected | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 2.69s | insufficient sample | 7.0 | 3,512 / 14 | - | control tokens visible; labelled fields not detected | not yet checked (earlier check out of date) |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 1.02s | 353 tok/s | 1.5 | 116 / 97 | - | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.73s | 379 tok/s | 2.3 | 3,438 / 1,000 | 144 | repeated text; cut off at token limit; duplicate keywords | not yet checked |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.32.5.dev20261121+8803ac3ca @ 8803ac3ca; mlx-vlm 0.8.0 @ f1618d6d5;
transformers 5.24.0) and experiment inputs (image, prompt, generation
settings, and each model's rendered prompt and effective generation
arguments). Earlier checks shown separately do not establish this result.

No recorded check applies to this run.

<details>
<summary>Earlier native checks (out of date)</summary>

| Model                                 | Observation            | Prior status | Prior stack                                                                                   | Checked            | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|---------------------------------------|------------------------|--------------|-----------------------------------------------------------------------------------------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | confirmed    | mlx 0.32.5.dev20261030+1ba7f969a @ 1ba7f969a; mlx-vlm 0.7.11 @ 918763c68; transformers 5.22.1 | Claude, 2026-10-30 | python -m mlx_vlm.generate --model mlx-community/llm-jp-4-vl-9b-mlx-4bit --revision 9c056d48b --image &lt;original 63MP JPEG, sha256 ecdd79a7...&gt; --prompt &lt;recorded prompt&gt; --max-tokens 1000 --temperature 0.0 --top-p 1.0 --top-k 0 --seed 0 --prefill-step-size 2048 --trust-remote-code reproduced the harness output byte-for-byte: '<\|channel\|> analysis<\|message\|> ...' visible in decoded text, 29 tokens; <\|channel\|>(9)/<\|message\|>(12) are special:true in tokenizer.json but are not stripped, and the processor has no clean_output/Harmony channel parser. Rendered-prompt and effective-argument digests added 2026-10-30 from the harness record, not measured natively: the native flags above are the recorded effective arguments, the llm-jp template has no date or time variable, and the output matched byte-for-byte. |

</details>

## Human accuracy notes

1 of 41 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed. Each note is a
person's judgement against the image, not one of the mechanical checks.

| Model                        | Reviewed         | Note                                                                                     |
|------------------------------|------------------|------------------------------------------------------------------------------------------|
| mlx-community/Molmo2-8B-4bit | john, 2026-10-25 | Describes the pictured tabby cats as calico cats and gives incorrect positional details. |

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 49 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | duplicate keywords: 3 runs since 2026-11-16; +2 more, newest back this run | Response repeats the same text; Response appears cut off at the token limit; Duplicate keywords: british english, british english examples of metadata, british english examples of details, british english examples of context, british english examples of information, british english examples of data, british english examples of content, british english examples of examples | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 40+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 3 runs (since 2026-11-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | back this run, first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | cut off at token limit | back this run, first 2026-10-16 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 40+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 40+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

35 completions without detected concerns; 4 more completed with answer-format or hint-reuse observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 640 x 480 pixels, 173,131 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.146
- *check_models revision:* 6f3a5278e06651d8f41e80c6f18f9540ebbab48b
- *check_models source dirty:* false
- *mlx-vlm:* 0.8.0
- *mlx-vlm source revision:* f1618d6d52be4ef25d3fe65270f7528a59c55df3
- *mlx:* 0.32.5.dev20261121+8803ac3ca
- *mlx source revision:* 8803ac3ca
- *transformers:* 5.24.0
- *macOS Version:* 27.2.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

## Full artifacts

**Evidence links** target the repository's mutable main branch: they show this
run only once these artifacts are committed, and a later run's commit replaces
them. Before sharing upstream, pin them to the commit that published these
artifacts.

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
