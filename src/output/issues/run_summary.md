# mlx-vlm compatibility findings across 51 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 3 other results need
reproducing with mlx-vlm alone before anything is reported (2 of 3 unchanged
since the baseline).

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

## Run summary

- *Run started:* 2026-10-02 14:01:35 BST
- *Run finished:* 2026-10-02 14:15:16 BST
- *Run duration:* 13m 40s
- *Time by phase:* generation 670s, model load 92s, prompt prep 40s, cleanup
  6s, outside the model loop 18s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,805 x 6,538 pixels (64.1 MP), 54.2 MB
- *Models attempted:* 51
- *Sampling settings:* checkpoint generation_config.json values for 25 of 51
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 51
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 3
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 0

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-09-26 17:36:29 UTC+01:00
- GPS: 50.682100°N, 3.466600°W

Descriptive hints:
- Description hint: A blue and light blue motor cabin cruiser with a tan canopy moored at a marina dock alongside other sailboats and leisure boats on calm water, casting clear reflections in the early evening light.
- Keyword hints: Boat, Boat canopy, Boat fender, Boating, Cabin cruiser, Calm Water, Dock, Harbor, Marina, Mast, Mooring, Motorboat, Nautical, Reflection, Rope, Sailboat, Sailing, Water reflection, Watercraft, Yacht

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

## Since the baseline sweep

- *Baseline:* 2a6bcd0b:src/output/results.jsonl
- *Baseline run timestamp:* 2026-09-27 23:23:22 BST
- *Baseline check_models:* 0.17.38 @ 602d1a39d
- *Baseline mlx:* 0.32.3.dev20260927+02ce1fb6a @ 02ce1fb6a
- *Baseline mlx-vlm:* 0.7.3 @ 967bf90b8
- *Baseline transformers:* 5.17.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Models compared:* 50
- *Identical generated text:* 45 of 50 completed in both
- *Generated text changed:* mlx-community/granite-4.0-3b-vision-4bit,
  mlx-community/gemma-3n-E4B-it-4bit,
  nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit,
  mlx-community/Kimi-VL-A3B-Thinking-2506-8bit,
  mlx-community/Muse-Glimmer-30B-OptiQ-4bit
- *Text changed, by decoding:* greedy 2 of 27; sampled, same settings and seed
  3 of 23
- *Generation tok/s ratio (now/baseline):* 1.008 (range 0.95-1.11, 50 models)
- *Prefill tok/s ratio (now/baseline):* 0.998 (range 0.85-1.17, 50 models)
- *Throughput noise band:* history (last 5 same-prompt runs, Tukey fence, at
  least ±10% of the median)

Run environment: macOS changed 27.0 -&gt; 27.0.1: the first run after an OS
upgrade compiles Metal pipelines cold, so prefill, time-to-first-token and
short-generation throughput are not comparable until a second run.

- New this run (no baseline): `mlx-community/InternVL3_5-1B-4bit`

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | major concerns | +duplicate keywords |

<details>
<summary>mlx: 50 upstream commit(s) since the baseline (02ce1fb6a..255328713)</summary>

- 255328713 Fix vmap of scatter placing the batch axis wrongly in the updates
  (#4322)
- f4232c7f8 [jaccl] second cable for the mesh (#4607)
- 1bed2e231 Fix complex to bool (#4593)
- 5ff1905b7 Fix conjugation in complex VJPs for inverse trig ops, cos, sqrt,
  and rsqrt (#4606)
- 2ae29a98a Reject non-aligned pointers when converting to mx.array (#4592)
- 831164aca Fix as_strided contiguity (#4575)
- 908148ee9 Sanitize '-' in make_template_hash for metal kernel (#4584)
- 34635d4f1 Fix fft size one (#4591)
- 10f116177 Fix MultiOptimizer crash when an optimizer's group is empty
  (#4570)
- c63c49dc6 docs: Correct the export callback type in the guide (#4578)
- 0840c42a7 Fix reduction over large arrays (#4549)
- 8803e5245 Do not cache failed distributed backend initialization (#4558)
- 6a1b7f778 Fix row/col contiguous flags in view (#4588)
- 99133e98c python: Fix bare ellipsis \_\_setitem\_\_ by squeezing leading
  singleton dimensions (#4561)
- 5c89fa11e [CUDA] Add gather_qmm_rhs_sm80 (#4554)
- ... and 35 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 13 upstream commit(s) since the baseline (967bf90b8..66e68ce36)</summary>

- 66e68ce3 Stop gradients on MoE router selection indices (MLX &gt;= 0.32.1)
  (#2409)
- d73f4b1a Keep the most likely MoG component for tiny top_p (#2405)
- 06ff63d6 Add Laya on the shared decision API (#2397)
- 42d5ef1b Add Decider and the shared typed decision API (#2396)
- ad6f4cfb Use a Qwen-compatible tokenizer in the processor regression test
  (#2401)
- fdcdaf46 Review/extraction test harness (#2394)
- c8da6659 Enable Qwen3-Omni multimodal batching (#2349)
- 00093678 Merge pull request #2388 from lucasnewman/version-074
- c0039a0a DeepSeek V4.1 vision and language model support (#2214)
- 539bf174 Support/internvl3 5 (#2375)
- 4e19f062 Bump version to 0.7.4.
- 5fe604cf Merge pull request #2385 from yaanfpv/fix/gemma4-audio-mask
- 1784a2e9 Fix Gemma 4 audio encoder attending one frame too far into the past

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures</summary>

- `mlx-community/granite-4.0-3b-vision-4bit` (`granite4_vision`): no commits
  touched `mlx_vlm/models/granite4_vision/`
- `mlx-community/gemma-3n-E4B-it-4bit` (`gemma3n`): no commits touched
  `mlx_vlm/models/gemma3n/`
- `nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit`
  (`nemotron_h_nano_omni`): no commits touched
  `mlx_vlm/models/nemotron_h_nano_omni/`
- `mlx-community/Kimi-VL-A3B-Thinking-2506-8bit` (`kimi_vl`): 66e68ce3 Stop
  gradients on MoE router selection indices (MLX &gt;= 0.32.1) (#2409)
- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`): no commits
  touched `mlx_vlm/models/muse_glimmer/`

Context, not attribution: shared generation, sampling and processor code can
change a model without touching its own package.

</details>

Mechanical diff only: one image, temperature as configured; single-observation
flips on one model are usually run-to-run variance, broad shifts are not.

## Model quality at a glance

Every attempted model, ordered by its mechanical checks, with counted facts.
"No concerns detected" is not an accuracy verdict: read the final answers in
the gallery. Prompt tokens include the image tokens, which drive prefill time.
Keywords are counted from the answer's Keywords field, with how many appear
verbatim in the prompt's keyword hints.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt tok | Keywords | Observed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.15s | 102 tok/s | 6.5 | 2,098 | 17 (10 from hints) | none |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 6.50s | 59.0 tok/s | 28 | 597 | 14 (12 from hints) | none |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 8.66s | 108 tok/s | 19 | 1,646 | 15 (12 from hints) | none |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.33s | 60.2 tok/s | 7.6 | 601 | 20 (20 from hints) | none |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.16s | 106 tok/s | 16 | 601 | 17 (15 from hints) | none |
| mlx-community/gemma-4-31b-it-4bit | no concerns detected | 9.60s | 25.3 tok/s | 20 | 601 | 17 (17 from hints) | none |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.35s | 161 tok/s | 4.6 | 1,384 | 14 (8 from hints) | none |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.05s | 55.9 tok/s | 10 | 2,120 | 20 (20 from hints) | none |
| mlx-community/InternVL3-8B-bf16 | no concerns detected | 6.03s | 36.8 tok/s | 17 | 2,120 | 13 (13 from hints) | none |
| mlx-community/InternVL3_5-1B-4bit | no concerns detected | 2.79s | 335 tok/s | 2.1 | 2,123 | 23 (19 from hints) | none |
| mlx-community/Llama-3.2-11B-Vision-Instruct-8bit | no concerns detected | 9.90s | 21.6 tok/s | 15 | 309 | 10 (2 from hints) | none |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.40s | 105 tok/s | 7.0 | 398 | 16 (11 from hints) | none |
| mlx-community/MiniCPM-V-4.6-4bit | no concerns detected | 5.05s | 239 tok/s | 3.2 | 938 | 17 (12 from hints) | none |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.89s | 64.6 tok/s | 13 | 2,935 | 20 (7 from hints) | none |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.08s | 186 tok/s | 7.8 | 2,934 | 13 (2 from hints) | none |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 5.87s | 71.9 tok/s | 8.5 | 1,531 | 36 (19 from hints) | none |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.48s | 155 tok/s | 3.9 | 4,091 | 19 (19 from hints) | none |
| mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit | no concerns detected | 6.33s | 74.4 tok/s | 24 | 1,295 | 18 (18 from hints) | none |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | no concerns detected | 32.55s | 80.5 tok/s | 8.4 | 16,556 | 16 (14 from hints) | none |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 44.60s | 74.4 tok/s | 23 | 16,554 | 20 (20 from hints) | none |
| mlx-community/Qwen3-VL-32B-Instruct-4bit | no concerns detected | 82.26s | 18.6 tok/s | 26 | 16,554 | 18 (17 from hints) | none |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 42.13s | 67.5 tok/s | 11 | 16,554 | 18 (15 from hints) | none |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 40.27s | 75.2 tok/s | 25 | 16,569 | 19 (19 from hints) | none |
| mlx-community/Qwen3.5-9B-MLX-4bit | no concerns detected | 41.41s | 89.8 tok/s | 11 | 16,569 | 15 (7 from hints) | none |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 61.97s | 27.5 tok/s | 21 | 16,569 | 16 (5 from hints) | none |
| nativ-community/MiMo-V2.6-Distill-Qwen-9B-MLX-4bit | no concerns detected | 38.12s | 85.3 tok/s | 11 | 16,566 | 15 (1 from hints) | none |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 8.70s | 36.2 tok/s | 18 | 1,281 | 18 (11 from hints) | none |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 9.80s | 127 tok/s | 23 | 3,636 | 17 (16 from hints) | none |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 2.23s | 481 tok/s | 1.9 | 2,123 | 14 (8 from hints) | duplicate keywords; prompt hint repeated |
| mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit | concerns detected | 10.39s | 30.0 tok/s | 23 | 2,402 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/gemma-3-27b-it-qat-4bit | concerns detected | 8.58s | 31.2 tok/s | 17 | 596 | 18 (14 from hints) | unsupplied place name |
| mlx-community/gemma-4-e4b-it-4bit | concerns detected | 4.00s | 122 tok/s | 5.9 | 597 | 15 (14 from hints) | duplicate keywords |
| mlx-community/GLM-4.6V-Flash-4bit | concerns detected | 8.81s | 75.7 tok/s | 8.7 | 6,393 | 9 (7 from hints) | prompt hint repeated |
| mlx-community/GLM-4.6V-nvfp4 | concerns detected | 21.21s | 41.0 tok/s | 78 | 6,393 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Idefics3-8B-Llama3-bf16 | concerns detected | 7.39s | 34.9 tok/s | 18 | 2,620 | 10 (1 from hints) | prompt hint repeated |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 4.05s | 57.5 tok/s | 9.3 | 1,144 | 10 (8 from hints) | prompt hint repeated |
| mlx-community/pixtral-12b-8bit | concerns detected | 8.07s | 39.7 tok/s | 16 | 3,125 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/Qwen2-VL-7B-Instruct-4bit | concerns detected | 48.09s | 87.4 tok/s | 9.3 | 16,565 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.49s | 128 tok/s | 5.6 | 1,438 | 19 (19 from hints) | prompt hint repeated |
| mlx-community/Step-3.7-Flash-oQ3e | concerns detected | 35.12s | 47.9 tok/s | 92 | 3,498 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 20.44s | 56.7 tok/s | 14 | 16,565 | 33 (18 from hints) | duplicate keywords |
| nativ-community/Mage-VL-OptiQ-4bit | concerns detected | 4.73s | 126 tok/s | 5.4 | 4,217 | 20 (20 from hints) | prompt hint repeated |
| mlx-community/FastVLM-0.5B-bf16 | major concerns | 3.11s | 335 tok/s | 2.2 | 341 | - | labelled fields not detected |
| mlx-community/gemma-3n-E4B-it-4bit | major concerns | 5.10s | 70.8 tok/s | 7.2 | 595 | - | labelled fields not detected |
| mlx-community/granite-vision-3.2-2b-nvfp4 | major concerns | 6.01s | 130 tok/s | 4.4 | 5,666 | 20 (20 from hints) | labelled fields not detected |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | 20.47s | 62.1 tok/s | 20 | 1,331 | 126 (19 from hints) | repeated text; labelled fields not detected; cut off at token limit; incomplete thinking block; duplicate keywords |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | major concerns | 3.73s | 210 tok/s | 4.0 | 2,115 | - | labelled fields not detected |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | 3.62s | 107 tok/s | 6.7 | 2,201 | - | control tokens visible; labelled fields not detected |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | 61.12s | 21.1 tok/s | 25 | 4,410 | 13 (0 from hints) | control tokens visible; labelled fields not detected; cut off at token limit; role tokens visible; duplicate keywords |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns | 2.00s | 207 tok/s | 1.8 | 337 | - | labelled fields not detected |
| mlx-community/SmolVLM-256M-Instruct-4bit | major concerns | 2.37s | 325 tok/s | 1.1 | 1,217 | - | labelled fields not detected |

## Completed attempts requiring review

| Model | Mechanical checks | Since baseline | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/Kimi-VL-A3B-Thinking-2506-8bit | major concerns | unchanged | Response repeats the same text; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Internal reasoning block appears incomplete; Duplicate keywords: boat, boat canopy, boat fender, boating, cabin cruiser, calm water, dock, harbor, marina, mast, nautical, reflection, rope, sailboat, sailing, water reflection, watercraft, mooring, motorboat, yacht wait | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-kimi-vl-a3b-thinking-2506-8bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns | unchanged | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns | changed | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description; Response appears cut off at the token limit; Conversation-role control tokens remain visible; Duplicate keywords: setting, action, lighting | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-muse-glimmer-30b-optiq-4bit) |

## Completions without detected concerns

28 completions without detected concerns; 20 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,805 x 6,538 pixels, 54,173,041 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.38
- *check_models revision:* 2a6bcd0b274a5ceb84a7dc7bd62de27b6d5536dc
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.4
- *mlx-vlm source revision:* 66e68ce36816cb5134acba3b0d72c6271f35ba24
- *mlx:* 0.32.4.dev20261002+255328713
- *mlx source revision:* 255328713
- *transformers:* 5.18.0
- *macOS Version:* 27.0.1
- *GPU/Chip:* Apple M5 Max
- *Python Version:* 3.14.7

GitHub links target the repository's mutable main branch; they resolve to this
run's evidence only once these artifacts are committed, and a later run's
commit supersedes them. Pin links to that artifact commit when durable issue
evidence is required.

## Full artifacts

| Artifact | Link |
| --- | --- |
| Diagnostics | [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md) |
| Model gallery | [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md) |
| Results JSONL | [results.jsonl](https://github.com/jrp2014/check_models/blob/main/src/output/results.jsonl) |
| Environment | [environment.log](https://github.com/jrp2014/check_models/blob/main/src/output/environment.log) |
| Log | [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log) |
