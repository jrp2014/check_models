# mlx-vlm compatibility findings across 41 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 2 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 31 with no mechanical observations, 8 with answer-format
or hint-reuse observations only (labels or form, or text reused from the
prompt's hints, which is allowed), and 2 with observations mlx-vlm can produce
(such as repetition, missing final answers or visible control tokens); only
those 2 are candidates for native reproduction.

## Run summary

- *Run started:* 2026-11-20 18:38:55 GMT
- *Run finished:* 2026-11-20 18:52:25 GMT
- *Run duration:* 13m 30s
- *Time by phase:* generation 662s, model load 88s, prompt prep 33s, cleanup
  5s, outside the model loop 22s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 9,984 x 6,656 pixels (66.5 MP), 49.0 MB
- *Models attempted:* 41
- *Sampling settings:* checkpoint generation_config.json values for 17 of 41
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 41
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 2
- *Reached token limit:* 2 (2 with incomplete output)
- *Stopped early for repetition:* 0

**What this run measures.** This run records model responses to one shared
image and prompt (evaluation lane: assisted). Mechanical checks are not
factual-accuracy judgments; inspect the image, prompt and final answers before
choosing a model. Results do not establish fitness for other tasks. Every
locally cached MLX vision-language model got the same image and prompt
(reproduced below) through mlx-vlm's generation pipeline.

<details>
<summary>Exact prompt sent to every model</summary>

```text
Create British-English catalogue metadata from the image and supplied context.

Treat any capture date/time and GPS as authoritative facts, but do not claim they are visible. Descriptive hints may be incomplete or wrong: retain details supported by the image, correct conflicts, and add important visible details. Prefer image evidence when a hint conflicts, and omit uncertain details.

Context: Authoritative context:
- Capture date/time: 2026-11-14 17:41:49 UTC+01:00

Descriptive hints:
- Description hint: A small mixed-breed dog wearing a harness looks attentively toward the camera while an older person relaxes in the background using a digital device in a living room.
- Keyword hints: Adobe Stock, Any Vision, Dog harness, Domestic life, Indoors, Man, Relaxing, Senior Man, Sitting, Tablet, alert, big ears, brown and black, canine, casual, chihuahua mix, close up, comfortable, companion, companion animal

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

- *Baseline:* 3ed61570:src/output/results.jsonl
- *Baseline run timestamp:* 2026-11-16 14:23:23 GMT
- *Baseline check_models:* 0.17.129 @ 5a40e3542
- *Baseline mlx:* 0.32.5.dev20261107+fc344c986 @ fc344c986
- *Baseline mlx-vlm:* 0.8.0 @ d204e74b7
- *Baseline transformers:* 5.23.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261120+bd2d1fb43

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; evaluation lane differs
(blind → assisted); image differs (sha256 dea9e7ef9738… → 6020d9876855…). The
roster, revisions and upstream changes below are facts about the runs and are
still shown.

Run environment: current run detected GPU activity outside harness inference
at the checks around 1 of 41 models (up to 14%); baseline run on battery power
for 41 of 41 models; baseline run detected GPU activity outside harness
inference at the checks around 31 of 41 models (up to 61%).

<details>
<summary>mlx: 16 upstream commit(s) since the baseline (fc344c986..bd2d1fb43)</summary>

- bd2d1fb43 [CUDA] Looped row reductions across blocks (#4916)
- 5078f27ec Fix flaky "make array from user buffer" test under heavy load
  (#4918)
- 90acc547e Add lock file for GitHub Actions (#4801)
- ebe3de603 Add bool to Metal sort kernel instantiations (#4861)
- 53353f40f Fix IndexError caused by a compiled function returning
  stop_gradient (#4880)
- 2905f33b1 [CUDA] Avoid releasing DXGI adapters during DLL teardown (#4867)
- 70d0d098c Fixed misaligned stores in `RandomBits` on the CPU (#4886)
- 1b3744cfb Do not wire user provided buffers (#4883)
- 74b1d72b1 docs: Correct the axis argument name in expand_dims (#4830)
- 8f0f391ec Fixed the build of the ring backend on FreeBSD (#4846)
- 9f6cb66a4 Fix missing API documentation and incorrect docstrings (#4813)
- ab5d01b7f Fix missing streams in l2_norm and scatter VJP (#4818)
- 07e7bdae8 Fix JIT compilation error caused by metal::remove_addrspace_t
  (#4833)
- 14dc9c673 MetalAllocator::malloc(0) leads to undefined behavior (#4881)
- fbedcee5f Fix: Respect dtype for array values in mx.full (#4901)
- ... and 1 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 48 upstream commit(s) since the baseline (d204e74b7..392052b44)</summary>

- 392052b4 Merge pull request #2770 from nassarofficial/feat/granite-docling
- bea7610f Use a model-local PIL processor for Granite Docling
- 161b9853 Merge branch 'main' into feat/granite-docling
- 6f5d742b docs: list Granite for Docling in the supported models
- e22cb59f Revert "docs(granite_docling): document the vision feature cache"
- d1099785 docs(granite_docling): document the vision feature cache
- f0e74652 feat(granite_docling): support the vision feature cache
- b561d0a4 fix(granite_docling): dense DeepStack features for batched
  generation
- 2ce3aa36 fix(utils): forward the options that the image processor declares
- e5fea63f feat(models): add Granite for Docling (granite_docling)
- a22a364a Merge pull request #2767 from Lornatang/fix/resize-image-zero-dim
- 2c3a565d Merge pull request #2766 from Lornatang/fix/legacy-learning-rate
- 12c0eb65 fix(utils): reject non-positive resize targets
- 22e2b94a Merge pull request #2765 from Lornatang/fix/min-p-zero
- f43d2cf3 Merge pull request #2771 from lucasnewman/fix-issue-2755
- ... and 33 more (see results.jsonl)

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
native confirmation. Keywords are counted from the answer's Keywords field,
with how many appear verbatim in the prompt's keyword hints. Hint text is the
percent of the description's words lying in four-word runs copied from the
prompt's description hint (of the whole answer when no field is labelled),
rounded down; 80% or more is reported as a repeated prompt hint.

| Model | Mechanical checks | Total | Gen tok/s | Peak GB | Prompt / output tok | Keywords | Hint text | Observed | Native repro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlx-community/AREX-2-4bit | no concerns detected | 82.30s | 29.3 tok/s | 21 | 16,576 / 928 | 17 (15 from hints) | 0% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 4.94s | 103 tok/s | 6.5 | 2,069 / 119 | 11 (2 from hints) | 26% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 7.51s | 54.9 tok/s | 28 | 567 / 86 | 16 (9 from hints) | 17% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 11.73s | 121 tok/s | 19 | 1,606 / 846 | 21 (11 from hints) | 12% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.37s | 63.4 tok/s | 7.6 | 571 / 100 | 18 (13 from hints) | 27% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.05s | 127 tok/s | 16 | 571 / 92 | 16 (13 from hints) | 18% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.25s | 126 tok/s | 6.0 | 567 / 105 | 14 (10 from hints) | 32% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 18.30s | 75.8 tok/s | 8.7 | 6,427 / 761 | 11 (8 from hints) | 16% | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | no concerns detected | 3.26s | 186 tok/s | 4.7 | 1,361 / 93 | 12 (9 from hints) | 65% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.41s | 55.6 tok/s | 10 | 2,087 / 112 | 20 (20 from hints) | 19% | none | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | no concerns detected | 3.33s | 207 tok/s | 4.0 | 2,087 / 79 | 13 (7 from hints) | 73% | none | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | no concerns detected | 5.49s | 307 tok/s | 3.6 | 905 / 392 | 17 (17 from hints) | 36% | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 15.42s | 83.7 tok/s | 25 | 905 / 806 | 18 (12 from hints) | 41% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.96s | 64.5 tok/s | 13 | 2,902 / 137 | 16 (7 from hints) | 29% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 3.82s | 189 tok/s | 7.7 | 2,901 / 126 | 13 (3 from hints) | 10% | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 24.46s | 76.9 tok/s | 73 | 2,381 / 116 | 17 (13 from hints) | 50% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 7.00s | 69.6 tok/s | 8.1 | 1,498 / 131 | 20 (20 from hints) | 55% | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 10.09s | 28.3 tok/s | 13 | 3,100 / 105 | 15 (9 from hints) | 0% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.34s | 212 tok/s | 3.7 | 4,060 / 127 | 18 (18 from hints) | 9% | none | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | no concerns detected | 4.30s | 55.8 tok/s | 9.3 | 1,103 / 96 | 10 (4 from hints) | 31% | none | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | no concerns detected | 24.54s | 72.3 tok/s | 26 | 12,764 / 108 | 18 (14 from hints) | 14% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 35.35s | 92.8 tok/s | 23 | 16,521 / 117 | 20 (16 from hints) | 11% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 36.25s | 69.4 tok/s | 11 | 16,521 / 104 | 18 (14 from hints) | 0% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 40.67s | 113 tok/s | 25 | 16,534 / 933 | 18 (8 from hints) | 0% | none | - |
| mlx-community/Qwen3.6-35B-A3B-4bit | no concerns detected | 48.41s | 114 tok/s | 25 | 16,534 / 927 | 20 (8 from hints) | 10% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 84.83s | 30.6 tok/s | 21 | 16,576 / 879 | 16 (7 from hints) | 9% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.94s | 130 tok/s | 5.4 | 4,184 / 140 | 18 (18 from hints) | 40% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.60s | 35.9 tok/s | 18 | 1,248 / 113 | 20 (14 from hints) | 62% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 11.19s | 164 tok/s | 23 | 3,603 / 837 | 14 (3 from hints) | 41% | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 74.90s | 30.7 tok/s | 87 | 3,602 / 928 | 17 (10 from hints) | 9% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 86.08s | 35.6 tok/s | 17 | 16,576 / 768 | 18 (16 from hints) | 11% | none | - |
| ibm-granite/granite-vision-4.1-4b | concerns detected | 6.70s | 60.6 tok/s | 9.7 | 1,361 / 91 | 13 (6 from hints) | 100% | prompt hint repeated | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | concerns detected | 2.43s | 520 tok/s | 1.9 | 2,095 / 126 | 19 (10 from hints) | 17% | duplicate keywords | - |
| mlx-community/InternVL3_5-1B-4bit | concerns detected | 2.68s | 454 tok/s | 2.1 | 2,090 / 105 | 20 (19 from hints) | 68% | duplicate keywords | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 4.16s | 386 tok/s | 1.8 | 308 / 34 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.39s | 107 tok/s | 6.7 | 2,173 / 25 | - | 0% of answer | control tokens visible; labelled fields not detected | not yet checked (earlier check out of date) |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 53.28s | 23.9 tok/s | 25 | 4,387 / 1,000 | - | 13% | labelled fields not detected; cut off at token limit | - |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.39s | 378 tok/s | 1.8 | 304 / 82 | - | 41% | labelled fields not detected | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | major concerns: answer format | 3.77s | 124 tok/s | 5.6 | 1,400 / 97 | 20 (20 from hints) | 100% | labelled fields not detected; prompt hint repeated | - |
| sahilchachra/LensVLM-9B-MXFP4 | major concerns: answer format | 6.85s | 101 tok/s | 7.4 | 1,846 / 304 | 18 (17 from hints) | 100% | answer emitted twice; prompt hint repeated | - |
| mlx-community/MiniCPM-o-4_5-4bit | major concerns: generation | 12.90s | 98.5 tok/s | 7.0 | 361 / 1,000 | - | 2% of answer | labelled fields not detected; cut off at token limit; incomplete thinking block | not yet checked |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.32.5.dev20261120+bd2d1fb43 @ bd2d1fb43; mlx-vlm 0.8.0 @ 392052b44;
transformers 5.23.0) and experiment inputs (image, prompt, generation
settings, and each model's rendered prompt and effective generation
arguments). A check made on another revision, build or experiment, or for a
model whose inputs were not recorded, does not apply and is not listed.

No recorded check applies to this run.

## Human accuracy notes

0 of 41 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed; 1 note for other
images or models. Each note is a person's judgement against the image, not one
of the mechanical checks.

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 48 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | all 2: 39+ runs since 2026-09-12 or earlier | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |
| mlx-community/MiniCPM-o-4_5-4bit | major concerns: generation | all 3: not in earlier runs read | Required labelled fields not detected: title, description, keywords; Response appears cut off at the token limit; Internal reasoning block appears incomplete | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-minicpm-o-45-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 39+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 39+ runs (since 2026-09-12 or earlier) |
| mlx-community/MiniCPM-o-4_5-4bit | labelled fields not detected | not in earlier runs read |
| mlx-community/MiniCPM-o-4_5-4bit | incomplete thinking block | not in earlier runs read |
| mlx-community/MiniCPM-o-4_5-4bit | cut off at token limit | not in earlier runs read |

</details>

## Completions without detected concerns

31 completions without detected concerns; 8 more completed with answer-format or hint-reuse observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 9,984 x 6,656 pixels, 49,027,330 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.142
- *check_models revision:* 3ed61570cb78c77df1bbec5e11d37ee9de23967f
- *check_models source dirty:* false
- *mlx-vlm:* 0.8.0
- *mlx-vlm source revision:* 392052b4439cf027ab42674d9977f62bb3dfa056
- *mlx:* 0.32.5.dev20261120+bd2d1fb43
- *mlx source revision:* bd2d1fb43
- *transformers:* 5.23.0
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
