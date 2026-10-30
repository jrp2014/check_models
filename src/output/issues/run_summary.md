# mlx-vlm compatibility findings across 43 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 2 other results need
reproducing with mlx-vlm alone before anything is reported.

**Completed models:** 30 with no mechanical observations, 11 with
prompt-compliance observations only (labels, form or copied hint text), and 2
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 2 are candidates for native
reproduction.

**Environment warnings** (from start-up checks; they can change which code
ran):

- mlx-vlm is installed at 2 versions (0.7.10, 0.7.11); which one is imported
  depends on sys.path order

## Run summary

- *Run started:* 2026-10-28 18:35:23 GMT
- *Run finished:* 2026-10-28 18:45:39 GMT
- *Run duration:* 10m 16s
- *Time by phase:* generation 475s, model load 82s, prompt prep 34s, cleanup
  5s, outside the model loop 20s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 6,656 x 9,483 pixels (63.1 MP), 60.5 MB
- *Models attempted:* 43
- *Sampling settings:* checkpoint generation_config.json values for 18 of 43
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 43
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 2
- *Reached token limit:* 0
- *Stopped early for repetition:* 1

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

## Since the baseline sweep

- *Baseline:* d170a488:src/output/results.jsonl
- *Baseline run timestamp:* 2026-10-25 01:57:57 BST
- *Baseline check_models:* 0.17.91 @ baa72d7e8
- *Baseline mlx:* 0.32.5.dev20261024+56c2bf923 @ 56c2bf923
- *Baseline mlx-vlm:* 0.7.10 @ e9af37662
- *Baseline transformers:* 5.20.2
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261028+7d2d94a55

**Not directly comparable** — output, quality and performance comparisons are
withheld because the runs differ in: prompt differs; evaluation lane differs
(blind → assisted); image differs (sha256 dea9e7ef9738… → ecdd79a7ae79…). The
roster, revisions and upstream changes below are facts about the runs and are
still shown.

Run environment: macOS changed since the baseline: 27.0.1 → 27.2; baseline run
on battery power for 43 of 43 models; baseline run detected GPU activity
outside harness inference at the checks around 3 of 43 models (up to 27%).

<details>
<summary>mlx: 21 upstream commit(s) since the baseline (56c2bf923..7d2d94a55)</summary>

- 7d2d94a55 Fix FP8 NaN preservation in to_fp8 and E8M0 scale dequantization
  (#4756)
- 32a183e12 Fix Metal and CUDA affine quantization for all-negative groups
  (#4775)
- df46f5842 Fix GPU concatenate for outputs larger than 2^31 elements (#4758)
- 0f4740dcd Use pytest to run CPU tests in parallel (#4774)
- 54a70511f [CUDA] Expose compilation targets via cuda.get_arch_list (#4761)
- c9ea4f1e9 Fix fast.cross_entropy VJP on non-contiguous Metal inputs (#4702)
- 87122b51a Fix fftn size one axis aliasing a temporary (#4705)
- 5ee1e8edf Debug time graph verification (#4711)
- a5f79fdde Add "config" module (#4647)
- c5afbf392  Fix strided_scan reading/writing out-of-bounds (#4430)
- d168c898e Simplify docs generation by ditching doxygen (#4690)
- 832991816 Cleanup metal quantized kernel instantiations (#4741)
- 7ca400331 Fix Windows CUDA 13 DLL loading (#4706)
- 946077b5f Keep `g` and `beta` in float32 in `gated_delta_update` (#4760)
- b996a43fa Fix Metal grid truncation in softmax/logsumexp/rms_norm/layer_norm
  past 2^27 rows (#4717)
- ... and 6 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm: 57 upstream commit(s) since the baseline (e9af37662..527c513a0)</summary>

- 527c513a Merge pull request #2617 from
  Blaizzy/codex/compaction-checkpoint-recovery
- ebced60b Update audio cache fixture for checkpoint keyword arguments
- 08d552a2 Reuse single-row decode caches and skip neutral penalties
- e5b6d565 Restore compacted tool evidence and checkpoint stable request
  prefixes
- f4064e95 Capture hybrid checkpoints before changing prompt suffixes
- a0b6e099 Keep fitting conversations alive after early compaction
- 7d96a647 Merge pull request #2613 from
  Blaizzy/codex/reasoning-tool-call-isolation
- dcdb3e81 Keep tool examples inside reasoning out of executable calls
- 8f8c91db Merge pull request #2612 from
  Blaizzy/codex/transient-context-compaction
- b6c430a2 Merge branch 'main' into codex/transient-context-compaction
- bf02cd02 Merge pull request #2618 from Blaizzy/pc/audio-stt-task-prompts
- 1009b626 Forward native STT prompts and audio task selection
- 9b64096e Merge pull request #2603 from
  bearcatty/fix/unlimited-ocr-visual-precision
- c49f756d Trim redundant OCR test assertions
- b84da5f1 Align OCR tests with shared model fixtures
- ... and 42 more (see results.jsonl)

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
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 6.11s | 65.1 tok/s | 9.6 | 1,430 / 109 | 20 (20 from hints) | 50% | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.38s | 484 tok/s | 1.9 | 2,104 / 69 | 8 (3 from hints) | 35% | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 51.02s | 30.3 tok/s | 21 | 16,594 / 114 | 20 (20 from hints) | 44% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.27s | 103 tok/s | 6.5 | 2,081 / 117 | 14 (8 from hints) | 36% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 8.00s | 48.4 tok/s | 28 | 586 / 76 | 14 (14 from hints) | 55% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 11.99s | 119 tok/s | 19 | 1,596 / 855 | 18 (15 from hints) | 19% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.73s | 62.7 tok/s | 7.6 | 590 / 104 | 20 (20 from hints) | 30% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 4.99s | 114 tok/s | 16 | 590 / 75 | 16 (13 from hints) | 57% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.19s | 124 tok/s | 6.0 | 586 / 83 | 14 (6 from hints) | 18% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 9.64s | 78.6 tok/s | 8.7 | 6,343 / 104 | 9 (6 from hints) | 29% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.96s | 56.7 tok/s | 10 | 2,104 / 108 | 18 (17 from hints) | 38% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.76s | 105 tok/s | 7.1 | 382 / 94 | 19 (19 from hints) | 20% | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 6.48s | 75.8 tok/s | 25 | 927 / 108 | 18 (13 from hints) | 33% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 6.83s | 65.8 tok/s | 13 | 3,046 / 119 | 16 (4 from hints) | 46% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.14s | 185 tok/s | 8.2 | 3,045 / 153 | 14 (2 from hints) | 22% | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 13.65s | 69.9 tok/s | 73 | 2,525 / 120 | 16 (8 from hints) | 13% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 6.63s | 72.2 tok/s | 8.0 | 1,318 / 133 | 20 (20 from hints) | 36% | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 10.25s | 31.2 tok/s | 12 | 3,136 / 112 | 16 (5 from hints) | 13% | none | - |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | no concerns detected | 51.60s | 24.9 tok/s | 25 | 4,367 / 973 | 18 (17 from hints) | 28% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.54s | 155 tok/s | 3.7 | 4,044 / 110 | 20 (20 from hints) | 44% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 35.05s | 78.0 tok/s | 23 | 16,578 / 155 | 19 (19 from hints) | 27% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 34.70s | 68.8 tok/s | 11 | 16,578 / 87 | 10 (3 from hints) | 40% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 32.74s | 73.3 tok/s | 25 | 16,594 / 112 | 18 (16 from hints) | 0% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 52.65s | 30.8 tok/s | 21 | 16,594 / 121 | 17 (12 from hints) | 0% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.68s | 129 tok/s | 5.4 | 4,173 / 99 | 20 (20 from hints) | 25% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.72s | 36.2 tok/s | 18 | 1,313 / 111 | 19 (10 from hints) | 50% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 7.05s | 141 tok/s | 23 | 3,641 / 135 | 20 (20 from hints) | 52% | none | - |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit | no concerns detected | 12.44s | 63.3 tok/s | 52 | 3,640 / 123 | 20 (20 from hints) | 60% | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 37.60s | 30.9 tok/s | 87 | 3,640 / 133 | 20 (20 from hints) | 44% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 61.35s | 35.4 tok/s | 17 | 16,594 / 118 | 18 (18 from hints) | 20% | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.27s | 179 tok/s | 4.6 | 1,454 / 68 | 11 (5 from hints) | 100% | prompt hint repeated | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.55s | 209 tok/s | 4.0 | 2,096 / 83 | 15 (15 from hints) | 90% | prompt hint repeated | - |
| mlx-community/MiniCPM-V-4.6-4bit | concerns detected | 6.97s | 250 tok/s | 3.2 | 925 / 752 | 21 (10 from hints) | 42% | duplicate keywords | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | concerns detected | 3.23s | 266 tok/s | 3.7 | 927 / 114 | 25 (18 from hints) | 34% | duplicate keywords | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 5.33s | 57.0 tok/s | 9.3 | 1,120 / 149 | 20 (15 from hints) | 27% | duplicate keywords | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | concerns detected | 22.36s | 70.5 tok/s | 26 | 12,683 / 90 | 18 (16 from hints) | 86% | prompt hint repeated | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.60s | 127 tok/s | 5.6 | 1,419 / 74 | 18 (18 from hints) | 100% | prompt hint repeated | - |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 19.14s | 59.4 tok/s | 14 | 16,589 / 229 | 51 (20 from hints) | 53% | duplicate keywords | - |
| sahilchachra/LensVLM-9B-MXFP4 | concerns detected | 4.59s | 101 tok/s | 7.3 | 1,802 / 83 | 15 (15 from hints) | 100% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 4.01s | 322 tok/s | 1.8 | 325 / 22 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.57s | 103 tok/s | 6.7 | 2,186 / 29 | - | 83% of answer | control tokens visible; labelled fields not detected; prompt hint repeated | not yet checked |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 2.96s | 206 tok/s | 1.8 | 321 / 148 | - | 19% | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.19s | 326 tok/s | 2.1 | 2,107 / 200 | 57 (10 from hints) | 29% | repeated text; stopped early: repeating; duplicate keywords | not yet checked |

## Human accuracy notes

0 of 43 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed; 1 note for other
images or models. Each note is a person's judgement against the image, not one
of the mechanical checks.

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 41 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | History | Observed result | Evidence |
| --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | duplicate keywords: 4 runs since 2026-10-16; +2 more, newest back this run | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: sky, sky view, skyline | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | labelled fields not detected: 32+ runs since 2026-09-12 or earlier; +2 more, newest not in earlier runs read | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords; Repeats the prompt's hint text: description | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 4 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 4 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | stopped early: repeating | back this run, first 2026-10-04 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 32+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | prompt hint repeated | not in earlier runs read |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 32+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

30 completions without detected concerns; 11 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 9,483 pixels, 60,507,861 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.97
- *check_models revision:* d170a488bfd7f56981ce2425160226e9919f57c2
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.10
- *mlx-vlm source revision:* 527c513a0768245644a70266a22126d880767b18
- *mlx:* 0.32.5.dev20261028+7d2d94a55
- *mlx source revision:* 7d2d94a55
- *transformers:* 5.21.0
- *macOS Version:* 27.2
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
