# mlx-vlm compatibility findings across 43 cached vision-language models

**For mlx-vlm maintainers:** no crashes need action; 2 other results need
reproducing with mlx-vlm alone before anything is reported (all 2 unchanged
since the baseline).

**Completed models:** 29 with no mechanical observations, 12 with
prompt-compliance observations only (labels, form or copied hint text), and 2
with observations mlx-vlm can produce (such as repetition, missing final
answers or visible control tokens); only those 2 are candidates for native
reproduction.

## Run summary

- *Run started:* 2026-10-30 14:11:30 GMT
- *Run finished:* 2026-10-30 14:22:29 GMT
- *Run duration:* 10m 58s
- *Time by phase:* generation 510s, model load 86s, prompt prep 35s, cleanup
  5s, outside the model loop 22s
- *Evaluation lane:* assisted
- *Prompt hints:* the image's description and keyword hints were included in
  the prompt, so field content may be copied from them rather than seen
- *Assessment:* General checks + metadata fields and duplicate keywords;
  length limits and factual accuracy not assessed
- *Input image:* JPEG, 6,656 x 9,483 pixels (63.1 MP), 60.5 MB
- *Provenance:* run captured by check_models 0.17.98 @ b93edf990; report
  rendered by 0.17.100 @ d8c13acf7; original assessments retained
- *Models attempted:* 43
- *Sampling settings:* checkpoint generation_config.json values for 18 of 43
  completed models where the command line left them unset; harness defaults
  elsewhere
- *Completed:* 43
- *Crashed:* 0
- *Indeterminate:* 0
- *Crashes requiring action:* 0
- *Other results requiring review:* 2
- *Reached token limit:* 1 (1 with incomplete output)
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

- *Baseline:* /private/tmp/check-models-update-sweep-20261030.O28tG2/before-output/results.jsonl
- *Baseline run timestamp:* 2026-10-28 18:45:39 GMT
- *Baseline check_models:* 0.17.97 @ d170a488b
- *Baseline mlx:* 0.32.5.dev20261028+7d2d94a55 @ 7d2d94a55
- *Baseline mlx-vlm:* 0.7.10 @ 527c513a0
- *Baseline transformers:* 5.21.0
- *Baseline python:* 3.14.7
- *Baseline hardware:* Apple M5 Max, 40 GPU cores, 128.0 GB RAM
- *Earlier retained runs on this mlx version:* none recorded on
  0.32.5.dev20261030+1ba7f969a
- *Models compared:* 43
- *Identical generated text:* 42 of 43 completed in both
- *Generated text changed:* 1 models (listed below)
- *Text changed, by decoding:* greedy 0 of 25; sampled, same settings and seed
  1 of 18
- *Greedy text changed, by checkpoint quantization:* quantized 0 of 20; not
  quantized 0 of 5
- *Generation tok/s ratio (now/baseline):* 0.980 over 42 models; lowest 0.90
  (LiquidAI/LFM2.5-VL-450M-MLX-bf16), highest 1.02
  (mlx-community/Qwen3.5-35B-A3B-4bit)
- *Prefill tok/s ratio (now/baseline):* 0.984 over 43 models; lowest 0.47
  (LiquidAI/LFM2.5-VL-450M-MLX-bf16), highest 1.24
  (mlx-community/InternVL3_5-1B-4bit)
- *Throughput noise band:* fixed ±15% fallback (insufficient history)

Run environment: current run detected GPU activity outside harness inference
at the checks around 19 of 43 models (up to 46%).

| Model | Execution | Usability | Observation delta |
| --- | --- | --- | --- |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | completed | no concerns detected → major concerns | +labelled fields not detected; +cut off at token limit |

<details>
<summary>Generated text changed for 1 models</summary>

| Model | Decoding | Weights | Shared prefix (chars) | Length (chars) | Output tokens | Prompt token count |
| --- | --- | --- | --- | --- | --- | --- |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | sampled | 4-bit affine | 30 | 504 → 342 | 973 → 1,000 | same |

</details>

Largest prefill changes outside 0.85-1.15x (prefill seconds are prompt tokens
divided by prefill tok/s; one timed generation per sweep, so observations, not
demonstrated speed changes):

| Model | Baseline prefill s | Now prefill s | Prefill tok/s ratio |
| --- | --- | --- | --- |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | 0.37 | 0.78 | 0.47 |
| mlx-community/Mistral-Small-4-119B-2603-4bit | 2.81 | 4.22 | 0.67 |
| mlx-community/InternVL3_5-1B-4bit | 0.52 | 0.42 | 1.24 |
| mlx-community/MiniCPM-o-4_5-4bit | 0.44 | 0.37 | 1.19 |
| mlx-community/Qwen3.5-35B-A3B-4bit | 26.57 | 31.37 | 0.85 |

<details>
<summary>mlx: 11 upstream commit(s) since the baseline (7d2d94a55..1ba7f969a)</summary>

- 1ba7f969a [jaccl] Detect dead RDMA links instead of crashing (#4767)
- 97462e188 Build and test for macOS 27 (#4786)
- 2b44b205b ci: Use self-hosted macOS runners (#4795)
- a21f892a6 Fix complex power VJP conjugation (#4778)
- 9e2e8e233 Fix 4D per-channel gate indexing in Metal gated_delta_update
  (#4791)
- 1c141875c Keep the existing file when save, save_safetensors or save_gguf
  fails (#4773)
- b45363d69 Add optional source IP binding for ring connections (#4757)
- bdbde252d Fix SliceUpdate vmap with unbatched update (#4770)
- 49366677f [Metal] Fix precision loss in addmm (#4769)
- 232eb68db Fix wrong results in left_shift/right_shift on overflow (#4694)
- 93007e9a8 Fix gguf tensor to f16 size overflow (#4764)

</details>

<details>
<summary>mlx-vlm: 21 upstream commit(s) since the baseline (527c513a0..918763c68)</summary>

- 918763c6 Merge pull request #2626 from Blaizzy/pc/diarized-text-only-models
- ef92c146 Support text-only ASR in diarized transcription requests
- 217548e2 Merge pull request #2622 from Blaizzy/pc/raon-1-mini
- faed4826 Merge branch 'main' into pc/raon-1-mini
- a5d3132f Trim Raon README validation details
- 533f13fd Remove Raon model license file
- cdd5da9e Remove Raon source checkpoint round-trip test
- d037e65c Keep Raon coverage in shared model and source checkpoint tests
- d7c10df8 Consolidate Raon checkpoint tests and move cache coverage
- e01bd0ab Match Raon source safetensors scale metadata
- caec611d Make native source loading explicit in Raon example
- 138d7617 Merge pull request #2621 from Blaizzy/pc/lfm3-thinking-parser
- 680a8f45 Guard response parser prefixes without an assistant anchor
- 924c0b6f Load original Raon-1 Mini MXFP8 checkpoint
- 06524ddf Restore standard model resolution
- ... and 6 more (see results.jsonl)

</details>

<details>
<summary>mlx-vlm commits touching the affected models' architectures and their imports</summary>

- `mlx-community/Muse-Glimmer-30B-OptiQ-4bit` (`muse_glimmer`; imports
  `activations.py`, `base.py`, `cache.py`, `rope_utils.py`): no commits
  touched `mlx_vlm/models/muse_glimmer/` or the entries it imports

Context, not attribution: shared generation, sampling and processor code can
change a model without touching its own package.

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
| ibm-granite/granite-vision-4.1-4b | no concerns detected | 6.35s | 64.4 tok/s | 9.6 | 1,430 / 109 | 20 (20 from hints) | 50% | none | - |
| LiquidAI/LFM2.5-VL-450M-MLX-bf16 | no concerns detected | 2.90s | 438 tok/s | 1.9 | 2,104 / 69 | 8 (3 from hints) | 35% | none | - |
| mlx-community/AREX-2-4bit | no concerns detected | 54.23s | 29.7 tok/s | 21 | 16,594 / 114 | 20 (20 from hints) | 44% | none | - |
| mlx-community/aya-vision-8b-4bit | no concerns detected | 5.32s | 101 tok/s | 6.5 | 2,081 / 117 | 14 (8 from hints) | 36% | none | - |
| mlx-community/diffusiongemma-26B-A4B-it-mxfp8 | no concerns detected | 8.03s | 46.8 tok/s | 28 | 586 / 76 | 14 (14 from hints) | 55% | none | - |
| mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit | no concerns detected | 12.08s | 117 tok/s | 19 | 1,596 / 855 | 18 (15 from hints) | 19% | none | - |
| mlx-community/gemma-4-12B-it-4bit | no concerns detected | 5.73s | 62.3 tok/s | 7.6 | 590 / 104 | 20 (20 from hints) | 30% | none | - |
| mlx-community/gemma-4-26b-a4b-it-4bit | no concerns detected | 5.01s | 108 tok/s | 16 | 590 / 75 | 16 (13 from hints) | 57% | none | - |
| mlx-community/gemma-4-e4b-it-4bit | no concerns detected | 4.24s | 121 tok/s | 6.0 | 586 / 83 | 14 (6 from hints) | 18% | none | - |
| mlx-community/GLM-4.6V-Flash-4bit | no concerns detected | 10.69s | 77.6 tok/s | 8.7 | 6,343 / 104 | 9 (6 from hints) | 29% | none | - |
| mlx-community/InternVL3-14B-4bit | no concerns detected | 6.84s | 56.5 tok/s | 10 | 2,104 / 108 | 18 (17 from hints) | 38% | none | - |
| mlx-community/MiniCPM-o-4_5-4bit | no concerns detected | 3.62s | 104 tok/s | 7.1 | 382 / 94 | 19 (19 from hints) | 20% | none | - |
| mlx-community/MiniCPM-V-4.7-35B-A3B-4bit | no concerns detected | 6.33s | 76.1 tok/s | 25 | 927 / 108 | 18 (13 from hints) | 33% | none | - |
| mlx-community/Ministral-3-14B-Instruct-2512-mxfp4 | no concerns detected | 7.13s | 64.5 tok/s | 13 | 3,046 / 119 | 16 (4 from hints) | 46% | none | - |
| mlx-community/Ministral-3-3B-Instruct-2512-4bit | no concerns detected | 4.17s | 183 tok/s | 8.2 | 3,045 / 153 | 14 (2 from hints) | 22% | none | - |
| mlx-community/Mistral-Small-4-119B-2603-4bit | no concerns detected | 17.22s | 69.0 tok/s | 73 | 2,525 / 120 | 16 (8 from hints) | 13% | none | - |
| mlx-community/Molmo2-8B-4bit | no concerns detected | 6.63s | 70.8 tok/s | 8.0 | 1,318 / 133 | 20 (20 from hints) | 36% | none | - |
| mlx-community/MolmoPoint-8B-4bit | no concerns detected | 10.10s | 30.2 tok/s | 13 | 3,136 / 112 | 16 (5 from hints) | 13% | none | - |
| mlx-community/North-Micro-Vision-Instruct-4bit | no concerns detected | 5.66s | 148 tok/s | 3.7 | 4,044 / 110 | 20 (20 from hints) | 44% | none | - |
| mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit | no concerns detected | 36.41s | 78.0 tok/s | 23 | 16,578 / 155 | 19 (19 from hints) | 27% | none | - |
| mlx-community/Qwen3-VL-8B-Instruct-4bit | no concerns detected | 37.49s | 67.5 tok/s | 11 | 16,578 / 87 | 10 (3 from hints) | 40% | none | - |
| mlx-community/Qwen3.5-35B-A3B-4bit | no concerns detected | 37.52s | 75.1 tok/s | 25 | 16,594 / 112 | 18 (16 from hints) | 0% | none | - |
| mlx-community/Qwen3.8-27B-nvfp4 | no concerns detected | 58.39s | 29.8 tok/s | 21 | 16,594 / 121 | 17 (12 from hints) | 0% | none | - |
| nativ-community/Mage-VL-OptiQ-4bit | no concerns detected | 4.89s | 127 tok/s | 5.4 | 4,173 / 99 | 20 (20 from hints) | 25% | none | - |
| nativ-community/Mistral-Small-3.2-24B-Instruct-2506-4bit | no concerns detected | 7.86s | 35.3 tok/s | 18 | 1,313 / 111 | 19 (10 from hints) | 50% | none | - |
| nativ-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-4bit | no concerns detected | 7.20s | 139 tok/s | 23 | 3,641 / 135 | 20 (20 from hints) | 52% | none | - |
| nativ-community/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-4bit | no concerns detected | 12.94s | 63.8 tok/s | 52 | 3,640 / 123 | 20 (20 from hints) | 60% | none | - |
| nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 | no concerns detected | 41.40s | 30.3 tok/s | 87 | 3,640 / 133 | 20 (20 from hints) | 44% | none | - |
| TechnoBaptist/Ternary-Bonsai-2-27B-mlx-2bit | no concerns detected | 66.45s | 33.3 tok/s | 17 | 16,594 / 118 | 18 (18 from hints) | 20% | none | - |
| mlx-community/granite-4.0-3b-vision-4bit | concerns detected | 3.39s | 179 tok/s | 4.6 | 1,454 / 68 | 11 (5 from hints) | 100% | prompt hint repeated | - |
| mlx-community/LFM2.5-VL-3B-OptiQ-4bit | concerns detected | 3.69s | 208 tok/s | 4.0 | 2,096 / 83 | 15 (15 from hints) | 90% | prompt hint repeated | - |
| mlx-community/MiniCPM-V-4.6-4bit | concerns detected | 6.87s | 244 tok/s | 3.2 | 925 / 752 | 21 (10 from hints) | 42% | duplicate keywords | - |
| mlx-community/MiniCPM-V-4.7-1B-4bit | concerns detected | 3.37s | 254 tok/s | 3.7 | 927 / 114 | 25 (18 from hints) | 34% | duplicate keywords | - |
| mlx-community/Phi-3.5-vision-instruct-bf16 | concerns detected | 5.59s | 52.9 tok/s | 9.3 | 1,120 / 149 | 20 (15 from hints) | 27% | duplicate keywords | - |
| mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit | concerns detected | 23.68s | 66.0 tok/s | 26 | 12,683 / 90 | 18 (16 from hints) | 86% | prompt hint repeated | - |
| mlx-community/SmolVLM2-2.2B-Instruct-mlx | concerns detected | 3.64s | 123 tok/s | 5.6 | 1,419 / 74 | 18 (18 from hints) | 100% | prompt hint repeated | - |
| mlx-community/X-Reasoner-7B-8bit | concerns detected | 20.79s | 58.6 tok/s | 14 | 16,589 / 229 | 51 (20 from hints) | 53% | duplicate keywords | - |
| sahilchachra/LensVLM-9B-MXFP4 | concerns detected | 5.13s | 98.1 tok/s | 7.3 | 1,802 / 83 | 15 (15 from hints) | 100% | prompt hint repeated | - |
| mlx-community/FastVLM-0.5B-bf16 | major concerns: answer format | 4.15s | 301 tok/s | 1.8 | 325 / 22 | - | 100% of answer | labelled fields not detected; prompt hint repeated | - |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | 3.62s | 101 tok/s | 6.7 | 2,186 / 29 | - | 83% of answer | control tokens visible; labelled fields not detected; prompt hint repeated | confirmed |
| mlx-community/Muse-Glimmer-30B-OptiQ-4bit | major concerns: answer format | 53.80s | 24.2 tok/s | 25 | 4,367 / 1,000 | - | 10% | labelled fields not detected; cut off at token limit | - |
| mlx-community/nanoLLaVA-1.5-4bit | major concerns: answer format | 3.03s | 203 tok/s | 1.5 | 321 / 148 | - | 19% | labelled fields not detected | - |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | 3.14s | 326 tok/s | 2.1 | 2,107 / 200 | 57 (10 from hints) | 29% | repeated text; stopped early: repeating; duplicate keywords | not yet checked |

## Native reproduction checks

Recorded in native_repro_notes.yaml for this run's model revisions, builds
(mlx 0.32.5.dev20261030+1ba7f969a @ 1ba7f969a; mlx-vlm 0.7.11 @ 918763c68;
transformers 5.22.1) and experiment inputs (image, prompt, generation
settings). A check made on another revision, build or experiment no longer
applies and is not listed.

| Model                                 | Observation            | Native    | Checked            | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|---------------------------------------|------------------------|-----------|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | confirmed | Claude, 2026-10-30 | python -m mlx_vlm.generate --model mlx-community/llm-jp-4-vl-9b-mlx-4bit --revision 9c056d48b --image &lt;original 63MP JPEG, sha256 ecdd79a7...&gt; --prompt &lt;recorded prompt&gt; --max-tokens 1000 --temperature 0.0 --top-p 1.0 --top-k 0 --seed 0 --prefill-step-size 2048 --trust-remote-code reproduced the harness output byte-for-byte: '<\|channel\|> analysis<\|message\|> ...' visible in decoded text, 29 tokens; <\|channel\|>(9)/<\|message\|>(12) are special:true in tokenizer.json but are not stripped, and the processor has no clean_output/Harmony channel parser. |

## Human accuracy notes

0 of 43 outputs reviewed by a person for this exact image, model revision and
output (accuracy_notes.yaml); the others are unreviewed; 1 note for other
images or models. Each note is a person's judgement against the image, not one
of the mechanical checks.

## Completed attempts requiring review

*History* dates each observation (a crash counts as one) over this model's
retained runs: 42 earlier retained runs from git
`HEAD:src/output/results.jsonl`, back to 2026-08-30; older runs were not read.
A run that did not attempt the model is skipped, and a report-only correction
of a run counts once. "Last N runs" and "N runs since" count consecutive runs
ending with this one; "first" is the earliest run read that showed it. The
table shows each model's longest-running observation; every observation's
dates are under *History by observation*.

| Model | Mechanical checks | Since baseline | History | Observed result | Evidence |
| --- | --- | --- | --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | major concerns: generation | unchanged | duplicate keywords: 5 runs since 2026-10-16; +2 more, newest 2 runs since 2026-10-28 | Response repeats the same text; Generation was stopped early after sustained repeated output; Duplicate keywords: sky, sky view, skyline | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-internvl35-1b-4bit) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | major concerns: answer format | unchanged | labelled fields not detected: 33+ runs since 2026-09-12 or earlier; +2 more, newest 2 runs since 2026-10-28 | Unrecognised model control tokens remain visible; Required labelled fields not detected: title, description, keywords; Repeats the prompt's hint text: description | [diagnostics](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md#diagnostic-mlx-community-llm-jp-4-vl-9b-mlx-4bit) |

<details>
<summary>History by observation</summary>

| Model | Observation | Persistence |
| --- | --- | --- |
| mlx-community/InternVL3_5-1B-4bit | duplicate keywords | last 5 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | repeated text | last 5 runs (since 2026-10-16), first 2026-10-04 |
| mlx-community/InternVL3_5-1B-4bit | stopped early: repeating | last 2 runs (since 2026-10-28), first 2026-10-04 |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | labelled fields not detected | last 33+ runs (since 2026-09-12 or earlier) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | prompt hint repeated | last 2 runs (since 2026-10-28) |
| mlx-community/llm-jp-4-vl-9b-mlx-4bit | control tokens visible | last 33+ runs (since 2026-09-12 or earlier) |

</details>

## Completions without detected concerns

29 completions without detected concerns; 12 more completed with prompt-compliance observations only (not maintainer issues). See the [full model gallery](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md).

## Run context

- *Image:* JPEG, 6,656 x 9,483 pixels, 60,507,861 bytes
- *Generation: max_tokens:* 1000
- *Generation: prefill_step_size:* 2048
- *Generation: seed:* 0
- *Trust remote code:* true
- *check_models version:* 0.17.98
- *check_models revision:* b93edf9909143aacc63836325c85cc65830dbd70
- *check_models source dirty:* false
- *mlx-vlm:* 0.7.11
- *mlx-vlm source revision:* 918763c6875c80e82f9f48344dfad4763346fea8
- *mlx:* 0.32.5.dev20261030+1ba7f969a
- *mlx source revision:* 1ba7f969a
- *transformers:* 5.22.1
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
