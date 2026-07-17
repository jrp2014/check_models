<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 0 failure(s), 7 harness issue(s), 0 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 61 locally-cached VLM model(s) checked; 0 hard failure(s), 7 harness/integration issue(s), 0 text-sanity/semantic issue(s), 0 preflight warning(s), 61 successful run(s).

Test image: `20260711-180426_DSC00975_DxO.jpg` (43.2 MB).

## Impact and Run Conditions

- _Models tested:_ 61
- _Crashed:_ 0
- _Completed:_ 61
- _Prompt:_ Analyze this image for cataloguing metadata, using British
  English.  Use only details that are clearly and definitely visible in the
  image. If a detail is uncertain, ambiguous, partially obscured, too small to
  verify, or not directly visible, leave it out. Do not guess.  Treat the
  metadata hints below as a draft catalog record. Keep only details that are
  clearly confirmed by the image, correct anything contradicted by the image,
  and add important visible details that are definitely present.  Return
  exactly these three sections, and nothing else:  Title: - 5-10 words,
  concrete and factual, limited to clearly visible content. - Output only the
  title text after the label. - Do not repeat or paraphrase these instructions
  in the title.  Description: - 1-2 factual sentences describing the main
  visible subject, setting, lighting, action, and other distinctive visible
  details. Omit anything uncertain or inferred. - Output only the description
  text after the label.  Keywords: - 10-18 unique comma-separated terms based
  only on clearly visible subjects, setting, colors, composition, and style.
  Omit uncertain tags rather than guessing. - Output only the keyword list
  after the label.  Rules: - Include only details that are definitely visible
  in the image. - Reuse metadata terms only when they are clearly supported by
  the image. - If metadata and image disagree, follow the image. - Prefer
  omission to speculation. - Do not copy prompt instructions into the Title,
  Description, or Keywords fields. - Do not infer identity, location, event,
  brand, species, time period, or intent unless visually obvious. - Do not
  output reasoning, notes, hedging, or extra sections.  Context: Authoritative
  context: - Location terms: England, Europe, Town, UK - Capture date/time:
  2026-07-11 19:04:26 BST 19:04:26 - GPS: 51.226814°N, 1.401142°E - Use this
  factual context where it improves the catalogue record; do not claim that
  contextual facts are visually observable.  Draft descriptive metadata: -
  Existing title: Seafront, Deal, England, UK, GBR, Europe - Existing
  description: A coastal view of Deal, Kent, UK, showing the shingle beach,
  sea, and seafront buildings on a partly cloudy day. - Existing keywords:
  Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People,
  Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water,
  Waterfront, Waves, architecture - Treat this draft as fallible. Retain
  supported details, correct errors, and add important visible information.
- _Image:_ 20260711-180426_DSC00975_DxO.jpg

## mlx-vlm / MLX Issue Matrix

Routing labels reflect evidence confidence only; a crash remains a conclusive
failed task at every confidence level.
<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                                    | Affected Models                                   | Confidence   | Evidence Type   | Fixed When                                          |
|----------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|--------------|-----------------|-----------------------------------------------------|
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt=1,205 \| output/prompt=6.56% \| stop=completed                                                                                          | 1: `mlx-community/MiniCPM-V-4.6-8bit`             | confirmed    | harness         | No leaked stop/control tokens.                      |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~4 \| prompt=853 \| output/prompt=0.47% \| stop=completed                                                                                                                            | 1: `mlx-community/gemma-4-31b-bf16`               | confirmed    | harness         | Requested sections render without template leakage. |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~2 \| prompt_tokens=16898, output_tokens=2, output/prompt=0.0%, weak text=truncated \| prompt=16,898 \| output/prompt=0.01% \| mixed burden=97% \| stop=completed \| 3 model cluster | 3: `Qwen/Qwen3-VL-2B-Instruct` (+2)               | confirmed    | context_budget  | Full and reduced reruns avoid context collapse.     |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | prompt_tokens=16909, repetitive output \| prompt=16,909 \| output/prompt=2.96% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 2 model cluster                                      | 2: `mlx-community/Qwen2-VL-2B-Instruct-4bit` (+1) | confirmed    | context_budget  | Full and reduced reruns avoid context collapse.     |
<!-- markdownlint-enable MD060 -->
---

## Issue 1: mlx-vlm — Stop-token leakage

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/MiniCPM-V-4.6-8bit
- _Expected:_ Affected reruns contain no leaked stop/control tokens and
  terminate cleanly before the configured max-token cap when the response is
  complete.
- _Actual:_ Stop/control tokens leaked into generated text
- _Filing target:_ mlx-vlm

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/MiniCPM-V-4.6-8bit --image 20260711-180426_DSC00975_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: England, Europe, Town, UK
- Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
- GPS: 51.226814°N, 1.401142°E
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Seafront, Deal, England, UK, GBR, Europe
- Existing description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.
- Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `b33a875fdf0b264dbfb24adaa03ac330ecede0f05832bd2bd6b0151e32d505c6`


### Expected Fix Signal

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 2: model repo first; mlx-vlm if template handling disagrees — Prompt-template / image-placeholder mismatch

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/gemma-4-31b-bf16
- _Expected:_ Affected reruns produce the requested sections without
  empty/filler output, template leakage, or image-placeholder mismatch
  symptoms.
- _Actual:_ Prompt/template output shape mismatch
- _Filing target:_ model repo first; mlx-vlm if template handling disagrees

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/gemma-4-31b-bf16 --image 20260711-180426_DSC00975_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: England, Europe, Town, UK
- Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
- GPS: 51.226814°N, 1.401142°E
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Seafront, Deal, England, UK, GBR, Europe
- Existing description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.
- Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `b33a875fdf0b264dbfb24adaa03ac330ecede0f05832bd2bd6b0151e32d505c6`


### Expected Fix Signal

- [ ] Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 3: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

### Expected and Actual Behaviour

- _Affected models:_ Qwen/Qwen3-VL-2B-Instruct,
  mlx-community/Qwen3-VL-2B-Instruct-bf16, mlx-community/X-Reasoner-7B-8bit
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model Qwen/Qwen3-VL-2B-Instruct --image 20260711-180426_DSC00975_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: England, Europe, Town, UK
- Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
- GPS: 51.226814°N, 1.401142°E
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Seafront, Deal, England, UK, GBR, Europe
- Existing description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.
- Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `b33a875fdf0b264dbfb24adaa03ac330ecede0f05832bd2bd6b0151e32d505c6`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 4: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/Qwen2-VL-2B-Instruct-4bit,
  mlx-community/Qwen3-VL-2B-Thinking-bf16
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --image 20260711-180426_DSC00975_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: England, Europe, Town, UK
- Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
- GPS: 51.226814°N, 1.401142°E
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Seafront, Deal, England, UK, GBR, Europe
- Existing description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.
- Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `b33a875fdf0b264dbfb24adaa03ac330ecede0f05832bd2bd6b0151e32d505c6`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

<!-- markdownlint-disable MD060 -->

## Prompt / Input Burden Evidence

| Model                                   | Burden   |   Total tokens |   Text estimate |   Non-text estimate | Source            |
|-----------------------------------------|----------|----------------|-----------------|---------------------|-------------------|
| Qwen/Qwen3-VL-2B-Instruct               | mixed    |          16898 |             488 |               16410 | estimated_nontext |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | mixed    |          16898 |             488 |               16410 | estimated_nontext |
| mlx-community/X-Reasoner-7B-8bit        | mixed    |          16909 |             488 |               16421 | estimated_nontext |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | mixed    |          16909 |             488 |               16421 | estimated_nontext |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | mixed    |          16900 |             488 |               16412 | estimated_nontext |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                           |
| mlx                        | 0.32.1.dev20260717+b7c3dd6d                                                                                                                     |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.4                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.24.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.5.0                                                                                                                                   |
| macOS Version              | 26.5.2                                                                                                                                          |
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
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=98e311dc5a6588305bef55d6f231605e3591120df70183b6cec2cf2d424d8362)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD060 -->
