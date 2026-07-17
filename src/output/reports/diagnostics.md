<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 1 failure(s), 6 harness issue(s), 1 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 61 locally-cached VLM model(s) checked; 1 hard failure(s), 6 harness/integration issue(s), 1 text-sanity/semantic issue(s), 0 preflight warning(s), 60 successful run(s).

Test image: `20260704-181004_DSC00862_DxO.jpg` (49.2 MB).

## Impact and Run Conditions

- _Models tested:_ 61
- _Crashed:_ 1
- _Completed:_ 60
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
  context: - Location terms: Deben Estuary, England, Europe, UK, Woodbridge -
  Capture date/time: 2026-07-04 19:10:04 BST 19:10:04 - Use this factual
  context where it improves the catalogue record; do not claim that contextual
  facts are visually observable.  Draft descriptive metadata: - Existing
  title: Deben Estuary, Woodbridge, England, UK, GBR, Europe - Existing
  description: Two sailing boats moored on a river with trees behind on the
  bank - Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary,
  Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors,
  Peaceful, Rigging, River, Riverbank, Sailboat - Treat this draft as
  fallible. Retain supported details, correct errors, and add important
  visible information.
- _Image:_ 20260704-181004_DSC00862_DxO.jpg
<!-- markdownlint-disable MD060 -->

## Crash / Failure Matrix

| Model                          | Outcome               | Phase   | Stage       | Primary exception                   | Suspected owner   |
|--------------------------------|-----------------------|---------|-------------|-------------------------------------|-------------------|
| mlx-community/gemma-4-31b-bf16 | Task outcome: crashed | decode  | Model Error | IndexError: list index out of range | model (medium)    |

<details>
<summary>Crash evidence: mlx-community/gemma-4-31b-bf16</summary>

```text
IndexError: list index out of range
```

```text
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
ValueError: Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

```text
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "~/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 171, in add_token
    if self.is_byte_token[token]:
       ~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: list index out of range
```

</details>
<!-- markdownlint-enable MD060 -->
## mlx-vlm / MLX Issue Matrix

Routing labels reflect evidence confidence only; a crash remains a conclusive
failed task at every confidence level.
<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                               | Evidence Snapshot                                                                                                                                                                                       | Affected Models                       | Confidence   | Evidence Type   | Fixed When                                                |
|------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|--------------|-----------------|-----------------------------------------------------------|
| `mlx`                                          | MLX: Decode / model error: list index out of range    | Model Error \| phase decode \| IndexError                                                                                                                                                               | 1: `mlx-community/gemma-4-31b-bf16`   | confirmed    | failure         | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt=1,179 \| output/prompt=6.87% \| stop=completed                                                                                             | 1: `mlx-community/MiniCPM-V-4.6-8bit` | confirmed    | harness         | No leaked stop/control tokens.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short | prompt_tokens=16779, repetitive output \| prompt=16,779 \| output/prompt=2.98% \| mixed burden=97% \| stop=max_tokens \| hit token cap (500) \| 4 model cluster                                         | 4: `Qwen/Qwen3-VL-2B-Instruct` (+3)   | confirmed    | context_budget  | Full and reduced reruns avoid context collapse.           |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short | output/prompt=0.1%, weak text=truncated \| prompt_tokens=16790, output_tokens=14, output/prompt=0.1%, weak text=truncated \| prompt=16,790 \| output/prompt=0.08% \| mixed burden=97% \| stop=completed | 1: `mlx-community/X-Reasoner-7B-8bit` | confirmed    | context_budget  | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## Issue 1: mlx — MLX: Decode / model error

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/gemma-4-31b-bf16
- _Expected:_ Affected reruns complete model load and generation, or fail with
  a narrower configuration/compatibility error that points to the owning
  layer.
- _Actual:_ MLX: Decode / model error: list index out of range
- _Filing target:_ mlx

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/gemma-4-31b-bf16 --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Location terms: Deben Estuary, England, Europe, UK, Woodbridge
- Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Existing description: Two sailing boats moored on a river with trees behind on the bank
- Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Image SHA256: `ca8d7f4e290d2f17ff550dd856e3cad8903013e2e0b5044bc926ee086199c806`


### Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 2: mlx-vlm — Stop-token leakage

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/MiniCPM-V-4.6-8bit
- _Expected:_ Affected reruns contain no leaked stop/control tokens and
  terminate cleanly before the configured max-token cap when the response is
  complete.
- _Actual:_ Stop/control tokens leaked into generated text
- _Filing target:_ mlx-vlm

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/MiniCPM-V-4.6-8bit --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Location terms: Deben Estuary, England, Europe, UK, Woodbridge
- Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Existing description: Two sailing boats moored on a river with trees behind on the bank
- Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Image SHA256: `ca8d7f4e290d2f17ff550dd856e3cad8903013e2e0b5044bc926ee086199c806`


### Expected Fix Signal

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 3: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

### Expected and Actual Behaviour

- _Affected models:_ Qwen/Qwen3-VL-2B-Instruct,
  mlx-community/Qwen2-VL-2B-Instruct-4bit,
  mlx-community/Qwen3-VL-2B-Instruct-bf16,
  mlx-community/Qwen3-VL-2B-Thinking-bf16
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model Qwen/Qwen3-VL-2B-Instruct --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Location terms: Deben Estuary, England, Europe, UK, Woodbridge
- Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Existing description: Two sailing boats moored on a river with trees behind on the bank
- Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Image SHA256: `ca8d7f4e290d2f17ff550dd856e3cad8903013e2e0b5044bc926ee086199c806`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 4: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/X-Reasoner-7B-8bit
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Location terms: Deben Estuary, England, Europe, UK, Woodbridge
- Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Existing description: Two sailing boats moored on a river with trees behind on the bank
- Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Image SHA256: `ca8d7f4e290d2f17ff550dd856e3cad8903013e2e0b5044bc926ee086199c806`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

<!-- markdownlint-disable MD060 -->

## Prompt / Input Burden Evidence

| Model                                   | Burden   |   Total tokens |   Text estimate |   Non-text estimate | Source            |
|-----------------------------------------|----------|----------------|-----------------|---------------------|-------------------|
| Qwen/Qwen3-VL-2B-Instruct               | mixed    |          16779 |             478 |               16301 | estimated_nontext |
| mlx-community/Qwen2-VL-2B-Instruct-4bit | mixed    |          16790 |             478 |               16312 | estimated_nontext |
| mlx-community/Qwen3-VL-2B-Instruct-bf16 | mixed    |          16779 |             478 |               16301 | estimated_nontext |
| mlx-community/Qwen3-VL-2B-Thinking-bf16 | mixed    |          16781 |             478 |               16303 | estimated_nontext |
| mlx-community/X-Reasoner-7B-8bit        | mixed    |          16790 |             478 |               16312 | estimated_nontext |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 -->

## Model/config observations

These signals are retained for model/configuration follow-up and are not
presented as confirmed mlx-vlm issues.

| Owner   | Models                                        | Observation                               | Evidence                                                                                                                                                                                | Routing   |
|---------|-----------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| model   | mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX | Generated text is mixed-script token-soup | Here are my reasoning steps:<br>We need to produce a catalog entry with Title, Description, Keywords. Use only details that are clearly visible. The image shows a river (likely Deb... | probable  |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                                    |
| mlx                        | 0.32.1.dev20260712+4367c73b                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.5                                                                                                                                                    |
| transformers               | 5.12.1                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.23.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.2                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.6                                                                                                                                                     |
| Xcode Build                | 17F113                                                                                                                                                   |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                               |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                           |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                                        |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                           |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                          |
| GPU/Chip                   | Apple M5 Max                                                                                                                                             |
| GPU Cores                  | 40                                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                                  |
| MLX Install Type           | editable local source                                                                                                                                    |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                                   |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                                       |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                             |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb)          |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=e62ebcb4631e77eda0aac74719a4b7df7639997787eb1144888ad11b02386ef6)           |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
