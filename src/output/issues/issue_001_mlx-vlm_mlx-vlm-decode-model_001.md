<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[mlx-vlm: Decode / model error\] mlx-vlm: Decode / model error: list index out of range affecting 1 model(s)

## Summary

1 model(s) show **mlx-vlm: Decode / model error** that should be filed against mlx-vlm.

- **Observed problem:** mlx-vlm: Decode / model error: list index out of range
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                            | Observed Behavior       | Token Counts   | Optional Context                                                                                                                                                                       |
|----------------------------------|-------------------------|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-4-31b-bf16` | list index out of range | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_007_mlx-community_gemma-4-31b-bf16_MLX_VLM_DECODE_MODEL_eddb255ac4b7.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/gemma-4-31b-bf16` fails with: Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
- Root exception: `builtins.IndexError`: list index out of range


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `20260704-181004_DSC00862_DxO.jpg` or replace it with an equivalent test image.
Image SHA256: `ca8d7f4e290d2f17ff550dd856e3cad8903013e2e0b5044bc926ee086199c806`

Native CLI:

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

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Description hint: Two sailing boats moored on a river with trees behind on the bank
- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging
- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'mlx-community/gemma-4-31b-bf16'
IMAGE = '20260704-181004_DSC00862_DxO.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe\n- Description hint: Two sailing boats moored on a river with trees behind on the bank\n- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging\n- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 500, 'temperature': 0.0, 'prefill_step_size': 4096}
model, processor = load(MODEL, **LOAD_KWARGS)
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    PROMPT,
    num_images=1,
)
if isinstance(formatted_prompt, list):
    formatted_prompt = "\n".join(str(message) for message in formatted_prompt)
result = generate(model, processor, formatted_prompt, image=IMAGE, **GENERATE_KWARGS)
print(result.text)
```

Prompt:

```text
Analyze this image for cataloguing metadata, using British English.

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

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Title hint: , Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Description hint: Two sailing boats moored on a river with trees behind on the bank
- Keyword hints: Bird, Boat, Boating, Buoy, Bushes, Coast, Deben Estuary, England, Estuary, Europe, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging
- Capture metadata: Taken on 2026-07-04 19:10:04 BST (at 19:10:04 local time).
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "20260704-181004_DSC00862_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/gemma-4-31b-bf16"
}
```

Optional advanced context:

- `mlx-community/gemma-4-31b-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260710T223701Z_007_mlx-community_gemma-4-31b-bf16_MLX_VLM_DECODE_MODEL_eddb255ac4b7.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Appendix: Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                                    |
| mlx                        | 0.32.1.dev20260710+4367c73b                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.5                                                                                                                                                    |
| transformers               | 5.13.0                                                                                                                                                   |
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
| MLX Distribution Root      | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                                       |
| MLX Core Extension         | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=e61c827cd79f978aa5eacc136f65d6dea065005787f3a1457dc9d4512d6ee9cf)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `mlx-community/gemma-4-31b-bf16`

Observed error:

```text
Model runtime error during generation for mlx-community/gemma-4-31b-bf16: [METAL] Command buffer execution failed: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

Root exception:

```text
builtins.IndexError: list index out of range
```

Traceback tail:

```text
    detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/tokenizer_utils.py", line 171, in add_token
    if self.is_byte_token[token]:
       ~~~~~~~~~~~~~~~~~~^^^^^^^
IndexError: list index out of range
```

