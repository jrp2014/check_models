<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model\]\[Text-sanity / token-soup output\] Generated text is mixed-script token-soup affecting 1 model(s)

## Summary

1 model(s) show **Text-sanity / token-soup output** that should be filed against model repository.

- **Observed problem:** Generated text is mixed-script token-soup
- **Target:** model repository
- **Affected models:** 1
- **Fixed when:** Generated text is readable natural language, not token soup.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                           | Observed Behavior                            | Token Counts                                                                                       | Optional Context                                                                                                                                                                          |
|-------------------------------------------------|----------------------------------------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` | token cap \| missing sections \| abrupt tail | prompt=3,457 \| output/prompt=14.46% \| mixed burden=86% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260713T045729Z_003_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_model_text_sanity_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output omitted required Title/Description/Keywords sections (title, keywords).
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: Output leaked reasoning or prompt-template text (here are my reasoning steps).
- Output excerpt: `Here are my reasoning steps: We need to produce a catalog entry with Title, Description, Keywords. Use only details that are clearly visible. The image shows a river (likely Deben Estuary) with two sailing boats moored. The larger boat is a sailboat with multiple masts (looks like a ketch or maybe a schooner). The h...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `20260704-181004_DSC00862_DxO.jpg` or replace it with an equivalent test image.
Image SHA256: `ca8d7f4e290d2f17ff550dd856e3cad8903013e2e0b5044bc926ee086199c806`

## Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX --image 20260704-181004_DSC00862_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX'
IMAGE = '20260704-181004_DSC00862_DxO.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Authoritative context:\n- Location terms: Deben Estuary, England, Europe, UK, Woodbridge\n- Capture date/time: 2026-07-04 19:10:04 BST 19:10:04\n- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.\n\nDraft descriptive metadata:\n- Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe\n- Existing description: Two sailing boats moored on a river with trees behind on the bank\n- Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat\n- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.'
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

Context: Authoritative context:
- Location terms: Deben Estuary, England, Europe, UK, Woodbridge
- Capture date/time: 2026-07-04 19:10:04 BST 19:10:04
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Deben Estuary, Woodbridge, England, UK, GBR, Europe
- Existing description: Two sailing boats moored on a river with trees behind on the bank
- Existing keywords: Bird, Boat, Boating, Buoy, Bushes, Coast, Estuary, Foliage, Forest, Landscape, Mast, Moored, Mudflat, Nature, Outdoors, Peaceful, Rigging, River, Riverbank, Sailboat
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.
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
  "model": "mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX"
}
```

Optional advanced context:

- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260713T045729Z_003_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_model_text_sanity_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns produce readable natural-language output without mixed-script token soup or decode artifacts.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Reproduce with the native command and confirm whether token soup appears without the check_models harness.
- [ ] Inspect tokenizer config, chat template, and decode cleanup for the model revision.
- [ ] Compare against a nearby quantization or base model to isolate model weights from tokenizer/runtime behavior.
- [ ] Add a focused regression check for mixed-script token-soup output if fixed.


## Appendix: Environment

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


## Appendix: Detailed Evidence

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

Observed signals:

- Output omitted required Title/Description/Keywords sections (title, keywords).
- Output leaked reasoning or prompt-template text (here are my reasoning steps).

Sample output:

```text
Here are my reasoning steps:
We need to produce a catalog entry with Title, Description, Keywords. Use only details that are clearly visible. The image shows a river (likely Deben Estuary) with two...
```

