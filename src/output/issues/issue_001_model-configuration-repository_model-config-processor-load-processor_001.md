<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model configuration/repository\]\[Model config: Processor load / processor error\] Model config: Processor load / processor error: Model preflight failed for mlx-community/Step affecting 1 model(s)

## Summary

1 model(s) show **Model config: Processor load / processor error** that should be filed against model configuration / repository.

- **Observed problem:** Model config: Processor load / processor error: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
- **Target:** model configuration / repository
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                               | Observed Behavior                                                                   | Token Counts   | Optional Context                                                                                                                                                                                 |
|-------------------------------------|-------------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Step-3.7-Flash-oQ2e` | ValueError: Loaded processor has no image_processor; expected multimodal processor. | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260721T073757Z_001_mlx-community_Step-3.7-Flash-oQ2e_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_3c.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Step-3.7-Flash-oQ2e` fails with: ValueError: Loaded processor has no image_processor; expected multimodal processor.
- Later exceptions: ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `20260718-165706_DSC01017.jpg` or replace it with an equivalent test image.
Image SHA256: `f4650e8f8fe8fc2d9926734489c5911b58364e9eba9182f75f81db6a3d1ee6c0`

## Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/Step-3.7-Flash-oQ2e --image 20260718-165706_DSC01017.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

Use authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the keyword list after the label.

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: Adobe Stock, Any Vision, England, Europe, UK
- Capture date/time: 2026-07-18 17:57:06 BST 17:57:06
- GPS: 50.817441°N, 0.134547°W
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
- Existing description: A lifeboats station with a ferris wheel in the background.
- Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = "mlx-community/Step-3.7-Flash-oQ2e"
IMAGE = "20260718-165706_DSC01017.jpg"
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nDescribe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.\n\nUse authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual; authoritative context may supply identity and location.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.\n- Output only the keyword list after the label.\n\nRules:\n- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.\n- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Authoritative context:\n- Location terms: Adobe Stock, Any Vision, England, Europe, UK\n- Capture date/time: 2026-07-18 17:57:06 BST 17:57:06\n- GPS: 50.817441°N, 0.134547°W\n- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.\n\nDraft descriptive metadata:\n- Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe\n- Existing description: A lifeboats station with a ferris wheel in the background.\n- Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna\n- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information."
LOAD_KWARGS = {"trust_remote_code": True}
GENERATE_KWARGS = {"max_tokens": 500, "temperature": 0.0, "prefill_step_size": 4096}
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

Describe visible details faithfully. If a visual detail is uncertain, ambiguous, partially obscured, or too small to verify, leave it out rather than guessing.

Use authoritative context as supplied fact, and treat the descriptive metadata as a draft catalog record. Retain draft details that are consistent with the image, correct contradictions, and add important visible details. Authoritative context may supply identity and location even when they are not visually readable.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual; authoritative context may supply identity and location.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences combining supplied authoritative context with the main visible subject, setting, lighting, action, and distinctive visible details.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms covering supplied authoritative context and clearly visible subjects, setting, colors, composition, and style.
- Output only the keyword list after the label.

Rules:
- Distinguish supplied authoritative facts from visible details; do not present contextual facts as though they were read from the image.
- Reuse draft metadata when it is consistent with the image; authoritative context does not require separate visual proof.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless supplied as authoritative context or visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Authoritative context:
- Location terms: Adobe Stock, Any Vision, England, Europe, UK
- Capture date/time: 2026-07-18 17:57:06 BST 17:57:06
- GPS: 50.817441°N, 0.134547°W
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Lifeboat Station, Southend-on-Sea, England, UK, GBR, Europe
- Existing description: A lifeboats station with a ferris wheel in the background.
- Existing keywords: Asphalt, British seaside, Cloudy, Essex, Fence, Flag, Hotel, Lifeboat Station, Modern Architecture, Objects, Overcast, Overcast Sky, Pier, RNLI, Sign, Southend-on-Sea, Thames Estuary, Waterfront, amusement ride, antenna
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
  "image": "20260718-165706_DSC01017.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Step-3.7-Flash-oQ2e"
}
```

Optional advanced context:

- `mlx-community/Step-3.7-Flash-oQ2e`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260721T073757Z_001_mlx-community_Step-3.7-Flash-oQ2e_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_3c.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect the retained exception chain, upstream boundary, and attributed package.
- [ ] Verify the requested and resolved model revisions against the captured environment.
- [ ] Reproduce with the single affected model before changing compatibility logic.
- [ ] Add a focused regression for the structured error signature when fixed.


## Appendix: Environment

| Component                  | Version                                                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.6                                                                                                                                           |
| mlx                        | 0.32.1.dev20260721+30a19f72                                                                                                                     |
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
| MLX Device                 | Apple M5 Max                                                                                                                                    |
| GPU Architecture           | applegpu_g17s                                                                                                                                   |
| Recommended Working Set    | 108 GB                                                                                                                                          |
| Fused Attention            | Available                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=179f02b689129393feff8c019e656046eda1ba0b224e6fa58a83501dd9ce959b)  |
| RAM                        | 128.0 GB                                                                                                                                        |


## Appendix: Detailed Evidence

### `mlx-community/Step-3.7-Flash-oQ2e`

Observed error:

```text
ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

Exception chain:

```text
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

Traceback:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/Step-3.7-Flash-oQ2e: Loaded processor has no image_processor; expected multimodal processor.
```

