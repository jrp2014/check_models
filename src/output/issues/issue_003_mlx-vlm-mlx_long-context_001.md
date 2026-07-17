<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 3 model(s)

## Summary

3 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 3
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior                                                                                   | Token Counts                                                               | Optional Context                                                                                                                                                                           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`               | generated_tokens~2 \| prompt_tokens=16898, output_tokens=2, output/prompt=0.0%, weak text=truncated | prompt=16,898 \| output/prompt=0.01% \| mixed burden=97% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_003_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)               |
| `mlx-community/Qwen3-VL-2B-Instruct-bf16` | generated_tokens~2 \| prompt_tokens=16898, output_tokens=2, output/prompt=0.0%, weak text=truncated | prompt=16,898 \| output/prompt=0.01% \| mixed burden=97% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_005_mlx-community_Qwen3-VL-2B-Instruct-bf16_mlx_vlm_mlx_long_context_001.json) |
| `mlx-community/X-Reasoner-7B-8bit`        | generated_tokens~5 \| prompt_tokens=16909, output_tokens=5, output/prompt=0.0%, weak text=truncated | prompt=16,909 \| output/prompt=0.03% \| mixed burden=97% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_004_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json)        |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `Qwen/Qwen3-VL-2B-Instruct`: Output appears truncated to about 2 tokens.
- `Qwen/Qwen3-VL-2B-Instruct`: At long prompt length (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Output excerpt: `-`
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: Output appears truncated to about 2 tokens.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.
Use a local copy of `20260711-180426_DSC00975_DxO.jpg` or replace it with an equivalent test image.
Image SHA256: `b33a875fdf0b264dbfb24adaa03ac330ecede0f05832bd2bd6b0151e32d505c6`

## Native mlx-vlm reproduction


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

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'Qwen/Qwen3-VL-2B-Instruct'
IMAGE = '20260711-180426_DSC00975_DxO.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Authoritative context:\n- Location terms: England, Europe, Town, UK\n- Capture date/time: 2026-07-11 19:04:26 BST 19:04:26\n- GPS: 51.226814°N, 1.401142°E\n- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.\n\nDraft descriptive metadata:\n- Existing title: Seafront, Deal, England, UK, GBR, Europe\n- Existing description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.\n- Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture\n- Treat this draft as fallible. Retain supported details, correct errors, and add important visible information.'
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
- Location terms: England, Europe, Town, UK
- Capture date/time: 2026-07-11 19:04:26 BST 19:04:26
- GPS: 51.226814°N, 1.401142°E
- Use this factual context where it improves the catalogue record; do not claim that contextual facts are visually observable.

Draft descriptive metadata:
- Existing title: Seafront, Deal, England, UK, GBR, Europe
- Existing description: A coastal view of Deal, Kent, UK, showing the shingle beach, sea, and seafront buildings on a partly cloudy day.
- Existing keywords: Beach, Buildings, Cars, Coastline, Deal, Horizon, Kent, Landscape, People, Promenade, Seaside, Shore, Sitting, Sky, Swimming, Walking, Water, Waterfront, Waves, architecture
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
  "image": "20260711-180426_DSC00975_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "Qwen/Qwen3-VL-2B-Instruct"
}
```

Optional advanced context:

- `Qwen/Qwen3-VL-2B-Instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_003_Qwen_Qwen3-VL-2B-Instruct_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3-VL-2B-Instruct-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_005_mlx-community_Qwen3-VL-2B-Instruct-bf16_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260717T221600Z_004_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Appendix: Environment

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


## Appendix: Detailed Evidence

### `Qwen/Qwen3-VL-2B-Instruct`

Observed signals:

- Output appears truncated to about 2 tokens.
- At long prompt length (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Beach, Buildings, Cars, Coastline, Deal).

Sample output:

```text
-
```

### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

Observed signals:

- Output appears truncated to about 2 tokens.
- At long prompt length (16898 tokens), output stayed unusually short (2 tokens; ratio 0.0%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Beach, Buildings, Cars, Coastline, Deal).

Sample output:

```text
-
```

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- Output appears truncated to about 5 tokens.
- At long prompt length (16909 tokens), output stayed unusually short (5 tokens; ratio 0.0%; weak text signal truncated).
- Model output may not follow prompt or image contents (missing: Beach, Buildings, Cars, Coastline, Deal).

Sample output:

```text
<|im_start|>
 addCriterion
```

