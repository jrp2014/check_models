<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 2 model(s)

## Summary

2 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 2
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior                      | Token Counts                                                                                         | Optional Context                                                                                                                                                                           |
|-------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | prompt_tokens=16736, repetitive output | prompt=16,736 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_007_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json) |
| `mlx-community/X-Reasoner-7B-8bit`        | prompt_tokens=16736, repetitive output | prompt=16,736 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_009_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json)        |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: At long prompt length (16736 tokens), output became repetitive.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Output became repetitive, indicating possible generation instability (token: phrase: "structure, architectural detai...").
- Output excerpt: `Title: St. Peter's Church, Colchester, England Description: The exterior of the tower of St. Peter's Church in Colchester, Essex, England, with a clock and stone walls, surrounded by a fence and trees, under a blue sky with clouds. Keywords: St. Peter's Church, Colchester, England, tower, clock, stone walls, fence,...`
- `mlx-community/X-Reasoner-7B-8bit`: At long prompt length (16736 tokens), output became repetitive.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --image /Users/jrp/Pictures/Processed/20260620-195128_DSC00512-Edit.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The exterior of the tower of St Peters Church in Colchester Essex
- Keyword hints: Adobe Stock, Any Vision, Church, Clouds, Colchester, England, Entrance, Essex, Europe, Fence, High Street, Path, Rural, Shadows, Sky, Stone, Tower, Tranquil, Tree, Trees
- Capture metadata: Taken on 2026-06-20 20:51:28 BST (at 20:51:28 local time). GPS: 51.854295°N, 0.958480°E.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image /Users/jrp/Pictures/Processed/20260620-195128_DSC00512-Edit.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The exterior of the tower of St Peters Church in Colchester Essex
- Keyword hints: Adobe Stock, Any Vision, Church, Clouds, Colchester, England, Entrance, Essex, Europe, Fence, High Street, Path, Rural, Shadows, Sky, Stone, Tower, Tranquil, Tree, Trees
- Capture metadata: Taken on 2026-06-20 20:51:28 BST (at 20:51:28 local time). GPS: 51.854295°N, 0.958480°E.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen2-VL-2B-Instruct-4bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260620-195128_DSC00512-Edit.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Description hint: The exterior of the tower of St Peters Church in Colchester Essex\n- Keyword hints: Adobe Stock, Any Vision, Church, Clouds, Colchester, England, Entrance, Essex, Europe, Fence, High Street, Path, Rural, Shadows, Sky, Stone, Tower, Tranquil, Tree, Trees\n- Capture metadata: Taken on 2026-06-20 20:51:28 BST (at 20:51:28 local time). GPS: 51.854295°N, 0.958480°E.'
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
- Description hint: The exterior of the tower of St Peters Church in Colchester Essex
- Keyword hints: Adobe Stock, Any Vision, Church, Clouds, Colchester, England, Entrance, Essex, Europe, Fence, High Street, Path, Rural, Shadows, Sky, Stone, Tower, Tranquil, Tree, Trees
- Capture metadata: Taken on 2026-06-20 20:51:28 BST (at 20:51:28 local time). GPS: 51.854295°N, 0.958480°E.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260620-195128_DSC00512-Edit.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_007_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260621T001325Z_009_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Appendix: Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.3                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260621+5abdd04b                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.12.1                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.20.1                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.1                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.5                                                                                                                                                     |
| Xcode Build                | 17F42                                                                                                                                                    |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,739,544 bytes, sha256=d7cb97dee8940843711ef5df4ff68b80a6e02f98bbd9e085c6794183655c94d1) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,692,160 bytes, sha256=82404f0ce166a876abbb18627634a6fd5361cab3d161c305a7bb36ab54ab086b)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Observed signals:

- At long prompt length (16736 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "structure, architectural detai...").

Sample output:

```text
Title: St. Peter's Church, Colchester, England
Description: The exterior of the tower of St. Peter's Church in Colchester, Essex, England, with a clock and stone walls, surrounded by a fence and tr...
```

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- At long prompt length (16736 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "stepped stairway, stepped stai...").

Sample output:

```text
Title:
- St Peter's Church Tower, Colchester

Description:
- A stone tower of St Peter's Church, Colchester, stands under a clear blue sky. The tower features a clock face, a pointed window, and a...
```

