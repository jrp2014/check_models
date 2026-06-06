<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 1 model(s)

## Summary

1 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 1
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                      | Observed Behavior   | Token Counts                                                                | Optional Context                                                                                                                                                                                        |
|--------------------------------------------|---------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/llava-v1.6-mistral-7b-8bit` | output/prompt=0.4%  | prompt=2,588 \| output/prompt=0.39% \| nontext burden=86% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_007_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output omitted required Title/Description/Keywords sections (title, description, keywords).
- Output excerpt: `The boy is on a jet ski.`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/llava-v1.6-mistral-7b-8bit --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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

Capture metadata hints: Taken on 2026-05-25 22:50:09 BST (at 22:50:09 local time).' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/llava-v1.6-mistral-7b-8bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nCapture metadata hints: Taken on 2026-05-25 22:50:09 BST (at 22:50:09 local time).'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 500, 'temperature': 0.0, 'prefill_step_size': 4096}
model, processor = load(MODEL, **LOAD_KWARGS)
result = generate(model, processor, PROMPT, image=IMAGE, **GENERATE_KWARGS)
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

Capture metadata hints: Taken on 2026-05-25 22:50:09 BST (at 22:50:09 local time).
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/llava-v1.6-mistral-7b-8bit"
}
```

Optional advanced context:

- `mlx-community/llava-v1.6-mistral-7b-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260606T195124Z_007_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect chat template selection and rendered message roles.
- [ ] Verify image placeholder count and order match the processor config.
- [ ] Check EOS defaults and whether the template expects explicit assistant prefixes.


## Appendix: Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.2                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260606+8f0e8b14                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.10.2                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.18.0                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,751,704 bytes, sha256=e50cd0c2d2ae16781f644476459cbc2ca23b0d428a897140740f86678b5e2bf5) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,675,136 bytes, sha256=8084f4df1833f78e6598f28ff40818d3c7a73343ced3f3c416b192f2292ad933)  |
| RAM                        | 128.0 GB                                                                                                                                                 |


## Appendix: Detailed Evidence

### `mlx-community/llava-v1.6-mistral-7b-8bit`

Observed signals:

- Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
The boy is on a jet ski.
```

