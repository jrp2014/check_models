# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 2 model(s)

## Summary

2 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 2
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

| Model                                            | Observed Behavior   | Token Counts                                                                | Optional Context                                                                                                                                                                                              |
|--------------------------------------------------|---------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/llava-v1.6-mistral-7b-8bit`       | output/prompt=0.4%  | prompt=2,789 \| output/prompt=0.39% \| nontext burden=83% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T202322Z_009_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json)       |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | output/prompt=0.6%  | prompt=1,585 \| output/prompt=0.57% \| nontext burden=70% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T202322Z_010_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json) |


## Minimal Evidence

- `mlx-community/llava-v1.6-mistral-7b-8bit`: Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues.
- `mlx-community/llava-v1.6-mistral-7b-8bit`: Model output may not follow prompt or image contents (missing: Bell Tower, Blue sky, Car, Chapel, Cross).
- Output excerpt: `The image is a photograph of a church.`
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/llava-v1.6-mistral-7b-8bit --image /Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A low-angle, wide shot of St Peter'"'"'s Church in Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style church, with its tall spire and flint walls, is pictured against a bright blue sky with wispy clouds. A black car is parked in the foreground.
- Keyword hints: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel, Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
- Capture metadata: Taken on 2026-05-09 17:50:09 BST (at 17:50:09 local time). GPS: 51.215500°N, 0.798500°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/paligemma2-10b-ft-docci-448-bf16 --image /Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A low-angle, wide shot of St Peter'"'"'s Church in Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style church, with its tall spire and flint walls, is pictured against a bright blue sky with wispy clouds. A black car is parked in the foreground.
- Keyword hints: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel, Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
- Capture metadata: Taken on 2026-05-09 17:50:09 BST (at 17:50:09 local time). GPS: 51.215500°N, 0.798500°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/llava-v1.6-mistral-7b-8bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg'
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Description hint: A low-angle, wide shot of St Peter's Church in Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style church, with its tall spire and flint walls, is pictured against a bright blue sky with wispy clouds. A black car is parked in the foreground.\n- Keyword hints: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel, Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey\n- Capture metadata: Taken on 2026-05-09 17:50:09 BST (at 17:50:09 local time). GPS: 51.215500°N, 0.798500°W."
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

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: A low-angle, wide shot of St Peter's Church in Petersfield, Hampshire, England, on a sunny day. The Gothic Revival style church, with its tall spire and flint walls, is pictured against a bright blue sky with wispy clouds. A black car is parked in the foreground.
- Keyword hints: Adobe Stock, Any Vision, Bell Tower, Blue sky, Car, Chapel, Church, Cross, Daylight, Dorking, England, Europe, Fence, Gothic Architecture, Objects, Sky, Station wagon, Steeple, Stone, Surrey
- Capture metadata: Taken on 2026-05-09 17:50:09 BST (at 17:50:09 local time). GPS: 51.215500°N, 0.798500°W.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/llava-v1.6-mistral-7b-8bit"
}
```

Optional advanced context:

- `mlx-community/llava-v1.6-mistral-7b-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T202322Z_009_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T202322Z_010_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect chat template selection and rendered message roles.
- [ ] Verify image placeholder count and order match the processor config.
- [ ] Check EOS defaults and whether the template expects explicit assistant prefixes.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260510+84961223 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.14.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `mlx-community/llava-v1.6-mistral-7b-8bit`

Observed signals:

- Output is very short relative to prompt size (0.4%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Bell Tower, Blue sky, Car, Chapel, Cross).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
The image is a photograph of a church.
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

Observed signals:

- Output is very short relative to prompt size (0.6%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Bell Tower, Blue sky, Car, Chapel, Church).

Sample output:

```text
- Use only the above metadata hints.
```

