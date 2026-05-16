# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 4 model(s)

## Summary

4 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 4
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

| Model                                           | Observed Behavior   | Token Counts                                                                | Optional Context                                                                                                                                                                                             |
|-------------------------------------------------|---------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/FastVLM-0.5B-bf16`               | generated_tokens~3  | prompt=491 \| output/prompt=0.61% \| nontext burden=15% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_003_mlx-community_FastVLM-0.5B-bf16_model_config_mlx_vlm_prompt_template_001.json)               |
| `mlx-community/InternVL3-14B-8bit`              | generated_tokens~6  | prompt=2,270 \| output/prompt=0.26% \| nontext burden=82% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_004_mlx-community_InternVL3-14B-8bit_model_config_mlx_vlm_prompt_template_001.json)              |
| `mlx-community/LFM2-VL-1.6B-8bit`               | generated_tokens~6  | prompt=745 \| output/prompt=0.81% \| nontext burden=44% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_006_mlx-community_LFM2-VL-1.6B-8bit_model_config_mlx_vlm_prompt_template_001.json)               |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16` | output/prompt=0.7%  | prompt=1,513 \| output/prompt=0.73% \| nontext burden=72% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_017_mlx-community_paligemma2-3b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json) |


## Minimal Evidence

- `mlx-community/FastVLM-0.5B-bf16`: Output appears truncated to about 3 tokens.
- `mlx-community/FastVLM-0.5B-bf16`: Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output excerpt: `weights:".`
- `mlx-community/InternVL3-14B-8bit`: Output appears truncated to about 6 tokens.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/FastVLM-0.5B-bf16 --image /Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A scenic view looking through open wrought-iron gates down a paved driveway lined with wooden fences, lush green trees, and blooming flowers, leading to the grand entrance of a historic gothic-style stone abbey.
- Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local time). GPS: 50.811559°N, 1.777085°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/InternVL3-14B-8bit --image /Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A scenic view looking through open wrought-iron gates down a paved driveway lined with wooden fences, lush green trees, and blooming flowers, leading to the grand entrance of a historic gothic-style stone abbey.
- Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local time). GPS: 50.811559°N, 1.777085°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/LFM2-VL-1.6B-8bit --image /Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A scenic view looking through open wrought-iron gates down a paved driveway lined with wooden fences, lush green trees, and blooming flowers, leading to the grand entrance of a historic gothic-style stone abbey.
- Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local time). GPS: 50.811559°N, 1.777085°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/paligemma2-3b-ft-docci-448-bf16 --image /Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A scenic view looking through open wrought-iron gates down a paved driveway lined with wooden fences, lush green trees, and blooming flowers, leading to the grand entrance of a historic gothic-style stone abbey.
- Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local time). GPS: 50.811559°N, 1.777085°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/FastVLM-0.5B-bf16'
IMAGE = '/Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Description hint: A scenic view looking through open wrought-iron gates down a paved driveway lined with wooden fences, lush green trees, and blooming flowers, leading to the grand entrance of a historic gothic-style stone abbey.\n- Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local time). GPS: 50.811559°N, 1.777085°W.'
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
- Description hint: A scenic view looking through open wrought-iron gates down a paved driveway lined with wooden fences, lush green trees, and blooming flowers, leading to the grand entrance of a historic gothic-style stone abbey.
- Capture metadata: Taken on 2026-05-16 14:37:59 BST (at 14:37:59 local time). GPS: 50.811559°N, 1.777085°W.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/FastVLM-0.5B-bf16"
}
```

Optional advanced context:

- `mlx-community/FastVLM-0.5B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_003_mlx-community_FastVLM-0.5B-bf16_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/InternVL3-14B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_004_mlx-community_InternVL3-14B-8bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/LFM2-VL-1.6B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_006_mlx-community_LFM2-VL-1.6B-8bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_017_mlx-community_paligemma2-3b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json)
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
| mlx             | 0.32.0.dev20260516+7b7c1240 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.1                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.15.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `mlx-community/FastVLM-0.5B-bf16`

Observed signals:

- Output appears truncated to about 3 tokens.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).

Sample output:

```text
weights:".
```

### `mlx-community/InternVL3-14B-8bit`

Observed signals:

- Output appears truncated to about 6 tokens.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).

Sample output:

```text
prefacing the same.
```

### `mlx-community/LFM2-VL-1.6B-8bit`

Observed signals:

- Output appears truncated to about 6 tokens.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).

Sample output:

```text
arak

It is a
```

_Additional affected models are listed in the Affected Models table above._

