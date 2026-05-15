# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 3 model(s)

## Summary

3 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 3
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

| Model                                            | Observed Behavior   | Token Counts                                                                | Optional Context                                                                                                                                                                                              |
|--------------------------------------------------|---------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/FastVLM-0.5B-bf16`                | generated_tokens~9  | prompt=571 \| output/prompt=1.58% \| nontext burden=16% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_003_mlx-community_FastVLM-0.5B-bf16_model_config_mlx_vlm_prompt_template_001.json)                |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit` | output/prompt=0.5%  | prompt=1,584 \| output/prompt=0.51% \| nontext burden=70% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_015_mlx-community_paligemma2-10b-ft-docci-448-6bit_model_config_mlx_vlm_prompt_template_001.json) |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | output/prompt=0.5%  | prompt=1,584 \| output/prompt=0.51% \| nontext burden=70% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_016_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json) |


## Minimal Evidence

- `mlx-community/FastVLM-0.5B-bf16`: Output appears truncated to about 9 tokens.
- `mlx-community/FastVLM-0.5B-bf16`: Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).
- Output excerpt: `teriorGREEGREEGREEGREEGREEGREE。`
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/FastVLM-0.5B-bf16 --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/paligemma2-10b-ft-docci-448-6bit --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/paligemma2-10b-ft-docci-448-bf16 --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: The tall spire of St John the Evangelist'"'"'s Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/FastVLM-0.5B-bf16'
IMAGE = '/Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg'
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Description hint: The tall spire of St John the Evangelist's Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.\n- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors\n- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W."
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
- Description hint: The tall spire of St John the Evangelist's Church in Upper St Leonards, Dorking, England, rises against a blue sky with wispy clouds on a sunny day. The Gothic Revival church is surrounded by a tranquil green churchyard with mature trees, and a bird is captured in flight near the steeple.
- Keyword hints: Architecture, Bench, Bird, Building, Bush, Church, Churchyard, Clock tower, Clouds, Dorking, England, Europe, Flying, Gothic, Gothic Revival, Gothic Revival architecture, Grass, Landscape, Lawn, Outdoors
- Capture metadata: Taken on 2026-05-09 17:54:42 BST (at 17:54:42 local time). GPS: 51.413600°N, 0.081900°W.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/FastVLM-0.5B-bf16"
}
```

Optional advanced context:

- `mlx-community/FastVLM-0.5B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_003_mlx-community_FastVLM-0.5B-bf16_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_015_mlx-community_paligemma2-10b-ft-docci-448-6bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_016_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json)
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
| mlx             | 0.32.0.dev20260515+7b7c1240 |
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

- Output appears truncated to about 9 tokens.
- Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).

Sample output:

```text
teriorGREEGREEGREEGREEGREEGREE。
```

### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

Observed signals:

- Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).

Sample output:

```text
- Use the following metadata terms:
```

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

Observed signals:

- Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).

Sample output:

```text
- Use the following metadata terms:
```

