# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 3 model(s)

## Summary

3 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 3
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

| Model                                                   | Observed Behavior                                                             | Token Counts                                                                | Optional Context                                                                                                                                                                                         |
|---------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | generated_tokens~2 \| prompt_tokens=3581, output_tokens=2, output/prompt=0.1% | prompt=3,581 \| output/prompt=0.06% \| nontext burden=88% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_002_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_mlx_long_context_001.json) |
| `mlx-community/InternVL3-14B-8bit`                      | generated_tokens~2 \| prompt_tokens=3031, output_tokens=2, output/prompt=0.1% | prompt=3,031 \| output/prompt=0.07% \| nontext burden=86% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_004_mlx-community_InternVL3-14B-8bit_mlx_vlm_mlx_long_context_001.json)                      |
| `mlx-community/InternVL3-8B-bf16`                       | generated_tokens~3 \| prompt_tokens=3031, output_tokens=3, output/prompt=0.1% | prompt=3,031 \| output/prompt=0.10% \| nontext burden=86% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_005_mlx-community_InternVL3-8B-bf16_mlx_vlm_mlx_long_context_001.json)                       |


## Minimal Evidence

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: Output appears truncated to about 2 tokens.
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: At long prompt length (3581 tokens), output stayed unusually short (2 tokens; ratio 0.1%).
- Output excerpt: `rien`
- `mlx-community/InternVL3-14B-8bit`: Output appears truncated to about 2 tokens.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit --image /Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Rochester Castle turns Red to celebrate Medway winning its bid to the European Footballing body UEFA to become the UK'"'"'s first ever completely 100 per cent carbon neutral city
- Capture metadata: Taken on 2026-05-15 21:17:14 BST (at 21:17:14 local time). GPS: 51.396828°N, 0.501581°E.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/InternVL3-14B-8bit --image /Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Rochester Castle turns Red to celebrate Medway winning its bid to the European Footballing body UEFA to become the UK'"'"'s first ever completely 100 per cent carbon neutral city
- Capture metadata: Taken on 2026-05-15 21:17:14 BST (at 21:17:14 local time). GPS: 51.396828°N, 0.501581°E.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/InternVL3-8B-bf16 --image /Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: Rochester Castle turns Red to celebrate Medway winning its bid to the European Footballing body UEFA to become the UK'"'"'s first ever completely 100 per cent carbon neutral city
- Capture metadata: Taken on 2026-05-15 21:17:14 BST (at 21:17:14 local time). GPS: 51.396828°N, 0.501581°E.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg'
PROMPT = "Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Description hint: Rochester Castle turns Red to celebrate Medway winning its bid to the European Footballing body UEFA to become the UK's first ever completely 100 per cent carbon neutral city\n- Capture metadata: Taken on 2026-05-15 21:17:14 BST (at 21:17:14 local time). GPS: 51.396828°N, 0.501581°E."
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
- Description hint: Rochester Castle turns Red to celebrate Medway winning its bid to the European Footballing body UEFA to become the UK's first ever completely 100 per cent carbon neutral city
- Capture metadata: Taken on 2026-05-15 21:17:14 BST (at 21:17:14 local time). GPS: 51.396828°N, 0.501581°E.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit"
}
```

Optional advanced context:

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_002_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/InternVL3-14B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_004_mlx-community_InternVL3-14B-8bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/InternVL3-8B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_005_mlx-community_InternVL3-8B-bf16_mlx_vlm_mlx_long_context_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260515+7b7c1240 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.0.dev0                  |
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

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

Observed signals:

- Output appears truncated to about 2 tokens.
- At long prompt length (3581 tokens), output stayed unusually short (2 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).

Sample output:

```text
rien
```

### `mlx-community/InternVL3-14B-8bit`

Observed signals:

- Output appears truncated to about 2 tokens.
- At long prompt length (3031 tokens), output stayed unusually short (2 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).

Sample output:

```text
ertz
```

### `mlx-community/InternVL3-8B-bf16`

Observed signals:

- Output appears truncated to about 3 tokens.
- At long prompt length (3031 tokens), output stayed unusually short (3 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).

Sample output:

```text
xposite
```

