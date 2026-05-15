# \[mlx-vlm\]\[Tokenizer / decoding artifact\] Tokenizer decode leaked BPE/byte markers affecting 1 model(s)

## Summary

1 model(s) show **Tokenizer / decoding artifact** that should be filed against mlx-vlm.

- **Observed problem:** Tokenizer decode leaked BPE/byte markers
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** No BPE/byte markers in output.


## Affected Models

| Model                                                   | Observed Behavior                           | Token Counts                                                                                         | Optional Context                                                                                                                                                                                 |
|---------------------------------------------------------|---------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | 323 BPE space markers found in decoded text | prompt=2,899 \| output/prompt=17.25% \| nontext burden=84% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_002_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) |


## Minimal Evidence

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 323 occurrences).
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).
- Output excerpt: `Ġram,ĠĠforĠthisĠwithĠaĠwhichĠisĠaĠaĠisĠaĠaĠisĠaĠdoesnĠwhichĠisĠaĠhaveĠaĠisĠaĠwhichĠisĠaĠwhichØ¹ÙĨÙĪØ§ÙĨĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠwhichĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠandĠaĠandĠaĠandĠaĠandĠaĠandĠaĠandĠaĠasĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠi...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit --image /Users/jrp/Pictures/Processed/20260509-165442_DSC09962_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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

MODEL = 'mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit'
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
  "model": "mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit"
}
```

Optional advanced context:

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T182226Z_002_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect tokenizer decode cleanup for byte-level/BPE marker leakage.
- [ ] Compare `decode` and `batch_decode` behavior with `skip_special_tokens=True`.
- [ ] Verify processor/tokenizer config does not require model-specific cleanup flags.


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

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

Observed signals:

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 323 occurrences).
- Model output may not follow prompt or image contents (missing: Architecture, Bench, Bird, Building, Bush).
- Output contains corrupted or malformed text segments (character_loop: '#Ġa' repeated).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
Ġram,ĠĠforĠthisĠwithĠaĠwhichĠisĠaĠaĠisĠaĠaĠisĠaĠdoesnĠwhichĠisĠaĠhaveĠaĠisĠaĠwhichĠisĠaĠwhichØ¹ÙĨÙĪØ§ÙĨĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠwhichĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠisĠaĠis...
```

