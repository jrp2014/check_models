# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 1 model(s)

## Summary

1 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 1
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

| Model                               | Observed Behavior                                                                  | Token Counts                                                                                         | Optional Context                                                                                                                                                                     |
|-------------------------------------|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit` | Output degeneration under long prompt length (incomplete_sentence: ends with 'fl') | prompt=16,804 \| output/prompt=2.98% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T003128Z_011_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_001.json) |


## Minimal Evidence

- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output degeneration under long prompt length (incomplete_sentence: ends with 'fl')
- Output excerpt: `The user wants me to analyze the provided image and generate cataloguing metadata in British English. I need to follow specific rules: - Use only details clearly visible in the image. - Do not guess or infer. - Use the provided hints as a draft but correct/omit anything not visible. - Output exactly three sections:...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --image /Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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

MODEL = 'mlx-community/Qwen3.5-9B-MLX-4bit'
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
  "model": "mlx-community/Qwen3.5-9B-MLX-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen3.5-9B-MLX-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T003128Z_011_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_001.json)
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

### Stack Signals

| Model                               |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                            | Owner           |
|-------------------------------------|--------------|--------------|-----------------|------------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit` |       16,804 |          500 | 2.98%           | Output degeneration under long prompt length (incomplete_sentence: ends with 'fl') | `mlx-vlm / mlx` |

### `mlx-community/Qwen3.5-9B-MLX-4bit`

Observed signals:

- Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'fl').
- Model refused or deflected the requested task (explicit_refusal).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).
- Output leaked reasoning or prompt-template text (description hint:).

Sample output:

```text
The user wants me to analyze the provided image and generate cataloguing metadata in British English.
I need to follow specific rules:
- Use only details clearly visible in the image.
- Do not gues...
```

