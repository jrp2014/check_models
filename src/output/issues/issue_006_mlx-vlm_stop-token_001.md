# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 1 model(s)

## Summary

1 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

| Model                               | Observed Behavior                                                                                                | Token Counts                                                                                         | Optional Context                                                                                                                                                               |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct` | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; | prompt=1,387 \| output/prompt=36.05% \| nontext burden=66% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T202322Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json) |


## Minimal Evidence

- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|end\|&gt; appeared in generated text.
- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output excerpt: `Title: Gothic Revival Church in Petersfield, Hampshire Description: A Gothic Revival style church with a tall spire and flint walls stands prominently against a bright blue sky with wispy clouds. A black car is parked in the foreground. Keywords: Gothic Revival, Church, Spire, Flint Walls, Blue Sky, Parked Car, Ha...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model microsoft/Phi-3.5-vision-instruct --image /Users/jrp/Pictures/Processed/20260509-165009_DSC09954.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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

MODEL = 'microsoft/Phi-3.5-vision-instruct'
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
  "model": "microsoft/Phi-3.5-vision-instruct"
}
```

Optional advanced context:

- `microsoft/Phi-3.5-vision-instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260510T202322Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect model EOS token IDs and tokenizer special-token mappings.
- [ ] Verify mlx-vlm stop criteria receive all configured EOS/stop tokens.
- [ ] Check `skip_special_tokens` handling during decode.
- [ ] Strip generated control tokens such as `<|end|>` and `</think>` only after confirming generation stopped at the right boundary.


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

### `microsoft/Phi-3.5-vision-instruct`

Observed signals:

- Special control token &lt;\|end\|&gt; appeared in generated text.
- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

Sample output:

```text
Title: Gothic Revival Church in Petersfield, Hampshire

Description: A Gothic Revival style church with a tall spire and flint walls stands prominently against a bright blue sky with wispy clouds....
```

