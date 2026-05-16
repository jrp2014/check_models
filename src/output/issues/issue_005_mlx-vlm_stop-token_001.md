# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 2 model(s)

## Summary

2 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 2
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

| Model                              | Observed Behavior                                                                                   | Token Counts                                                                                         | Optional Context                                                                                                                                                              |
|------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.6-27B-mxfp8`  | decoded text contains control token &lt;/think&gt; \| prompt_tokens=16730, repetitive output        | prompt=16,730 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_015_mlx-community_Qwen3.6-27B-mxfp8_mlx_vlm_stop_token_001.json)  |
| `mlx-community/X-Reasoner-7B-8bit` | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt_tokens=16715, repetitive output | prompt=16,715 \| output/prompt=2.99% \| nontext burden=97% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_016_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json) |


## Minimal Evidence

- `mlx-community/Qwen3.6-27B-mxfp8`: Special control token &lt;/think&gt; appeared in generated text.
- `mlx-community/Qwen3.6-27B-mxfp8`: At long prompt length (16730 tokens), output became repetitive.
- Output excerpt: `- Neubпіiharuga (2000000000) &lt;think&gt; &lt;think&gt; &lt;/think&gt; ``` 2500000000 ``` ``` ``` ``` 2500000000 ``` ``` ``` ``` 2500000000 ``` ``` ``` ``` 2500000000 ``` ``` ``` ``` 2500000000 ``` ``` ``` ``` 2500000000 ``` ``` ``` ``` 2500000000 ``` ``` ``` ``` 2500000000 ``` ``` ``` ``` 2500000000 ```...`
- `mlx-community/X-Reasoner-7B-8bit`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.6-27B-mxfp8 --image /Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image /Users/jrp/Pictures/Processed/20260516-133759_DSC00002_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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

MODEL = 'mlx-community/Qwen3.6-27B-mxfp8'
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
  "model": "mlx-community/Qwen3.6-27B-mxfp8"
}
```

Optional advanced context:

- `mlx-community/Qwen3.6-27B-mxfp8`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_015_mlx-community_Qwen3.6-27B-mxfp8_mlx_vlm_stop_token_001.json)
- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T223018Z_016_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json)
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

### `mlx-community/Qwen3.6-27B-mxfp8`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- At long prompt length (16730 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output became repetitive, indicating possible generation instability (token: phrase: "``` 2500000000 ``` ```...").
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output omitted required Title/Description/Keywords sections (title, description, keywords).
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

````text
- Neubпіiharuga (2000000000)

<think>

<think>

</think>

```
2500000000
```

```

```

```
2500000000
```

```

```

```
2500000000
```

```

```

```
2500000000
```

```

```

```
2500000000
```...
````

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- At long prompt length (16715 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: scenic, view, looking, through, open).
- Output became repetitive, indicating possible generation instability (token: phrase: "1.<\|endoftext\|>3 1.<\|endoftext...").
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
1.<|endoftext|>1<|endoftext|>1<|endoftext|>1<|endoftext|>The 1.<|endoftext|>The 1.<|endoftext|>3 2.<|endoftext|>3 2.<|endoftext|>The 1. The 1.<|endoftext|>The 2. The 1.<|endoftext|>The 1.<|endoftex...
```

