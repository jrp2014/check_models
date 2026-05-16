# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 1 model(s)

## Summary

1 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 1
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

| Model                              | Observed Behavior                                                                                              | Token Counts                                                                                         | Optional Context                                                                                                                                                              |
|------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/X-Reasoner-7B-8bit` | decoded text contains control token &lt;\|endoftext\|&gt; \| prompt_tokens=16868, prompt/image context dropped | prompt=16,868 \| output/prompt=2.96% \| nontext burden=98% \| stop=max_tokens \| hit token cap (500) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_016_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json) |


## Minimal Evidence

- `mlx-community/X-Reasoner-7B-8bit`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- `mlx-community/X-Reasoner-7B-8bit`: At long prompt length (16868 tokens), output may stop following prompt/image context.
- Output excerpt: `<\|endoftext\|>1.<\|endoftext\|>1<\|endoftext\|>1<\|endoftext\|>10<\|endoftext\|>10<\|endoftext\|>10<\|endoftext\|>10<\|endoftext\|>10<\|endoftext\|>10.<\|endoftext\|>10.<\|endoftext\|>The 1.<\|endoftext\|>The 1.<\|endoftext\|>1<\|endoftext\|>100 1.<\|endoftext\|>100 - 1.<\|endoftext\|>The 1.<\|endoftext\|>The 1.<\|endoftext\|>The 1.<\|endoftext\|>The 1...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/X-Reasoner-7B-8bit --image /Users/jrp/Pictures/Processed/20260515-201714_DSC09998_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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

MODEL = 'mlx-community/X-Reasoner-7B-8bit'
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
  "model": "mlx-community/X-Reasoner-7B-8bit"
}
```

Optional advanced context:

- `mlx-community/X-Reasoner-7B-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260515T231645Z_016_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_stop_token_001.json)
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

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- At long prompt length (16868 tokens), output may stop following prompt/image context.
- Model output may not follow prompt or image contents (missing: Rochester, Castle, turns, Red, celebrate).
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
<|endoftext|>1.<|endoftext|>1<|endoftext|>1<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10<|endoftext|>10.<|endoftext|>10.<|endoftext|>The 1.<|endoftext|>The 1.<|endofte...
```

