# \[mlx\]\[MLX: Model load / model error\] Weight/config mismatch during model load affecting 2 model(s)

## Summary

2 model(s) show **MLX: Model load / model error** that should be filed against mlx.

- **Observed problem:** Weight/config mismatch during model load
- **Target:** mlx
- **Affected models:** 2
- **Fixed when:** Load/generation completes or fails with a narrower owner.


## Affected Models

| Model                                     | Observed Behavior                   | Token Counts   | Optional Context                                                                                                                                                                                |
|-------------------------------------------|-------------------------------------|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`        | model error \| mlx model load model | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)        |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | model error \| mlx model load model | stop=exception | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_005_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json) |


## Minimal Evidence

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16` fails with: Model loading failed: Received 2 parameters not in model:
- Root exception: `builtins.ValueError`: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm.weight.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit` fails with: Model loading failed: Received 4 parameters not in model:
- Root exception: `builtins.ValueError`: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_projector.linear_1.scales,<br>multi_modal_projector.linear_2.biases,<br>multi_modal_projector.l...


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model LiquidAI/LFM2.5-VL-450M-MLX-bf16 --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A classic-style sailboat with a dark hull and wooden mast is moored in a calm estuary during low tide. The water has receded, exposing a vast expanse of green, algae-covered mudflats behind the vessel. The boat, adorned with a string of small flags, floats peacefully, waiting for the tide to rise again.
- Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local time). GPS: 52.089294°N, 1.317741°E.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Kimi-VL-A3B-Thinking-8bit --image /Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg --prompt 'Analyze this image for cataloguing metadata, using British English.

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
- Description hint: A classic-style sailboat with a dark hull and wooden mast is moored in a calm estuary during low tide. The water has receded, exposing a vast expanse of green, algae-covered mudflats behind the vessel. The boat, adorned with a string of small flags, floats peacefully, waiting for the tide to rise again.
- Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local time). GPS: 52.089294°N, 1.317741°E.' --max-tokens 500 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'LiquidAI/LFM2.5-VL-450M-MLX-bf16'
IMAGE = '/Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg'
PROMPT = 'Analyze this image for cataloguing metadata, using British English.\n\nUse only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.\n\nTreat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.\n\nReturn exactly these three sections, and nothing else:\n\nTitle:\n- 5-10 words, concrete and factual, limited to clearly visible content.\n- Output only the title text after the label.\n- Do not repeat or paraphrase these instructions in the title.\n\nDescription:\n- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.\n- Output only the description text after the label.\n\nKeywords:\n- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.\n- Output only the keyword list after the label.\n\nRules:\n- Include only details that are definitely visible in the image.\n- Reuse metadata terms only when they are clearly supported by the image.\n- If metadata and image disagree, follow the image.\n- Prefer omission to speculation.\n- Do not copy prompt instructions into the Title, Description, or Keywords fields.\n- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.\n- Do not output reasoning, notes, hedging, or extra sections.\n\nContext: Existing metadata hints (high confidence; use only when visually confirmed):\n- Description hint: A classic-style sailboat with a dark hull and wooden mast is moored in a calm estuary during low tide. The water has receded, exposing a vast expanse of green, algae-covered mudflats behind the vessel. The boat, adorned with a string of small flags, floats peacefully, waiting for the tide to rise again.\n- Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local time). GPS: 52.089294°N, 1.317741°E.'
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
- Description hint: A classic-style sailboat with a dark hull and wooden mast is moored in a calm estuary during low tide. The water has receded, exposing a vast expanse of green, algae-covered mudflats behind the vessel. The boat, adorned with a string of small flags, floats peacefully, waiting for the tide to rise again.
- Capture metadata: Taken on 2026-05-02 18:33:45 BST (at 18:33:45 local time). GPS: 52.089294°N, 1.317741°E.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 500,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260502-173345_DSC09912_DxO.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "LiquidAI/LFM2.5-VL-450M-MLX-bf16"
}
```

Optional advanced context:

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260509T112302Z_005_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Compare checkpoint keys with the selected model class and model config.
- [ ] Inspect missing/unexpected projector, scale, bias, and quantized-weight parameter names.
- [ ] Verify the model repo revision matches the mlx-vlm/mlx loader expectations.
- [ ] Reproduce after upgrading/downgrading mlx-vlm and mlx to isolate version compatibility.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260509+84961223 |
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

### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

Observed error:

```text
Model loading failed: Received 2 parameters not in model: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

Root exception:

```text
builtins.ValueError: Received 2 parameters not in model: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

Traceback tail:

```text
multi_modal_projector.layer_norm.weight.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Received 2 parameters not in model: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

Observed error:

```text
Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

Root exception:

```text
builtins.ValueError: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

