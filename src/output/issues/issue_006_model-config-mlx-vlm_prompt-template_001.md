<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 2 model(s)

## Summary

2 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 2
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                             | Observed Behavior   | Token Counts               | Optional Context                                                                                                                                                                               |
|-----------------------------------|---------------------|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/gemma-3n-E2B-4bit` | generated_tokens=0  | prompt=0 \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_010_mlx-community_gemma-3n-E2B-4bit_model_config_mlx_vlm_prompt_template_001.json) |
| `mlx-community/gemma-4-31b-bf16`  | generated_tokens=0  | prompt=0 \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_011_mlx-community_gemma-4-31b-bf16_model_config_mlx_vlm_prompt_template_001.json)  |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/gemma-3n-E2B-4bit`: No generated tokens were recorded.
- `mlx-community/gemma-4-31b-bf16`: No generated tokens were recorded.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/gemma-3n-E2B-4bit --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/gemma-4-31b-bf16 --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/gemma-3n-E2B-4bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg'
PROMPT = 'Describe this image briefly.'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 200, 'temperature': 0.0, 'prefill_step_size': 4096}
model, processor = load(MODEL, **LOAD_KWARGS)
result = generate(model, processor, PROMPT, image=IMAGE, **GENERATE_KWARGS)
print(result.text)
```

Prompt:

```text
Describe this image briefly.
```

Generation/load config:

```json
{
  "generate_kwargs": {
    "max_tokens": 200,
    "prefill_step_size": 4096,
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/gemma-3n-E2B-4bit"
}
```

Optional advanced context:

- `mlx-community/gemma-3n-E2B-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_010_mlx-community_gemma-3n-E2B-4bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/gemma-4-31b-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_011_mlx-community_gemma-4-31b-bf16_model_config_mlx_vlm_prompt_template_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Inspect chat template selection and rendered message roles.
- [ ] Verify image placeholder count and order match the processor config.
- [ ] Check EOS defaults and whether the template expects explicit assistant prefixes.


## Appendix: Environment

| Component       | Version       |
|-----------------|---------------|
| mlx-vlm         | 0.5.0         |
| mlx             | 0.31.2        |
| mlx-lm          | 0.31.3        |
| mlx-audio       | 0.4.3         |
| transformers    | 5.9.0         |
| tokenizers      | 0.22.2        |
| huggingface-hub | 1.16.1        |
| Python Version  | 3.13.13       |
| OS              | Darwin 25.5.0 |
| macOS Version   | 26.5          |
| GPU/Chip        | Apple M5 Max  |
| GPU Cores       | 40            |
| Metal Support   | Metal 4       |
| RAM             | 128.0 GB      |


## Appendix: Detailed Evidence

### `mlx-community/gemma-3n-E2B-4bit`

Observed signals:

- No generated tokens were recorded.

### `mlx-community/gemma-4-31b-bf16`

Observed signals:

- No generated tokens were recorded.

