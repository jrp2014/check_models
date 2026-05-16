# \[model-config / mlx-vlm\]\[Prompt-template / image-placeholder mismatch\] Prompt/template output shape mismatch affecting 7 model(s)

## Summary

7 model(s) show **Prompt-template / image-placeholder mismatch** that should be filed against model repo first; mlx-vlm if template handling disagrees.

- **Observed problem:** Prompt/template output shape mismatch
- **Target:** model repo first; mlx-vlm if template handling disagrees
- **Affected models:** 7
- **Fixed when:** Requested sections render without template leakage.


## Affected Models

| Model                                                   | Observed Behavior   | Token Counts                                                                | Optional Context                                                                                                                                                                                                     |
|---------------------------------------------------------|---------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`                      | generated_tokens~4  | prompt=269 \| output/prompt=1.49% \| nontext burden=98% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_model_config_mlx_vlm_prompt_template_001.json)                      |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | generated_tokens~3  | prompt=807 \| output/prompt=0.37% \| nontext burden=99% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_model_config_mlx_vlm_prompt_template_001.json) |
| `mlx-community/GLM-4.6V-nvfp4`                          | generated_tokens~8  | prompt=790 \| output/prompt=1.01% \| nontext burden=99% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_004_mlx-community_GLM-4.6V-nvfp4_model_config_mlx_vlm_prompt_template_001.json)                          |
| `mlx-community/InternVL3-8B-bf16`                       | output/prompt=1.5%  | prompt=781 \| output/prompt=1.54% \| nontext burden=99% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_005_mlx-community_InternVL3-8B-bf16_model_config_mlx_vlm_prompt_template_001.json)                       |
| `mlx-community/Qwen3.5-27B-4bit`                        | generated_tokens~3  | prompt=593 \| output/prompt=0.51% \| nontext burden=99% \| stop=completed   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_008_mlx-community_Qwen3.5-27B-4bit_model_config_mlx_vlm_prompt_template_001.json)                        |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | generated_tokens~6  | prompt=1,031 \| output/prompt=0.58% \| nontext burden=99% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_010_mlx-community_paligemma2-10b-ft-docci-448-6bit_model_config_mlx_vlm_prompt_template_001.json)        |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | generated_tokens~6  | prompt=1,031 \| output/prompt=0.58% \| nontext burden=99% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_011_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json)        |


## Minimal Evidence

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: Output appears truncated to about 4 tokens.
- Output excerpt: `судy.`
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: Output appears truncated to about 3 tokens.
- Output excerpt: `ê¸°ê°Ģ.`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model LiquidAI/LFM2.5-VL-450M-MLX-bf16 --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-nvfp4 --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/InternVL3-8B-bf16 --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-27B-4bit --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/paligemma2-10b-ft-docci-448-6bit --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/paligemma2-10b-ft-docci-448-bf16 --image /Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --resize-shape 1024 1024 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'LiquidAI/LFM2.5-VL-450M-MLX-bf16'
IMAGE = '/Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg'
PROMPT = 'Describe this image briefly.'
LOAD_KWARGS = {'trust_remote_code': True}
GENERATE_KWARGS = {'max_tokens': 200, 'temperature': 0.0, 'prefill_step_size': 4096, 'resize_shape': (1024, 1024)}
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
    "resize_shape": [
      1024,
      1024
    ],
    "temperature": 0.0
  },
  "image": "/Users/jrp/Pictures/Processed/20260516-143411_DSC00013.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "LiquidAI/LFM2.5-VL-450M-MLX-bf16"
}
```

Optional advanced context:

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/GLM-4.6V-nvfp4`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_004_mlx-community_GLM-4.6V-nvfp4_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/InternVL3-8B-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_005_mlx-community_InternVL3-8B-bf16_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/Qwen3.5-27B-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_008_mlx-community_Qwen3.5-27B-4bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_010_mlx-community_paligemma2-10b-ft-docci-448-6bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260516T231646Z_011_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json)
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

### `LiquidAI/LFM2.5-VL-450M-MLX-bf16`

Observed signals:

- Output appears truncated to about 4 tokens.

Sample output:

```text
судy.
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

Observed signals:

- Output appears truncated to about 3 tokens.

Sample output:

```text
ê¸°ê°Ģ.
```

### `mlx-community/GLM-4.6V-nvfp4`

Observed signals:

- Output appears truncated to about 8 tokens.

Sample output:

```text
NullIn     co
```

_Additional affected models are listed in the Affected Models table above._

