# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 2 model(s)

## Summary

2 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 2
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

| Model                                     | Observed Behavior                                                               | Token Counts                                                                  | Optional Context                                                                                                                                                                           |
|-------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit`       | output/prompt=0.1% \| prompt_tokens=16167, output_tokens=12, output/prompt=0.1% | prompt=16,167 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_013_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_002.json)       |
| `mlx-community/paligemma2-3b-pt-896-4bit` | output/prompt=0.2% \| prompt_tokens=4103, output_tokens=9, output/prompt=0.2%   | prompt=4,103 \| output/prompt=0.22% \| nontext burden=100% \| stop=completed  | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_018_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_002.json) |


## Minimal Evidence

- `mlx-community/Qwen3.5-9B-MLX-4bit`: Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- `mlx-community/Qwen3.5-9B-MLX-4bit`: At long prompt length (16167 tokens), output stayed unusually short (12 tokens; ratio 0.1%).
- Output excerpt: `云峰 rouluterждronk {item}`
- `mlx-community/paligemma2-3b-pt-896-4bit`: Output is very short relative to prompt size (0.2%), suggesting possible early-stop or prompt-handling issues.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-9B-MLX-4bit --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/paligemma2-3b-pt-896-4bit --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen3.5-9B-MLX-4bit'
IMAGE = '/Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg'
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
  "image": "/Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg",
  "load_kwargs": {
    "trust_remote_code": true
  },
  "model": "mlx-community/Qwen3.5-9B-MLX-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen3.5-9B-MLX-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_013_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_002.json)
- `mlx-community/paligemma2-3b-pt-896-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260517T213817Z_018_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_002.json)
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
| mlx             | 0.32.0.dev20260517+7b7c1240 |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.8.1                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.15.0                      |
| Python Version  | 3.13.13                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `mlx-community/Qwen3.5-9B-MLX-4bit`

Observed signals:

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16167 tokens), output stayed unusually short (12 tokens; ratio 0.1%).

Sample output:

```text
云峰 rouluterждronk
{item}
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

Observed signals:

- Output is very short relative to prompt size (0.2%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (4103 tokens), output stayed unusually short (9 tokens; ratio 0.2%).

Sample output:

```text
It has been tagged with #1.
```

