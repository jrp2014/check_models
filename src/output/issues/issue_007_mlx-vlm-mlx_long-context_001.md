# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 3 model(s)

## Summary

3 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 3
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

| Model                                | Observed Behavior                      | Token Counts                                                                                          | Optional Context                                                                                                            |
|--------------------------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-27B-mxfp8`    | prompt_tokens=16167, repetitive output | prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](../repro_bundles/20260521T221248Z_010_mlx-community_Qwen3.5-27B-mxfp8_mlx_vlm_mlx_long_context_001.json)    |
| `mlx-community/Qwen3.5-35B-A3B-6bit` | prompt_tokens=16167, repetitive output | prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](../repro_bundles/20260521T221248Z_011_mlx-community_Qwen3.5-35B-A3B-6bit_mlx_vlm_mlx_long_context_001.json) |
| `mlx-community/Qwen3.5-35B-A3B-bf16` | prompt_tokens=16167, repetitive output | prompt=16,167 \| output/prompt=1.24% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](../repro_bundles/20260521T221248Z_012_mlx-community_Qwen3.5-35B-A3B-bf16_mlx_vlm_mlx_long_context_001.json) |


## Minimal Evidence

- `mlx-community/Qwen3.5-27B-mxfp8`: At long prompt length (16167 tokens), output became repetitive.
- `mlx-community/Qwen3.5-27B-mxfp8`: Output became repetitive, indicating possible generation instability (token: 2v,).
- Output excerpt: `orda2v,, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v, 2v`
- `mlx-community/Qwen3.5-35B-A3B-6bit`: At long prompt length (16167 tokens), output became repetitive.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-27B-mxfp8 --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-35B-A3B-6bit --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/Qwen3.5-35B-A3B-bf16 --image /Users/jrp/Pictures/Processed/20260516-143527_DSC00014.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen3.5-27B-mxfp8'
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
  "model": "mlx-community/Qwen3.5-27B-mxfp8"
}
```

Optional advanced context:

- `mlx-community/Qwen3.5-27B-mxfp8`: [optional JSON](../repro_bundles/20260521T221248Z_010_mlx-community_Qwen3.5-27B-mxfp8_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-35B-A3B-6bit`: [optional JSON](../repro_bundles/20260521T221248Z_011_mlx-community_Qwen3.5-35B-A3B-6bit_mlx_vlm_mlx_long_context_001.json)
- `mlx-community/Qwen3.5-35B-A3B-bf16`: [optional JSON](../repro_bundles/20260521T221248Z_012_mlx-community_Qwen3.5-35B-A3B-bf16_mlx_vlm_mlx_long_context_001.json)
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
| mlx             | 0.32.0.dev20260521+5d1c0e4c |
| mlx-lm          | 0.31.3                      |
| mlx-audio       | 0.4.3                       |
| transformers    | 5.9.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.16.1                      |
| Python Version  | 3.13.13                     |
| OS              | Darwin 25.5.0               |
| macOS Version   | 26.5                        |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `mlx-community/Qwen3.5-27B-mxfp8`

Observed signals:

- At long prompt length (16167 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: 2v,).
- Output contains corrupted or malformed text segments (incomplete_sentence: ends with '2v').

Sample output:

```text
orda2v,,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v,  2v...
```

### `mlx-community/Qwen3.5-35B-A3B-6bit`

Observed signals:

- At long prompt length (16167 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: ,).
- Output contains corrupted or malformed text segments (character_loop: ' ,' repeated).

Sample output:

```text
,rew, , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

Observed signals:

- At long prompt length (16167 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: all,).

Sample output:

```text
5, each, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all, all...
```

