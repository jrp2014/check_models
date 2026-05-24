<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 1 model(s)

## Summary

1 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Affected models:** 1
- **Fixed when:** Full and reduced reruns avoid context collapse.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                     | Observed Behavior                      | Token Counts                                                                                          | Optional Context                                                                                                                                                                           |
|-------------------------------------------|----------------------------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | prompt_tokens=16346, repetitive output | prompt=16,346 \| output/prompt=1.22% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_009_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json) |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: At long prompt length (16346 tokens), output became repetitive.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Output became repetitive, indicating possible generation instability (token: phrase: "answered by answered by...").
- Output excerpt: `Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answered by Answe...`


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen2-VL-2B-Instruct-4bit --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'mlx-community/Qwen2-VL-2B-Instruct-4bit'
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
  "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit"
}
```

Optional advanced context:

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_009_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)
- JSON bundles contain extended local diagnostics only; the model, prompt, image reference, and generation settings needed to reproduce are inline above.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


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

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Observed signals:

- At long prompt length (16346 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "answered by answered by...").
- Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'by').

Sample output:

```text
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answered by
Answe...
```

