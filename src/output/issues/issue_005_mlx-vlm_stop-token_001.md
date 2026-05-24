<!-- markdownlint-disable MD012 MD013 MD033 MD060 -->

# \[mlx-vlm\]\[Stop-token leakage\] Stop/control tokens leaked into generated text affecting 4 model(s)

## Summary

4 model(s) show **Stop-token leakage** that should be filed against mlx-vlm.

- **Observed problem:** Stop/control tokens leaked into generated text
- **Target:** mlx-vlm
- **Affected models:** 4
- **Fixed when:** No leaked stop/control tokens.


## Affected Models

<!-- markdownlint-disable MD060 -->

| Model                                | Observed Behavior                                                                                                | Token Counts                                                                                         | Optional Context                                                                                                                                                                |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct`  | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; | prompt=770 \| output/prompt=25.97% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200)   | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)  |
| `mlx-community/GLM-4.6V-Flash-6bit`  | decoded text contains control token &lt;/think&gt;                                                               | prompt=6,045 \| output/prompt=2.88% \| nontext burden=100% \| stop=completed                         | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_004_mlx-community_GLM-4.6V-Flash-6bit_mlx_vlm_stop_token_001.json)  |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | decoded text contains control token &lt;/think&gt;                                                               | prompt=6,045 \| output/prompt=3.31% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_005_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json) |
| `mlx-community/GLM-4.6V-nvfp4`       | decoded text contains control token &lt;/think&gt;                                                               | prompt=6,045 \| output/prompt=3.13% \| nontext burden=100% \| stop=completed                         | [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_006_mlx-community_GLM-4.6V-nvfp4_mlx_vlm_stop_token_001.json)       |
<!-- markdownlint-enable MD060 -->


## Minimal Evidence

- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|end\|&gt; appeared in generated text.
- `microsoft/Phi-3.5-vision-instruct`: Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output excerpt: `The image shows a street view with a row of tall, cylindrical silos in the background, a gated entrance with a 'No Entry' sign, and a red building with a sign that reads 'COOKIE'. There are people standing near the gate, and a car is parked on the side of the road. The sky is blue with some clouds, and the overall s...`
- `mlx-community/GLM-4.6V-Flash-6bit`: Special control token &lt;/think&gt; appeared in generated text.


## Minimal Reproduction

These commands use `mlx-vlm` directly so the issue can be reproduced without installing the `check_models` harness.

Native CLI:

```bash
python -m mlx_vlm.generate --model microsoft/Phi-3.5-vision-instruct --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-Flash-6bit --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-Flash-mxfp4 --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
python -m mlx_vlm.generate --model mlx-community/GLM-4.6V-nvfp4 --image /Users/jrp/Pictures/Processed/20260523-180223_DSC00153.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```

Minimal Python repro (representative model):

```python
from mlx_vlm.generate import generate
from mlx_vlm.utils import load

MODEL = 'microsoft/Phi-3.5-vision-instruct'
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
  "model": "microsoft/Phi-3.5-vision-instruct"
}
```

Optional advanced context:

- `microsoft/Phi-3.5-vision-instruct`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_002_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)
- `mlx-community/GLM-4.6V-Flash-6bit`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_004_mlx-community_GLM-4.6V-Flash-6bit_mlx_vlm_stop_token_001.json)
- `mlx-community/GLM-4.6V-Flash-mxfp4`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_005_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json)
- `mlx-community/GLM-4.6V-nvfp4`: [optional JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260524T204643Z_006_mlx-community_GLM-4.6V-nvfp4_mlx_vlm_stop_token_001.json)
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

### `microsoft/Phi-3.5-vision-instruct`

Observed signals:

- Special control token &lt;\|end\|&gt; appeared in generated text.
- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

Sample output:

```text
The image shows a street view with a row of tall, cylindrical silos in the background, a gated entrance with a 'No Entry' sign, and a red building with a sign that reads 'COOKIE'. There are people...
```

### `mlx-community/GLM-4.6V-Flash-6bit`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

```text
<think>Got it, let's describe this image briefly. The scene shows the entrance to Burton Brewery, with large stainless steel fermentation tanks in the background. There's a black iron gate with sto...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

```text
<think>Got it, let's describe this image briefly. The image shows the Burton Brewery, with large silver cylindrical storage tanks (brewing vessels) dominating the background. In the foreground, the...
```

_Additional affected models are listed in the Affected Models table above._

