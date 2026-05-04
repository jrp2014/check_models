# \[mlx\]\[MLX: Model load / model error\] Weight/config mismatch during model load affecting 2 model(s)

## Summary

2 model(s) show **MLX: Model load / model error** that should be filed against mlx.

- **Observed problem:** Weight/config mismatch during model load
- **Target:** mlx
- **Raw owner hint:** `mlx`
- **Affected models:** 2
- **Fixed when:** Load/generation completes or fails with a narrower owner.
- **Issue kind:** `runtime_failure`
- **Raw cluster:** `mlx_mlx-model-load-model_001` (`MLX_MODEL_LOAD_MODEL`)


## Affected Models

| Model                                     | Representative Signal               | Token Context   | Repro JSON                                                                                                                         |
|-------------------------------------------|-------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------|
| `LiquidAI/LFM2.5-VL-450M-MLX-bf16`        | model error \| mlx model load model | stop=exception  | [repro JSON](../repro_bundles/20260504T192124Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)        |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | model error \| mlx model load model | stop=exception  | [repro JSON](../repro_bundles/20260504T192124Z_003_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json) |


## Minimal Evidence

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16` fails with: Model loading failed: Received 2 parameters not in model:
- Root exception: `builtins.ValueError`: Received 2 parameters not in model: <br>multi_modal_projector.layer_norm.bias,<br>multi_modal_projector.layer_norm.weight.
- `mlx-community/Kimi-VL-A3B-Thinking-8bit` fails with: Model loading failed: Received 4 parameters not in model:
- Root exception: `builtins.ValueError`: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_projector.linear_1.scales,<br>multi_modal_projector.linear_2.biases,<br>multi_modal_projector.l...


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models LiquidAI/LFM2.5-VL-450M-MLX-bf16 mlx-community/Kimi-VL-A3B-Thinking-8bit
```

Repro bundles:

- `LiquidAI/LFM2.5-VL-450M-MLX-bf16`: [repro JSON](../repro_bundles/20260504T192124Z_001_LiquidAI_LFM2.5-VL-450M-MLX-bf16_MLX_MODEL_LOAD_MODEL_853049863f38.json)
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: [repro JSON](../repro_bundles/20260504T192124Z_003_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)
- Note: these are local artifact links; attach or publish the JSON when filing upstream.

Per-model failure reruns:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models LiquidAI/LFM2.5-VL-450M-MLX-bf16
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
```


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `mlx_mlx-model-load-model_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Compare checkpoint keys with the selected model class and model config.
- [ ] Inspect missing/unexpected projector, scale, bias, and quantized-weight parameter names.
- [ ] Verify the model repo revision matches the mlx-vlm/mlx loader expectations.
- [ ] Reproduce after upgrading/downgrading mlx-vlm and mlx to isolate version compatibility.


## Likely Root Cause

- _Filing target:_ mlx
- _Likely owner:_ `mlx`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MLX_MODEL_LOAD_MODEL`
- _Why this classification is credible:_ model error \| mlx model load model
- _Suggested next action:_ Compare checkpoint keys with the selected model
  class/config, especially projector scale/bias parameters and quantized
  weight naming, before judging model quality.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260504+e8ebdebe |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.7.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.13.0                      |
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

