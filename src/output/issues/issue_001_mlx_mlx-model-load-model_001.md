# \[mlx\]\[MLX-MODEL-LOAD-MODEL\] Received 4 parameters not in model: affecting 1 model(s)

## Summary

1 model(s) share a `MLX_MODEL_LOAD_MODEL` signal that clusters under `mlx`.

- **Issue kind:** `runtime_failure`
- **Cluster ID:** `mlx_mlx-model-load-model_001`
- **Symptom family:** `mlx_model_load_model received parameters not in model multi_modal_projector linear_1 biases multi_modal_projector linear`
- **Acceptance signal:** Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.


## Affected Models

| Model                                     | Representative Signal               | Token Context   | Repro Bundle                                                                                                                                                                                                                  |
|-------------------------------------------|-------------------------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | model error \| mlx model load model | stop=exception  | [`20260502T225507Z_006_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json`](../repro_bundles/20260502T225507Z_006_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json) |


## Evidence

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


## Likely Root Cause

- _Likely owner:_ `mlx`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MLX_MODEL_LOAD_MODEL`
- _Why this classification is credible:_ model error \| mlx model load model
- _Suggested next action:_ Inspect KV/cache behavior, memory pressure, and
  long-context execution.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
```

Repro bundles:

- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: [`20260502T225507Z_006_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json`](../repro_bundles/20260502T225507Z_006_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Acceptance Criteria

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `mlx_mlx-model-load-model_001` maintainer-triage cluster.


## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260502+e8ebdebe |
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

