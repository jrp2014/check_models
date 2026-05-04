# \[mlx\]\[MLX-MODEL-LOAD-MODEL\] Received 4 parameters not in model: affecting 1 model(s)

## Summary

1 model(s) show **MLX: Model load / model error** that appears to belong with `mlx`.

- **Observed problem:** Received 4 parameters not in model:
- **Target:** `mlx`
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.
- **Issue kind:** `runtime_failure`
- **Raw cluster:** `mlx_mlx-model-load-model_001` (`MLX_MODEL_LOAD_MODEL`)


## Affected Models

| Model                                     | Representative Signal               | Token Context   | Repro JSON                                                                                                                         |
|-------------------------------------------|-------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Kimi-VL-A3B-Thinking-8bit` | model error \| mlx model load model | stop=exception  | [repro JSON](../repro_bundles/20260503T224052Z_002_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json) |


## Minimal Evidence

- `mlx-community/Kimi-VL-A3B-Thinking-8bit` fails with: Model loading failed: Received 4 parameters not in model:
- Root exception: `builtins.ValueError`: Received 4 parameters not in model: <br>multi_modal_projector.linear_1.biases,<br>multi_modal_projector.linear_1.scales,<br>multi_modal_projector.linear_2.biases,<br>multi_modal_projector.l...


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
```

Repro bundles:

- `mlx-community/Kimi-VL-A3B-Thinking-8bit`: [repro JSON](../repro_bundles/20260503T224052Z_002_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json)


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `mlx_mlx-model-load-model_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Likely Root Cause

- _Likely owner:_ `mlx`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MLX_MODEL_LOAD_MODEL`
- _Why this classification is credible:_ model error \| mlx model load model
- _Suggested next action:_ Inspect KV/cache behavior, memory pressure, and
  long-context execution.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260503+e8ebdebe |
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

