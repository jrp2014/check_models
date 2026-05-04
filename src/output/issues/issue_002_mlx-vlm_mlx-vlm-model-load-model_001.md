# \[mlx-vlm\]\[MLX-VLM-MODEL-LOAD-MODEL\] Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite' affecting 1 model(s)

## Summary

1 model(s) show **mlx-vlm: Model load / model error** that appears to belong with `mlx-vlm`.

- **Observed problem:** Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
- **Target:** `mlx-vlm`
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.
- **Issue kind:** `runtime_failure`
- **Raw cluster:** `mlx-vlm_mlx-vlm-model-load-model_001` (`MLX_VLM_MODEL_LOAD_MODEL`)


## Affected Models

| Model                                | Representative Signal                   | Token Context   | Repro JSON                                                                                                                        |
|--------------------------------------|-----------------------------------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/granite-4.1-8b-mxfp8` | model error \| mlx vlm model load model | stop=exception  | [repro JSON](../repro_bundles/20260503T224052Z_005_mlx-community_granite-4.1-8b-mxfp8_MLX_VLM_MODEL_LOAD_MODEL_5af3e849109f.json) |


## Minimal Evidence

- `mlx-community/granite-4.1-8b-mxfp8` fails with: Model loading failed: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
- Root exception: `builtins.ValueError`: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/granite-4.1-8b-mxfp8
```

Repro bundles:

- `mlx-community/granite-4.1-8b-mxfp8`: [repro JSON](../repro_bundles/20260503T224052Z_005_mlx-community_granite-4.1-8b-mxfp8_MLX_VLM_MODEL_LOAD_MODEL_5af3e849109f.json)


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `mlx-vlm_mlx-vlm-model-load-model_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Likely Root Cause

- _Likely owner:_ `mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MLX_VLM_MODEL_LOAD_MODEL`
- _Why this classification is credible:_ model error \| mlx vlm model load
  model
- _Suggested next action:_ Inspect prompt-template, stop-token, and decode
  post-processing behavior.


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

### `mlx-community/granite-4.1-8b-mxfp8`

Observed error:

```text
Model loading failed: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
```

Root exception:

```text
builtins.ValueError: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
```

Traceback tail:

```text
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 136, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Model type granite not supported. Error: No module named 'mlx_vlm.speculative.drafters.granite'
```

