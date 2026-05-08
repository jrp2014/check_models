# \[mlx-vlm\]\[mlx-vlm: Model load / model error\] Missing module/import during model load affecting 1 model(s)

## Summary

1 model(s) show **mlx-vlm: Model load / model error** that should be filed against mlx-vlm.

- **Observed problem:** Missing module/import during model load
- **Target:** mlx-vlm
- **Raw owner hint:** `mlx-vlm`
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.
- **Issue kind:** `runtime_failure`
- **Raw cluster:** `mlx-vlm_mlx-vlm-model-load-model_001` (`MLX_VLM_MODEL_LOAD_MODEL`)


## Affected Models

| Model                  | Representative Signal                   | Token Context   | Repro JSON                                                                                                          |
|------------------------|-----------------------------------------|-----------------|---------------------------------------------------------------------------------------------------------------------|
| `facebook/pe-av-large` | model error \| mlx vlm model load model | stop=exception  | [repro JSON](../repro_bundles/20260508T130439Z_002_facebook_pe-av-large_MLX_VLM_MODEL_LOAD_MODEL_8b244da8c605.json) |


## Minimal Evidence

- `facebook/pe-av-large` fails with: Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
- Root exception: `builtins.ValueError`: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models facebook/pe-av-large
```

Repro bundles:

- `facebook/pe-av-large`: [repro JSON](../repro_bundles/20260508T130439Z_002_facebook_pe-av-large_MLX_VLM_MODEL_LOAD_MODEL_8b244da8c605.json)
- Note: these are local artifact links; attach or publish the JSON when filing upstream.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `mlx-vlm_mlx-vlm-model-load-model_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Likely Root Cause

- _Filing target:_ mlx-vlm
- _Likely owner:_ `mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MLX_VLM_MODEL_LOAD_MODEL`
- _Why this classification is credible:_ model error \| mlx vlm model load
  model
- _Suggested next action:_ Inspect the import path and installed package
  version that owns the missing module before treating this as a model
  failure.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.5.0                       |
| mlx             | 0.32.0.dev20260508+a1c0b6f9 |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.8.0.dev0                  |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.14.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |


## Appendix: Detailed Evidence

### `facebook/pe-av-large`

Observed error:

```text
Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
```

Root exception:

```text
builtins.ValueError: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
```

Traceback tail:

```text
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 137, in get_model_and_args
    raise ValueError(msg)
ValueError: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: Model type pe_audio_video not supported. Error: No module named 'mlx_vlm.speculative.drafters.pe_audio_video'
```

