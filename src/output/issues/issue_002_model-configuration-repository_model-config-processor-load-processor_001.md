# \[model configuration/repository\]\[MODEL-CONFIG-PROCESSOR-LOAD-PROCESSOR\] Loaded processor has no image_processor; expected multimodal processor affecting 1 model(s)

## Summary

1 model(s) share a `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR` signal that clusters under `model configuration/repository`.

- **Issue kind:** `runtime_failure`
- **Cluster ID:** `model-configuration-repository_model-config-processor-load-processor_001`
- **Symptom family:** `model_config_processor_load_processor model preflight failed for mlx-community/molmopoint-8b-fp16 loaded processor has n`
- **Acceptance signal:** Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.


## Affected Models

| Model                              | Representative Signal                                    | Token Context   | Repro Bundle                                                                                                                                                                                                                  |
|------------------------------------|----------------------------------------------------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | processor error \| model config processor load processor | stop=exception  | [`20260502T105726Z_002_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json`](../repro_bundles/20260502T105726Z_002_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json) |


## Evidence

### `mlx-community/MolmoPoint-8B-fp16`

Observed error:

```text
Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```


## Likely Root Cause

- _Likely owner:_ `model configuration/repository`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Why this classification is credible:_ processor error \| model config
  processor load processor
- _Suggested next action:_ Inspect model repo config, chat template, and EOS
  settings.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
```

Repro bundles:

- `mlx-community/MolmoPoint-8B-fp16`: [`20260502T105726Z_002_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json`](../repro_bundles/20260502T105726Z_002_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json)


## Fix Checklist

- [ ] Inspect the exported error package, load phase, and traceback owner.
- [ ] Check model config, tokenizer files, and weight shape compatibility.
- [ ] Compare against installed mlx, mlx-vlm, mlx-lm, transformers, and tokenizers versions.
- [ ] Reproduce with the single affected model before judging output quality.


## Acceptance Criteria

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `model-configuration-repository_model-config-processor-load-processor_001` maintainer-triage cluster.


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

