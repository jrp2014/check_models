# \[model configuration/repository\]\[Model config: Processor load / processor error\] Processor config is missing image processor affecting 1 model(s)

## Summary

1 model(s) show **Model config: Processor load / processor error** that should be filed against model configuration / repository.

- **Observed problem:** Processor config is missing image processor
- **Target:** model configuration / repository
- **Raw owner hint:** `model configuration/repository`
- **Affected models:** 1
- **Fixed when:** Load/generation completes or fails with a narrower owner.
- **Issue kind:** `runtime_failure`
- **Raw cluster:** `model-configuration-repository_model-config-processor-load-processor_001` (`MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`)


## Affected Models

| Model                              | Representative Signal                                    | Token Context   | Repro JSON                                                                                                                         |
|------------------------------------|----------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | processor error \| model config processor load processor | stop=exception  | [repro JSON](../repro_bundles/20260504T192124Z_004_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json) |


## Minimal Evidence

- `mlx-community/MolmoPoint-8B-fp16` fails with: Loaded processor has no image_processor; expected multimodal processor.
- Root exception: `builtins.ValueError`: Loaded processor has no image_processor; expected multimodal processor.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
```

Repro bundles:

- `mlx-community/MolmoPoint-8B-fp16`: [repro JSON](../repro_bundles/20260504T192124Z_004_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json)
- Note: these are local artifact links; attach or publish the JSON when filing upstream.


## Expected Fix Signal

- [ ] Affected reruns complete model load and generation, or fail with a narrower configuration/compatibility error that points to the owning layer.
- [ ] The cluster rerun no longer produces this `model-configuration-repository_model-config-processor-load-processor_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Inspect `preprocessor_config.json`, `processor_config.json`, and AutoProcessor mapping.
- [ ] Verify the loaded processor exposes the image processor expected by mlx-vlm.
- [ ] Check whether the model repo needs processor files or mlx-vlm needs a fallback path.
- [ ] Reproduce with the single affected model before judging output quality.


## Likely Root Cause

- _Filing target:_ model configuration / repository
- _Likely owner:_ `model configuration/repository`
- _Confidence:_ high
- _Issue kind:_ `runtime_failure`
- _Issue subtype:_ `MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR`
- _Why this classification is credible:_ processor error \| model config
  processor load processor
- _Suggested next action:_ Inspect the model repo processor/preprocessor
  config and AutoProcessor mapping; the multimodal processor is missing or not
  exposing the image processor expected by mlx-vlm.


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

### `mlx-community/MolmoPoint-8B-fp16`

Observed error:

```text
Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

Root exception:

```text
builtins.ValueError: Loaded processor has no image_processor; expected multimodal processor.
```

Traceback tail:

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

