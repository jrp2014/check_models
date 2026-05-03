# \[model-config / mlx-vlm\]\[prompt-template\] Model returned zero output tokens affecting 2 model(s)

## Summary

2 model(s) share a `prompt_template` signal that clusters under `model-config / mlx-vlm`.

- **Issue kind:** `harness`
- **Cluster ID:** `model-config-mlx-vlm_prompt-template_001`
- **Symptom family:** `prompt_template`
- **Acceptance signal:** Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.


## Affected Models

| Model                                            | Representative Signal                                                                                          | Token Context                                                               | Repro Bundle                                                                                                                                                                                                                                              |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/llava-v1.6-mistral-7b-8bit`       | Model returned zero output tokens.                                                                             | prompt=0 \| stop=completed                                                  | [`20260503T011608Z_009_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json`](../repro_bundles/20260503T011608Z_009_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json)             |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16` | Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues. | prompt=1,587 \| output/prompt=0.88% \| nontext burden=70% \| stop=completed | [`20260503T011608Z_010_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json`](../repro_bundles/20260503T011608Z_010_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json) |


## Evidence

### `mlx-community/llava-v1.6-mistral-7b-8bit`

Observed signals:

- Model returned zero output tokens.

### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

Observed signals:

- Output is very short relative to prompt size (0.9%), suggesting possible early-stop or prompt-handling issues.
- Model output may not follow prompt or image contents (missing: Bench, Blue sky, Clouds, East Anglia, English countryside).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
- Use only the metadata that is clearly supported by the image.
```


## Likely Root Cause

- _Likely owner:_ `model-config / mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `harness`
- _Issue subtype:_ `prompt_template`
- _Why this classification is credible:_ Model returned zero output tokens.
- _Suggested next action:_ Inspect model repo config, chat template, and EOS
  settings.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/llava-v1.6-mistral-7b-8bit mlx-community/paligemma2-10b-ft-docci-448-bf16
```

Repro bundles:

- `mlx-community/llava-v1.6-mistral-7b-8bit`: [`20260503T011608Z_009_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json`](../repro_bundles/20260503T011608Z_009_mlx-community_llava-v1.6-mistral-7b-8bit_model_config_mlx_vlm_prompt_template_001.json)
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`: [`20260503T011608Z_010_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json`](../repro_bundles/20260503T011608Z_010_mlx-community_paligemma2-10b-ft-docci-448-bf16_model_config_mlx_vlm_prompt_template_001.json)


## Fix Checklist

- [ ] Inspect chat template selection and rendered message roles.
- [ ] Verify image placeholder count and order match the processor config.
- [ ] Check EOS defaults and whether the template expects explicit assistant prefixes.


## Acceptance Criteria

- [ ] Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.
- [ ] The cluster rerun no longer produces this `model-config-mlx-vlm_prompt-template_001` maintainer-triage cluster.


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

