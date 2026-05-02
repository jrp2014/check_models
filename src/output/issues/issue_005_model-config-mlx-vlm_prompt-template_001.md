# [model-config / mlx-vlm][prompt-template] Output appears truncated to about 4 tokens affecting 2 model(s)

## Summary

2 model(s) share a `prompt_template` signal that clusters under `model-config / mlx-vlm`.

- **Issue kind:** `harness`
- **Cluster ID:** `model-config-mlx-vlm_prompt-template_001`
- **Symptom family:** `prompt_template`
- **Acceptance signal:** Affected reruns produce the requested sections without empty/filler output, template leakage, or image-placeholder mismatch symptoms.


## Affected Models

| Model                             | Representative Signal                       | Token Context                                                             | Repro Bundle   |
|-----------------------------------|---------------------------------------------|---------------------------------------------------------------------------|----------------|
| `mlx-community/gemma-3n-E2B-4bit` | Output appears truncated to about 4 tokens. | prompt=264 \| output/prompt=1.52% \| nontext burden=98% \| stop=completed |                |
| `mlx-community/gemma-4-31b-bf16`  | Output appears truncated to about 5 tokens. | prompt=266 \| output/prompt=1.88% \| nontext burden=98% \| stop=completed |                |


## Evidence

### `mlx-community/gemma-3n-E2B-4bit`

Observed signals:

- Output appears truncated to about 4 tokens.

Sample output:

```text
in the park
```

### `mlx-community/gemma-4-31b-bf16`

Observed signals:

- Output appears truncated to about 5 tokens.

Sample output:

```text
in one sentence.
```


## Likely Root Cause

- _Likely owner:_ `model-config / mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `harness`
- _Issue subtype:_ `prompt_template`
- _Why this classification is credible:_ Output appears truncated to about 4
  tokens. \| nontext prompt burden=98%
- _Suggested next action:_ Inspect model repo config, chat template, and EOS
  settings.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-3n-E2B-4bit mlx-community/gemma-4-31b-bf16
```


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

