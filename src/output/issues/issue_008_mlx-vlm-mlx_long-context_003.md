# \[mlx-vlm / mlx\]\[long-context\] Output degeneration under long prompt length (repeated_punctuation: ':**...') affecting 1 model(s)

## Summary

1 model(s) share a `long_context` signal that clusters under `mlx-vlm / mlx`.

- **Issue kind:** `stack_signal`
- **Cluster ID:** `mlx-vlm-mlx_long-context_003`
- **Symptom family:** `long_context`
- **Acceptance signal:** A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.


## Affected Models

| Model                               | Representative Signal                                                         | Token Context                                                                                        | Repro Bundle                                                                                                                                                                                            |
|-------------------------------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit` | Output degeneration under long prompt length (repeated_punctuation: ':**...') | prompt=16,290 \| output/prompt=3.07% \| nontext burden=100% \| stop=completed \| hit token cap (500) | [`20260502T233440Z_009_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_003.json`](../repro_bundles/20260502T233440Z_009_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_003.json) |


## Evidence

### Stack Signals

| Model                               |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                       | Owner           |
|-------------------------------------|--------------|--------------|-----------------|-------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-9B-MLX-4bit` |       16,290 |          500 | 3.07%           | Output degeneration under long prompt length (repeated_punctuation: ':**...') | `mlx-vlm / mlx` |

### `mlx-community/Qwen3.5-9B-MLX-4bit`

Observed signals:

- Output contains corrupted or malformed text segments (repeated_punctuation: ':**...').

Sample output:

```text
The user wants a description of the provided image.

1.  **Identify the main subject:** The image shows a tranquil, somewhat melancholic scene in a park or garden, likely on a rainy or overcast day...
```


## Likely Root Cause

- _Likely owner:_ `mlx-vlm / mlx`
- _Confidence:_ medium
- _Issue kind:_ `stack_signal`
- _Issue subtype:_ `long_context`
- _Why this classification is credible:_ hit token cap (500) \| nontext prompt
  burden=100% \| degeneration=repeated_punctuation: ':**...'
- _Suggested next action:_ Treat as a model-quality limitation for this prompt
  and image.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-9B-MLX-4bit
```

Repro bundles:

- `mlx-community/Qwen3.5-9B-MLX-4bit`: [`20260502T233440Z_009_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_003.json`](../repro_bundles/20260502T233440Z_009_mlx-community_Qwen3.5-9B-MLX-4bit_mlx_vlm_mlx_long_context_003.json)


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Acceptance Criteria

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The cluster rerun no longer produces this `mlx-vlm-mlx_long-context_003` maintainer-triage cluster.


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

