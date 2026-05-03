# \[mlx-vlm / mlx\]\[long-context\] Output degeneration under long prompt length (incomplete_sentence: ends with 'in') affecting 1 model(s)

## Summary

1 model(s) share a `long_context` signal that clusters under `mlx-vlm / mlx`.

- **Issue kind:** `stack_signal`
- **Cluster ID:** `mlx-vlm-mlx_long-context_002`
- **Symptom family:** `long_context`
- **Acceptance signal:** A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.


## Affected Models

| Model                            | Representative Signal                                                              | Token Context                                                                                       | Repro Bundle                                                                                                                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen3.5-27B-4bit` | Output degeneration under long prompt length (incomplete_sentence: ends with 'in') | prompt=16,807 \| output/prompt=2.97% \| nontext burden=97% \| stop=completed \| hit token cap (500) | [`20260503T011608Z_007_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_002.json`](../repro_bundles/20260503T011608Z_007_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_002.json) |


## Evidence

### Stack Signals

| Model                            |   Prompt Tok |   Output Tok | Output/Prompt   | Symptom                                                                            | Owner           |
|----------------------------------|--------------|--------------|-----------------|------------------------------------------------------------------------------------|-----------------|
| `mlx-community/Qwen3.5-27B-4bit` |       16,807 |          500 | 2.97%           | Output degeneration under long prompt length (incomplete_sentence: ends with 'in') | `mlx-vlm / mlx` |

### `mlx-community/Qwen3.5-27B-4bit`

Observed signals:

- Output contains corrupted or malformed text segments (incomplete_sentence: ends with 'in').
- Output omitted required Title/Description/Keywords sections (title, description, keywords).
- Output leaked reasoning or prompt-template text (description hint:, keyword hints:).

Sample output:

```text
The user wants me to analyze the image and generate cataloguing metadata based on specific rules.

**1. Analyze the Image:**
*   **Foreground:** A long, narrow boat (looks like a punt or similar fl...
```


## Likely Root Cause

- _Likely owner:_ `mlx-vlm / mlx`
- _Confidence:_ medium
- _Issue kind:_ `stack_signal`
- _Issue subtype:_ `long_context`
- _Why this classification is credible:_ hit token cap (500) \| nontext prompt
  burden=97% \| missing sections: title, description, keywords \| nonvisual
  metadata reused
- _Suggested next action:_ Treat as a model limitation for this prompt; the
  requested output contract is not being met.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-27B-4bit
```

Repro bundles:

- `mlx-community/Qwen3.5-27B-4bit`: [`20260503T011608Z_007_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_002.json`](../repro_bundles/20260503T011608Z_007_mlx-community_Qwen3.5-27B-4bit_mlx_vlm_mlx_long_context_002.json)


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Acceptance Criteria

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The cluster rerun no longer produces this `mlx-vlm-mlx_long-context_002` maintainer-triage cluster.


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

