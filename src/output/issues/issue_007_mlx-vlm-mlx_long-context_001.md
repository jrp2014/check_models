# \[mlx-vlm / mlx\]\[long-context\] At long prompt length (16807 tokens), output became repetitive affecting 1 model(s)

## Summary

1 model(s) share a `long_context` signal that clusters under `mlx-vlm / mlx`.

- **Issue kind:** `cutoff_degraded`
- **Cluster ID:** `mlx-vlm-mlx_long-context_001`
- **Symptom family:** `long_context`
- **Acceptance signal:** A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.


## Affected Models

| Model                              | Representative Signal                                           | Token Context                                                                                       | Repro Bundle                                                                                                                                                                                          |
|------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/X-Reasoner-7B-8bit` | At long prompt length (16807 tokens), output became repetitive. | prompt=16,807 \| output/prompt=2.97% \| nontext burden=97% \| stop=completed \| hit token cap (500) | [`20260503T205313Z_008_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json`](../repro_bundles/20260503T205313Z_008_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json) |


## Evidence

### `mlx-community/X-Reasoner-7B-8bit`

Observed signals:

- At long prompt length (16807 tokens), output became repetitive.
- Output became repetitive, indicating possible generation instability (token: phrase: "quay, river, boats, sailing,...").

Sample output:

```text
Title:
Woodbridge Tide Mill Museum and Boats

Description:
A historic quayside scene at Woodbridge, Suffolk, featuring traditional sailing barges moored on the River Deben at low tide. The backgrou...
```


## Likely Root Cause

- _Likely owner:_ `mlx-vlm / mlx`
- _Confidence:_ high
- _Issue kind:_ `cutoff_degraded`
- _Issue subtype:_ `long_context`
- _Why this classification is credible:_ At long prompt length (16807 tokens),
  output became repetitive. \| hit token cap (500) \| nontext prompt
  burden=97% \| missing terms: 10 Best (structured), Bird, Gull, Mooring,
  Mudflats
- _Suggested next action:_ Inspect long-context cache behavior under heavy
  image-token burden.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
```

Repro bundles:

- `mlx-community/X-Reasoner-7B-8bit`: [`20260503T205313Z_008_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json`](../repro_bundles/20260503T205313Z_008_mlx-community_X-Reasoner-7B-8bit_mlx_vlm_mlx_long_context_001.json)


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Acceptance Criteria

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The cluster rerun no longer produces this `mlx-vlm-mlx_long-context_001` maintainer-triage cluster.


## Environment

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

