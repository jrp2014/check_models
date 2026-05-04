# \[mlx-vlm / mlx\]\[long-context\] At long prompt length (16901 tokens), output became repetitive affecting 1 model(s)

## Summary

1 model(s) show **Long-context collapse** that appears to belong with `mlx-vlm / mlx`.

- **Observed problem:** At long prompt length (16901 tokens), output became repetitive.
- **Target:** `mlx-vlm / mlx`
- **Affected models:** 1
- **Fixed when:** Full and reduced reruns avoid context collapse.
- **Issue kind:** `cutoff_degraded`
- **Raw cluster:** `mlx-vlm-mlx_long-context_001` (`long_context`)


## Affected Models

| Model                                     | Representative Signal                                           | Token Context                                                                                       | Repro JSON                                                                                                                    |
|-------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | At long prompt length (16901 tokens), output became repetitive. | prompt=16,901 \| output/prompt=2.96% \| nontext burden=97% \| stop=completed \| hit token cap (500) | [repro JSON](../repro_bundles/20260503T224052Z_004_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json) |


## Minimal Evidence

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: At long prompt length (16901 tokens), output became repetitive.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- Output excerpt: `Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat...`


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen2-VL-2B-Instruct-4bit
```

Repro bundles:

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: [repro JSON](../repro_bundles/20260503T224052Z_004_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The cluster rerun no longer produces this `mlx-vlm-mlx_long-context_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Likely Root Cause

- _Likely owner:_ `mlx-vlm / mlx`
- _Confidence:_ high
- _Issue kind:_ `cutoff_degraded`
- _Issue subtype:_ `long_context`
- _Why this classification is credible:_ At long prompt length (16901 tokens),
  output became repetitive. \| hit token cap (500) \| nontext prompt
  burden=97% \| missing sections: title, description, keywords
- _Suggested next action:_ Inspect long-context cache behavior under heavy
  image-token burden.


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

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Observed signals:

- At long prompt length (16901 tokens), output became repetitive.
- Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- Output became repetitive, indicating possible generation instability (token: phrase: "boat anchor boat anchor...").
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat Anchor Boat...
```

