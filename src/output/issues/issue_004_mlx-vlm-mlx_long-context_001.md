# \[mlx-vlm / mlx\]\[Long-context collapse\] Long-context generation collapsed or became too short affecting 1 model(s)

## Summary

1 model(s) show **Long-context collapse** that should be filed against mlx-vlm first; MLX if cache/runtime reproduces.

- **Observed problem:** Long-context generation collapsed or became too short
- **Target:** mlx-vlm first; MLX if cache/runtime reproduces
- **Raw owner hint:** `mlx-vlm / mlx`
- **Affected models:** 1
- **Fixed when:** Full and reduced reruns avoid context collapse.
- **Issue kind:** `context_budget`
- **Raw cluster:** `mlx-vlm-mlx_long-context_001` (`long_context`)


## Affected Models

| Model                                     | Representative Signal                                                                                                                                                                                          | Token Context                                                                | Repro JSON                                                                                                                    |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16901 tokens), output stayed unusually short (11 tokens; ratio 0.1%). | prompt=16,901 \| output/prompt=0.07% \| nontext burden=97% \| stop=completed | [repro JSON](../repro_bundles/20260504T192124Z_005_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json) |


## Minimal Evidence

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: At long prompt length (16901 tokens), output stayed unusually short (11 tokens; ratio 0.1%).
- Output excerpt: `Boat, boat, boat, boat, boat`


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen2-VL-2B-Instruct-4bit
```

Repro bundles:

- `mlx-community/Qwen2-VL-2B-Instruct-4bit`: [repro JSON](../repro_bundles/20260504T192124Z_005_mlx-community_Qwen2-VL-2B-Instruct-4bit_mlx_vlm_mlx_long_context_001.json)
- Note: these are local artifact links; attach or publish the JSON when filing upstream.


## Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The cluster rerun no longer produces this `mlx-vlm-mlx_long-context_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Rerun with reduced image/text burden and compare output recovery.
- [ ] Compare prompt-token accounting with text-only and image+text prompts.
- [ ] Inspect cache allocation, prefill step size, and long-context generation behavior.


## Likely Root Cause

- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces
- _Likely owner:_ `mlx-vlm / mlx`
- _Confidence:_ high
- _Issue kind:_ `context_budget`
- _Issue subtype:_ `long_context`
- _Why this classification is credible:_ Output is very short relative to
  prompt size (0.1%), suggesting possible early-stop or prompt-handling
  issues. \| At long prompt length (16901 tokens), output stayed unusually
  short (11 tokens; ratio 0.1%). \| output/prompt=0.07% \| nontext prompt
  burden=97%
- _Suggested next action:_ Treat this as a prompt-budget issue first; nontext
  prompt burden is 97% and the output stays weak under that load.


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

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Observed signals:

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16901 tokens), output stayed unusually short (11 tokens; ratio 0.1%).
- Model output may not follow prompt or image contents (missing: classic, style, sailboat, dark, hull).
- Output omitted required Title/Description/Keywords sections (title, description, keywords).

Sample output:

```text
Boat, boat, boat, boat, boat
```

