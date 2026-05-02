# \[mlx-vlm / mlx\]\[long-context\] Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt affecting 2 model(s)

## Summary

2 model(s) share a `long_context` signal that clusters under `mlx-vlm / mlx`.

- **Issue kind:** `context_budget`
- **Cluster ID:** `mlx-vlm-mlx_long-context_001`
- **Symptom family:** `long_context`
- **Acceptance signal:** A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.


## Affected Models

| Model                                     | Representative Signal                                                                                                                                                                                          | Token Context                                                                 | Repro Bundle   |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|----------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues. \| At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%). | prompt=16,299 \| output/prompt=0.08% \| nontext burden=100% \| stop=completed |                |
| `mlx-community/paligemma2-3b-pt-896-4bit` | Output appears truncated to about 3 tokens. \| At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).                                                                      | prompt=4,101 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed  |                |


## Evidence

### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Observed signals:

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%).

Sample output:

```text
I'm sorry, but the context didn't show up.
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

Observed signals:

- Output appears truncated to about 3 tokens.
- At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).

Sample output:

```text
The garden
```


## Likely Root Cause

- _Likely owner:_ `mlx-vlm / mlx`
- _Confidence:_ high
- _Issue kind:_ `context_budget`
- _Issue subtype:_ `long_context`
- _Why this classification is credible:_ Output is very short relative to
  prompt size (0.1%), suggesting possible early-stop or prompt-handling
  issues. \| At long prompt length (16299 tokens), output stayed unusually
  short (13 tokens; ratio 0.1%). \| output/prompt=0.08% \| nontext prompt
  burden=100%
- _Suggested next action:_ Treat this as a prompt-budget issue first; nontext
  prompt burden is 100% and the output stays weak under that load.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt 'Describe this picture' --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen2-VL-2B-Instruct-4bit mlx-community/paligemma2-3b-pt-896-4bit
```


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

