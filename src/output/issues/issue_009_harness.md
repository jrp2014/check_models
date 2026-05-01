# [Harness Issue] The garden in mlx-community/paligemma2-3b-pt-896-4bit

## Description

Integration/harness warning detected for `mlx-community/paligemma2-3b-pt-896-4bit`.

### Details

- Output appears truncated to about 3 tokens.
- At long prompt length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).

## At a Glance

- _Observed:_ Output indicates a likely integration issue.
- _Likely owner:_ `mlx-vlm / mlx`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate long-context handling and stop-token
  behavior across mlx-vlm + mlx runtime.
- _Token summary:_ prompt=4,101, output=3, output/prompt=0.07%


## Maintainer Triage

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ context_budget \| long_context
- _Summary:_ Output appears truncated to about 3 tokens. \| At long prompt
  length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).
  \| output/prompt=0.07% \| nontext prompt burden=100%
- _Evidence:_ Output appears truncated to about 3 tokens. \| At long prompt
  length (4101 tokens), output stayed unusually short (3 tokens; ratio 0.1%).
- _Token context:_ prompt=4,101 \| output/prompt=0.07% \| nontext burden=100%
  \| stop=completed
- _Next action:_ Treat this as a prompt-budget issue first; nontext prompt
  burden is 100% and the output stays weak under that load.


## Reproducibility

### Repro Command

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt Describe this picture --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-3b-pt-896-4bit
```

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260501+e8ebdebe |
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

