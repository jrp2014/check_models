# [Harness Issue] prompt_template in mlx-community/gemma-3n-E2B-4bit

## Description

Integration/harness warning detected for `mlx-community/gemma-3n-E2B-4bit`.

### Details

- Output appears truncated to about 4 tokens.

## At a Glance

- _Observed:_ Output shape suggests a prompt-template or stop-condition
  mismatch.
- _Likely owner:_ `model-config / mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ validate chat-template/config expectations and
  mlx-vlm prompt formatting for this model.
- _Token summary:_ prompt=264, output=4, output/prompt=1.52%


## Maintainer Triage

- _Likely owner:_ model-config \| confidence=high
- _Classification:_ harness \| prompt_template
- _Summary:_ Output appears truncated to about 4 tokens. \| nontext prompt
  burden=98%
- _Evidence:_ Output appears truncated to about 4 tokens.
- _Token context:_ prompt=264 \| output/prompt=1.52% \| nontext burden=98% \|
  stop=completed
- _Next action:_ Inspect model repo config, chat template, and EOS settings.


## Reproducibility

### Repro Command

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt Describe this picture --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-3n-E2B-4bit
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

