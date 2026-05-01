# [Harness Issue] encoding in mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

## Description

Integration/harness warning detected for `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`.

### Details

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 139 occurrences).

## At a Glance

- _Observed:_ Decoded output contains tokenizer artifacts that should not
  appear in user-facing text.
- _Likely owner:_ `mlx-vlm`
- _Why it matters:_ The run completed, but the output pattern points to
  stack/runtime behavior rather than a clean model-quality limitation.
- _Suggested next step:_ check processor/chat-template wiring and generation
  kwargs.
- _Token summary:_ prompt=2,097, output=172, output/prompt=8.20%


## Maintainer Triage

- _Likely owner:_ mlx-vlm \| confidence=high
- _Classification:_ harness \| encoding
- _Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 139 occurrences). \| nontext prompt burden=100%
- _Evidence:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
  output (about 139 occurrences).
- _Token context:_ prompt=2,097 \| output/prompt=8.20% \| nontext burden=100%
  \| stop=completed
- _Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
  user-facing text.


## Reproducibility

### Repro Command

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt Describe this picture --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit
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

