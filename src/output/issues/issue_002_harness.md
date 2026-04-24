# [Harness Issue] Title: Windsor Castle Round Tower\n\nDescription: The Round Tower of Windsor Castle is seen across the River Thames in mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit

## Description

Integration/harness warning detected for `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`.

### Details

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 58 occurrences).

## Maintainer Triage

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| encoding
_Summary:_ Tokenizer space-marker artifacts (for example Ġ) appeared in output
           (about 58 occurrences). \| nontext prompt burden=83% \| missing
           sections: description, keywords \| missing terms: view, royal,
           residence, Berkshire, which
_Evidence:_ Tokenizer space-marker artifacts (for example Ġ) appeared in
            output (about 58 occurrences).
_Token context:_ prompt=2,654 \| output/prompt=3.17% \| nontext burden=83% \|
                 stop=completed
_Next action:_ Inspect decode cleanup; tokenizer markers are leaking into
               user-facing text.


## Reproducibility

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit
```

---

## Environment

| Component       | Version                      |
|-----------------|------------------------------|
| mlx-vlm         | 0.4.5                        |
| mlx             | 0.32.0.dev20260424+211e57be5 |
| mlx-lm          | 0.31.3                       |
| transformers    | 5.6.2                        |
| tokenizers      | 0.22.2                       |
| huggingface-hub | 1.12.0                       |
| Python Version  | 3.13.12                      |
| OS              | Darwin 25.4.0                |
| macOS Version   | 26.4.1                       |
| GPU/Chip        | Apple M5 Max                 |
| GPU Cores       | 40                           |
| Metal Support   | Metal 4                      |
| RAM             | 128.0 GB                     |

