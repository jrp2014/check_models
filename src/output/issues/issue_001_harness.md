# [Harness Issue] Title: Windsor Castle at Night

Description: The iconic Windsor Castle stands prominently at night, its lights castin... in microsoft/Phi-3.5-vision-instruct

## Description
Integration/harness warning detected for `microsoft/Phi-3.5-vision-instruct`.

### Details
- token_leak:<|end|>
- token_leak:<|endoftext|>
- training_leak:code_example

## Reproducibility

### Repro Command
```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Phi-3.5-vision-instruct
```

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.4                       |
| mlx             | 0.31.2.dev20260419+fa4320d5 |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.5.4                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.11.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

