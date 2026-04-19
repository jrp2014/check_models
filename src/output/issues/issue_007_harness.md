# [Harness Issue] - The photograph is in color in mlx-community/paligemma2-10b-ft-docci-448-bf16

## Description

Integration/harness warning detected for `mlx-community/paligemma2-10b-ft-docci-448-bf16`.

### Details

- Output is very short relative to prompt size (0.5%), suggesting possible early-stop or prompt-handling issues.

## Maintainer Triage

_Likely owner:_ model-config \| confidence=high
_Classification:_ harness \| prompt_template
_Summary:_ Output is very short relative to prompt size (0.5%), suggesting
           possible early-stop or prompt-handling issues. \| nontext prompt
           burden=71% \| missing terms: view, Round, Tower, Windsor, Castle
_Evidence:_ Output is very short relative to prompt size (0.5%), suggesting
            possible early-stop or prompt-handling issues.
_Token context:_ prompt=1,531 \| output/prompt=0.52% \| nontext burden=71% \|
                 stop=completed
_Next action:_ Inspect model repo config, chat template, and EOS settings.


## Reproducibility

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-10b-ft-docci-448-bf16
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

