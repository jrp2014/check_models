# [Harness Issue] Title: - White Striped Pattern  Description: - A repeating pattern of white stripes on a dark background, resembling in mlx-community/X-Reasoner-7B-8bit

## Description

Integration/harness warning detected for `mlx-community/X-Reasoner-7B-8bit`.

### Details

- At long prompt length (16731 tokens), output may stop following prompt/image context.

## Maintainer Triage

_Likely owner:_ mlx \| confidence=high
_Classification:_ token_cap \| long_context
_Summary:_ At long prompt length (16731 tokens), output may stop following
           prompt/image context. \| hit token cap (500) \| nontext prompt
           burden=97% \| missing terms: view, Round, Tower, Windsor, Castle
_Evidence:_ At long prompt length (16731 tokens), output may stop following
            prompt/image context.
_Token context:_ prompt=16,731 \| output/prompt=2.99% \| nontext burden=97% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect long-context cache behavior under heavy image-token
               burden.


## Reproducibility

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
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

