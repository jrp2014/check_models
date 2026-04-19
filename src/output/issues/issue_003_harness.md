# [Harness Issue] <|endoftext|> in mlx-community/Qwen2-VL-2B-Instruct-4bit

## Description

Integration/harness warning detected for `mlx-community/Qwen2-VL-2B-Instruct-4bit`.

### Details

- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output appears truncated to about 2 tokens.
- At long prompt length (16731 tokens), output stayed unusually short (2 tokens; ratio 0.0%).

## Maintainer Triage

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;\|endoftext\|&gt; appeared in generated
           text. \| Output appears truncated to about 2 tokens. \| nontext
           prompt burden=97% \| missing terms: view, Round, Tower, Windsor,
           Castle
_Evidence:_ Special control token &lt;\|endoftext\|&gt; appeared in generated
            text. \| Output appears truncated to about 2 tokens.
_Token context:_ prompt=16,731 \| output/prompt=0.01% \| nontext burden=97% \|
                 stop=completed
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.


## Reproducibility

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen2-VL-2B-Instruct-4bit
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

