# [Harness Issue] Title: Windsor Castle from River Thames  Description: A view of the Round Tower of Windsor Castle, a royal residence in microsoft/Phi-3.5-vision-instruct

## Description

Integration/harness warning detected for `microsoft/Phi-3.5-vision-instruct`.

### Details

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.

## Maintainer Triage

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;\|end\|&gt; appeared in generated text.
           \| Special control token &lt;\|endoftext\|&gt; appeared in
           generated text. \| hit token cap (500) \| nontext prompt burden=66%
_Evidence:_ Special control token &lt;\|end\|&gt; appeared in generated text.
            \| Special control token &lt;\|endoftext\|&gt; appeared in
            generated text.
_Token context:_ prompt=1,317 \| output/prompt=37.97% \| nontext burden=66% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.


## Reproducibility

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Phi-3.5-vision-instruct
```

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260425+211e57be |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.6.2                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.12.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

