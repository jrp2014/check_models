# [Harness Issue] <think>Got it, let's describe this picture step by step. First, the the scene is a serene park or garden with a lake in mlx-community/GLM-4.6V-nvfp4

## Description

Integration/harness warning detected for `mlx-community/GLM-4.6V-nvfp4`.

### Details

- Special control token &lt;/think&gt; appeared in generated text.

## Maintainer Triage

_Likely owner:_ mlx-vlm \| confidence=high
_Classification:_ harness \| stop_token
_Summary:_ Special control token &lt;/think&gt; appeared in generated text. \|
           hit token cap (500) \| nontext prompt burden=100% \| reasoning leak
_Evidence:_ Special control token &lt;/think&gt; appeared in generated text.
_Token context:_ prompt=6,091 \| output/prompt=8.21% \| nontext burden=100% \|
                 stop=completed \| hit token cap (500)
_Next action:_ Inspect EOS/stop-token stripping; control tokens are leaking
               into user-facing text.


## Reproducibility

### Repro Command

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt Describe this picture --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/GLM-4.6V-nvfp4
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

