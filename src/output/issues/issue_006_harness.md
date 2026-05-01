# [Harness Issue] I'm sorry, but the context didn't show up in mlx-community/Qwen2-VL-2B-Instruct-4bit

## Description

Integration/harness warning detected for `mlx-community/Qwen2-VL-2B-Instruct-4bit`.

### Details

- Output is very short relative to prompt size (0.1%), suggesting possible early-stop or prompt-handling issues.
- At long prompt length (16299 tokens), output stayed unusually short (13 tokens; ratio 0.1%).

## Maintainer Triage

_Likely owner:_ mlx \| confidence=high
_Classification:_ context_budget \| long_context
_Summary:_ Output is very short relative to prompt size (0.1%), suggesting
           possible early-stop or prompt-handling issues. \| At long prompt
           length (16299 tokens), output stayed unusually short (13 tokens;
           ratio 0.1%). \| output/prompt=0.08% \| nontext prompt burden=100%
_Evidence:_ Output is very short relative to prompt size (0.1%), suggesting
            possible early-stop or prompt-handling issues. \| At long prompt
            length (16299 tokens), output stayed unusually short (13 tokens;
            ratio 0.1%).
_Token context:_ prompt=16,299 \| output/prompt=0.08% \| nontext burden=100%
                 \| stop=completed
_Next action:_ Treat this as a prompt-budget issue first; nontext prompt
               burden is 100% and the output stays weak under that load.


## Reproducibility

### Repro Command

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt Describe this picture --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen2-VL-2B-Instruct-4bit
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

