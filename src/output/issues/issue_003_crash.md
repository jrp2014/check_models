# [Bug] MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR:a6e1fb5a9278

## Description
A runtime failure occurred affecting **1 model(s)**.

### Affected Models
- `mlx-community/MolmoPoint-8B-fp16`

## Traceback / Error Message
```text
Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

## Reproducibility
A reproduction bundle is available at: `20260419T010359Z_003_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json`

### Repro Command
```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
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

