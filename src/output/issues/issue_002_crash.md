# [Bug] MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR:a6e1fb5a9278

## Description

A runtime failure occurred affecting **1 model(s)**.

### Affected Models

- `mlx-community/MolmoPoint-8B-fp16`

## Maintainer Triage

_Likely owner:_ model-config \| confidence=high
_Classification:_ runtime_failure \| MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR
_Summary:_ processor error \| model config processor load processor
_Evidence:_ processor error \| model config processor load processor
_Token context:_ stop=exception
_Next action:_ Inspect model repo config, chat template, and EOS settings.


## Traceback / Error Message

```text
Model preflight failed for mlx-community/MolmoPoint-8B-fp16: Loaded processor has no image_processor; expected multimodal processor.
```

## Reproducibility

A reproduction bundle is available at: `20260424T215035Z_002_mlx-community_MolmoPoint-8B-fp16_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_a6.json`

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
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

