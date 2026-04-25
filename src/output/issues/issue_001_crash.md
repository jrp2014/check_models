# [Bug] MODEL_CONFIG_MODEL_LOAD_MODEL:d01bffe45ed7

## Description

A runtime failure occurred affecting **1 model(s)**.

### Affected Models

- `ggml-org/gemma-3-1b-it-GGUF`

## Maintainer Triage

_Likely owner:_ model-config \| confidence=high
_Classification:_ runtime_failure \| MODEL_CONFIG_MODEL_LOAD_MODEL
_Summary:_ model error \| model config model load model
_Evidence:_ model error \| model config model load model
_Token context:_ stop=exception
_Next action:_ Inspect model repo config, chat template, and EOS settings.


## Traceback / Error Message

```text
Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27
```

## Reproducibility

A reproduction bundle is available at: `20260425T231030Z_001_ggml-org_gemma-3-1b-it-GGUF_MODEL_CONFIG_MODEL_LOAD_MODEL_d01bffe45e.json`

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models ggml-org/gemma-3-1b-it-GGUF
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

