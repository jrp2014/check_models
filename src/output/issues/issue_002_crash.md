# [Bug] MODEL_CONFIG_MODEL_LOAD_MODEL:d01bffe45ed7

## Description

A runtime failure occurred affecting **1 model(s)**.

### Affected Models

- `ggml-org/gemma-3-1b-it-GGUF`

## Traceback / Error Message

```text
Model loading failed: Config not found at /Users/jrp/.cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27

```

## Reproducibility

A reproduction bundle is available at: `20260419T010359Z_002_ggml-org_gemma-3-1b-it-GGUF_MODEL_CONFIG_MODEL_LOAD_MODEL_d01bffe45e.json`

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models ggml-org/gemma-3-1b-it-GGUF

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
