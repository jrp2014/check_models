# [Bug] MLX_VLM_DECODE_MODEL:1e0557a38f91

## Description

A runtime failure occurred affecting **4 model(s)**.

### Affected Models

- `Qwen/Qwen3-VL-2B-Instruct`
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`
- `mlx-community/X-Reasoner-7B-8bit`

## Traceback / Error Message

```text
Model generation failed for Qwen/Qwen3-VL-2B-Instruct: [broadcast_shapes] Shapes (3,1,4096) and (3,1,16228) cannot be broadcast.

```

## Reproducibility

A reproduction bundle is available at: `20260419T010359Z_001_Qwen_Qwen3-VL-2B-Instruct_MLX_VLM_DECODE_MODEL_1e0557a38f91.json`

### Repro Command

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models Qwen/Qwen3-VL-2B-Instruct

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
