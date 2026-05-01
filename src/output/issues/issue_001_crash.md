# [Bug] MLX_MODEL_LOAD_MODEL:e82eb35e5965

## Description

A runtime failure occurred affecting **1 model(s)**.

### Affected Models

- `mlx-community/Kimi-VL-A3B-Thinking-8bit`

## At a Glance

- _Observed:_ Model loading failed: Received 4 parameters not in model:
- _Likely owner:_ `mlx`
- _Why it matters:_ This prevented a complete model response; stage `Model
  Error`; phase `model_load`; code `MLX_MODEL_LOAD_MODEL`; type `ValueError`.
- _Suggested next step:_ check tensor/cache behavior and memory pressure
  handling.
- _Affected models:_ `mlx-community/Kimi-VL-A3B-Thinking-8bit`


## Maintainer Triage

- _Likely owner:_ mlx \| confidence=high
- _Classification:_ runtime_failure \| MLX_MODEL_LOAD_MODEL
- _Summary:_ model error \| mlx model load model
- _Evidence:_ model error \| mlx model load model
- _Token context:_ stop=exception
- _Next action:_ Inspect KV/cache behavior, memory pressure, and long-context
  execution.


## Traceback / Error Message

```text
Model loading failed: Received 4 parameters not in model: 
multi_modal_projector.linear_1.biases,
multi_modal_projector.linear_1.scales,
multi_modal_projector.linear_2.biases,
multi_modal_projector.linear_2.scales.
```

## Reproducibility

A reproduction bundle is available at: `20260501T205016Z_001_mlx-community_Kimi-VL-A3B-Thinking-8bit_MLX_MODEL_LOAD_MODEL_e82eb35e5965.json`

### Repro Command

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260403-124049_DSC09541.jpg --trust-remote-code --prompt Describe this picture --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
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

