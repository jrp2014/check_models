# [Bug] HUGGINGFACE_HUB_MODEL_LOAD_MODEL:254f1efe4cda

## Description
A runtime failure occurred affecting **53 model(s)**.

### Affected Models
- `Qwen/Qwen3-VL-2B-Instruct`
- `ggml-org/gemma-3-1b-it-GGUF`
- `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`
- `meta-llama/Llama-3.2-11B-Vision-Instruct`
- `microsoft/Phi-3.5-vision-instruct`
- `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`
- `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`
- `mlx-community/FastVLM-0.5B-bf16`
- `mlx-community/GLM-4.6V-Flash-6bit`
- `mlx-community/GLM-4.6V-Flash-mxfp4`
- `mlx-community/GLM-4.6V-nvfp4`
- `mlx-community/Idefics3-8B-Llama3-bf16`
- `mlx-community/InternVL3-14B-8bit`
- `mlx-community/InternVL3-8B-bf16`
- `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`
- `mlx-community/Kimi-VL-A3B-Thinking-8bit`
- `mlx-community/LFM2-VL-1.6B-8bit`
- `mlx-community/LFM2.5-VL-1.6B-bf16`
- `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`
- `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`
- `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`
- `mlx-community/Ministral-3-3B-Instruct-2512-4bit`
- `mlx-community/Molmo-7B-D-0924-8bit`
- `mlx-community/Molmo-7B-D-0924-bf16`
- `mlx-community/MolmoPoint-8B-fp16`
- `mlx-community/Phi-3.5-vision-instruct-bf16`
- `mlx-community/Qwen2-VL-2B-Instruct-4bit`
- `mlx-community/Qwen3-VL-2B-Thinking-bf16`
- `mlx-community/Qwen3.5-27B-4bit`
- `mlx-community/Qwen3.5-27B-mxfp8`
- `mlx-community/Qwen3.5-35B-A3B-4bit`
- `mlx-community/Qwen3.5-35B-A3B-6bit`
- `mlx-community/Qwen3.5-35B-A3B-bf16`
- `mlx-community/Qwen3.5-9B-MLX-4bit`
- `mlx-community/SmolVLM-Instruct-bf16`
- `mlx-community/SmolVLM2-2.2B-Instruct-mlx`
- `mlx-community/X-Reasoner-7B-8bit`
- `mlx-community/gemma-3-27b-it-qat-4bit`
- `mlx-community/gemma-3-27b-it-qat-8bit`
- `mlx-community/gemma-3n-E2B-4bit`
- `mlx-community/gemma-3n-E4B-it-bf16`
- `mlx-community/gemma-4-31b-bf16`
- `mlx-community/gemma-4-31b-it-4bit`
- `mlx-community/llava-v1.6-mistral-7b-8bit`
- `mlx-community/nanoLLaVA-1.5-4bit`
- `mlx-community/paligemma2-10b-ft-docci-448-6bit`
- `mlx-community/paligemma2-10b-ft-docci-448-bf16`
- `mlx-community/paligemma2-3b-ft-docci-448-bf16`
- `mlx-community/paligemma2-3b-pt-896-4bit`
- `mlx-community/pixtral-12b-8bit`
- `mlx-community/pixtral-12b-bf16`
- `qnguyen3/nanoLLaVA`

## Traceback / Error Message
```text
Model loading failed: [Errno 32] Broken pipe
```

## Reproducibility
A reproduction bundle is available at: `20260418T234313Z_001_Qwen_Qwen3-VL-2B-Instruct_HUGGINGFACE_HUB_MODEL_LOAD_MODEL_254f1ef.json`

### Repro Command
```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models Qwen/Qwen3-VL-2B-Instruct
```

---

## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.4                       |
| mlx             | 0.31.2.dev20260418+fa4320d5 |
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

