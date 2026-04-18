# Diagnostics Report — 53 failure(s), 0 harness issue(s) (mlx-vlm 0.4.4)

## Summary

Automated benchmarking of **54 locally-cached VLM models** found **53 hard
failure(s)** and **0 harness/integration issue(s)** plus **1 preflight
compatibility warning(s)** in successful models. 1 of 54 models succeeded.

Test image: `20260418-171010_DSC09799.jpg` (35.9 MB).

---

## Action Summary

Quick triage list with likely owner and next action for each issue class.

- **[Critical] [huggingface_hub]** Model loading failed: [Errno 32] Broken pipe (53 model(s)). Next: check cache/revision availability and network/auth state; Hub disconnects may be transient outages rather than model defects.
- **[Medium] [transformers]** Preflight compatibility warnings (1 issue(s)). Next: verify API compatibility and pinned version floor.

---

## Priority Summary

| Priority     | Issue                                        | Models Affected                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Owner             | Next Action                                                                                                                   |
|--------------|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Critical** | Model loading failed: [Errno 32] Broken pipe | 53 (Qwen3-VL-2B-Instruct, gemma-3-1b-it-GGUF, Kimi-VL-A3B-Thinking-2506-6bit, Llama-3.2-11B-Vision-Instruct, Phi-3.5-vision-instruct, Apriel-1.5-15b-Thinker-6bit-MLX, Devstral-Small-2-24B-Instruct-2512-5bit, ERNIE-4.5-VL-28B-A3B-Thinking-bf16, FastVLM-0.5B-bf16, GLM-4.6V-Flash-6bit, GLM-4.6V-Flash-mxfp4, GLM-4.6V-nvfp4, Idefics3-8B-Llama3-bf16, InternVL3-14B-8bit, InternVL3-8B-bf16, Kimi-VL-A3B-Thinking-2506-bf16, Kimi-VL-A3B-Thinking-8bit, LFM2-VL-1.6B-8bit, LFM2.5-VL-1.6B-bf16, Llama-3.2-11B-Vision-Instruct-8bit, Ministral-3-14B-Instruct-2512-mxfp4, Ministral-3-14B-Instruct-2512-nvfp4, Ministral-3-3B-Instruct-2512-4bit, Molmo-7B-D-0924-8bit, Molmo-7B-D-0924-bf16, MolmoPoint-8B-fp16, Phi-3.5-vision-instruct-bf16, Qwen2-VL-2B-Instruct-4bit, Qwen3-VL-2B-Thinking-bf16, Qwen3.5-27B-4bit, Qwen3.5-27B-mxfp8, Qwen3.5-35B-A3B-4bit, Qwen3.5-35B-A3B-6bit, Qwen3.5-35B-A3B-bf16, Qwen3.5-9B-MLX-4bit, SmolVLM-Instruct-bf16, SmolVLM2-2.2B-Instruct-mlx, X-Reasoner-7B-8bit, gemma-3-27b-it-qat-4bit, gemma-3-27b-it-qat-8bit, gemma-3n-E2B-4bit, gemma-3n-E4B-it-bf16, gemma-4-31b-bf16, gemma-4-31b-it-4bit, llava-v1.6-mistral-7b-8bit, nanoLLaVA-1.5-4bit, paligemma2-10b-ft-docci-448-6bit, paligemma2-10b-ft-docci-448-bf16, paligemma2-3b-ft-docci-448-bf16, paligemma2-3b-pt-896-4bit, pixtral-12b-8bit, pixtral-12b-bf16, nanoLLaVA) | `huggingface_hub` | check cache/revision availability and network/auth state; Hub disconnects may be transient outages rather than model defects. |
| **Medium**   | Preflight compatibility warning              | 1 issue(s)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `transformers`    | verify API compatibility and pinned version floor.                                                                            |

---

## 1. Failure affecting 53 models (Priority: Critical)

**Observed behavior:** Model loading failed: [Errno 32] Broken pipe
**Owner (likely component):** `huggingface_hub`
**Suggested next action:** check cache/revision availability and network/auth state; Hub disconnects may be transient outages rather than model defects.
**Affected models:** `Qwen/Qwen3-VL-2B-Instruct`, `ggml-org/gemma-3-1b-it-GGUF`, `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`, `meta-llama/Llama-3.2-11B-Vision-Instruct`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`, `mlx-community/FastVLM-0.5B-bf16`, `mlx-community/GLM-4.6V-Flash-6bit`, `mlx-community/GLM-4.6V-Flash-mxfp4`, `mlx-community/GLM-4.6V-nvfp4`, `mlx-community/Idefics3-8B-Llama3-bf16`, `mlx-community/InternVL3-14B-8bit`, `mlx-community/InternVL3-8B-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`, `mlx-community/Kimi-VL-A3B-Thinking-8bit`, `mlx-community/LFM2-VL-1.6B-8bit`, `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`, `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`, `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`, `mlx-community/Ministral-3-3B-Instruct-2512-4bit`, `mlx-community/Molmo-7B-D-0924-8bit`, `mlx-community/Molmo-7B-D-0924-bf16`, `mlx-community/MolmoPoint-8B-fp16`, `mlx-community/Phi-3.5-vision-instruct-bf16`, `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-27B-4bit`, `mlx-community/Qwen3.5-27B-mxfp8`, `mlx-community/Qwen3.5-35B-A3B-4bit`, `mlx-community/Qwen3.5-35B-A3B-6bit`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.5-9B-MLX-4bit`, `mlx-community/SmolVLM-Instruct-bf16`, `mlx-community/SmolVLM2-2.2B-Instruct-mlx`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/gemma-3-27b-it-qat-4bit`, `mlx-community/gemma-3-27b-it-qat-8bit`, `mlx-community/gemma-3n-E2B-4bit`, `mlx-community/gemma-3n-E4B-it-bf16`, `mlx-community/gemma-4-31b-bf16`, `mlx-community/gemma-4-31b-it-4bit`, `mlx-community/llava-v1.6-mistral-7b-8bit`, `mlx-community/nanoLLaVA-1.5-4bit`, `mlx-community/paligemma2-10b-ft-docci-448-6bit`, `mlx-community/paligemma2-10b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-ft-docci-448-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`, `mlx-community/pixtral-12b-8bit`, `mlx-community/pixtral-12b-bf16`, `qnguyen3/nanoLLaVA`

| Model                                                   | Observed Behavior                            | First Seen Failing      | Recent Repro           |
|---------------------------------------------------------|----------------------------------------------|-------------------------|------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`                             | Model loading failed: [Errno 32] Broken pipe | 2026-04-17 13:13:03 BST | 3/3 recent runs failed |
| `ggml-org/gemma-3-1b-it-GGUF`                           | Model loading failed: [Errno 32] Broken pipe | 2026-04-18 00:45:49 BST | 3/3 recent runs failed |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `microsoft/Phi-3.5-vision-instruct`                     | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/FastVLM-0.5B-bf16`                       | Model loading failed: [Errno 32] Broken pipe | 2026-02-13 23:33:19 GMT | 3/3 recent runs failed |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/GLM-4.6V-nvfp4`                          | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-14B-8bit`                      | Model loading failed: [Errno 32] Broken pipe | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-8B-bf16`                       | Model loading failed: [Errno 32] Broken pipe | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | Model loading failed: [Errno 32] Broken pipe | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`                      | Model loading failed: [Errno 32] Broken pipe | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | Model loading failed: [Errno 32] Broken pipe | 2026-04-17 13:13:03 BST | 3/3 recent runs failed |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | Model loading failed: [Errno 32] Broken pipe | 2026-04-17 13:13:03 BST | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-27B-4bit`                        | Model loading failed: [Errno 32] Broken pipe | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | Model loading failed: [Errno 32] Broken pipe | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | Model loading failed: [Errno 32] Broken pipe | 2026-04-06 17:15:49 BST | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | Model loading failed: [Errno 32] Broken pipe | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | Model loading failed: [Errno 32] Broken pipe | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/SmolVLM-Instruct-bf16`                   | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/X-Reasoner-7B-8bit`                      | Model loading failed: [Errno 32] Broken pipe | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/gemma-3n-E2B-4bit`                       | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/gemma-4-31b-bf16`                        | Model loading failed: [Errno 32] Broken pipe | 2026-04-10 14:13:07 BST | 3/3 recent runs failed |
| `mlx-community/gemma-4-31b-it-4bit`                     | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | Model loading failed: [Errno 32] Broken pipe | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | Model loading failed: [Errno 32] Broken pipe | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | Model loading failed: [Errno 32] Broken pipe | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | Model loading failed: [Errno 32] Broken pipe | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/pixtral-12b-8bit`                        | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/pixtral-12b-bf16`                        | Model loading failed: [Errno 32] Broken pipe | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `qnguyen3/nanoLLaVA`                                    | Model loading failed: [Errno 32] Broken pipe | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |

### To reproduce

- Exact model-specific repro command appears below in the `Reproducibility` section under `Target specific failing models`.
- Representative failing model: `Qwen/Qwen3-VL-2B-Instruct`

<details>
<summary>Detailed trace logs (affected models)</summary>

#### `Qwen/Qwen3-VL-2B-Instruct`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `ggml-org/gemma-3-1b-it-GGUF`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `meta-llama/Llama-3.2-11B-Vision-Instruct`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `microsoft/Phi-3.5-vision-instruct`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/FastVLM-0.5B-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/GLM-4.6V-Flash-6bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/GLM-4.6V-Flash-mxfp4`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/GLM-4.6V-nvfp4`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Idefics3-8B-Llama3-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/InternVL3-14B-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/InternVL3-8B-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Kimi-VL-A3B-Thinking-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/LFM2-VL-1.6B-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/LFM2.5-VL-1.6B-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Ministral-3-3B-Instruct-2512-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Molmo-7B-D-0924-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Molmo-7B-D-0924-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/MolmoPoint-8B-fp16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Phi-3.5-vision-instruct-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen2-VL-2B-Instruct-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen3.5-27B-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen3.5-27B-mxfp8`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen3.5-35B-A3B-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen3.5-35B-A3B-6bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen3.5-35B-A3B-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/Qwen3.5-9B-MLX-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/SmolVLM-Instruct-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/SmolVLM2-2.2B-Instruct-mlx`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/X-Reasoner-7B-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/gemma-3-27b-it-qat-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/gemma-3-27b-it-qat-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/gemma-3n-E2B-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/gemma-3n-E4B-it-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/gemma-4-31b-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/gemma-4-31b-it-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/llava-v1.6-mistral-7b-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/nanoLLaVA-1.5-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/paligemma2-10b-ft-docci-448-6bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/paligemma2-10b-ft-docci-448-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/paligemma2-3b-ft-docci-448-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/paligemma2-3b-pt-896-4bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/pixtral-12b-8bit`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `mlx-community/pixtral-12b-bf16`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

#### `qnguyen3/nanoLLaVA`

Traceback tail:

```text
    getattr(sys.stderr, 'flush', lambda: None)()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
BrokenPipeError: [Errno 32] Broken pipe
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model loading failed: [Errno 32] Broken pipe
```

Captured stdout/stderr:

```text
=== STDERR ===
--- Logging error ---
--- Logging error ---
```

</details>

---

## Preflight Compatibility Warnings (1 issue(s))

These warnings were detected before inference. They are informational by
default and do not invalidate successful runs on their own.
Keep running if outputs look healthy. Escalate only when the warnings line up
with backend-import side effects, startup hangs, or runtime crashes.
Do not treat these warnings alone as a reason to set MLX_VLM_ALLOW_TF=1 or to
assume the benchmark results are bad.

### `transformers`

- Suggested tracker: `transformers` (<https://github.com/huggingface/transformers/issues/new>)
- Suggested next action: verify API compatibility and pinned version floor.
- Triage guidance: continue if runs are otherwise healthy; investigate only if this warning matches real backend symptoms in the same run.
- Warnings:
  - `transformers import utils no longer reference the TF/FLAX/JAX backend guard env vars used by check_models; backend guard hints for those backends may be ignored with this version.`


---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** `HuggingFaceTB/SmolVLM-Instruct`

| Model                                                   | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|---------------------------------------------------------|--------------------------|-------------------------|------------------------|
| `Qwen/Qwen3-VL-2B-Instruct`                             | still failing            | 2026-04-17 13:13:03 BST | 3/3 recent runs failed |
| `ggml-org/gemma-3-1b-it-GGUF`                           | still failing            | 2026-04-18 00:45:49 BST | 3/3 recent runs failed |
| `jqlive/Kimi-VL-A3B-Thinking-2506-6bit`                 | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `meta-llama/Llama-3.2-11B-Vision-Instruct`              | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `microsoft/Phi-3.5-vision-instruct`                     | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`         | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16`      | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/FastVLM-0.5B-bf16`                       | still failing            | 2026-02-13 23:33:19 GMT | 3/3 recent runs failed |
| `mlx-community/GLM-4.6V-Flash-6bit`                     | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/GLM-4.6V-Flash-mxfp4`                    | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/GLM-4.6V-nvfp4`                          | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Idefics3-8B-Llama3-bf16`                 | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-14B-8bit`                      | still failing            | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |
| `mlx-community/InternVL3-8B-bf16`                       | still failing            | 2026-02-23 12:54:48 GMT | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-2506-bf16`          | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Kimi-VL-A3B-Thinking-8bit`               | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2-VL-1.6B-8bit`                       | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/LFM2.5-VL-1.6B-bf16`                     | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/Llama-3.2-11B-Vision-Instruct-8bit`      | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Ministral-3-14B-Instruct-2512-mxfp4`     | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4`     | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Ministral-3-3B-Instruct-2512-4bit`       | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Molmo-7B-D-0924-8bit`                    | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Molmo-7B-D-0924-bf16`                    | still failing            | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`                      | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/Phi-3.5-vision-instruct-bf16`            | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`               | still failing            | 2026-04-17 13:13:03 BST | 3/3 recent runs failed |
| `mlx-community/Qwen3-VL-2B-Thinking-bf16`               | still failing            | 2026-04-17 13:13:03 BST | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-27B-4bit`                        | still failing            | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-27B-mxfp8`                       | still failing            | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-4bit`                    | still failing            | 2026-04-06 17:15:49 BST | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                    | still failing            | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                    | still failing            | 2026-03-01 22:26:38 GMT | 3/3 recent runs failed |
| `mlx-community/Qwen3.5-9B-MLX-4bit`                     | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/SmolVLM-Instruct-bf16`                   | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/SmolVLM2-2.2B-Instruct-mlx`              | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/X-Reasoner-7B-8bit`                      | still failing            | 2026-02-08 22:32:28 GMT | 3/3 recent runs failed |
| `mlx-community/gemma-3-27b-it-qat-4bit`                 | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/gemma-3-27b-it-qat-8bit`                 | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/gemma-3n-E2B-4bit`                       | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/gemma-3n-E4B-it-bf16`                    | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/gemma-4-31b-bf16`                        | still failing            | 2026-04-10 14:13:07 BST | 3/3 recent runs failed |
| `mlx-community/gemma-4-31b-it-4bit`                     | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/llava-v1.6-mistral-7b-8bit`              | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/nanoLLaVA-1.5-4bit`                      | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/paligemma2-10b-ft-docci-448-6bit`        | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-10b-ft-docci-448-bf16`        | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-3b-ft-docci-448-bf16`         | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/paligemma2-3b-pt-896-4bit`               | still failing            | 2026-02-07 20:59:01 GMT | 3/3 recent runs failed |
| `mlx-community/pixtral-12b-8bit`                        | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |
| `mlx-community/pixtral-12b-bf16`                        | still failing            | 2026-03-22 01:27:09 GMT | 3/3 recent runs failed |
| `qnguyen3/nanoLLaVA`                                    | still failing            | 2026-04-19 00:39:44 BST | 3/3 recent runs failed |

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 53
- **Summary diagnostics models:** 1
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 45.03s (45.03s)
- **Average runtime per model:** 0.83s (0.83s)
- **Dominant runtime phase:** decode dominated 1/54 measured model runs (52% of tracked runtime).
- **Phase totals:** model load=9.29s, prompt prep=0.00s, decode=15.88s, cleanup=5.11s
- **Observed stop reasons:** completed=1, exception=53
- **Validation overhead:** 19.77s total (avg 0.37s across 54 model(s)).
- **First-token latency:** Avg 4.83s | Min 4.83s | Max 4.83s across 1 model(s).
- **What this likely means:** Most measured runtime is spent inside generation rather than load or prompt setup.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (1 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly).

### Ran, but with quality warnings (1 model(s))

- `HuggingFaceTB/SmolVLM-Instruct`: Model output may not follow prompt or image contents (missing: 10 Best, 10 Best (structured), Bay Window, Berkshire,...

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

## Reproducibility

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

### Portable triage (no local image required)

```bash
# Capture dependency versions
python -m pip show mlx mlx-vlm mlx-lm transformers huggingface-hub tokenizers

# Verify imports with explicit pass/fail output
python - <<'PY'
import importlib
packages = ('mlx', 'mlx_vlm', 'mlx_lm', 'transformers', 'huggingface_hub', 'tokenizers')
for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, '__version__', 'unknown')
        print(f'{name} OK {version}')
    except Exception as exc:
        print(f'{name} FAIL {type(exc).__name__}: {exc}')
PY
```

### Target specific failing models

**Note:** A comprehensive JSON reproduction bundle including system info and
the exact prompt trace has been exported to
[repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles)
for each failing model.

```bash
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models Qwen/Qwen3-VL-2B-Instruct
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models ggml-org/gemma-3-1b-it-GGUF
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models jqlive/Kimi-VL-A3B-Thinking-2506-6bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models meta-llama/Llama-3.2-11B-Vision-Instruct
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Phi-3.5-vision-instruct
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/FastVLM-0.5B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/GLM-4.6V-Flash-6bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/GLM-4.6V-Flash-mxfp4
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/GLM-4.6V-nvfp4
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Idefics3-8B-Llama3-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/InternVL3-14B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/InternVL3-8B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-2506-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Kimi-VL-A3B-Thinking-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2-VL-1.6B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/LFM2.5-VL-1.6B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Llama-3.2-11B-Vision-Instruct-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Ministral-3-14B-Instruct-2512-mxfp4
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Ministral-3-14B-Instruct-2512-nvfp4
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Ministral-3-3B-Instruct-2512-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Molmo-7B-D-0924-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Molmo-7B-D-0924-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/MolmoPoint-8B-fp16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Phi-3.5-vision-instruct-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen2-VL-2B-Instruct-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3-VL-2B-Thinking-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-27B-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-27B-mxfp8
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-6bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-35B-A3B-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Qwen3.5-9B-MLX-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/SmolVLM-Instruct-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/SmolVLM2-2.2B-Instruct-mlx
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/X-Reasoner-7B-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-3-27b-it-qat-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-3-27b-it-qat-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-3n-E2B-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-3n-E4B-it-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-4-31b-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/gemma-4-31b-it-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/llava-v1.6-mistral-7b-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/nanoLLaVA-1.5-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-10b-ft-docci-448-6bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-10b-ft-docci-448-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-3b-ft-docci-448-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/paligemma2-3b-pt-896-4bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/pixtral-12b-8bit
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/pixtral-12b-bf16
python -m check_models --image /Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models qnguyen3/nanoLLaVA
```

### Prompt Used

```text
Analyze this image for cataloguing metadata, using British English.

Use only details that are clearly and definitely visible in the image. If a detail is uncertain, ambiguous, partially obscured, too small to verify, or not directly visible, leave it out. Do not guess.

Treat the metadata hints below as a draft catalog record. Keep only details that are clearly confirmed by the image, correct anything contradicted by the image, and add important visible details that are definitely present.

Return exactly these three sections, and nothing else:

Title:
- 5-10 words, concrete and factual, limited to clearly visible content.
- Output only the title text after the label.
- Do not repeat or paraphrase these instructions in the title.

Description:
- 1-2 factual sentences describing the main visible subject, setting, lighting, action, and other distinctive visible details. Omit anything uncertain or inferred.
- Output only the description text after the label.

Keywords:
- 10-18 unique comma-separated terms based only on clearly visible subjects, setting, colors, composition, and style. Omit uncertain tags rather than guessing.
- Output only the keyword list after the label.

Rules:
- Include only details that are definitely visible in the image.
- Reuse metadata terms only when they are clearly supported by the image.
- If metadata and image disagree, follow the image.
- Prefer omission to speculation.
- Do not copy prompt instructions into the Title, Description, or Keywords fields.
- Do not infer identity, location, event, brand, species, time period, or intent unless visually obvious.
- Do not output reasoning, notes, hedging, or extra sections.

Context: Existing metadata hints (high confidence; use only when visually confirmed):
- Description hint: The late afternoon sun illuminates the Tudor Revival architecture of the buildings at Lincoln's Inn in London, England. The historic red brick structures, with their distinctive diaper patterning, stone quoins, and large bay windows, are part of one of the four Inns of Court in London to which barristers of England and Wales belong.
- Keyword hints: 10 Best, 10 Best (structured), Adobe Stock, Any Vision, Bay Window, Berkshire, Blue sky, Bollard, Brickwork, Car, Chimney, Chimneys, Daylight, Door, Driving, England, Eton, Eton College, Europe, Gable
- Capture metadata: Taken on 2026-04-18 18:10:10 BST (at 18:10:10 local time). GPS: 52.203500°N, 0.113500°E.
```

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260418-171010_DSC09799.jpg`
- Generation settings: max_tokens=500, temperature=0.0, top_p=1.0

_Report generated on 2026-04-19 00:43:13 BST by [check_models](https://github.com/jrp2014/check_models)._
