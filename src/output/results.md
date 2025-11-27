# Model Performance Results

_Generated on 2025-11-25 21:02:19 GMT_

## ⚠️ Quality Issues

- **❌ Failed Models (1):**
  - `mlx-community/Qwen2-VL-2B-Instruct-4bit` (`OOM`)

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 2.51s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                                |   Generation (s) |   Load (s) |   Total (s) | Quality Issues   |
|:------------------------------------------|-----------------:|-----------:|------------:|:-----------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |                  |            |             |                  |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:

<!-- markdownlint-disable MD013 -->

### ❌ mlx-community/Qwen2-VL-2B-Instruct-4bit

**Status:** Failed (OOM)
**Error:**

> Model runtime error during generation for
> mlx-community/Qwen2-VL-2B-Instruct-4bit: [metal::malloc] Attempting to
> allocate 135383101952 bytes which is greater than the maximum allowed buffer
> size of 86586540032 bytes.
**Type:** `ValueError`

---

<!-- markdownlint-enable MD013 -->

---

## System/Hardware Information

- **OS**: Darwin 25.1.0
- **macOS Version**: 26.1
- **SDK Version**: 26.1
- **Python Version**: 3.13.9
- **Architecture**: arm64
- **GPU/Chip**: Apple M4 Max
- **GPU Cores**: 40
- **Metal Support**: Metal 4
- **RAM**: 128.0 GB
- **CPU Cores (Physical)**: 16
- **CPU Cores (Logical)**: 16

## Library Versions

- `Pillow`: `12.0.0`
- `huggingface-hub`: `0.36.0`
- `mlx`: `0.30.1.dev20251125+c9f4dc85`
- `mlx-lm`: `0.28.4`
- `mlx-vlm`: `0.3.7`
- `tokenizers`: `0.22.1`
- `transformers`: `4.57.3`

_Report generated on: 2025-11-25 21:02:19 GMT_
