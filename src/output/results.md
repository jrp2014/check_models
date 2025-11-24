# Model Performance Results

_Generated on 2025-11-24 23:09:08 GMT_

## 🏆 Performance Highlights

- **Fastest:** `Qwen/Qwen3-VL-2B-Instruct` (103.3 tps)
- **💾 Most efficient:** `Qwen/Qwen3-VL-2B-Instruct` (5.8 GB)
- **⚡ Fastest load:** `Qwen/Qwen3-VL-2B-Instruct` (0.00s)
- **📊 Average TPS:** 103.3 across 1 models

## 📈 Resource Usage

- **Total peak memory:** 5.8 GB
- **Average peak memory:** 5.8 GB
- **Memory efficiency:** 448 tokens/GB

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 103 | Min: 103 | Max: 103
- **Peak Memory**: Avg: 5.8 | Min: 5.8 | Max: 5.8
- **Total Time**: Avg: 7.16s | Min: 7.16s | Max: 7.16s
- **Generation Time**: Avg: 6.45s | Min: 6.45s | Max: 6.45s
- **Model Load Time**: Avg: 0.71s | Min: 0.71s | Max: 0.71s

> **Prompt used:**
>
> Provide a factual caption, description, and keywords suitable for cataloguing, or searching for, the image.
>
> The photo was taken around 2010-09-30 12:58:31 BST . Focus on visual content, drawing on any available contextual information for specificity. Do not speculate.

**Note:** Results sorted: errors first, then by generation time (fastest to slowest).

**Overall runtime:** 8.48s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                  |   Token |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues   |
|:----------------------------|--------:|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------|
| `Qwen/Qwen3-VL-2B-Instruct` |      11 |             2,117 |                   500 |          2,617 |        1,348 |       103 |         5.8 |            6.45s |      0.71s |       7.16s |                  |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

## Model Gallery

Full output from each model:

<!-- markdownlint-disable MD013 -->

### ✅ Qwen/Qwen3-VL-2B-Instruct

**Metrics:** 103 TPS | 500 tokens

```text
### Factual Caption, Description, and Keywords

**Caption:**  
A panoramic view of the Calton Hill Memorial in Edinburgh, Scotland, with the cityscape and surrounding greenery visible in the background. The image captures the historic stone structure with its prominent columns and a grassy foreground, taken from an elevated vantage point.

**Description:**  
The photograph showcases the Calton Hill Memorial, a prominent monument located on Calton Hill in Edinburgh. The structure, built in the neoclassical style, features a series of tall, fluted columns supporting a flat roof. The memorial is set on a raised stone platform, surrounded by lush green grass and trees with autumnal foliage. In the background, the city of Edinburgh stretches out, with a mix of residential buildings and modern structures, all under a bright, partly cloudy sky. The scene is bathed in natural daylight, highlighting the textures of the stone and the vibrant colors of the landscape.

**Keywords:**  
calton hill memorial, edinburgh, scotland, neoclassical architecture, historical monument, cityscape, greenery, autumn foliage, stone structure, elevated view, public park, urban landscape, historical site, calton hill, monument, columns, grass, sky, daylight, urban environment, scenic view, historical architecture, calton hill, edinburgh, scotland, neoclassical, memorial, stone, platform, green, trees, autumn, sky, daylight, urban, landscape, scenic, view, historical, site, calton hill, edinburgh, scotland, neoclassical, architecture, monument, stone, platform, green, trees, autumn, sky, daylight, urban, landscape, scenic, view, historical, site, calton hill, edinburgh, scotland, neoclassical, architecture, monument, stone, platform, green, trees, autumn, sky, daylight, urban, landscape, scenic, view, historical, site, calton hill, edinburgh, scotland, neoclassical, architecture, monument, stone, platform, green, trees, autumn, sky, daylight, urban, landscape, scenic, view, historical, site, calton hill, edinburgh, scotland, neoclassical, architecture, monument, stone, platform, green, trees, autumn, sky, daylight, urban, landscape, scenic, view, historical, site, calton hill, edinburgh, scotland, neoclassical,
```

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
- `mlx`: `0.30.1.dev20251123+3e05cea9`
- `mlx-lm`: `0.28.4`
- `mlx-vlm`: `0.3.7`
- `tokenizers`: `0.22.1`
- `transformers`: `4.57.1`

_Report generated on: 2025-11-24 23:09:08 GMT_
