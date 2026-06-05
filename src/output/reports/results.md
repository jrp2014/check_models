# Model Performance Results

_Generated on 2026-06-05 22:45:14 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ none.
- _Maintainer signals:_ harness-risk successes=0, mechanically clean
  outputs=1/1.
- _Useful now:_ 1 mechanically clean A/B model(s) with useful signals.
- _Review watchlist:_ none.

### Quality & Metadata

- _Quality signal frequency:_ none.

### Runtime

- _Runtime pattern:_ upstream prefill / first-token dominates measured phase
  time (87%; 1/1 measured model(s)).
- _Phase totals:_ model load=0.70s, local prompt prep=0.04s, upstream prefill
  / first-token=13.73s, post-prefill decode=1.32s, cleanup=0.06s.
- _Generation total:_ 15.05s across 1 model(s); upstream prefill / first-token
  split available for 1/1 model(s).
- _What this likely means:_ Most measured runtime is spent inside upstream
  generation before the first token is available.
- _Suggested next action:_ Inspect prompt/image token accounting,
  dynamic-resolution image burden, prefill step sizing, and cache/prefill
  behavior.
- _Termination reasons:_ completed=1.
- _Validation overhead:_ 0.11s total (avg 0.11s across 1 model(s)).
- _Upstream prefill / first-token latency:_ Avg 13.73s | Min 13.73s | Max
  13.73s across 1 model(s).

## 🏆 Performance Highlights

- **Fastest:** `Qwen/Qwen3-VL-2B-Instruct` (93.5 tps)
- **💾 Most efficient:** `Qwen/Qwen3-VL-2B-Instruct` (8.6 GB)
- **⚡ Fastest load:** `Qwen/Qwen3-VL-2B-Instruct` (0.70s)
- **📊 Average TPS:** 93.5 across 1 models

## 📈 Resource Usage

- **Input image size:** 66.45 MP
- **Average peak delta from post-load:** 4.62 GB
- **Peak memory delta / MP:** 71 MB/MP
- **Average peak memory:** 8.6 GB
- **Memory efficiency:** 1899 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** ✅ B: 1

**Average Utility Score:** 75/100

- **Best for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (✅ B, 75/100)
- **Best descriptions:** `Qwen/Qwen3-VL-2B-Instruct` (87/100)
- **Best keywording:** `Qwen/Qwen3-VL-2B-Instruct` (0/100)
- **Worst for cataloging:** `Qwen/Qwen3-VL-2B-Instruct` (✅ B, 75/100)

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 93.5 | Min: 93.5 | Max: 93.5
- **Peak Memory**: Avg: 8.6 | Min: 8.6 | Max: 8.6
- **Total Time**: Avg: 15.91s | Min: 15.91s | Max: 15.91s
- **Generation Time**: Avg: 15.05s | Min: 15.05s | Max: 15.05s
- **Model Load Time**: Avg: 0.70s | Min: 0.70s | Max: 0.70s

## ✅ Usable Diagnostic Candidates

Models that passed mechanical diagnostics and retained useful cataloguing or
metadata-alignment signals.

- _Best end-to-end cataloging:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 93.5 TPS | Memory
  8.6 | Caveat nontext prompt burden=100%)
- _Best descriptions:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 93.5 TPS | Memory
  8.6 | Caveat nontext prompt burden=100%)
- _Best keywording:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 93.5 TPS | Memory
  8.6 | Caveat nontext prompt burden=100%)
- _Fastest generation:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 93.5 TPS | Memory
  8.6 | Caveat nontext prompt burden=100%)
- _Lowest memory footprint:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 93.5 TPS | Memory
  8.6 | Caveat nontext prompt burden=100%)
- _Best balance:_ [`Qwen/Qwen3-VL-2B-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-qwen-qwen3-vl-2b-instruct)
  (Utility B 75/100 | Description 87 | Keywords 0 | Speed 93.5 TPS | Memory
  8.6 | Caveat nontext prompt burden=100%)

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 16.47s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                  |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Cached Tokens | Finish Reason   |   Diffusion Canvas Tokens | Diffusion Denoising Steps   |   Diffusion Work Tokens |   Diffusion Canvas Tps |   Diffusion Work Tps | Is Draft   | Draft Text   | Text Already Printed   | Diffusion Step   | Diffusion Total Steps   | Diffusion Canvas Index   |   Generation (s) |   Load (s) |   Total (s) | Quality Issues   |   Error Package |
|:----------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|----------------:|:----------------|--------------------------:|:----------------------------|------------------------:|-----------------------:|---------------------:|:-----------|:-------------|:-----------------------|:-----------------|:------------------------|:-------------------------|-----------------:|-----------:|------------:|:-----------------|----------------:|
| `Qwen/Qwen3-VL-2B-Instruct` |            16,239 |                    64 |         16,303 |        1,182 |      93.5 |         8.6 |               0 | stop            |                         0 | 0                           |                       0 |                      0 |                    0 | 0          |              | 0                      | 0                | 0                       | 0                        |           15.05s |      0.70s |      15.91s |                  |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5.1
- _SDK Version:_ 26.5
- _Active Developer Directory:_ /Applications/Xcode.app/Contents/Developer
- _Xcode Version:_ 26.5
- _Xcode Build:_ 17F42
- _SDK Path:_ /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
- _Metal SDK:_ MacOSX26.5.sdk
- _Metal Compiler Version:_ Apple metal version 32023.883 (metalfe-32023.883)
- _Metallib Linker Version:_ AIR-LLD 32023.883 (metalfe-32023.883) (compatible
  with legacy metallib linker)
- _Apple Clang Version:_ Apple clang version 21.0.0 (clang-2100.1.1.101)
- _Python Version:_ 3.13.13
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18
- _MLX Install Type:_ editable local source
- _MLX Distribution Root:_ /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages
- _mlx-metal Distribution:_ not installed; local editable mlx supplies backend
- _MLX Core Extension:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so
- _MLX Metallib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib
  (157,751,704 bytes,
  sha256=e50cd0c2d2ae16781f644476459cbc2ca23b0d428a897140740f86678b5e2bf5)
- _MLX libmlx.dylib:_ /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib
  (21,675,136 bytes,
  sha256=7c19dc6a8bf6b56db0155bed7a1d2b0a54182b42eba2fcf0f42b97d3d4c57eff)

## Library Versions

- `numpy`: `2.4.6`
- `mlx`: `0.32.0.dev20260605+6ea7a00d`
- `mlx-vlm`: `0.6.1`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.18.0`
- `transformers`: `5.10.2`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-06-05 22:45:14 BST_
