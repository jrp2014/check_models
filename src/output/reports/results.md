# Model Performance Results

_Generated on 2026-05-17 00:26:31 BST_

## 🎯 Action Snapshot

### Failures & Triage

- _Framework/runtime failures:_ none.
- _Maintainer signals:_ harness-risk successes=0, clean outputs=0/1.
- _Useful now:_ none (no clean A/B shortlist for this run).
- _Review watchlist:_ 1 model(s) with breaking or lower-value output.

### Quality & Metadata

- _Quality signal frequency:_ cutoff=1.

### Runtime

- _Runtime pattern:_ post-prefill decode dominates measured phase time (62%;
  1/1 measured model(s)).
- _Phase totals:_ model load=0.61s, local prompt prep=0.03s, upstream prefill
  / first-token=0.43s, post-prefill decode=1.85s, cleanup=0.05s.
- _Generation total:_ 2.28s across 1 model(s); upstream prefill / first-token
  split available for 1/1 model(s).
- _What this likely means:_ Most measured runtime is spent generating after
  the first token is available.
- _Suggested next action:_ Prioritize early-stop policies, lower long-tail
  token budgets, or upstream decode-path work.
- _Termination reasons:_ max_tokens=1.
- _Validation overhead:_ 0.18s total (avg 0.18s across 1 model(s)).
- _Upstream prefill / first-token latency:_ Avg 0.43s | Min 0.43s | Max 0.43s
  across 1 model(s).

## 🏆 Performance Highlights

- **Fastest:** `HuggingFaceTB/SmolVLM-Instruct` (129.3 tps)
- **💾 Most efficient:** `HuggingFaceTB/SmolVLM-Instruct` (5.5 GB)
- **⚡ Fastest load:** `HuggingFaceTB/SmolVLM-Instruct` (0.61s)
- **📊 Average TPS:** 129.3 across 1 models

## 📈 Resource Usage

- **Total peak memory:** 5.5 GB
- **Average peak memory:** 5.5 GB
- **Memory efficiency:** 255 tokens/GB

## 📚 Cataloging Utility Summary

**Grade Distribution:** 🟠 D: 1

**Average Utility Score:** 45/100

- **Best for cataloging:** `HuggingFaceTB/SmolVLM-Instruct` (🟠 D, 45/100)
- **Worst for cataloging:** `HuggingFaceTB/SmolVLM-Instruct` (🟠 D, 45/100)

### ⚠️ 1 Models with Low Utility (D/F)

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) - Keywords are not specific or diverse enough

## 📊 Aggregate Statistics (Successful Runs)

- **Generation Tps**: Avg: 129 | Min: 129 | Max: 129
- **Peak Memory**: Avg: 5.5 | Min: 5.5 | Max: 5.5
- **Total Time**: Avg: 3.10s | Min: 3.10s | Max: 3.10s
- **Generation Time**: Avg: 2.28s | Min: 2.28s | Max: 2.28s
- **Model Load Time**: Avg: 0.61s | Min: 0.61s | Max: 0.61s

## ✅ Recommended Models

Quick picks based on end-to-end utility plus description and keyword strength.

- _Fastest generation:_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (Utility D 45/100 | Description 60 | Keywords 0 | Speed 129 TPS | Memory 5.5
  | Caveat hit token cap (200); nontext prompt burden=99%)
- _Lowest memory footprint:_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct)
  (Utility D 45/100 | Description 60 | Keywords 0 | Speed 129 TPS | Memory 5.5
  | Caveat hit token cap (200); nontext prompt burden=99%)

## 🔍 Quality Pattern Breakdown

Common weaknesses and failure patterns from this run, linked to the gallery
when available.

- _Low-utility outputs (1):_ [`HuggingFaceTB/SmolVLM-Instruct`](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md#model-huggingfacetb-smolvlm-instruct).
  Common weakness: Keywords are not specific or diverse enough.

_Prompt used:_

<!-- markdownlint-disable MD011 MD028 MD037 MD045 -->
>
> Describe this image briefly.
<!-- markdownlint-enable MD011 MD028 MD037 MD045 -->

_Note:_ Results sorted: errors first, then by generation time (fastest to
slowest).

_Overall runtime:_ 3.43s

<!-- markdownlint-disable MD013 MD033 MD034 MD037 MD049 -->

| Model Name                       |   Prompt (tokens) |   Generation (tokens) |   Total Tokens |   Prompt Tps |   Gen TPS |   Peak (GB) |   Generation (s) |   Load (s) |   Total (s) | Quality Issues   |   Error Package |
|:---------------------------------|------------------:|----------------------:|---------------:|-------------:|----------:|------------:|-----------------:|-----------:|------------:|:-----------------|----------------:|
| `HuggingFaceTB/SmolVLM-Instruct` |             1,196 |                   200 |          1,396 |        2,808 |       129 |         5.5 |            2.28s |      0.61s |       3.10s | cutoff           |                 |

<!-- markdownlint-enable MD013 MD033 MD034 MD037 MD049 -->

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Automated review digest:_ [review.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/review.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

---

## System/Hardware Information

- _OS:_ Darwin 25.5.0
- _macOS Version:_ 26.5
- _SDK Version:_ 26.5
- _Xcode Version:_ 26.5
- _Xcode Build:_ 17F42
- _Metal SDK:_ MacOSX.sdk
- _Python Version:_ 3.13.13
- _Architecture:_ arm64
- _GPU/Chip:_ Apple M5 Max
- _GPU Cores:_ 40
- _Metal Support:_ Metal 4
- _RAM:_ 128.0 GB
- _CPU Cores (Physical):_ 18
- _CPU Cores (Logical):_ 18

## Library Versions

- `numpy`: `2.4.5`
- `mlx`: `0.32.0.dev20260516+7b7c1240`
- `mlx-vlm`: `0.5.0`
- `mlx-lm`: `0.31.3`
- `mlx-audio`: `0.4.3`
- `huggingface-hub`: `1.15.0`
- `transformers`: `5.8.1`
- `tokenizers`: `0.22.2`
- `Pillow`: `12.2.0`

_Report generated on: 2026-05-17 00:26:31 BST_
