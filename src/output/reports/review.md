<!-- markdownlint-disable MD012 MD013 -->

# Automated Review Digest

_Generated on 2026-06-05 22:45:14 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Strong Candidates

- `Qwen/Qwen3-VL-2B-Instruct`: ✅ B (75/100) | Desc 87 | Keywords 0 | 93.5 tps

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

| Model                       | Verdict   | Hint Handling           | Key Evidence               |
|-----------------------------|-----------|-------------------------|----------------------------|
| `Qwen/Qwen3-VL-2B-Instruct` | `clean`   | preserves trusted hints | nontext prompt burden=100% |

### `caveat`

- None.

### `needs_triage`

- None.

### `avoid`

- None.

## Model Verdicts

### `Qwen/Qwen3-VL-2B-Instruct`

- _Recommendation:_ recommended; review verdict: clean
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ nontext prompt burden=100%
- _Tokens:_ prompt 16239 tok; estimated text 6 tok; estimated non-text 16233
  tok; generated 64 tok; requested max 200 tok; stop reason completed

