# Automated Review Digest

_Generated on 2026-05-17 00:26:31 BST_

Trusted-hint review uses only prompt title/description/keyword hints for utility comparison. Capture metadata, GPS, timestamps, source labels, and location labels are treated as nonvisual metadata and are not required visual evidence.

_Review artifacts:_

- _Standalone output gallery:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Canonical run log:_ [check_models.log](https://github.com/jrp2014/check_models/blob/main/src/output/check_models.log)

## 🧭 Review Shortlist

### Watchlist

- `HuggingFaceTB/SmolVLM-Instruct`: 🟠 D (45/100) | Desc 60 | Keywords 0 | 129.3 tps | cutoff

## User Buckets

User-first summary grouped by recommendation bucket.

### `recommended`

- None — no models produced clean output meeting all quality thresholds for this prompt.

### `caveat`

- None.

### `needs_triage`

- None.

### `avoid`

| Model                            | Verdict           | Hint Handling           | Key Evidence                                     |
|----------------------------------|-------------------|-------------------------|--------------------------------------------------|
| `HuggingFaceTB/SmolVLM-Instruct` | `cutoff_degraded` | preserves trusted hints | hit token cap (200) \| nontext prompt burden=99% |

## Model Verdicts

### `HuggingFaceTB/SmolVLM-Instruct`

- _Recommendation:_ avoid for now; review verdict: cutoff degraded
- _Owner:_ likely owner `model`
- _Next step:_ Treat as a model-quality limitation for this prompt and image.
- _Key signals:_ hit token cap (200); nontext prompt burden=99%
- _Tokens:_ prompt 1196 tok; estimated text 6 tok; estimated non-text 1190
  tok; generated 200 tok; requested max 200 tok; stop reason max_tokens

