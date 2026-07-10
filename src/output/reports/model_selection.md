# Model Selection Brief

Generated on: 2026-07-10 23:37:01 BST

- Evaluation lane: assisted
- Metadata exposed to prompt: yes
- Semantic rankings: grounded (metadata-assisted visual verification)
- Primary use cases: brief captions; structured title/description/keywords
- Scope: ranked shortlist, not the complete run; complete per-model outputs and diagnostics are in `model_gallery.md`.

## Evidence Links

- _Output evidence:_ [model_gallery.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/model_gallery.md)
- _Maintainer diagnostics:_ [diagnostics.md](https://github.com/jrp2014/check_models/blob/main/src/output/reports/diagnostics.md)

## Quick Chooser

Practical current-run buckets for model users. These are triage signals, not grounded visual-quality claims.

### Best under 4 GB

- No clean current-run candidates fit under this memory budget.

### Best under 8 GB

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|-------------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |       5.1 |       221 |           96 | `caveat` | [metadata-borrowing; context-budget; missing-sections] -001.jpg - Title: Deben Estuary, Woodb... |

### Fastest usable

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                         |
|-------------------------------------------|-----------|-----------|--------------|----------|--------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |       5.1 |     221   |           96 | `caveat` | [metadata-borrowing; context-budget; missing-sections] -001.jpg - Title: Deben Estuary, Woodb... |
| `mlx-community/Qwen3.5-9B-MLX-4bit`       |      11   |      92.4 |           76 | `caveat` | [metadata-borrowing; context-budget] Title: Two moored sailboats in a river estuary ...          |
| `mlx-community/Qwen3.5-35B-A3B-6bit`      |      35   |      92.2 |           82 | `caveat` | [metadata-borrowing; context-budget] Title: Two sailing boats moored on a river with trees be... |

### Quality if memory allows

| Model                                     |   Peak GB |   Gen TPS |   Usefulness | Status        | Evidence                                                                                         |
|-------------------------------------------|-----------|-----------|--------------|---------------|--------------------------------------------------------------------------------------------------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` |       5.1 |     221   |           96 | `caveat`      | [metadata-borrowing; context-budget; missing-sections] -001.jpg - Title: Deben Estuary, Woodb... |
| `mlx-community/pixtral-12b-8bit`          |      16   |      39   |           89 | `caveat`      | [metadata-borrowing; context-budget] Title: Two Boats Moored on River ...                        |
| `microsoft/Phi-3.5-vision-instruct`       |       9.5 |      57.7 |           86 | `recommended` | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, s... |

### Current failures / avoid

| Model                                |   Peak GB |   Gen TPS |   Usefulness | Status   | Evidence                                                                                                                           |
|--------------------------------------|-----------|-----------|--------------|----------|------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/LFM2-VL-1.6B-8bit`    |       3   |     325   |           96 | `avoid`  | missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused                                                     |
| `mlx-community/Molmo-7B-D-0924-bf16` |      48   |      27   |           96 | `avoid`  | missing sections: title, description, keywords \| missing terms: Bird, Boating, Buoy, Bushes, Foliage \| nonvisual metadata reused |
| `mlx-community/Molmo-7B-D-0924-8bit` |      41   |      46.7 |           94 | `avoid`  | missing sections: title, description, keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused   |
| `qnguyen3/nanoLLaVA`                 |       4.6 |     112   |           92 | `avoid`  | missing sections: keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast \| nonvisual metadata reused                       |
| `mlx-community/InternVL3-14B-8bit`   |      18   |      31.4 |           90 | `avoid`  | missing terms: Bird, Bushes, Coast, Deben Estuary, Estuary                                                                         |

## Brief Caption Candidates

Top 10 ranked candidates for brief captions. This is a selection aid, not the complete result set.

| Model                                               |   Hygiene |   Usefulness |   Gen TPS |   Peak GB | Verdict          | Caption Preview                                                                                                                                                                      | Caveat                                                                                                                                    |
|-----------------------------------------------------|-----------|--------------|-----------|-----------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct`                 |       100 |           86 |      57.7 |       9.5 | `clean`          | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greene ... [tail] h one boat having a prominent mast and rigging. ... | missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                         |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        |       100 |           86 |      55.5 |       9.5 | `clean`          | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greene ... [tail] h one boat having a prominent mast and rigging. ... | missing terms: Bird, Boating, Buoy, Bushes, Coast                                                                                         |
| `mlx-community/Ornith-1.0-35B-bf16`                 |       100 |           79 |      60.9 |      76   | `clean`          | Title: Two sailboats moored on a river with a wooded bank Description: Two sailing boats with orange masts are moored ... [tail] oreground near the stern of the nearest vessel. ... | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                |
| `mlx-community/Qwen3.6-27B-mxfp8`                   |       100 |           78 |      17.6 |      38   | `clean`          | Title: Two sailing boats moored on a river ...                                                                                                                                       | missing terms: Bird, Boating, Bushes, Coast, Deben Estuary                                                                                |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`           |        75 |           96 |     221   |       5.1 | `context_budget` | [metadata-borrowing; context-budget; missing-sections] -001.jpg - Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe - Description: Two sailing boats moored on a river w... | output/prompt=0.26% \| nontext prompt burden=97% \| missing sections: keywords \| missing terms: Bird, Boating, Buoy, Bushes, Coast       |
| `mlx-community/pixtral-12b-8bit`                    |        75 |           89 |      39   |      16   | `context_budget` | [metadata-borrowing; context-budget] Title: Two Boats Moored on River ...                                                                                                            | output/prompt=2.35% \| nontext prompt burden=87% \| missing terms: Bird, Buoy, Bushes, Coast, Deben Estuary \| nonvisual metadata reused  |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                |        75 |           82 |      92.2 |      35   | `context_budget` | [metadata-borrowing; context-budget] Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a river, backed by... | output/prompt=0.59% \| nontext prompt burden=97% \| missing terms: Bird, Bushes, Coast, Peaceful, Woodbridge \| nonvisual metadata reused |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                |        75 |           81 |      64.6 |      76   | `context_budget` | [metadata-borrowing; context-budget] Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a calm river, back... | output/prompt=0.61% \| nontext prompt burden=97% \| missing terms: Bird, Bushes, Coast, Peaceful, Woodbridge \| nonvisual metadata reused |
| `mlx-community/GLM-4.6V-Flash-6bit`                 |        75 |           80 |      58.4 |      11   | `context_budget` | [context-budget; missing-sections] <\|begin_of_box\|>Title: Two sailing boats moored on water with trees behind ...                                                                  | output/prompt=1.41% \| nontext prompt burden=93% \| missing sections: title \| missing terms: Bird, Boating, Bushes, Coast, Deben Estuary |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` |        75 |           78 |      61.8 |      13   | `context_budget` | [metadata-borrowing; context-budget] **Title:** Sailing Boat Moored on Deben Estuary ...                                                                                             | output/prompt=3.84% \| nontext prompt burden=86% \| missing terms: Bird, Boating, Bushes, Coast, Forest \| nonvisual metadata reused      |

## Structured Metadata Candidates

Top 10 ranked candidates for structured title/description/keywords. Use the gallery for complete per-model evidence.

| Model                                               |   Metadata agreement | Verdict          | Output Preview                                                                                                                                                                       |
|-----------------------------------------------------|----------------------|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct`                 |                   26 | `clean`          | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greene ... [tail] h one boat having a prominent mast and rigging. ... |
| `mlx-community/Phi-3.5-vision-instruct-bf16`        |                   26 | `clean`          | Title: Sailing Boats at Deben Estuary Description: Two sailing boats are moored on a river, surrounded by lush greene ... [tail] h one boat having a prominent mast and rigging. ... |
| `mlx-community/Ornith-1.0-35B-bf16`                 |                   25 | `clean`          | Title: Two sailboats moored on a river with a wooded bank Description: Two sailing boats with orange masts are moored ... [tail] oreground near the stern of the nearest vessel. ... |
| `mlx-community/Qwen3.6-27B-mxfp8`                   |                   26 | `clean`          | Title: Two sailing boats moored on a river ...                                                                                                                                       |
| `mlx-community/Qwen2-VL-2B-Instruct-4bit`           |                    6 | `context_budget` | [metadata-borrowing; context-budget; missing-sections] -001.jpg - Title: Deben Estuary, Woodbridge, England, UK, GBR, Europe - Description: Two sailing boats moored on a river w... |
| `mlx-community/pixtral-12b-8bit`                    |                   17 | `context_budget` | [metadata-borrowing; context-budget] Title: Two Boats Moored on River ...                                                                                                            |
| `mlx-community/Qwen3.5-35B-A3B-6bit`                |                   12 | `context_budget` | [metadata-borrowing; context-budget] Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a river, backed by... |
| `mlx-community/Qwen3.5-35B-A3B-bf16`                |                   13 | `context_budget` | [metadata-borrowing; context-budget] Title: Two sailing boats moored on a river with trees behind Description: Two sailing boats with tall masts are moored on a calm river, back... |
| `mlx-community/GLM-4.6V-Flash-6bit`                 |                   22 | `context_budget` | [context-budget; missing-sections] <\|begin_of_box\|>Title: Two sailing boats moored on water with trees behind ...                                                                  |
| `mlx-community/Ministral-3-14B-Instruct-2512-nvfp4` |                   18 | `context_budget` | [metadata-borrowing; context-budget] **Title:** Sailing Boat Moored on Deben Estuary ...                                                                                             |
