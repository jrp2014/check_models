# Check Models Issue Queue

Generated on: 2026-07-19 01:16:17 BST
Test Environment: Darwin 25.5.0, Apple M5 Max, mlx-vlm 0.6.5, mlx 0.32.1.dev20260719+b7c3dd6d
Summary: 60 models tested. 0 hard failures. 2 harness issues. 0 stack signals. 0 text-sanity issues. 1 issue draft.

| Target    | Problem                                        | Evidence Snapshot                                                                                          | Affected Models                              | Issue Draft                                                                                                            | Evidence Bundle                                                                                                                                                                   | Fixed When                     |
|-----------|------------------------------------------------|------------------------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `mlx-vlm` | Stop/control tokens leaked into generated text | decoded text contains control token &lt;/think&gt; \| prompt=317 \| output/prompt=59.31% \| stop=completed | 1: `mlx-community/Qwen3-VL-2B-Thinking-bf16` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx-vlm_stop-token_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260719T001617Z_001_mlx-community_Qwen3-VL-2B-Thinking-bf16_mlx_vlm_stop_token_001.json) | No leaked stop/control tokens. |

