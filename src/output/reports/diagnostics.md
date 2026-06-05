<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 0 failure(s), 1 harness issue(s) (mlx-vlm 0.6.1)

**Run summary:** 1 locally-cached VLM model(s) checked; 0 hard failure(s), 1 harness/integration issue(s), 0 preflight warning(s), 1 successful run(s).

Test image: `20260525-164635_DSC00024.jpg` (2.4 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target    | Problem                                        | Evidence Snapshot                                                                                                                                           | Affected Models                              | Issue Draft                                                                                                            | Evidence Bundle                                                                                                                                                                   | Fixed When                     |
|-----------|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `mlx-vlm` | Stop/control tokens leaked into generated text | decoded text contains control token &lt;/think&gt; \| prompt=16,241 \| output/prompt=1.23% \| nontext burden=100% \| stop=max_tokens \| hit token cap (200) | 1: `mlx-community/Qwen3-VL-2B-Thinking-bf16` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx-vlm_stop-token_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260605T213953Z_001_mlx-community_Qwen3-VL-2B-Thinking-bf16_mlx_vlm_stop_token_001.json) | No leaked stop/control tokens. |
<!-- markdownlint-enable MD060 -->

---

## Harness/Integration Issues (1 model(s))

1 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
So, let's see. The image shows a person riding a Sea-Doo jet ski on the water. The jet ski is green and black, with the Sea-Doo logo visible. The rider is wearing a life jacket and a black shirt, and ...
```

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 1
- **Summary diagnostics models:** 0
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 17.80s (17.80s)
- **Average runtime per model:** 17.80s (17.80s)
- **Dominant runtime phase:** upstream prefill / first-token dominated 1/1 measured model runs (80% of tracked runtime).
- **Phase totals:** model load=0.63s, local prompt prep=0.04s, upstream prefill / first-token=14.15s, post-prefill decode=2.85s, cleanup=0.07s
- **Generation total:** 17.01s across 1 model(s); upstream prefill / first-token split available for 1/1 model(s).
- **Observed stop reasons:** max_tokens=1
- **Validation overhead:** 0.12s total (avg 0.12s across 1 model(s)).
- **Upstream prefill / first-token latency:** Avg 14.15s | Min 14.15s | Max 14.15s across 1 model(s).
- **What this likely means:** Most measured runtime is spent inside upstream generation before the first token is available.
- **Suggested next action:** Inspect prompt/image token accounting, dynamic-resolution image burden, prefill step sizing, and cache/prefill behavior.

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg --models mlx-community/Qwen3-VL-2B-Thinking-bf16 --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `mlx-community/Qwen3-VL-2B-Thinking-bf16`.

### Run details

- Input image: `/Users/jrp/Pictures/Processed/20260525-164635_DSC00024.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

_Report generated on 2026-06-05 22:39:53 BST by [check_models](https://github.com/jrp2014/check_models)._

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.1                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260605+6ea7a00d                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.3                                                                                                                                                    |
| transformers               | 5.10.2                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.18.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.1                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.5                                                                                                                                                     |
| Xcode Build                | 17F42                                                                                                                                                    |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                               |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                           |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                                        |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                           |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                          |
| GPU/Chip                   | Apple M5 Max                                                                                                                                             |
| GPU Cores                  | 40                                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                                  |
| MLX Install Type           | editable local source                                                                                                                                    |
| MLX Distribution Root      | /Users/jrp/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                                       |
| MLX Core Extension         | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (157,751,704 bytes, sha256=e50cd0c2d2ae16781f644476459cbc2ca23b0d428a897140740f86678b5e2bf5) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,675,136 bytes, sha256=7c19dc6a8bf6b56db0155bed7a1d2b0a54182b42eba2fcf0f42b97d3d4c57eff)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
