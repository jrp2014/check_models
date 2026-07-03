<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 2 failure(s), 5 harness issue(s) (mlx-vlm 0.6.3)

**Run summary:** 60 locally-cached VLM model(s) checked; 2 hard failure(s), 5 harness/integration issue(s), 0 preflight warning(s), 58 successful run(s).

Test image: `cats.jpg` (0.2 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                                                                          | Evidence Snapshot                                                                                                                                                                                                                         | Affected Models                                            | Issue Draft                                                                                                                            | Evidence Bundle                                                                                                                                                                               | Fixed When                                                |
|------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                          | Weight/config mismatch during model load                                                         | Weight Mismatch \| phase model_load \| ValueError                                                                                                                                                                                         | 1: `mlx-community/LFM2.5-VL-1.6B-bf16`                     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-weight-mismatch_001.md) | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260628T211108Z_004_mlx-community_LFM2.5-VL-1.6B-bf16_MLX_MODEL_LOAD_WEIGHT_MISMATCH_7574b1189.json) | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | mlx-vlm: Model load / model error: property 'eos_token_id' of 'ModelConfig' object has no setter | Model Error \| phase model_load \| AttributeError                                                                                                                                                                                         | 1: `mlx-community/MolmoPoint-8B-fp16`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_mlx-vlm_mlx-vlm-model-load-model_001.md)   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260628T211108Z_005_mlx-community_MolmoPoint-8B-fp16_MLX_VLM_MODEL_LOAD_MODEL_7cbd53695717.json)     | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                      | Tokenizer decode leaked BPE/byte markers                                                         | 76 BPE space markers found in decoded text \| prompt=417 \| output/prompt=21.58% \| nontext burden=99% \| stop=completed                                                                                                                  | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_encoding_001.md)                   | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260628T211108Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) | No BPE/byte markers in output.                            |
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text                                                   | decoded text contains control token &lt;\|end\|&gt; \| decoded text contains control token &lt;\|endoftext\|&gt; \| prompt=770 \| output/prompt=25.97% \| nontext burden=99% \| stop=max_tokens \| hit token cap (200) \| 3 model cluster | 3: `microsoft/Phi-3.5-vision-instruct` (+2)                | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                 | [3 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260628T211108Z_001_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)                | No leaked stop/control tokens.                            |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short                                            | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed                                                        | 1: `mlx-community/paligemma2-3b-pt-896-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_mlx-vlm-mlx_long-context_001.md)           | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260628T211108Z_007_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_001.json)       | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 1 model

**Affected models:** `mlx-community/LFM2.5-VL-1.6B-bf16`

**Observed error:**

```text
Model loading failed: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

**Relevant upstream traceback:**

```text
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/jrp/Documents/AI/mlx/mlx/python/mlx/nn/layers/base.py", line 191, in load_weights
    raise ValueError(f"Missing {num_missing} parameters: \n{missing}.")
ValueError: Missing 2 parameters: 
multi_modal_projector.layer_norm.bias,
multi_modal_projector.layer_norm.weight.
```

<!-- markdownlint-disable MD060 -->

| Model                               | Observed Behavior                                                                                                           | First Seen Failing      | Recent Repro           |
|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | Model loading failed: Missing 2 parameters: multi_modal_projector.layer_norm.bias, multi_modal_projector.layer_norm.weight. | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/LFM2.5-VL-1.6B-bf16`

## 2. Failure affecting 1 model

**Affected models:** `mlx-community/MolmoPoint-8B-fp16`

**Observed error:**

```text
Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter
```

**Relevant upstream traceback:**

```text
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 534, in load_model
    model_config = apply_generation_config_defaults(model_config, config)
  File "/Users/jrp/Documents/AI/mlx/mlx-vlm/mlx_vlm/utils.py", line 69, in apply_generation_config_defaults
    setattr(model_config, key, config[key])
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: property 'eos_token_id' of 'ModelConfig' object has no setter
```

<!-- markdownlint-disable MD060 -->

| Model                              | Observed Behavior                                                                   | First Seen Failing      | Recent Repro           |
|------------------------------------|-------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/MolmoPoint-8B-fp16` | Model loading failed: property 'eos_token_id' of 'ModelConfig' object has no setter | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/MolmoPoint-8B-fp16`

---

## Harness/Integration Issues (5 model(s))

5 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `microsoft/Phi-3.5-vision-instruct`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Special control token &lt;|endoftext|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).

**Sample output:**

```text
Two cats are sleeping on a pink couch with remote controls beside them.<|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|end|><|endoftext|><|e...
```

### `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;|end|&gt; appeared in generated text.
- Output switched language/script unexpectedly (tokenizer_artifact).
- Output leaked reasoning or prompt-template text (here are my reasoning steps, the user asks:).

**Sample output:**

```text
...at's it. No need for extra. We'll comply.
[BEGIN FINAL RESPONSE]
Two tabby cats are curled up sleeping side‑by‑side on a pink couch, with a TV remote resting nearby.
[END FINAL RESPONSE]
<|end|>
```

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

**Why this appears to be an integration/runtime issue:**

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 76 occurrences).

**Sample output:**

```text
TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠblanketĠorĠaĠcouch.ĠTheĠcatĠonĠtheĠleftĠisĠaĠkitten,ĠandĠtheĠoneĠonĠtheĠrightĠisĠanĠadultĠcat.ĠBothĠcatsĠareĠinĠrelaxedĠpostures,Ġwit...
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
...need to describe this briefly, so focus on the key elements: two cats, pink couch, remotes.
</think>

Two tabby cats are lying on a bright pink couch, each resting with their paws extended. A w...
```

### `mlx-community/paligemma2-3b-pt-896-4bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.
- At long prompt length (4103 tokens), output stayed unusually short (3 tokens; ratio 0.1%; weak text signal truncated).

**Sample output:**

```text
Cat.
```

---

## History Context

Recent reproducibility is measured from history (up to last 3 runs where each
model appears).

**Regressions since previous run:** none
**Recoveries since previous run:** none
**Quality regressions in history window:** `mlx-community/Qwen3-VL-2B-Thinking-bf16`
**Core library changes in window:** `mlx=0.32.0.dev20260612+337f736a->0.32.0.dev20260628+e94b4150`, `transformers=5.11.0->5.12.1`

<!-- markdownlint-disable MD060 -->

| Model                               | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/LFM2.5-VL-1.6B-bf16` | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
| `mlx-community/MolmoPoint-8B-fp16`  | still failing            | 2026-03-27 13:06:07 GMT | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 7
- **Summary diagnostics models:** 53
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 470.17s (470.17s)
- **Average runtime per model:** 7.84s (7.84s)
- **Dominant runtime phase:** model load dominated 41/60 measured model runs (47% of tracked runtime).
- **Phase totals:** model load=225.60s, local prompt prep=0.19s, upstream prefill / first-token=62.74s, post-prefill decode=181.15s, cleanup=6.57s
- **Generation total:** 243.89s across 58 model(s); upstream prefill / first-token split available for 58/58 model(s).
- **Observed stop reasons:** completed=50, exception=2, max_tokens=8
- **Validation overhead:** 0.18s total (avg 0.00s across 60 model(s)).
- **Upstream prefill / first-token latency:** Avg 1.08s | Min 0.02s | Max 7.26s across 58 model(s).
- **What this likely means:** Cold model load time is a major share of runtime for this cohort.
- **Suggested next action:** Consider staged runs, model reuse, or narrowing the model set before reruns.

---

## Models Not Flagged (53 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 45 model(s).
- **Ran, but with quality warnings:** 8 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --exclude mlx-community/Ornith-1.0-35B-bf16 --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --presence-context-size 20 --frequency-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/MolmoPoint-8B-fp16`, `microsoft/Phi-3.5-vision-instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/paligemma2-3b-pt-896-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

Report generated on: 2026-06-28 22:11:08 BST by [check_models](https://github.com/jrp2014/check_models).

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.3                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260628+e94b4150                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.12.1                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.21.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.1                                                                                                                                                   |
| SDK Version                | 26.5                                                                                                                                                     |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                                       |
| Xcode Version              | 26.6                                                                                                                                                     |
| Xcode Build                | 17F113                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,451,352 bytes, sha256=c2b26a5583dd2776ccab3b125611a6924bdc21c4f0880c222e3e8975933f962e) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,746,656 bytes, sha256=e58e30c3625106bcb0e150263126a0de69e71cc5ae21c066901112cadac758d7)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
