<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 0 failure(s), 0 indeterminate, 2 harness issue(s), 0 text-sanity issue(s) (mlx-vlm 0.6.5)

**Run summary:** 61 attempted, 61 evaluated; 0 hard failure(s), 2 harness/integration issue(s), 0 text-sanity/semantic issue(s), 0 indeterminate, 0 preflight warning(s), 61 successful run(s).

Test image: `cats.jpg` (0.2 MB).

## Impact and Run Conditions

- _Models attempted:_ 61
- _Models evaluated:_ 61
- _Indeterminate:_ 0
- _Crashed:_ 0
- _Completed:_ 61
- _Prompt:_ Describe this image briefly.
- _Image:_ cats.jpg

## mlx-vlm / MLX Issue Matrix

Routing labels reflect evidence confidence only; a crash remains a conclusive
failed task at every confidence level.
<!-- markdownlint-disable MD060 -->

| Target                                         | Problem                                               | Evidence Snapshot                                                                                                                                                                       | Affected Models                              | Confidence   | Evidence Type   | Fixed When                                      |
|------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|--------------|-----------------|-------------------------------------------------|
| `mlx-vlm`                                      | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;/think&gt; \| prompt=317 \| output/prompt=59.31% \| stop=completed                                                                              | 1: `mlx-community/Qwen3-VL-2B-Thinking-bf16` | confirmed    | harness         | No leaked stop/control tokens.                  |
| mlx-vlm first; MLX if cache/runtime reproduces | Long-context generation collapsed or became too short | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| visual input burden=100% \| stop=completed | 1: `mlx-community/paligemma2-3b-pt-896-4bit` | confirmed    | context_budget  | Full and reduced reruns avoid context collapse. |
<!-- markdownlint-enable MD060 -->
---

## Issue 1: mlx-vlm — Stop-token leakage

<!-- markdownlint-disable MD033 -->

### Generated Output Evidence

<details>
<summary>Complete generated output: mlx-community/Qwen3-VL-2B-Thinking-bf16</summary>

```text
So, let's see. The image shows two cats lying on a pink couch. The couch is a bright pink color, and there are two remote controls next to the cats. One cat is on the left, with a striped pattern, and the other is on the right, also striped but maybe a bit different. Both cats are relaxed, lying down, maybe sleeping. The remotes are white with some buttons, and one has a blue button. The background is the pink couch, so the main elements are the two cats, the remotes, and the pink fabric. I need to describe this briefly.
</think>

Two tabby cats are lying on a bright pink couch, relaxed and asleep. A white remote control with a blue button sits between them, and another remote control is positioned near the left cat. The scene is cozy, with the cats' striped fur and the vibrant pink fabric of the couch forming the backdrop.
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/Qwen3-VL-2B-Thinking-bf16
- _Expected:_ Affected reruns contain no leaked stop/control tokens and
  terminate cleanly before the configured max-token cap when the response is
  complete.
- _Actual:_ Stop/control tokens leaked into generated text
- _Filing target:_ mlx-vlm

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3-VL-2B-Thinking-bf16 --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`


### Expected Fix Signal

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

---

## Issue 2: mlx-vlm first; MLX if cache/runtime reproduces — Long-context collapse

<!-- markdownlint-disable MD033 -->

### Generated Output Evidence

<details>
<summary>Complete generated output: mlx-community/paligemma2-3b-pt-896-4bit</summary>

```text
Cat.
```

</details>
<!-- markdownlint-enable MD033 -->

### Expected and Actual Behaviour

- _Affected models:_ mlx-community/paligemma2-3b-pt-896-4bit
- _Expected:_ A same-command rerun and a reduced image/text burden rerun show
  consistent prompt-token accounting and no long-context collapse.
- _Actual:_ Long-context generation collapsed or became too short
- _Filing target:_ mlx-vlm first; MLX if cache/runtime reproduces

### Native mlx-vlm reproduction


```bash
python -m mlx_vlm.generate --model mlx-community/paligemma2-3b-pt-896-4bit --image cats.jpg --prompt 'Describe this image briefly.' --max-tokens 200 --temperature 0.0 --trust-remote-code --prefill-step-size 4096
```


Image SHA256: `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`


### Expected Fix Signal

- [ ] A same-command rerun and a reduced image/text burden rerun show consistent prompt-token accounting and no long-context collapse.
- [ ] The native `mlx-vlm` CLI/Python repro no longer shows the observed problem.

<!-- markdownlint-disable MD060 -->

## Prompt / Input Burden Evidence

| Model                                   | Burden       |   Total tokens |   Text estimate |   Non-text estimate | Source            |
|-----------------------------------------|--------------|----------------|-----------------|---------------------|-------------------|
| mlx-community/paligemma2-3b-pt-896-4bit | visual_input |           4103 |               6 |                4097 | estimated_nontext |
<!-- markdownlint-enable MD060 -->
<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                         |
|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.5                                                                                                                                           |
| mlx                        | 0.32.1.dev20260718+b7c3dd6d                                                                                                                     |
| mlx-lm                     | 0.31.3                                                                                                                                          |
| mlx-audio                  | 0.4.4                                                                                                                                           |
| transformers               | 5.14.1                                                                                                                                          |
| tokenizers                 | 0.22.2                                                                                                                                          |
| huggingface-hub            | 1.24.0                                                                                                                                          |
| Python Version             | 3.13.13                                                                                                                                         |
| OS                         | Darwin 25.5.0                                                                                                                                   |
| macOS Version              | 26.5.2                                                                                                                                          |
| SDK Version                | 26.5                                                                                                                                            |
| SDK Path                   | /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk                                              |
| Xcode Version              | 26.6                                                                                                                                            |
| Xcode Build                | 17F113                                                                                                                                          |
| Active Developer Directory | /Applications/Xcode.app/Contents/Developer                                                                                                      |
| Metal SDK                  | MacOSX26.5.sdk                                                                                                                                  |
| Metal Compiler Version     | Apple metal version 32023.883 (metalfe-32023.883)                                                                                               |
| Metallib Linker Version    | AIR-LLD 32023.883 (metalfe-32023.883) (compatible with legacy metallib linker)                                                                  |
| Apple Clang Version        | Apple clang version 21.0.0 (clang-2100.1.1.101)                                                                                                 |
| GPU/Chip                   | Apple M5 Max                                                                                                                                    |
| GPU Cores                  | 40                                                                                                                                              |
| MLX Device                 | Apple M5 Max                                                                                                                                    |
| GPU Architecture           | applegpu_g17s                                                                                                                                   |
| Recommended Working Set    | 108 GB                                                                                                                                          |
| Fused Attention            | Available                                                                                                                                       |
| Metal Support              | Metal 4                                                                                                                                         |
| MLX Install Type           | editable local source                                                                                                                           |
| MLX Distribution Root      | ~/miniconda3/envs/mlx-vlm/lib/python3.13/site-packages                                                                                          |
| mlx-metal Distribution     | not installed; local editable mlx supplies backend                                                                                              |
| MLX Core Extension         | ~/Documents/AI/mlx/mlx/python/mlx/core.cpython-313-darwin.so                                                                                    |
| MLX Metallib               | ~/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,449,848 bytes, sha256=1078bd042297dbbf704a414617a7988c55b0001ea69d7cb478bcafa2fdfdeecb) |
| MLX libmlx.dylib           | ~/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,697,568 bytes, sha256=bef52521877d51c72e90bb0a5dc2823d77e222799855fc5f7bbb21ac8bcd129f)  |
| RAM                        | 128.0 GB                                                                                                                                        |
<!-- markdownlint-enable MD060 -->
