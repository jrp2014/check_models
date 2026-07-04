<!-- markdownlint-disable MD013 -->

# Diagnostics Report — 3 failure(s), 9 harness issue(s) (mlx-vlm 0.6.4)

**Run summary:** 61 locally-cached VLM model(s) checked; 3 hard failure(s), 9 harness/integration issue(s), 0 preflight warning(s), 58 successful run(s).

Test image: `cats.jpg` (0.2 MB).

---

## Issue Queue

Root-cause issue drafts are generated in
[issues/index.md](https://github.com/jrp2014/check_models/blob/main/src/output/issues/index.md).
Each row is intended to become one focused upstream GitHub issue.

<!-- markdownlint-disable MD060 -->

| Target                                                   | Problem                                               | Evidence Snapshot                                                                                                                                                                  | Affected Models                                            | Issue Draft                                                                                                                                                              | Evidence Bundle                                                                                                                                                                                             | Fixed When                                                |
|----------------------------------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| `mlx`                                                    | Weight/config mismatch during model load              | Model Error \| phase model_load \| ValueError                                                                                                                                      | 1: `mlx-community/MiniCPM-V-4.6-8bit`                      | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_001_mlx_mlx-model-load-model_001.md)                                             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260704T185505Z_004_mlx-community_MiniCPM-V-4.6-8bit_MLX_MODEL_LOAD_MODEL_b4b095ff6a33.json)                       | Load/generation completes or fails with a narrower owner. |
| model configuration / repository                         | Processor config is missing image processor           | Processor Error \| phase processor_load \| ValueError \| 2 model cluster                                                                                                           | 2: `mlx-community/diffusiongemma-26B-A4B-it-8bit` (+1)     | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_002_model-configuration-repository_model-config-processor-load-processor_001.md) | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260704T185505Z_010_mlx-community_diffusiongemma-26B-A4B-it-8bit_MODEL_CONFIG_PROCESSOR_LOAD_PROCESSOR_49.json) | Load/generation completes or fails with a narrower owner. |
| `mlx-vlm`                                                | Tokenizer decode leaked BPE/byte markers              | 56 BPE space markers found in decoded text \| prompt=441 \| output/prompt=14.74% \| nontext burden=99% \| stop=completed                                                           | 1: `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_003_mlx-vlm_encoding_001.md)                                                     | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260704T185505Z_003_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)               | No BPE/byte markers in output.                            |
| `mlx-vlm`                                                | Stop/control tokens leaked into generated text        | decoded text contains control token &lt;\|end\|&gt; \| prompt=1,330 \| output/prompt=13.08% \| nontext burden=100% \| stop=completed \| 2 model cluster                            | 2: `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX` (+1)    | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_004_mlx-vlm_stop-token_001.md)                                                   | [2 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260704T185505Z_002_mlx-community_Apriel-1.5-15b-Thinker-6bit-MLX_mlx_vlm_stop_token_001.json)                  | No leaked stop/control tokens.                            |
| model repo first; mlx-vlm if template handling disagrees | Prompt/template output shape mismatch                 | generated_tokens~3 \| prompt=315 \| output/prompt=0.95% \| nontext burden=98% \| stop=completed \| 5 model cluster                                                                 | 5: `Qwen/Qwen3-VL-2B-Instruct` (+4)                        | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_005_model-config-mlx-vlm_prompt-template_001.md)                                 | [5 repro JSONs](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260704T185505Z_001_Qwen_Qwen3-VL-2B-Instruct_model_config_mlx_vlm_prompt_template_001.json)                    | Requested sections render without template leakage.       |
| mlx-vlm first; MLX if cache/runtime reproduces           | Long-context generation collapsed or became too short | generated_tokens~3 \| prompt_tokens=4103, output_tokens=3, output/prompt=0.1%, weak text=truncated \| prompt=4,103 \| output/prompt=0.07% \| nontext burden=100% \| stop=completed | 1: `mlx-community/paligemma2-3b-pt-896-4bit`               | [issue draft](https://github.com/jrp2014/check_models/blob/main/src/output/issues/issue_006_mlx-vlm-mlx_long-context_001.md)                                             | [repro JSON](https://github.com/jrp2014/check_models/blob/main/src/output/repro_bundles/20260704T185505Z_012_mlx-community_paligemma2-3b-pt-896-4bit_mlx_vlm_mlx_long_context_001.json)                     | Full and reduced reruns avoid context collapse.           |
<!-- markdownlint-enable MD060 -->

---

## 1. Failure affecting 2 models

**Affected models:** `mlx-community/diffusiongemma-26B-A4B-it-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`

**Observed error:**

```text
Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

**Relevant upstream traceback:**

```text
Traceback (most recent call last):
ValueError: Loaded processor has no image_processor; expected multimodal processor.
The above exception was the direct cause of the following exception:
Traceback (most recent call last):
ValueError: Model preflight failed for mlx-community/diffusiongemma-26B-A4B-it-8bit: Loaded processor has no image_processor; expected multimodal processor.
```

<!-- markdownlint-disable MD060 -->

| Model                                           | Observed Behavior                                                       | First Seen Failing      | Recent Repro           |
|-------------------------------------------------|-------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`  | Loaded processor has no image_processor; expected multimodal processor. | 2026-06-12 13:18:08 BST | 3/3 recent runs failed |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` | Loaded processor has no image_processor; expected multimodal processor. | 2026-07-03 13:59:09 BST | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/diffusiongemma-26B-A4B-it-8bit`

## 2. Failure affecting 1 model

**Affected models:** `mlx-community/MiniCPM-V-4.6-8bit`

**Observed error:**

```text
Model loading failed: Received 512 parameters not in model: 
language_model.model.model.embed_tokens.weight,
language_model.model.model.layers.0.input_layernorm.weight,
language_model.model.model.layers.0.linear_attn.A_log,
language_model.model.model.layers.0.linear_attn.conv1d.weight,
language_model.model.model.layers.0.linear_attn.dt_bias,
language_model.model.model.layers.0.linear_attn.in_proj_a.weight,
language_model.model.model.layers.0.linear_attn.in_proj_b.weight,
language_model.model.model.layers.0.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.0.linear_attn.in_proj_z.weight,
language_model.model.model.layers.0.linear_attn.norm.weight,
language_model.model.model.layers.0.linear_attn.out_proj.weight,
language_model.model.model.layers.0.mlp.down_proj.biases,
language_model.model.model.layers.0.mlp.down_proj.scales,
language_model.model.model.layers.0.mlp.down_proj.weight,
language_model.model.model.layers.0.mlp.gate_proj.biases,
language_model.model.model.layers.0.mlp.gate_proj.scales,
language_model.model.model.layers.0.mlp.gate_proj.weight,
language_model.model.model.layers.0.mlp.up_proj.biases,
language_model.model.model.layers.0.mlp.up_proj.scales,
language_model.model.model.layers.0.mlp.up_proj.weight,
language_model.model.model.layers.0.post_attention_layernorm.weight,
language_model.model.model.layers.1.input_layernorm.weight,
language_model.model.model.layers.1.linear_attn.A_log,
language_model.model.model.layers.1.linear_attn.conv1d.weight,
language_model.model.model.layers.1.linear_attn.dt_bias,
language_model.model.model.layers.1.linear_attn.in_proj_a.weight,
language_model.model.model.layers.1.linear_attn.in_proj_b.weight,
language_model.model.model.layers.1.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.1.linear_attn.in_proj_z.weight,
language_model.model.model.layers.1.linear_attn.norm.weight,
language_model.model.model.layers.1.linear_attn.out_proj.weight,
language_model.model.model.layers.1.mlp.down_proj.biases,
language_model.model.model.layers.1.mlp.down_proj.scales,
language_model.model.model.layers.1.mlp.down_proj.weight,
language_model.model.model.layers.1.mlp.gate_proj.biases,
language_model.model.model.layers.1.mlp.gate_proj.scales,
language_model.model.model.layers.1.mlp.gate_proj.weight,
language_model.model.model.layers.1.mlp.up_proj.biases,
language_model.model.model.layers.1.mlp.up_proj.scales,
language_model.model.model.layers.1.mlp.up_proj.weight,
language_model.model.model.layers.1.post_attention_layernorm.weight,
language_model.model.model.layers.10.input_layernorm.weight,
language_model.model.model.layers.10.linear_attn.A_log,
language_model.model.model.layers.10.linear_attn.conv1d.weight,
language_model.model.model.layers.10.linear_attn.dt_bias,
language_model.model.model.layers.10.linear_attn.in_proj_a.weight,
language_model.model.model.layers.10.linear_attn.in_proj_b.weight,
language_model.model.model.layers.10.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.10.linear_attn.in_proj_z.weight,
language_model.model.model.layers.10.linear_attn.norm.weight,
language_model.model.model.layers.10.linear_attn.out_proj.weight,
language_model.model.model.layers.10.mlp.down_proj.biases,
language_model.model.model.layers.10.mlp.down_proj.scales,
language_model.model.model.layers.10.mlp.down_proj.weight,
language_model.model.model.layers.10.mlp.gate_proj.biases,
language_model.model.model.layers.10.mlp.gate_proj.scales,
language_model.model.model.layers.10.mlp.gate_proj.weight,
language_model.model.model.layers.10.mlp.up_proj.biases,
language_model.model.model.layers.10.mlp.up_proj.scales,
language_model.model.model.layers.10.mlp.up_proj.weight,
language_model.model.model.layers.10.post_attention_layernorm.weight,
language_model.model.model.layers.11.input_layernorm.weight,
language_model.model.model.layers.11.mlp.down_proj.biases,
language_model.model.model.layers.11.mlp.down_proj.scales,
language_model.model.model.layers.11.mlp.down_proj.weight,
language_model.model.model.layers.11.mlp.gate_proj.biases,
language_model.model.model.layers.11.mlp.gate_proj.scales,
language_model.model.model.layers.11.mlp.gate_proj.weight,
language_model.model.model.layers.11.mlp.up_proj.biases,
language_model.model.model.layers.11.mlp.up_proj.scales,
language_model.model.model.layers.11.mlp.up_proj.weight,
language_model.model.model.layers.11.post_attention_layernorm.weight,
language_model.model.model.layers.11.self_attn.k_norm.weight,
language_model.model.model.layers.11.self_attn.k_proj.biases,
language_model.model.model.layers.11.self_attn.k_proj.scales,
language_model.model.model.layers.11.self_attn.k_proj.weight,
language_model.model.model.layers.11.self_attn.o_proj.biases,
language_model.model.model.layers.11.self_attn.o_proj.scales,
language_model.model.model.layers.11.self_attn.o_proj.weight,
language_model.model.model.layers.11.self_attn.q_norm.weight,
language_model.model.model.layers.11.self_attn.q_proj.biases,
language_model.model.model.layers.11.self_attn.q_proj.scales,
language_model.model.model.layers.11.self_attn.q_proj.weight,
language_model.model.model.layers.11.self_attn.v_proj.biases,
language_model.model.model.layers.11.self_attn.v_proj.scales,
language_model.model.model.layers.11.self_attn.v_proj.weight,
language_model.model.model.layers.12.input_layernorm.weight,
language_model.model.model.layers.12.linear_attn.A_log,
language_model.model.model.layers.12.linear_attn.conv1d.weight,
language_model.model.model.layers.12.linear_attn.dt_bias,
language_model.model.model.layers.12.linear_attn.in_proj_a.weight,
language_model.model.model.layers.12.linear_attn.in_proj_b.weight,
language_model.model.model.layers.12.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.12.linear_attn.in_proj_z.weight,
language_model.model.model.layers.12.linear_attn.norm.weight,
language_model.model.model.layers.12.linear_attn.out_proj.weight,
language_model.model.model.layers.12.mlp.down_proj.biases,
language_model.model.model.layers.12.mlp.down_proj.scales,
language_model.model.model.layers.12.mlp.down_proj.weight,
language_model.model.model.layers.12.mlp.gate_proj.biases,
language_model.model.model.layers.12.mlp.gate_proj.scales,
language_model.model.model.layers.12.mlp.gate_proj.weight,
language_model.model.model.layers.12.mlp.up_proj.biases,
language_model.model.model.layers.12.mlp.up_proj.scales,
language_model.model.model.layers.12.mlp.up_proj.weight,
language_model.model.model.layers.12.post_attention_layernorm.weight,
language_model.model.model.layers.13.input_layernorm.weight,
language_model.model.model.layers.13.linear_attn.A_log,
language_model.model.model.layers.13.linear_attn.conv1d.weight,
language_model.model.model.layers.13.linear_attn.dt_bias,
language_model.model.model.layers.13.linear_attn.in_proj_a.weight,
language_model.model.model.layers.13.linear_attn.in_proj_b.weight,
language_model.model.model.layers.13.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.13.linear_attn.in_proj_z.weight,
language_model.model.model.layers.13.linear_attn.norm.weight,
language_model.model.model.layers.13.linear_attn.out_proj.weight,
language_model.model.model.layers.13.mlp.down_proj.biases,
language_model.model.model.layers.13.mlp.down_proj.scales,
language_model.model.model.layers.13.mlp.down_proj.weight,
language_model.model.model.layers.13.mlp.gate_proj.biases,
language_model.model.model.layers.13.mlp.gate_proj.scales,
language_model.model.model.layers.13.mlp.gate_proj.weight,
language_model.model.model.layers.13.mlp.up_proj.biases,
language_model.model.model.layers.13.mlp.up_proj.scales,
language_model.model.model.layers.13.mlp.up_proj.weight,
language_model.model.model.layers.13.post_attention_layernorm.weight,
language_model.model.model.layers.14.input_layernorm.weight,
language_model.model.model.layers.14.linear_attn.A_log,
language_model.model.model.layers.14.linear_attn.conv1d.weight,
language_model.model.model.layers.14.linear_attn.dt_bias,
language_model.model.model.layers.14.linear_attn.in_proj_a.weight,
language_model.model.model.layers.14.linear_attn.in_proj_b.weight,
language_model.model.model.layers.14.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.14.linear_attn.in_proj_z.weight,
language_model.model.model.layers.14.linear_attn.norm.weight,
language_model.model.model.layers.14.linear_attn.out_proj.weight,
language_model.model.model.layers.14.mlp.down_proj.biases,
language_model.model.model.layers.14.mlp.down_proj.scales,
language_model.model.model.layers.14.mlp.down_proj.weight,
language_model.model.model.layers.14.mlp.gate_proj.biases,
language_model.model.model.layers.14.mlp.gate_proj.scales,
language_model.model.model.layers.14.mlp.gate_proj.weight,
language_model.model.model.layers.14.mlp.up_proj.biases,
language_model.model.model.layers.14.mlp.up_proj.scales,
language_model.model.model.layers.14.mlp.up_proj.weight,
language_model.model.model.layers.14.post_attention_layernorm.weight,
language_model.model.model.layers.15.input_layernorm.weight,
language_model.model.model.layers.15.mlp.down_proj.biases,
language_model.model.model.layers.15.mlp.down_proj.scales,
language_model.model.model.layers.15.mlp.down_proj.weight,
language_model.model.model.layers.15.mlp.gate_proj.biases,
language_model.model.model.layers.15.mlp.gate_proj.scales,
language_model.model.model.layers.15.mlp.gate_proj.weight,
language_model.model.model.layers.15.mlp.up_proj.biases,
language_model.model.model.layers.15.mlp.up_proj.scales,
language_model.model.model.layers.15.mlp.up_proj.weight,
language_model.model.model.layers.15.post_attention_layernorm.weight,
language_model.model.model.layers.15.self_attn.k_norm.weight,
language_model.model.model.layers.15.self_attn.k_proj.biases,
language_model.model.model.layers.15.self_attn.k_proj.scales,
language_model.model.model.layers.15.self_attn.k_proj.weight,
language_model.model.model.layers.15.self_attn.o_proj.biases,
language_model.model.model.layers.15.self_attn.o_proj.scales,
language_model.model.model.layers.15.self_attn.o_proj.weight,
language_model.model.model.layers.15.self_attn.q_norm.weight,
language_model.model.model.layers.15.self_attn.q_proj.biases,
language_model.model.model.layers.15.self_attn.q_proj.scales,
language_model.model.model.layers.15.self_attn.q_proj.weight,
language_model.model.model.layers.15.self_attn.v_proj.biases,
language_model.model.model.layers.15.self_attn.v_proj.scales,
language_model.model.model.layers.15.self_attn.v_proj.weight,
language_model.model.model.layers.16.input_layernorm.weight,
language_model.model.model.layers.16.linear_attn.A_log,
language_model.model.model.layers.16.linear_attn.conv1d.weight,
language_model.model.model.layers.16.linear_attn.dt_bias,
language_model.model.model.layers.16.linear_attn.in_proj_a.weight,
language_model.model.model.layers.16.linear_attn.in_proj_b.weight,
language_model.model.model.layers.16.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.16.linear_attn.in_proj_z.weight,
language_model.model.model.layers.16.linear_attn.norm.weight,
language_model.model.model.layers.16.linear_attn.out_proj.weight,
language_model.model.model.layers.16.mlp.down_proj.biases,
language_model.model.model.layers.16.mlp.down_proj.scales,
language_model.model.model.layers.16.mlp.down_proj.weight,
language_model.model.model.layers.16.mlp.gate_proj.biases,
language_model.model.model.layers.16.mlp.gate_proj.scales,
language_model.model.model.layers.16.mlp.gate_proj.weight,
language_model.model.model.layers.16.mlp.up_proj.biases,
language_model.model.model.layers.16.mlp.up_proj.scales,
language_model.model.model.layers.16.mlp.up_proj.weight,
language_model.model.model.layers.16.post_attention_layernorm.weight,
language_model.model.model.layers.17.input_layernorm.weight,
language_model.model.model.layers.17.linear_attn.A_log,
language_model.model.model.layers.17.linear_attn.conv1d.weight,
language_model.model.model.layers.17.linear_attn.dt_bias,
language_model.model.model.layers.17.linear_attn.in_proj_a.weight,
language_model.model.model.layers.17.linear_attn.in_proj_b.weight,
language_model.model.model.layers.17.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.17.linear_attn.in_proj_z.weight,
language_model.model.model.layers.17.linear_attn.norm.weight,
language_model.model.model.layers.17.linear_attn.out_proj.weight,
language_model.model.model.layers.17.mlp.down_proj.biases,
language_model.model.model.layers.17.mlp.down_proj.scales,
language_model.model.model.layers.17.mlp.down_proj.weight,
language_model.model.model.layers.17.mlp.gate_proj.biases,
language_model.model.model.layers.17.mlp.gate_proj.scales,
language_model.model.model.layers.17.mlp.gate_proj.weight,
language_model.model.model.layers.17.mlp.up_proj.biases,
language_model.model.model.layers.17.mlp.up_proj.scales,
language_model.model.model.layers.17.mlp.up_proj.weight,
language_model.model.model.layers.17.post_attention_layernorm.weight,
language_model.model.model.layers.18.input_layernorm.weight,
language_model.model.model.layers.18.linear_attn.A_log,
language_model.model.model.layers.18.linear_attn.conv1d.weight,
language_model.model.model.layers.18.linear_attn.dt_bias,
language_model.model.model.layers.18.linear_attn.in_proj_a.weight,
language_model.model.model.layers.18.linear_attn.in_proj_b.weight,
language_model.model.model.layers.18.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.18.linear_attn.in_proj_z.weight,
language_model.model.model.layers.18.linear_attn.norm.weight,
language_model.model.model.layers.18.linear_attn.out_proj.weight,
language_model.model.model.layers.18.mlp.down_proj.biases,
language_model.model.model.layers.18.mlp.down_proj.scales,
language_model.model.model.layers.18.mlp.down_proj.weight,
language_model.model.model.layers.18.mlp.gate_proj.biases,
language_model.model.model.layers.18.mlp.gate_proj.scales,
language_model.model.model.layers.18.mlp.gate_proj.weight,
language_model.model.model.layers.18.mlp.up_proj.biases,
language_model.model.model.layers.18.mlp.up_proj.scales,
language_model.model.model.layers.18.mlp.up_proj.weight,
language_model.model.model.layers.18.post_attention_layernorm.weight,
language_model.model.model.layers.19.input_layernorm.weight,
language_model.model.model.layers.19.mlp.down_proj.biases,
language_model.model.model.layers.19.mlp.down_proj.scales,
language_model.model.model.layers.19.mlp.down_proj.weight,
language_model.model.model.layers.19.mlp.gate_proj.biases,
language_model.model.model.layers.19.mlp.gate_proj.scales,
language_model.model.model.layers.19.mlp.gate_proj.weight,
language_model.model.model.layers.19.mlp.up_proj.biases,
language_model.model.model.layers.19.mlp.up_proj.scales,
language_model.model.model.layers.19.mlp.up_proj.weight,
language_model.model.model.layers.19.post_attention_layernorm.weight,
language_model.model.model.layers.19.self_attn.k_norm.weight,
language_model.model.model.layers.19.self_attn.k_proj.biases,
language_model.model.model.layers.19.self_attn.k_proj.scales,
language_model.model.model.layers.19.self_attn.k_proj.weight,
language_model.model.model.layers.19.self_attn.o_proj.biases,
language_model.model.model.layers.19.self_attn.o_proj.scales,
language_model.model.model.layers.19.self_attn.o_proj.weight,
language_model.model.model.layers.19.self_attn.q_norm.weight,
language_model.model.model.layers.19.self_attn.q_proj.biases,
language_model.model.model.layers.19.self_attn.q_proj.scales,
language_model.model.model.layers.19.self_attn.q_proj.weight,
language_model.model.model.layers.19.self_attn.v_proj.biases,
language_model.model.model.layers.19.self_attn.v_proj.scales,
language_model.model.model.layers.19.self_attn.v_proj.weight,
language_model.model.model.layers.2.input_layernorm.weight,
language_model.model.model.layers.2.linear_attn.A_log,
language_model.model.model.layers.2.linear_attn.conv1d.weight,
language_model.model.model.layers.2.linear_attn.dt_bias,
language_model.model.model.layers.2.linear_attn.in_proj_a.weight,
language_model.model.model.layers.2.linear_attn.in_proj_b.weight,
language_model.model.model.layers.2.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.2.linear_attn.in_proj_z.weight,
language_model.model.model.layers.2.linear_attn.norm.weight,
language_model.model.model.layers.2.linear_attn.out_proj.weight,
language_model.model.model.layers.2.mlp.down_proj.biases,
language_model.model.model.layers.2.mlp.down_proj.scales,
language_model.model.model.layers.2.mlp.down_proj.weight,
language_model.model.model.layers.2.mlp.gate_proj.biases,
language_model.model.model.layers.2.mlp.gate_proj.scales,
language_model.model.model.layers.2.mlp.gate_proj.weight,
language_model.model.model.layers.2.mlp.up_proj.biases,
language_model.model.model.layers.2.mlp.up_proj.scales,
language_model.model.model.layers.2.mlp.up_proj.weight,
language_model.model.model.layers.2.post_attention_layernorm.weight,
language_model.model.model.layers.20.input_layernorm.weight,
language_model.model.model.layers.20.linear_attn.A_log,
language_model.model.model.layers.20.linear_attn.conv1d.weight,
language_model.model.model.layers.20.linear_attn.dt_bias,
language_model.model.model.layers.20.linear_attn.in_proj_a.weight,
language_model.model.model.layers.20.linear_attn.in_proj_b.weight,
language_model.model.model.layers.20.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.20.linear_attn.in_proj_z.weight,
language_model.model.model.layers.20.linear_attn.norm.weight,
language_model.model.model.layers.20.linear_attn.out_proj.weight,
language_model.model.model.layers.20.mlp.down_proj.biases,
language_model.model.model.layers.20.mlp.down_proj.scales,
language_model.model.model.layers.20.mlp.down_proj.weight,
language_model.model.model.layers.20.mlp.gate_proj.biases,
language_model.model.model.layers.20.mlp.gate_proj.scales,
language_model.model.model.layers.20.mlp.gate_proj.weight,
language_model.model.model.layers.20.mlp.up_proj.biases,
language_model.model.model.layers.20.mlp.up_proj.scales,
language_model.model.model.layers.20.mlp.up_proj.weight,
language_model.model.model.layers.20.post_attention_layernorm.weight,
language_model.model.model.layers.21.input_layernorm.weight,
language_model.model.model.layers.21.linear_attn.A_log,
language_model.model.model.layers.21.linear_attn.conv1d.weight,
language_model.model.model.layers.21.linear_attn.dt_bias,
language_model.model.model.layers.21.linear_attn.in_proj_a.weight,
language_model.model.model.layers.21.linear_attn.in_proj_b.weight,
language_model.model.model.layers.21.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.21.linear_attn.in_proj_z.weight,
language_model.model.model.layers.21.linear_attn.norm.weight,
language_model.model.model.layers.21.linear_attn.out_proj.weight,
language_model.model.model.layers.21.mlp.down_proj.biases,
language_model.model.model.layers.21.mlp.down_proj.scales,
language_model.model.model.layers.21.mlp.down_proj.weight,
language_model.model.model.layers.21.mlp.gate_proj.biases,
language_model.model.model.layers.21.mlp.gate_proj.scales,
language_model.model.model.layers.21.mlp.gate_proj.weight,
language_model.model.model.layers.21.mlp.up_proj.biases,
language_model.model.model.layers.21.mlp.up_proj.scales,
language_model.model.model.layers.21.mlp.up_proj.weight,
language_model.model.model.layers.21.post_attention_layernorm.weight,
language_model.model.model.layers.22.input_layernorm.weight,
language_model.model.model.layers.22.linear_attn.A_log,
language_model.model.model.layers.22.linear_attn.conv1d.weight,
language_model.model.model.layers.22.linear_attn.dt_bias,
language_model.model.model.layers.22.linear_attn.in_proj_a.weight,
language_model.model.model.layers.22.linear_attn.in_proj_b.weight,
language_model.model.model.layers.22.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.22.linear_attn.in_proj_z.weight,
language_model.model.model.layers.22.linear_attn.norm.weight,
language_model.model.model.layers.22.linear_attn.out_proj.weight,
language_model.model.model.layers.22.mlp.down_proj.biases,
language_model.model.model.layers.22.mlp.down_proj.scales,
language_model.model.model.layers.22.mlp.down_proj.weight,
language_model.model.model.layers.22.mlp.gate_proj.biases,
language_model.model.model.layers.22.mlp.gate_proj.scales,
language_model.model.model.layers.22.mlp.gate_proj.weight,
language_model.model.model.layers.22.mlp.up_proj.biases,
language_model.model.model.layers.22.mlp.up_proj.scales,
language_model.model.model.layers.22.mlp.up_proj.weight,
language_model.model.model.layers.22.post_attention_layernorm.weight,
language_model.model.model.layers.23.input_layernorm.weight,
language_model.model.model.layers.23.mlp.down_proj.biases,
language_model.model.model.layers.23.mlp.down_proj.scales,
language_model.model.model.layers.23.mlp.down_proj.weight,
language_model.model.model.layers.23.mlp.gate_proj.biases,
language_model.model.model.layers.23.mlp.gate_proj.scales,
language_model.model.model.layers.23.mlp.gate_proj.weight,
language_model.model.model.layers.23.mlp.up_proj.biases,
language_model.model.model.layers.23.mlp.up_proj.scales,
language_model.model.model.layers.23.mlp.up_proj.weight,
language_model.model.model.layers.23.post_attention_layernorm.weight,
language_model.model.model.layers.23.self_attn.k_norm.weight,
language_model.model.model.layers.23.self_attn.k_proj.biases,
language_model.model.model.layers.23.self_attn.k_proj.scales,
language_model.model.model.layers.23.self_attn.k_proj.weight,
language_model.model.model.layers.23.self_attn.o_proj.biases,
language_model.model.model.layers.23.self_attn.o_proj.scales,
language_model.model.model.layers.23.self_attn.o_proj.weight,
language_model.model.model.layers.23.self_attn.q_norm.weight,
language_model.model.model.layers.23.self_attn.q_proj.biases,
language_model.model.model.layers.23.self_attn.q_proj.scales,
language_model.model.model.layers.23.self_attn.q_proj.weight,
language_model.model.model.layers.23.self_attn.v_proj.biases,
language_model.model.model.layers.23.self_attn.v_proj.scales,
language_model.model.model.layers.23.self_attn.v_proj.weight,
language_model.model.model.layers.3.input_layernorm.weight,
language_model.model.model.layers.3.mlp.down_proj.biases,
language_model.model.model.layers.3.mlp.down_proj.scales,
language_model.model.model.layers.3.mlp.down_proj.weight,
language_model.model.model.layers.3.mlp.gate_proj.biases,
language_model.model.model.layers.3.mlp.gate_proj.scales,
language_model.model.model.layers.3.mlp.gate_proj.weight,
language_model.model.model.layers.3.mlp.up_proj.biases,
language_model.model.model.layers.3.mlp.up_proj.scales,
language_model.model.model.layers.3.mlp.up_proj.weight,
language_model.model.model.layers.3.post_attention_layernorm.weight,
language_model.model.model.layers.3.self_attn.k_norm.weight,
language_model.model.model.layers.3.self_attn.k_proj.biases,
language_model.model.model.layers.3.self_attn.k_proj.scales,
language_model.model.model.layers.3.self_attn.k_proj.weight,
language_model.model.model.layers.3.self_attn.o_proj.biases,
language_model.model.model.layers.3.self_attn.o_proj.scales,
language_model.model.model.layers.3.self_attn.o_proj.weight,
language_model.model.model.layers.3.self_attn.q_norm.weight,
language_model.model.model.layers.3.self_attn.q_proj.biases,
language_model.model.model.layers.3.self_attn.q_proj.scales,
language_model.model.model.layers.3.self_attn.q_proj.weight,
language_model.model.model.layers.3.self_attn.v_proj.biases,
language_model.model.model.layers.3.self_attn.v_proj.scales,
language_model.model.model.layers.3.self_attn.v_proj.weight,
language_model.model.model.layers.4.input_layernorm.weight,
language_model.model.model.layers.4.linear_attn.A_log,
language_model.model.model.layers.4.linear_attn.conv1d.weight,
language_model.model.model.layers.4.linear_attn.dt_bias,
language_model.model.model.layers.4.linear_attn.in_proj_a.weight,
language_model.model.model.layers.4.linear_attn.in_proj_b.weight,
language_model.model.model.layers.4.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.4.linear_attn.in_proj_z.weight,
language_model.model.model.layers.4.linear_attn.norm.weight,
language_model.model.model.layers.4.linear_attn.out_proj.weight,
language_model.model.model.layers.4.mlp.down_proj.biases,
language_model.model.model.layers.4.mlp.down_proj.scales,
language_model.model.model.layers.4.mlp.down_proj.weight,
language_model.model.model.layers.4.mlp.gate_proj.biases,
language_model.model.model.layers.4.mlp.gate_proj.scales,
language_model.model.model.layers.4.mlp.gate_proj.weight,
language_model.model.model.layers.4.mlp.up_proj.biases,
language_model.model.model.layers.4.mlp.up_proj.scales,
language_model.model.model.layers.4.mlp.up_proj.weight,
language_model.model.model.layers.4.post_attention_layernorm.weight,
language_model.model.model.layers.5.input_layernorm.weight,
language_model.model.model.layers.5.linear_attn.A_log,
language_model.model.model.layers.5.linear_attn.conv1d.weight,
language_model.model.model.layers.5.linear_attn.dt_bias,
language_model.model.model.layers.5.linear_attn.in_proj_a.weight,
language_model.model.model.layers.5.linear_attn.in_proj_b.weight,
language_model.model.model.layers.5.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.5.linear_attn.in_proj_z.weight,
language_model.model.model.layers.5.linear_attn.norm.weight,
language_model.model.model.layers.5.linear_attn.out_proj.weight,
language_model.model.model.layers.5.mlp.down_proj.biases,
language_model.model.model.layers.5.mlp.down_proj.scales,
language_model.model.model.layers.5.mlp.down_proj.weight,
language_model.model.model.layers.5.mlp.gate_proj.biases,
language_model.model.model.layers.5.mlp.gate_proj.scales,
language_model.model.model.layers.5.mlp.gate_proj.weight,
language_model.model.model.layers.5.mlp.up_proj.biases,
language_model.model.model.layers.5.mlp.up_proj.scales,
language_model.model.model.layers.5.mlp.up_proj.weight,
language_model.model.model.layers.5.post_attention_layernorm.weight,
language_model.model.model.layers.6.input_layernorm.weight,
language_model.model.model.layers.6.linear_attn.A_log,
language_model.model.model.layers.6.linear_attn.conv1d.weight,
language_model.model.model.layers.6.linear_attn.dt_bias,
language_model.model.model.layers.6.linear_attn.in_proj_a.weight,
language_model.model.model.layers.6.linear_attn.in_proj_b.weight,
language_model.model.model.layers.6.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.6.linear_attn.in_proj_z.weight,
language_model.model.model.layers.6.linear_attn.norm.weight,
language_model.model.model.layers.6.linear_attn.out_proj.weight,
language_model.model.model.layers.6.mlp.down_proj.biases,
language_model.model.model.layers.6.mlp.down_proj.scales,
language_model.model.model.layers.6.mlp.down_proj.weight,
language_model.model.model.layers.6.mlp.gate_proj.biases,
language_model.model.model.layers.6.mlp.gate_proj.scales,
language_model.model.model.layers.6.mlp.gate_proj.weight,
language_model.model.model.layers.6.mlp.up_proj.biases,
language_model.model.model.layers.6.mlp.up_proj.scales,
language_model.model.model.layers.6.mlp.up_proj.weight,
language_model.model.model.layers.6.post_attention_layernorm.weight,
language_model.model.model.layers.7.input_layernorm.weight,
language_model.model.model.layers.7.mlp.down_proj.biases,
language_model.model.model.layers.7.mlp.down_proj.scales,
language_model.model.model.layers.7.mlp.down_proj.weight,
language_model.model.model.layers.7.mlp.gate_proj.biases,
language_model.model.model.layers.7.mlp.gate_proj.scales,
language_model.model.model.layers.7.mlp.gate_proj.weight,
language_model.model.model.layers.7.mlp.up_proj.biases,
language_model.model.model.layers.7.mlp.up_proj.scales,
language_model.model.model.layers.7.mlp.up_proj.weight,
language_model.model.model.layers.7.post_attention_layernorm.weight,
language_model.model.model.layers.7.self_attn.k_norm.weight,
language_model.model.model.layers.7.self_attn.k_proj.biases,
language_model.model.model.layers.7.self_attn.k_proj.scales,
language_model.model.model.layers.7.self_attn.k_proj.weight,
language_model.model.model.layers.7.self_attn.o_proj.biases,
language_model.model.model.layers.7.self_attn.o_proj.scales,
language_model.model.model.layers.7.self_attn.o_proj.weight,
language_model.model.model.layers.7.self_attn.q_norm.weight,
language_model.model.model.layers.7.self_attn.q_proj.biases,
language_model.model.model.layers.7.self_attn.q_proj.scales,
language_model.model.model.layers.7.self_attn.q_proj.weight,
language_model.model.model.layers.7.self_attn.v_proj.biases,
language_model.model.model.layers.7.self_attn.v_proj.scales,
language_model.model.model.layers.7.self_attn.v_proj.weight,
language_model.model.model.layers.8.input_layernorm.weight,
language_model.model.model.layers.8.linear_attn.A_log,
language_model.model.model.layers.8.linear_attn.conv1d.weight,
language_model.model.model.layers.8.linear_attn.dt_bias,
language_model.model.model.layers.8.linear_attn.in_proj_a.weight,
language_model.model.model.layers.8.linear_attn.in_proj_b.weight,
language_model.model.model.layers.8.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.8.linear_attn.in_proj_z.weight,
language_model.model.model.layers.8.linear_attn.norm.weight,
language_model.model.model.layers.8.linear_attn.out_proj.weight,
language_model.model.model.layers.8.mlp.down_proj.biases,
language_model.model.model.layers.8.mlp.down_proj.scales,
language_model.model.model.layers.8.mlp.down_proj.weight,
language_model.model.model.layers.8.mlp.gate_proj.biases,
language_model.model.model.layers.8.mlp.gate_proj.scales,
language_model.model.model.layers.8.mlp.gate_proj.weight,
language_model.model.model.layers.8.mlp.up_proj.biases,
language_model.model.model.layers.8.mlp.up_proj.scales,
language_model.model.model.layers.8.mlp.up_proj.weight,
language_model.model.model.layers.8.post_attention_layernorm.weight,
language_model.model.model.layers.9.input_layernorm.weight,
language_model.model.model.layers.9.linear_attn.A_log,
language_model.model.model.layers.9.linear_attn.conv1d.weight,
language_model.model.model.layers.9.linear_attn.dt_bias,
language_model.model.model.layers.9.linear_attn.in_proj_a.weight,
language_model.model.model.layers.9.linear_attn.in_proj_b.weight,
language_model.model.model.layers.9.linear_attn.in_proj_qkv.weight,
language_model.model.model.layers.9.linear_attn.in_proj_z.weight,
language_model.model.model.layers.9.linear_attn.norm.weight,
language_model.model.model.layers.9.linear_attn.out_proj.weight,
language_model.model.model.layers.9.mlp.down_proj.biases,
language_model.model.model.layers.9.mlp.down_proj.scales,
language_model.model.model.layers.9.mlp.down_proj.weight,
language_model.model.model.layers.9.mlp.gate_proj.biases,
language_model.model.model.layers.9.mlp.gate_proj.scales,
language_model.model.model.layers.9.mlp.gate_proj.weight,
language_model.model.model.layers.9.mlp.up_proj.biases,
language_model.model.model.layers.9.mlp.up_proj.scales,
language_model.model.model.layers.9.mlp.up_proj.weight,
language_model.model.model.layers.9.post_attention_layernorm.weight,
language_model.model.model.norm.weight.
```

**Relevant upstream traceback:**

```text
language_model.model.model.layers.9.mlp.gate_proj.weight,
language_model.model.model.layers.9.mlp.up_proj.biases,
language_model.model.model.layers.9.mlp.up_proj.scales,
language_model.model.model.layers.9.mlp.up_proj.weight,
language_model.model.model.layers.9.post_attention_layernorm.weight,
language_model.model.model.norm.weight.
```

<!-- markdownlint-disable MD060 -->

| Model                              | Observed Behavior                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | First Seen Failing      | Recent Repro           |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------|
| `mlx-community/MiniCPM-V-4.6-8bit` | Model loading failed: Received 512 parameters not in model: language_model.model.model.embed_tokens.weight, language_model.model.model.layers.0.input_layernorm.weight, language_model.model.model.layers.0.linear_attn.A_log, language_model.model.model.layers.0.linear_attn.conv1d.weight, language_model.model.model.layers.0.linear_attn.dt_bias, language_model.model.model.layers.0.linear_attn.in_proj_a.weight, language_model.model.model.layers.0.linear_attn.in_proj_b.weight, language_model.model.model.layers.0.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.0.linear_attn.in_proj_z.weight, language_model.model.model.layers.0.linear_attn.norm.weight, language_model.model.model.layers.0.linear_attn.out_proj.weight, language_model.model.model.layers.0.mlp.down_proj.biases, language_model.model.model.layers.0.mlp.down_proj.scales, language_model.model.model.layers.0.mlp.down_proj.weight, language_model.model.model.layers.0.mlp.gate_proj.biases, language_model.model.model.layers.0.mlp.gate_proj.scales, language_model.model.model.layers.0.mlp.gate_proj.weight, language_model.model.model.layers.0.mlp.up_proj.biases, language_model.model.model.layers.0.mlp.up_proj.scales, language_model.model.model.layers.0.mlp.up_proj.weight, language_model.model.model.layers.0.post_attention_layernorm.weight, language_model.model.model.layers.1.input_layernorm.weight, language_model.model.model.layers.1.linear_attn.A_log, language_model.model.model.layers.1.linear_attn.conv1d.weight, language_model.model.model.layers.1.linear_attn.dt_bias, language_model.model.model.layers.1.linear_attn.in_proj_a.weight, language_model.model.model.layers.1.linear_attn.in_proj_b.weight, language_model.model.model.layers.1.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.1.linear_attn.in_proj_z.weight, language_model.model.model.layers.1.linear_attn.norm.weight, language_model.model.model.layers.1.linear_attn.out_proj.weight, language_model.model.model.layers.1.mlp.down_proj.biases, language_model.model.model.layers.1.mlp.down_proj.scales, language_model.model.model.layers.1.mlp.down_proj.weight, language_model.model.model.layers.1.mlp.gate_proj.biases, language_model.model.model.layers.1.mlp.gate_proj.scales, language_model.model.model.layers.1.mlp.gate_proj.weight, language_model.model.model.layers.1.mlp.up_proj.biases, language_model.model.model.layers.1.mlp.up_proj.scales, language_model.model.model.layers.1.mlp.up_proj.weight, language_model.model.model.layers.1.post_attention_layernorm.weight, language_model.model.model.layers.10.input_layernorm.weight, language_model.model.model.layers.10.linear_attn.A_log, language_model.model.model.layers.10.linear_attn.conv1d.weight, language_model.model.model.layers.10.linear_attn.dt_bias, language_model.model.model.layers.10.linear_attn.in_proj_a.weight, language_model.model.model.layers.10.linear_attn.in_proj_b.weight, language_model.model.model.layers.10.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.10.linear_attn.in_proj_z.weight, language_model.model.model.layers.10.linear_attn.norm.weight, language_model.model.model.layers.10.linear_attn.out_proj.weight, language_model.model.model.layers.10.mlp.down_proj.biases, language_model.model.model.layers.10.mlp.down_proj.scales, language_model.model.model.layers.10.mlp.down_proj.weight, language_model.model.model.layers.10.mlp.gate_proj.biases, language_model.model.model.layers.10.mlp.gate_proj.scales, language_model.model.model.layers.10.mlp.gate_proj.weight, language_model.model.model.layers.10.mlp.up_proj.biases, language_model.model.model.layers.10.mlp.up_proj.scales, language_model.model.model.layers.10.mlp.up_proj.weight, language_model.model.model.layers.10.post_attention_layernorm.weight, language_model.model.model.layers.11.input_layernorm.weight, language_model.model.model.layers.11.mlp.down_proj.biases, language_model.model.model.layers.11.mlp.down_proj.scales, language_model.model.model.layers.11.mlp.down_proj.weight, language_model.model.model.layers.11.mlp.gate_proj.biases, language_model.model.model.layers.11.mlp.gate_proj.scales, language_model.model.model.layers.11.mlp.gate_proj.weight, language_model.model.model.layers.11.mlp.up_proj.biases, language_model.model.model.layers.11.mlp.up_proj.scales, language_model.model.model.layers.11.mlp.up_proj.weight, language_model.model.model.layers.11.post_attention_layernorm.weight, language_model.model.model.layers.11.self_attn.k_norm.weight, language_model.model.model.layers.11.self_attn.k_proj.biases, language_model.model.model.layers.11.self_attn.k_proj.scales, language_model.model.model.layers.11.self_attn.k_proj.weight, language_model.model.model.layers.11.self_attn.o_proj.biases, language_model.model.model.layers.11.self_attn.o_proj.scales, language_model.model.model.layers.11.self_attn.o_proj.weight, language_model.model.model.layers.11.self_attn.q_norm.weight, language_model.model.model.layers.11.self_attn.q_proj.biases, language_model.model.model.layers.11.self_attn.q_proj.scales, language_model.model.model.layers.11.self_attn.q_proj.weight, language_model.model.model.layers.11.self_attn.v_proj.biases, language_model.model.model.layers.11.self_attn.v_proj.scales, language_model.model.model.layers.11.self_attn.v_proj.weight, language_model.model.model.layers.12.input_layernorm.weight, language_model.model.model.layers.12.linear_attn.A_log, language_model.model.model.layers.12.linear_attn.conv1d.weight, language_model.model.model.layers.12.linear_attn.dt_bias, language_model.model.model.layers.12.linear_attn.in_proj_a.weight, language_model.model.model.layers.12.linear_attn.in_proj_b.weight, language_model.model.model.layers.12.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.12.linear_attn.in_proj_z.weight, language_model.model.model.layers.12.linear_attn.norm.weight, language_model.model.model.layers.12.linear_attn.out_proj.weight, language_model.model.model.layers.12.mlp.down_proj.biases, language_model.model.model.layers.12.mlp.down_proj.scales, language_model.model.model.layers.12.mlp.down_proj.weight, language_model.model.model.layers.12.mlp.gate_proj.biases, language_model.model.model.layers.12.mlp.gate_proj.scales, language_model.model.model.layers.12.mlp.gate_proj.weight, language_model.model.model.layers.12.mlp.up_proj.biases, language_model.model.model.layers.12.mlp.up_proj.scales, language_model.model.model.layers.12.mlp.up_proj.weight, language_model.model.model.layers.12.post_attention_layernorm.weight, language_model.model.model.layers.13.input_layernorm.weight, language_model.model.model.layers.13.linear_attn.A_log, language_model.model.model.layers.13.linear_attn.conv1d.weight, language_model.model.model.layers.13.linear_attn.dt_bias, language_model.model.model.layers.13.linear_attn.in_proj_a.weight, language_model.model.model.layers.13.linear_attn.in_proj_b.weight, language_model.model.model.layers.13.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.13.linear_attn.in_proj_z.weight, language_model.model.model.layers.13.linear_attn.norm.weight, language_model.model.model.layers.13.linear_attn.out_proj.weight, language_model.model.model.layers.13.mlp.down_proj.biases, language_model.model.model.layers.13.mlp.down_proj.scales, language_model.model.model.layers.13.mlp.down_proj.weight, language_model.model.model.layers.13.mlp.gate_proj.biases, language_model.model.model.layers.13.mlp.gate_proj.scales, language_model.model.model.layers.13.mlp.gate_proj.weight, language_model.model.model.layers.13.mlp.up_proj.biases, language_model.model.model.layers.13.mlp.up_proj.scales, language_model.model.model.layers.13.mlp.up_proj.weight, language_model.model.model.layers.13.post_attention_layernorm.weight, language_model.model.model.layers.14.input_layernorm.weight, language_model.model.model.layers.14.linear_attn.A_log, language_model.model.model.layers.14.linear_attn.conv1d.weight, language_model.model.model.layers.14.linear_attn.dt_bias, language_model.model.model.layers.14.linear_attn.in_proj_a.weight, language_model.model.model.layers.14.linear_attn.in_proj_b.weight, language_model.model.model.layers.14.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.14.linear_attn.in_proj_z.weight, language_model.model.model.layers.14.linear_attn.norm.weight, language_model.model.model.layers.14.linear_attn.out_proj.weight, language_model.model.model.layers.14.mlp.down_proj.biases, language_model.model.model.layers.14.mlp.down_proj.scales, language_model.model.model.layers.14.mlp.down_proj.weight, language_model.model.model.layers.14.mlp.gate_proj.biases, language_model.model.model.layers.14.mlp.gate_proj.scales, language_model.model.model.layers.14.mlp.gate_proj.weight, language_model.model.model.layers.14.mlp.up_proj.biases, language_model.model.model.layers.14.mlp.up_proj.scales, language_model.model.model.layers.14.mlp.up_proj.weight, language_model.model.model.layers.14.post_attention_layernorm.weight, language_model.model.model.layers.15.input_layernorm.weight, language_model.model.model.layers.15.mlp.down_proj.biases, language_model.model.model.layers.15.mlp.down_proj.scales, language_model.model.model.layers.15.mlp.down_proj.weight, language_model.model.model.layers.15.mlp.gate_proj.biases, language_model.model.model.layers.15.mlp.gate_proj.scales, language_model.model.model.layers.15.mlp.gate_proj.weight, language_model.model.model.layers.15.mlp.up_proj.biases, language_model.model.model.layers.15.mlp.up_proj.scales, language_model.model.model.layers.15.mlp.up_proj.weight, language_model.model.model.layers.15.post_attention_layernorm.weight, language_model.model.model.layers.15.self_attn.k_norm.weight, language_model.model.model.layers.15.self_attn.k_proj.biases, language_model.model.model.layers.15.self_attn.k_proj.scales, language_model.model.model.layers.15.self_attn.k_proj.weight, language_model.model.model.layers.15.self_attn.o_proj.biases, language_model.model.model.layers.15.self_attn.o_proj.scales, language_model.model.model.layers.15.self_attn.o_proj.weight, language_model.model.model.layers.15.self_attn.q_norm.weight, language_model.model.model.layers.15.self_attn.q_proj.biases, language_model.model.model.layers.15.self_attn.q_proj.scales, language_model.model.model.layers.15.self_attn.q_proj.weight, language_model.model.model.layers.15.self_attn.v_proj.biases, language_model.model.model.layers.15.self_attn.v_proj.scales, language_model.model.model.layers.15.self_attn.v_proj.weight, language_model.model.model.layers.16.input_layernorm.weight, language_model.model.model.layers.16.linear_attn.A_log, language_model.model.model.layers.16.linear_attn.conv1d.weight, language_model.model.model.layers.16.linear_attn.dt_bias, language_model.model.model.layers.16.linear_attn.in_proj_a.weight, language_model.model.model.layers.16.linear_attn.in_proj_b.weight, language_model.model.model.layers.16.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.16.linear_attn.in_proj_z.weight, language_model.model.model.layers.16.linear_attn.norm.weight, language_model.model.model.layers.16.linear_attn.out_proj.weight, language_model.model.model.layers.16.mlp.down_proj.biases, language_model.model.model.layers.16.mlp.down_proj.scales, language_model.model.model.layers.16.mlp.down_proj.weight, language_model.model.model.layers.16.mlp.gate_proj.biases, language_model.model.model.layers.16.mlp.gate_proj.scales, language_model.model.model.layers.16.mlp.gate_proj.weight, language_model.model.model.layers.16.mlp.up_proj.biases, language_model.model.model.layers.16.mlp.up_proj.scales, language_model.model.model.layers.16.mlp.up_proj.weight, language_model.model.model.layers.16.post_attention_layernorm.weight, language_model.model.model.layers.17.input_layernorm.weight, language_model.model.model.layers.17.linear_attn.A_log, language_model.model.model.layers.17.linear_attn.conv1d.weight, language_model.model.model.layers.17.linear_attn.dt_bias, language_model.model.model.layers.17.linear_attn.in_proj_a.weight, language_model.model.model.layers.17.linear_attn.in_proj_b.weight, language_model.model.model.layers.17.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.17.linear_attn.in_proj_z.weight, language_model.model.model.layers.17.linear_attn.norm.weight, language_model.model.model.layers.17.linear_attn.out_proj.weight, language_model.model.model.layers.17.mlp.down_proj.biases, language_model.model.model.layers.17.mlp.down_proj.scales, language_model.model.model.layers.17.mlp.down_proj.weight, language_model.model.model.layers.17.mlp.gate_proj.biases, language_model.model.model.layers.17.mlp.gate_proj.scales, language_model.model.model.layers.17.mlp.gate_proj.weight, language_model.model.model.layers.17.mlp.up_proj.biases, language_model.model.model.layers.17.mlp.up_proj.scales, language_model.model.model.layers.17.mlp.up_proj.weight, language_model.model.model.layers.17.post_attention_layernorm.weight, language_model.model.model.layers.18.input_layernorm.weight, language_model.model.model.layers.18.linear_attn.A_log, language_model.model.model.layers.18.linear_attn.conv1d.weight, language_model.model.model.layers.18.linear_attn.dt_bias, language_model.model.model.layers.18.linear_attn.in_proj_a.weight, language_model.model.model.layers.18.linear_attn.in_proj_b.weight, language_model.model.model.layers.18.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.18.linear_attn.in_proj_z.weight, language_model.model.model.layers.18.linear_attn.norm.weight, language_model.model.model.layers.18.linear_attn.out_proj.weight, language_model.model.model.layers.18.mlp.down_proj.biases, language_model.model.model.layers.18.mlp.down_proj.scales, language_model.model.model.layers.18.mlp.down_proj.weight, language_model.model.model.layers.18.mlp.gate_proj.biases, language_model.model.model.layers.18.mlp.gate_proj.scales, language_model.model.model.layers.18.mlp.gate_proj.weight, language_model.model.model.layers.18.mlp.up_proj.biases, language_model.model.model.layers.18.mlp.up_proj.scales, language_model.model.model.layers.18.mlp.up_proj.weight, language_model.model.model.layers.18.post_attention_layernorm.weight, language_model.model.model.layers.19.input_layernorm.weight, language_model.model.model.layers.19.mlp.down_proj.biases, language_model.model.model.layers.19.mlp.down_proj.scales, language_model.model.model.layers.19.mlp.down_proj.weight, language_model.model.model.layers.19.mlp.gate_proj.biases, language_model.model.model.layers.19.mlp.gate_proj.scales, language_model.model.model.layers.19.mlp.gate_proj.weight, language_model.model.model.layers.19.mlp.up_proj.biases, language_model.model.model.layers.19.mlp.up_proj.scales, language_model.model.model.layers.19.mlp.up_proj.weight, language_model.model.model.layers.19.post_attention_layernorm.weight, language_model.model.model.layers.19.self_attn.k_norm.weight, language_model.model.model.layers.19.self_attn.k_proj.biases, language_model.model.model.layers.19.self_attn.k_proj.scales, language_model.model.model.layers.19.self_attn.k_proj.weight, language_model.model.model.layers.19.self_attn.o_proj.biases, language_model.model.model.layers.19.self_attn.o_proj.scales, language_model.model.model.layers.19.self_attn.o_proj.weight, language_model.model.model.layers.19.self_attn.q_norm.weight, language_model.model.model.layers.19.self_attn.q_proj.biases, language_model.model.model.layers.19.self_attn.q_proj.scales, language_model.model.model.layers.19.self_attn.q_proj.weight, language_model.model.model.layers.19.self_attn.v_proj.biases, language_model.model.model.layers.19.self_attn.v_proj.scales, language_model.model.model.layers.19.self_attn.v_proj.weight, language_model.model.model.layers.2.input_layernorm.weight, language_model.model.model.layers.2.linear_attn.A_log, language_model.model.model.layers.2.linear_attn.conv1d.weight, language_model.model.model.layers.2.linear_attn.dt_bias, language_model.model.model.layers.2.linear_attn.in_proj_a.weight, language_model.model.model.layers.2.linear_attn.in_proj_b.weight, language_model.model.model.layers.2.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.2.linear_attn.in_proj_z.weight, language_model.model.model.layers.2.linear_attn.norm.weight, language_model.model.model.layers.2.linear_attn.out_proj.weight, language_model.model.model.layers.2.mlp.down_proj.biases, language_model.model.model.layers.2.mlp.down_proj.scales, language_model.model.model.layers.2.mlp.down_proj.weight, language_model.model.model.layers.2.mlp.gate_proj.biases, language_model.model.model.layers.2.mlp.gate_proj.scales, language_model.model.model.layers.2.mlp.gate_proj.weight, language_model.model.model.layers.2.mlp.up_proj.biases, language_model.model.model.layers.2.mlp.up_proj.scales, language_model.model.model.layers.2.mlp.up_proj.weight, language_model.model.model.layers.2.post_attention_layernorm.weight, language_model.model.model.layers.20.input_layernorm.weight, language_model.model.model.layers.20.linear_attn.A_log, language_model.model.model.layers.20.linear_attn.conv1d.weight, language_model.model.model.layers.20.linear_attn.dt_bias, language_model.model.model.layers.20.linear_attn.in_proj_a.weight, language_model.model.model.layers.20.linear_attn.in_proj_b.weight, language_model.model.model.layers.20.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.20.linear_attn.in_proj_z.weight, language_model.model.model.layers.20.linear_attn.norm.weight, language_model.model.model.layers.20.linear_attn.out_proj.weight, language_model.model.model.layers.20.mlp.down_proj.biases, language_model.model.model.layers.20.mlp.down_proj.scales, language_model.model.model.layers.20.mlp.down_proj.weight, language_model.model.model.layers.20.mlp.gate_proj.biases, language_model.model.model.layers.20.mlp.gate_proj.scales, language_model.model.model.layers.20.mlp.gate_proj.weight, language_model.model.model.layers.20.mlp.up_proj.biases, language_model.model.model.layers.20.mlp.up_proj.scales, language_model.model.model.layers.20.mlp.up_proj.weight, language_model.model.model.layers.20.post_attention_layernorm.weight, language_model.model.model.layers.21.input_layernorm.weight, language_model.model.model.layers.21.linear_attn.A_log, language_model.model.model.layers.21.linear_attn.conv1d.weight, language_model.model.model.layers.21.linear_attn.dt_bias, language_model.model.model.layers.21.linear_attn.in_proj_a.weight, language_model.model.model.layers.21.linear_attn.in_proj_b.weight, language_model.model.model.layers.21.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.21.linear_attn.in_proj_z.weight, language_model.model.model.layers.21.linear_attn.norm.weight, language_model.model.model.layers.21.linear_attn.out_proj.weight, language_model.model.model.layers.21.mlp.down_proj.biases, language_model.model.model.layers.21.mlp.down_proj.scales, language_model.model.model.layers.21.mlp.down_proj.weight, language_model.model.model.layers.21.mlp.gate_proj.biases, language_model.model.model.layers.21.mlp.gate_proj.scales, language_model.model.model.layers.21.mlp.gate_proj.weight, language_model.model.model.layers.21.mlp.up_proj.biases, language_model.model.model.layers.21.mlp.up_proj.scales, language_model.model.model.layers.21.mlp.up_proj.weight, language_model.model.model.layers.21.post_attention_layernorm.weight, language_model.model.model.layers.22.input_layernorm.weight, language_model.model.model.layers.22.linear_attn.A_log, language_model.model.model.layers.22.linear_attn.conv1d.weight, language_model.model.model.layers.22.linear_attn.dt_bias, language_model.model.model.layers.22.linear_attn.in_proj_a.weight, language_model.model.model.layers.22.linear_attn.in_proj_b.weight, language_model.model.model.layers.22.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.22.linear_attn.in_proj_z.weight, language_model.model.model.layers.22.linear_attn.norm.weight, language_model.model.model.layers.22.linear_attn.out_proj.weight, language_model.model.model.layers.22.mlp.down_proj.biases, language_model.model.model.layers.22.mlp.down_proj.scales, language_model.model.model.layers.22.mlp.down_proj.weight, language_model.model.model.layers.22.mlp.gate_proj.biases, language_model.model.model.layers.22.mlp.gate_proj.scales, language_model.model.model.layers.22.mlp.gate_proj.weight, language_model.model.model.layers.22.mlp.up_proj.biases, language_model.model.model.layers.22.mlp.up_proj.scales, language_model.model.model.layers.22.mlp.up_proj.weight, language_model.model.model.layers.22.post_attention_layernorm.weight, language_model.model.model.layers.23.input_layernorm.weight, language_model.model.model.layers.23.mlp.down_proj.biases, language_model.model.model.layers.23.mlp.down_proj.scales, language_model.model.model.layers.23.mlp.down_proj.weight, language_model.model.model.layers.23.mlp.gate_proj.biases, language_model.model.model.layers.23.mlp.gate_proj.scales, language_model.model.model.layers.23.mlp.gate_proj.weight, language_model.model.model.layers.23.mlp.up_proj.biases, language_model.model.model.layers.23.mlp.up_proj.scales, language_model.model.model.layers.23.mlp.up_proj.weight, language_model.model.model.layers.23.post_attention_layernorm.weight, language_model.model.model.layers.23.self_attn.k_norm.weight, language_model.model.model.layers.23.self_attn.k_proj.biases, language_model.model.model.layers.23.self_attn.k_proj.scales, language_model.model.model.layers.23.self_attn.k_proj.weight, language_model.model.model.layers.23.self_attn.o_proj.biases, language_model.model.model.layers.23.self_attn.o_proj.scales, language_model.model.model.layers.23.self_attn.o_proj.weight, language_model.model.model.layers.23.self_attn.q_norm.weight, language_model.model.model.layers.23.self_attn.q_proj.biases, language_model.model.model.layers.23.self_attn.q_proj.scales, language_model.model.model.layers.23.self_attn.q_proj.weight, language_model.model.model.layers.23.self_attn.v_proj.biases, language_model.model.model.layers.23.self_attn.v_proj.scales, language_model.model.model.layers.23.self_attn.v_proj.weight, language_model.model.model.layers.3.input_layernorm.weight, language_model.model.model.layers.3.mlp.down_proj.biases, language_model.model.model.layers.3.mlp.down_proj.scales, language_model.model.model.layers.3.mlp.down_proj.weight, language_model.model.model.layers.3.mlp.gate_proj.biases, language_model.model.model.layers.3.mlp.gate_proj.scales, language_model.model.model.layers.3.mlp.gate_proj.weight, language_model.model.model.layers.3.mlp.up_proj.biases, language_model.model.model.layers.3.mlp.up_proj.scales, language_model.model.model.layers.3.mlp.up_proj.weight, language_model.model.model.layers.3.post_attention_layernorm.weight, language_model.model.model.layers.3.self_attn.k_norm.weight, language_model.model.model.layers.3.self_attn.k_proj.biases, language_model.model.model.layers.3.self_attn.k_proj.scales, language_model.model.model.layers.3.self_attn.k_proj.weight, language_model.model.model.layers.3.self_attn.o_proj.biases, language_model.model.model.layers.3.self_attn.o_proj.scales, language_model.model.model.layers.3.self_attn.o_proj.weight, language_model.model.model.layers.3.self_attn.q_norm.weight, language_model.model.model.layers.3.self_attn.q_proj.biases, language_model.model.model.layers.3.self_attn.q_proj.scales, language_model.model.model.layers.3.self_attn.q_proj.weight, language_model.model.model.layers.3.self_attn.v_proj.biases, language_model.model.model.layers.3.self_attn.v_proj.scales, language_model.model.model.layers.3.self_attn.v_proj.weight, language_model.model.model.layers.4.input_layernorm.weight, language_model.model.model.layers.4.linear_attn.A_log, language_model.model.model.layers.4.linear_attn.conv1d.weight, language_model.model.model.layers.4.linear_attn.dt_bias, language_model.model.model.layers.4.linear_attn.in_proj_a.weight, language_model.model.model.layers.4.linear_attn.in_proj_b.weight, language_model.model.model.layers.4.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.4.linear_attn.in_proj_z.weight, language_model.model.model.layers.4.linear_attn.norm.weight, language_model.model.model.layers.4.linear_attn.out_proj.weight, language_model.model.model.layers.4.mlp.down_proj.biases, language_model.model.model.layers.4.mlp.down_proj.scales, language_model.model.model.layers.4.mlp.down_proj.weight, language_model.model.model.layers.4.mlp.gate_proj.biases, language_model.model.model.layers.4.mlp.gate_proj.scales, language_model.model.model.layers.4.mlp.gate_proj.weight, language_model.model.model.layers.4.mlp.up_proj.biases, language_model.model.model.layers.4.mlp.up_proj.scales, language_model.model.model.layers.4.mlp.up_proj.weight, language_model.model.model.layers.4.post_attention_layernorm.weight, language_model.model.model.layers.5.input_layernorm.weight, language_model.model.model.layers.5.linear_attn.A_log, language_model.model.model.layers.5.linear_attn.conv1d.weight, language_model.model.model.layers.5.linear_attn.dt_bias, language_model.model.model.layers.5.linear_attn.in_proj_a.weight, language_model.model.model.layers.5.linear_attn.in_proj_b.weight, language_model.model.model.layers.5.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.5.linear_attn.in_proj_z.weight, language_model.model.model.layers.5.linear_attn.norm.weight, language_model.model.model.layers.5.linear_attn.out_proj.weight, language_model.model.model.layers.5.mlp.down_proj.biases, language_model.model.model.layers.5.mlp.down_proj.scales, language_model.model.model.layers.5.mlp.down_proj.weight, language_model.model.model.layers.5.mlp.gate_proj.biases, language_model.model.model.layers.5.mlp.gate_proj.scales, language_model.model.model.layers.5.mlp.gate_proj.weight, language_model.model.model.layers.5.mlp.up_proj.biases, language_model.model.model.layers.5.mlp.up_proj.scales, language_model.model.model.layers.5.mlp.up_proj.weight, language_model.model.model.layers.5.post_attention_layernorm.weight, language_model.model.model.layers.6.input_layernorm.weight, language_model.model.model.layers.6.linear_attn.A_log, language_model.model.model.layers.6.linear_attn.conv1d.weight, language_model.model.model.layers.6.linear_attn.dt_bias, language_model.model.model.layers.6.linear_attn.in_proj_a.weight, language_model.model.model.layers.6.linear_attn.in_proj_b.weight, language_model.model.model.layers.6.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.6.linear_attn.in_proj_z.weight, language_model.model.model.layers.6.linear_attn.norm.weight, language_model.model.model.layers.6.linear_attn.out_proj.weight, language_model.model.model.layers.6.mlp.down_proj.biases, language_model.model.model.layers.6.mlp.down_proj.scales, language_model.model.model.layers.6.mlp.down_proj.weight, language_model.model.model.layers.6.mlp.gate_proj.biases, language_model.model.model.layers.6.mlp.gate_proj.scales, language_model.model.model.layers.6.mlp.gate_proj.weight, language_model.model.model.layers.6.mlp.up_proj.biases, language_model.model.model.layers.6.mlp.up_proj.scales, language_model.model.model.layers.6.mlp.up_proj.weight, language_model.model.model.layers.6.post_attention_layernorm.weight, language_model.model.model.layers.7.input_layernorm.weight, language_model.model.model.layers.7.mlp.down_proj.biases, language_model.model.model.layers.7.mlp.down_proj.scales, language_model.model.model.layers.7.mlp.down_proj.weight, language_model.model.model.layers.7.mlp.gate_proj.biases, language_model.model.model.layers.7.mlp.gate_proj.scales, language_model.model.model.layers.7.mlp.gate_proj.weight, language_model.model.model.layers.7.mlp.up_proj.biases, language_model.model.model.layers.7.mlp.up_proj.scales, language_model.model.model.layers.7.mlp.up_proj.weight, language_model.model.model.layers.7.post_attention_layernorm.weight, language_model.model.model.layers.7.self_attn.k_norm.weight, language_model.model.model.layers.7.self_attn.k_proj.biases, language_model.model.model.layers.7.self_attn.k_proj.scales, language_model.model.model.layers.7.self_attn.k_proj.weight, language_model.model.model.layers.7.self_attn.o_proj.biases, language_model.model.model.layers.7.self_attn.o_proj.scales, language_model.model.model.layers.7.self_attn.o_proj.weight, language_model.model.model.layers.7.self_attn.q_norm.weight, language_model.model.model.layers.7.self_attn.q_proj.biases, language_model.model.model.layers.7.self_attn.q_proj.scales, language_model.model.model.layers.7.self_attn.q_proj.weight, language_model.model.model.layers.7.self_attn.v_proj.biases, language_model.model.model.layers.7.self_attn.v_proj.scales, language_model.model.model.layers.7.self_attn.v_proj.weight, language_model.model.model.layers.8.input_layernorm.weight, language_model.model.model.layers.8.linear_attn.A_log, language_model.model.model.layers.8.linear_attn.conv1d.weight, language_model.model.model.layers.8.linear_attn.dt_bias, language_model.model.model.layers.8.linear_attn.in_proj_a.weight, language_model.model.model.layers.8.linear_attn.in_proj_b.weight, language_model.model.model.layers.8.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.8.linear_attn.in_proj_z.weight, language_model.model.model.layers.8.linear_attn.norm.weight, language_model.model.model.layers.8.linear_attn.out_proj.weight, language_model.model.model.layers.8.mlp.down_proj.biases, language_model.model.model.layers.8.mlp.down_proj.scales, language_model.model.model.layers.8.mlp.down_proj.weight, language_model.model.model.layers.8.mlp.gate_proj.biases, language_model.model.model.layers.8.mlp.gate_proj.scales, language_model.model.model.layers.8.mlp.gate_proj.weight, language_model.model.model.layers.8.mlp.up_proj.biases, language_model.model.model.layers.8.mlp.up_proj.scales, language_model.model.model.layers.8.mlp.up_proj.weight, language_model.model.model.layers.8.post_attention_layernorm.weight, language_model.model.model.layers.9.input_layernorm.weight, language_model.model.model.layers.9.linear_attn.A_log, language_model.model.model.layers.9.linear_attn.conv1d.weight, language_model.model.model.layers.9.linear_attn.dt_bias, language_model.model.model.layers.9.linear_attn.in_proj_a.weight, language_model.model.model.layers.9.linear_attn.in_proj_b.weight, language_model.model.model.layers.9.linear_attn.in_proj_qkv.weight, language_model.model.model.layers.9.linear_attn.in_proj_z.weight, language_model.model.model.layers.9.linear_attn.norm.weight, language_model.model.model.layers.9.linear_attn.out_proj.weight, language_model.model.model.layers.9.mlp.down_proj.biases, language_model.model.model.layers.9.mlp.down_proj.scales, language_model.model.model.layers.9.mlp.down_proj.weight, language_model.model.model.layers.9.mlp.gate_proj.biases, language_model.model.model.layers.9.mlp.gate_proj.scales, language_model.model.model.layers.9.mlp.gate_proj.weight, language_model.model.model.layers.9.mlp.up_proj.biases, language_model.model.model.layers.9.mlp.up_proj.scales, language_model.model.model.layers.9.mlp.up_proj.weight, language_model.model.model.layers.9.post_attention_layernorm.weight, language_model.model.model.norm.weight. | 2026-07-04 19:55:05 BST | 1/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

### To reproduce

- Use the linked issue draft for the native upstream repro command.
- Representative failing model: `mlx-community/MiniCPM-V-4.6-8bit`

---

## Harness/Integration Issues (9 model(s))

9 model(s) show potential harness/integration issues; see per-model breakdown
below.
These models completed successfully but show integration problems (for example
stop-token leakage, decoding artifacts, or long-context breakdown) that likely
point to stack/runtime behavior rather than inherent model quality limits.

### `Qwen/Qwen3-VL-2B-Instruct`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
This image
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

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 56 occurrences).

**Sample output:**

```text
TheĠimageĠfeaturesĠtwoĠcatsĠlyingĠonĠaĠpinkĠsurface,ĠpossiblyĠaĠcouchĠorĠbed.ĠTheĠcatĠonĠtheĠleftĠisĠlyingĠonĠitsĠbackĠwithĠitsĠlegsĠstretchedĠout,ĠwhileĠtheĠcatĠonĠtheĠrightĠisĠlyingĠonĠitsĠsideĠw...
```

### `mlx-community/Qwen3-VL-2B-Instruct-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
This image
```

### `mlx-community/Qwen3-VL-2B-Thinking-bf16`

**Why this appears to be an integration/runtime issue:**

- Special control token &lt;/think&gt; appeared in generated text.

**Sample output:**

```text
So,,
</think>

Two cats are lying on a bright pink blanket. One cat is a tabby with darker stripes, and the other is a calico with a mix of orange, black, and white fur. Both cats are relaxed, w...
```

### `mlx-community/Qwen3.5-35B-A3B-bf16`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
os,
```

### `mlx-community/Qwen3.6-27B-mxfp8`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 2 tokens.

**Sample output:**

```text
.
```

### `mlx-community/X-Reasoner-7B-8bit`

**Why this appears to be an integration/runtime issue:**

- Output appears truncated to about 3 tokens.

**Sample output:**

```text
The image
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

**Regressions since previous run:** `mlx-community/MiniCPM-V-4.6-8bit`
**Recoveries since previous run:** `mlx-community/LFM2.5-VL-1.6B-bf16`, `mlx-community/MolmoPoint-8B-fp16`
**Quality regressions in history window:** `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/MiniCPM-V-4.6-8bit`, `mlx-community/Qwen3-VL-2B-Instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`
**Core library changes in window:** `mlx=0.32.0.dev20260614+89064477->0.32.0.dev20260704+de7b4ed9`, `mlx-vlm=0.6.3->0.6.4`, `transformers=5.12.0->5.12.1`

<!-- markdownlint-disable MD060 -->

| Model                                           | Status vs Previous Run   | First Seen Failing      | Recent Repro           |
|-------------------------------------------------|--------------------------|-------------------------|------------------------|
| `mlx-community/MiniCPM-V-4.6-8bit`              | new regression           | 2026-07-04 19:55:05 BST | 1/3 recent runs failed |
| `mlx-community/diffusiongemma-26B-A4B-it-8bit`  | still failing            | 2026-06-12 13:18:08 BST | 3/3 recent runs failed |
| `mlx-community/diffusiongemma-26B-A4B-it-mxfp8` | still failing            | 2026-07-03 13:59:09 BST | 3/3 recent runs failed |
<!-- markdownlint-enable MD060 -->

---

## Coverage & Runtime Metrics

- **Detailed diagnostics models:** 12
- **Summary diagnostics models:** 49
- **Coverage check:** ✅ Complete (each model appears exactly once).
- **Total model runtime (sum):** 407.57s (407.57s)
- **Average runtime per model:** 6.68s (6.68s)
- **Dominant runtime phase:** post-prefill decode dominated 17/61 measured model runs (52% of tracked runtime).
- **Phase totals:** model load=135.36s, local prompt prep=0.20s, upstream prefill / first-token=55.58s, post-prefill decode=213.37s, cleanup=7.04s
- **Generation total:** 268.95s across 58 model(s); upstream prefill / first-token split available for 58/58 model(s).
- **Observed stop reasons:** completed=51, exception=3, max_tokens=7
- **Validation overhead:** 0.10s total (avg 0.00s across 61 model(s)).
- **Upstream prefill / first-token latency:** Avg 0.96s | Min 0.02s | Max 6.37s across 58 model(s).
- **What this likely means:** Most measured runtime is spent generating after the first token is available.
- **Suggested next action:** Prioritize early-stop policies, lower long-tail token budgets, or upstream decode-path work.

---

## Models Not Flagged (49 model(s))

These models completed without diagnostics flags (no hard failure, harness
warning, or stack-signal anomaly). The detailed per-model rows remain in the
generated results and review reports.

- **Clean output:** 41 model(s).
- **Ran, but with quality warnings:** 8 model(s).

## Reproducibility

Issue-specific native repro commands are in the linked issue drafts.
Prompt text is in the linked issue drafts and repro bundles.

```bash
# Install check_models benchmarking tool
pip install -e "src/[dev]"

# Re-run with the same CLI arguments
python -m check_models --image /Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg --trust-remote-code --max-tokens 200 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --presence-context-size 20 --frequency-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose
```

Queued issue models: `mlx-community/MiniCPM-V-4.6-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-8bit`, `mlx-community/diffusiongemma-26B-A4B-it-mxfp8`, `Qwen/Qwen3-VL-2B-Instruct`, `mlx-community/Apriel-1.5-15b-Thinker-6bit-MLX`, `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`, `mlx-community/Qwen3-VL-2B-Instruct-bf16`, `mlx-community/Qwen3-VL-2B-Thinking-bf16`, `mlx-community/Qwen3.5-35B-A3B-bf16`, `mlx-community/Qwen3.6-27B-mxfp8`, `mlx-community/X-Reasoner-7B-8bit`, `mlx-community/paligemma2-3b-pt-896-4bit`.

Repro bundles with prompt traces and environment details are available in [repro_bundles/](https://github.com/jrp2014/check_models/tree/main/src/output/repro_bundles).

### Run details

- Input image: `/Users/jrp/Documents/AI/mlx/mlx-vlm/examples/images/cats.jpg`
- Generation settings: max_tokens=200, temperature=0.0, top_p=1.0

Report generated on: 2026-07-04 19:55:05 BST by [check_models](https://github.com/jrp2014/check_models).

<!-- markdownlint-disable MD060 -->

---

## Environment

| Component                  | Version                                                                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| mlx-vlm                    | 0.6.4                                                                                                                                                    |
| mlx                        | 0.32.0.dev20260704+de7b4ed9                                                                                                                              |
| mlx-lm                     | 0.31.3                                                                                                                                                   |
| mlx-audio                  | 0.4.4                                                                                                                                                    |
| transformers               | 5.12.1                                                                                                                                                   |
| tokenizers                 | 0.22.2                                                                                                                                                   |
| huggingface-hub            | 1.22.0                                                                                                                                                   |
| Python Version             | 3.13.13                                                                                                                                                  |
| OS                         | Darwin 25.5.0                                                                                                                                            |
| macOS Version              | 26.5.2                                                                                                                                                   |
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
| MLX Metallib               | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/mlx.metallib (162,451,352 bytes, sha256=7e5c9a3a3225bf3b04a5fe67c50602975d3698a45e2113433465848af47fd70c) |
| MLX libmlx.dylib           | /Users/jrp/Documents/AI/mlx/mlx/python/mlx/lib/libmlx.dylib (21,747,136 bytes, sha256=53b0e529da8969b02cd891b10e5c7b24413dc65c0ccc092343d438e39e13a7d0)  |
| RAM                        | 128.0 GB                                                                                                                                                 |
<!-- markdownlint-enable MD060 -->
