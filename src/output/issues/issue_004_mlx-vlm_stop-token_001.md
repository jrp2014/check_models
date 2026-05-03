# \[mlx-vlm\]\[stop-token\] Special control token &lt;|end|&gt; appeared in generated text affecting 2 model(s)

## Summary

2 model(s) share a `stop_token` signal that clusters under `mlx-vlm`.

- **Issue kind:** `harness`
- **Cluster ID:** `mlx-vlm_stop-token_001`
- **Symptom family:** `stop_token`
- **Acceptance signal:** Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.


## Affected Models

| Model                                | Representative Signal                                                                                                                        | Token Context                                                                                       | Repro Bundle                                                                                                                                                                                  |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `microsoft/Phi-3.5-vision-instruct`  | Special control token &lt;\|end\|&gt; appeared in generated text. \| Special control token &lt;\|endoftext\|&gt; appeared in generated text. | prompt=1,394 \| output/prompt=35.87% \| nontext burden=65% \| stop=completed \| hit token cap (500) | [`20260503T011608Z_001_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json`](../repro_bundles/20260503T011608Z_001_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)   |
| `mlx-community/GLM-4.6V-Flash-mxfp4` | Special control token &lt;/think&gt; appeared in generated text.                                                                             | prompt=6,681 \| output/prompt=7.48% \| nontext burden=93% \| stop=completed \| hit token cap (500)  | [`20260503T011608Z_003_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json`](../repro_bundles/20260503T011608Z_003_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json) |


## Evidence

### `microsoft/Phi-3.5-vision-instruct`

Observed signals:

- Special control token &lt;\|end\|&gt; appeared in generated text.
- Special control token &lt;\|endoftext\|&gt; appeared in generated text.
- Generated text appears to continue into example-code templates mid-output.
- Output switched language/script unexpectedly (tokenizer_artifact, code_snippet).

Sample output:

```text
Title: Woodbridge Tide Mill Museum and Boat

Description: The Woodbridge Tide Mill Museum in Woodbridge, Suffolk, England, is seen at low tide on the River Deben. In the foreground, a long boat cov...
```

### `mlx-community/GLM-4.6V-Flash-mxfp4`

Observed signals:

- Special control token &lt;/think&gt; appeared in generated text.
- Output formatting deviated from the requested structure. Details: Unknown tags: &lt;think&gt;.
- Output omitted required Title/Description/Keywords sections (title, keywords).
- Output leaked reasoning or prompt-template text (&lt;think&gt;).

Sample output:

```text
<think>Got it, let's tackle this. First, I need to analyze the image for cataloguing metadata. The user provided a lot of context, but I need to focus on what's clearly visible.

First, the Title....
```


## Likely Root Cause

- _Likely owner:_ `mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `harness`
- _Issue subtype:_ `stop_token`
- _Why this classification is credible:_ Special control token &lt;\|end\|&gt;
  appeared in generated text. \| Special control token &lt;\|endoftext\|&gt;
  appeared in generated text. \| hit token cap (500) \| nontext prompt
  burden=65%
- _Suggested next action:_ Inspect EOS/stop-token stripping; control tokens
  are leaking into user-facing text.


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models microsoft/Phi-3.5-vision-instruct mlx-community/GLM-4.6V-Flash-mxfp4
```

Repro bundles:

- `microsoft/Phi-3.5-vision-instruct`: [`20260503T011608Z_001_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json`](../repro_bundles/20260503T011608Z_001_microsoft_Phi-3.5-vision-instruct_mlx_vlm_stop_token_001.json)
- `mlx-community/GLM-4.6V-Flash-mxfp4`: [`20260503T011608Z_003_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json`](../repro_bundles/20260503T011608Z_003_mlx-community_GLM-4.6V-Flash-mxfp4_mlx_vlm_stop_token_001.json)


## Fix Checklist

- [ ] Inspect model EOS token IDs and tokenizer special-token mappings.
- [ ] Verify mlx-vlm stop criteria receive all configured EOS/stop tokens.
- [ ] Check `skip_special_tokens` handling during decode.
- [ ] Strip generated control tokens such as `<|end|>` and `</think>` only after confirming generation stopped at the right boundary.


## Acceptance Criteria

- [ ] Affected reruns contain no leaked stop/control tokens and terminate cleanly before the configured max-token cap when the response is complete.
- [ ] The cluster rerun no longer produces this `mlx-vlm_stop-token_001` maintainer-triage cluster.


## Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260502+e8ebdebe |
| mlx-lm          | 0.31.3                      |
| transformers    | 5.7.0                       |
| tokenizers      | 0.22.2                      |
| huggingface-hub | 1.13.0                      |
| Python Version  | 3.13.12                     |
| OS              | Darwin 25.4.0               |
| macOS Version   | 26.4.1                      |
| GPU/Chip        | Apple M5 Max                |
| GPU Cores       | 40                          |
| Metal Support   | Metal 4                     |
| RAM             | 128.0 GB                    |

