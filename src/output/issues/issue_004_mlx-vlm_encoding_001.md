# \[mlx-vlm\]\[Tokenizer / decoding artifact\] Tokenizer decode leaked BPE/byte markers affecting 1 model(s)

## Summary

1 model(s) show **Tokenizer / decoding artifact** that should be filed against mlx-vlm.

- **Observed problem:** Tokenizer decode leaked BPE/byte markers
- **Target:** mlx-vlm
- **Raw owner hint:** `mlx-vlm`
- **Affected models:** 1
- **Fixed when:** No BPE/byte markers in output.
- **Issue kind:** `harness`
- **Raw cluster:** `mlx-vlm_encoding_001` (`encoding`)


## Affected Models

| Model                                                   | Representative Signal                                                                       | Token Context                                                               | Repro JSON                                                                                                                          |
|---------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit` | Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences). | prompt=3,619 \| output/prompt=2.98% \| nontext burden=88% \| stop=completed | [repro JSON](../repro_bundles/20260503T224052Z_001_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json) |


## Minimal Evidence

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences).
- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: Output omitted required Title/Description/Keywords sections (description, keywords).
- Output excerpt: `Title:ĠClassicĠsailboatĠmooredĠinĠestuaryĊĊDescription:ĠAĠclassic-styleĠsailboatĠwithĠaĠdarkĠhullĠandĠwoodenĠmastĠisĠmooredĠinĠaĠcalmĠestuaryĠduringĠlowĠtide.ĠTheĠwaterĠhasĠreceded,ĠexposingĠgreen,Ġalgae-coveredĠmudflatsĠbehindĠtheĠvessel.ĊĊKeywords:Ġsailboat,ĠwoodenĠmast,ĠdarkĠhull,Ġestuary,ĠlowĠtide,Ġmudflats,Ġgre...`


## Repro Commands

Cluster rerun:

```bash
python -m check_models --folder /Users/jrp/Pictures/Processed --trust-remote-code --max-tokens 500 --temperature 0.0 --top-p 1.0 --repetition-context-size 20 --prefill-step-size 4096 --timeout 300.0 --verbose --models mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit
```

Repro bundles:

- `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`: [repro JSON](../repro_bundles/20260503T224052Z_001_mlx-community_Devstral-Small-2-24B-Instruct-2512-5bit_mlx_vlm_encoding_001.json)
- Note: these are local artifact links; attach or publish the JSON when filing upstream.


## Expected Fix Signal

- [ ] Affected reruns contain no leaked BPE, byte-level, or tokenizer marker text.
- [ ] The cluster rerun no longer produces this `mlx-vlm_encoding_001` maintainer-triage cluster.


## Fix Checklist

- [ ] Inspect tokenizer decode cleanup for byte-level/BPE marker leakage.
- [ ] Compare `decode` and `batch_decode` behavior with `skip_special_tokens=True`.
- [ ] Verify processor/tokenizer config does not require model-specific cleanup flags.


## Likely Root Cause

- _Filing target:_ mlx-vlm
- _Likely owner:_ `mlx-vlm`
- _Confidence:_ high
- _Issue kind:_ `harness`
- _Issue subtype:_ `encoding`
- _Why this classification is credible:_ Tokenizer space-marker artifacts (for
  example Ġ) appeared in output (about 61 occurrences). \| nontext prompt
  burden=88% \| missing sections: description, keywords \| missing terms:
  vast, expanse, adorned, small, floats
- _Suggested next action:_ Inspect decode cleanup; tokenizer markers are
  leaking into user-facing text.


## Appendix: Environment

| Component       | Version                     |
|-----------------|-----------------------------|
| mlx-vlm         | 0.4.5                       |
| mlx             | 0.32.0.dev20260503+e8ebdebe |
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


## Appendix: Detailed Evidence

### `mlx-community/Devstral-Small-2-24B-Instruct-2512-5bit`

Observed signals:

- Tokenizer space-marker artifacts (for example Ġ) appeared in output (about 61 occurrences).
- Output omitted required Title/Description/Keywords sections (description, keywords).
- Output appears to copy prompt context verbatim (32% overlap).

Sample output:

```text
Title:ĠClassicĠsailboatĠmooredĠinĠestuaryĊĊDescription:ĠAĠclassic-styleĠsailboatĠwithĠaĠdarkĠhullĠandĠwoodenĠmastĠisĠmooredĠinĠaĠcalmĠestuaryĠduringĠlowĠtide.ĠTheĠwaterĠhasĠreceded,ĠexposingĠgreen,...
```

