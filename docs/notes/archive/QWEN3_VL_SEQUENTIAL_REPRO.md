# Archived Qwen3-VL Sequential Reproduction Probe

Archived on 2026-07-26.

`src/tools/qwen3_vl_sequential_repro.py` was a narrow, upstream-only diagnostic
for comparing single-model, reversed-order, repeated-model, and sequential
in-process Qwen3-VL failures. It was never part of the normal test or CI paths.

The probe was retired because paired and repeated runs could trigger a native
Metal abort and leave WindowServer or GPU state unstable. Keeping a hazardous,
one-off executable among maintained tools made its support status unclear.

Its former invocation shape was:

```bash
cd src
python tools/qwen3_vl_sequential_repro.py /path/to/image.jpg --plan
```

The final source remains available from Git history:

```bash
git show 9225c527:src/tools/qwen3_vl_sequential_repro.py
```

Use the maintained `check_models.py` harness and its captured traceback,
environment, provenance, and native reproduction evidence for current runtime
diagnostics. Restore the archived probe only for a specific upstream
investigation that requires its exact sequential-process experiment.
