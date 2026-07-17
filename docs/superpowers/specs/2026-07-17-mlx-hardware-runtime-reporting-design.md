# MLX Hardware and Runtime Reporting Design

**Date:** 2026-07-17
**Status:** Approved for implementation planning

## Purpose

Improve hardware and runtime observability without adding dependencies or changing
the meaning of existing benchmark metrics. The implementation will use MLX's
native device information where available, retain macOS command fallbacks, and
surface the resulting facts in both human-readable and structured outputs.

This first slice deliberately excludes kernel process-footprint and VM-statistics
wrappers. Those metrics need a separate sampling design before they can be treated
as per-model benchmark evidence.

## Goals

- Obtain MLX device name, GPU architecture, unified-memory size, and Metal's
  recommended working-set ceiling through a fast, cached `mx.device_info()` probe.
- Continue producing chip and total-memory facts when optional `psutil` is absent.
- Record fused scaled-dot-product-attention availability as explicit runtime state.
- Contextualize upstream peak memory against the recommended working-set ceiling.
- Carry new facts into console, HTML, Markdown, TSV, JSONL, history, diagnostics,
  and reproduction environment output where those formats already expose the
  corresponding system or performance information.
- Preserve existing `peak_memory` and `peak_memory_gb` semantics and compatibility.

## Non-goals

- Do not replace upstream `GenerationResult.peak_memory` with a locally sampled
  value.
- Do not add a hardcoded total-memory fallback or present an estimate as detected
  hardware.
- Do not add `libproc`, `RUSAGE_INFO_V4`, `host_statistics64`, or background memory
  polling in this change.
- Do not remove `system_profiler`; it remains the fallback for fields such as GPU
  core count that MLX does not provide.
- Do not split `src/check_models.py` or add a new runtime dependency.

## Architecture

### Cached MLX device probe

Add a small cached helper in the console/system metadata section. It will:

1. Return no data when MLX is unavailable.
2. Call `mx.device_info()` once per process when callable.
3. Normalize only recognized values with valid types and non-negative numeric
   ranges.
4. Return no data and log at debug level when MLX raises a known runtime or value
   error.

The normalized result will retain raw byte counts internally. Human-readable
formatting will happen only at output boundaries.

Recognized fields are:

- `device_name`
- `architecture`
- `memory_size`
- `max_recommended_working_set_size`

Unknown keys remain ignored so MLX can extend its result without changing this
tool's output contract.

### Chip and total-memory fallback chain

Chip identity will use this order:

1. Existing cached `system_profiler` GPU/chip name.
2. `/usr/sbin/sysctl -n machdep.cpu.brand_string` on macOS.
3. MLX `device_name`.
4. Omit the value if every source is unavailable.

Total unified memory will use this order:

1. `psutil.virtual_memory().total` when available and valid, preserving current
   behavior.
2. `/usr/sbin/sysctl -n hw.memsize` on macOS.
3. MLX `memory_size`.
4. Omit the value if every source is unavailable.

The command probes will use the existing bounded macOS command helper. Invalid,
empty, zero, or negative results will fall through rather than becoming report
facts. No default memory amount will be invented.

GPU core count and Metal family will continue to come from `system_profiler`.

### Typed run context

The recommended working-set size must not be recovered by parsing a formatted
system-information string. Add an optional `recommended_working_set_bytes` field
to both `FinalizationInputs` and `ReportRenderContext`, and populate it from the
normalized MLX device probe. Finalization passes the same value to every report
and persistence builder that needs it.

The raw value is run-level context, not a model result. Report builders use it to
derive a model-level percentage:

```text
peak_memory_working_set_pct = peak_memory_gb * 1_000_000_000
                              / recommended_working_set_bytes * 100
```

This uses the same decimal-gigabyte convention already used for MLX allocator
metrics. A percentage is emitted only when peak memory is finite and non-negative
and the denominator is a positive integer.

The existing peak-memory value remains canonical for sorting, efficiency scoring,
and compatibility. The percentage is contextual metadata, not a replacement or
new ranking input.

### Fused-attention capability probe

Extend `collect_runtime_fingerprint()` with a `fused_attention` entry. Availability
requires both:

- `mx.fast` to exist; and
- `mx.fast.scaled_dot_product_attention` to be callable.

The probe follows the existing `RuntimeProbeResult` states:

- `ok` when callable;
- `unavailable` when MLX or the callable is absent; and
- `errored` with bounded detail if attribute access raises.

The probe must always be present in the fingerprint, matching the existing
"never silently omit a probe" rule.

Capability detection will live in one shared helper. The runtime fingerprint uses
its structured result directly, while `get_system_characteristics()` maps the same
result to human-readable text. This prevents the two output paths from reporting
different states.

## Output Contract

### System and environment facts

When available, `get_system_characteristics()` adds stable human-readable rows:

- `GPU/Chip`
- `MLX Device`
- `GPU Architecture`
- `RAM`
- `Recommended Working Set`
- `Fused Attention`

Existing keys retain their current names and meanings. The new rows automatically
flow through the current system-information mappings into:

- the console version/system block;
- HTML and Markdown environment sections;
- diagnostics environment sections;
- JSONL metadata and history run records; and
- reproduction-bundle environment data.

Unavailable hardware facts are omitted. `Fused Attention` is different because it
describes capability state: it is rendered as `Available`, `Unavailable`, or
`Error`, consistent with the runtime fingerprint.

### Peak-memory context

Human-readable console, HTML, and Markdown model metrics render a valid ratio in
this form:

```text
18.2 GB (19.0% of 96.0 GB recommended working set)
```

When the denominator is unavailable, rendering remains exactly the current bare
peak-memory value. A missing ratio must not suppress peak memory.

Structured outputs add the optional field
`peak_memory_working_set_pct` alongside existing peak-memory fields:

- per-model JSONL metrics;
- per-model history facts; and
- a TSV column with the same machine-oriented field name.

The field is numeric, expressed as a percentage rather than a fraction, and is not
clamped at 100 because exceeding the recommended working set is meaningful
diagnostic evidence. Additive fields do not require a format-version bump.

## Error Handling

- Hardware detection is best effort and must never prevent model execution or
  report generation.
- MLX device-probe and macOS command failures are logged at debug level.
- Probe errors must not be replaced with guessed values.
- One failing source advances to the next fallback without discarding facts already
  collected from other sources.
- Numeric values must be checked for type, finiteness where applicable, and valid
  range before use.
- A fused-attention attribute-access failure becomes an explicit `errored` runtime
  state rather than a missing key.

## Testing

Add tests to existing files only.

### Hardware detection tests

In `src/tests/test_pure_logic_functions.py`, cover:

- successful normalization and caching of MLX device information;
- missing MLX and exception paths;
- chip fallback order;
- total-memory fallback order with and without `psutil`;
- rejection of invalid sysctl and MLX values; and
- system-characteristic rows for each available fact.

All tests mock platform and subprocess boundaries and make no real hardware calls.

### Runtime fingerprint tests

In `src/tests/test_jsonl_output.py`, update the exact probe-key assertion and cover
available, unavailable, and errored fused-attention states.

### Report and structured-output tests

Use the existing report, TSV, and JSONL test files to verify:

- human peak-memory rendering with and without a denominator;
- exact percentage calculation using decimal gigabytes;
- JSONL, history, and TSV additive fields;
- percentages above 100 remain visible;
- invalid denominators omit only the percentage; and
- system facts reach human and machine environment sections.

Tests that generate files must use `tmp_path` or existing gitignored test-output
locations and must not rewrite tracked `src/output/` snapshots.

## Documentation and Change Record

Update `CHANGELOG.md` under `[Unreleased]` to describe the new hardware facts,
working-set context, and fused-attention probe. Update `src/README.md` only if its
documented report schema or sample output enumerates the affected fields.

## Verification

Follow the repository workflow:

1. Activate the `mlx-vlm` conda environment and validate it.
2. Run focused hardware, JSONL, report, and TSV tests.
3. Run `make format`.
4. Run `make -C src lint-fix` and `make lint`.
5. Run the commit-hygiene check.
6. Run `make quality` as the final gate.
