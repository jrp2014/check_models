# MLX Hardware and Runtime Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dependency-free MLX/macOS hardware facts, fused-attention capability state, and recommended-working-set context to every relevant human and structured benchmark output.

**Architecture:** Normalize `mx.device_info()` once behind a cached typed helper, preserve the current `system_profiler` behavior, and add bounded `sysctl` fallbacks for chip and memory. Carry the raw recommended-working-set byte count through `ReportRenderContext`; derive one additive percentage in `MachineArtifactFacts` so JSONL, history, and TSV share the same value, while human renderers use a single formatter.

**Tech Stack:** Python 3.13+, MLX (`mlx.core`), standard-library `functools`, `math`, `subprocess`, existing dataclasses/TypedDicts, pytest/unittest.mock, Rich, tabulate.

## Global Constraints

- Always activate the `mlx-vlm` conda environment before Python, pytest, or make commands.
- Keep `src/check_models.py` as the intentional single-file monolith.
- Add tests only to existing `src/tests/test_*.py` files.
- Do not add a runtime dependency.
- Do not invent a hardcoded memory amount when hardware probes fail.
- Do not replace or reinterpret upstream `GenerationResult.peak_memory`.
- Do not add `libproc`, `RUSAGE_INFO_V4`, `host_statistics64`, or memory polling.
- Keep all new structured fields additive; do not bump JSONL/history format versions.
- Route generated test artifacts to `tmp_path`; do not rewrite tracked `src/output/` files.
- Update `CHANGELOG.md` under `[Unreleased]` for this maintainer-visible feature.
- Before the final quality gate, run `make format`, `make -C src lint-fix`, and `make lint` in that order.

## File Map

- Modify `src/check_models.py`: typed device facts, fallbacks, capability probe, run-context propagation, ratio calculation, and all output formatting.
- Modify `src/tests/test_pure_logic_functions.py`: cached MLX device facts, fallback order, invalid values, and system-information rows.
- Modify `src/tests/test_jsonl_output.py`: fused-attention states plus JSONL/history working-set facts.
- Modify `src/tests/test_tsv_output.py`: additive working-set percentage column.
- Modify `src/tests/test_report_generation.py`: HTML/Markdown system facts and peak-memory context.
- Modify `src/tests/test_metrics_modes.py`: compact and detailed console memory context.
- Modify `CHANGELOG.md`: `[Unreleased]` feature entry.
- Conditionally modify `src/README.md`: only if its checked-in schema/sample sections enumerate the changed output fields.

---

### Task 1: Cached MLX Device Facts and macOS Fallbacks

**Files:**

- Modify: `src/check_models.py` at `SystemProfilerDict`, `get_system_info()`, `_run_macos_toolchain_command()`, and `get_system_characteristics()`
- Test: `src/tests/test_pure_logic_functions.py` in `TestSystemProfilerParsing` or an adjacent `TestHardwareFacts` class

**Interfaces:**

- Produces: `MlxDeviceInfo`, `_get_mlx_device_info() -> MlxDeviceInfo`, `_get_macos_sysctl_value(name: str) -> str | None`, `_get_total_memory_bytes() -> int | None`, `_get_recommended_working_set_bytes() -> int | None`
- Consumes: module-level `mx`, `psutil`, `_run_macos_toolchain_command()`, `get_system_info()`, `DECIMAL_GB`

- [ ] **Step 1: Write failing tests for normalized and cached MLX device facts**

Add tests that patch `mod.mx`, clear the new cache before and after each assertion, and verify both normalization and a single runtime call:

```python
class TestHardwareFacts:
    def test_get_mlx_device_info_normalizes_and_caches(
        self,
        mod: types.ModuleType,
    ) -> None:
        runtime = MagicMock()
        runtime.device_info.return_value = {
            "device_name": "Apple M5 Max",
            "architecture": "applegpu_g17s",
            "memory_size": 128 * 1024**3,
            "max_recommended_working_set_size": 96 * 1024**3,
            "ignored": "value",
        }
        mod._get_mlx_device_info.cache_clear()
        with patch.object(mod, "mx", runtime):
            first = mod._get_mlx_device_info()
            second = mod._get_mlx_device_info()
        mod._get_mlx_device_info.cache_clear()

        assert first == second == {
            "device_name": "Apple M5 Max",
            "architecture": "applegpu_g17s",
            "memory_size": 128 * 1024**3,
            "max_recommended_working_set_size": 96 * 1024**3,
        }
        runtime.device_info.assert_called_once_with()

    @pytest.mark.parametrize(
        "payload",
        [
            {"memory_size": True},
            {"memory_size": 0},
            {"memory_size": -1},
            {"memory_size": "128 GB"},
            {"device_name": "  ", "architecture": 17},
        ],
    )
    def test_get_mlx_device_info_rejects_invalid_values(
        self,
        mod: types.ModuleType,
        payload: dict[str, object],
    ) -> None:
        runtime = MagicMock()
        runtime.device_info.return_value = payload
        mod._get_mlx_device_info.cache_clear()
        with patch.object(mod, "mx", runtime):
            assert mod._get_mlx_device_info() == {}
        mod._get_mlx_device_info.cache_clear()
```

Import `MagicMock` beside the existing `patch` import. Also add one test each for
`mx is None`, a missing/non-callable `device_info`, and `device_info()` raising
`RuntimeError`; every case must return `{}`.

- [ ] **Step 2: Run the new hardware tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py -k "mlx_device_info" -q'
```

Expected: FAIL because `MlxDeviceInfo` and `_get_mlx_device_info` do not exist.

- [ ] **Step 3: Add the typed cached MLX device helper**

Near the existing system-profiler types, add:

```python
class MlxDeviceInfo(TypedDict, total=False):
    """Validated subset of ``mx.device_info()`` used by reports."""

    device_name: str
    architecture: str
    memory_size: int
    max_recommended_working_set_size: int
```

In the console/system metadata section, add:

```python
@lru_cache(maxsize=1)
def _get_mlx_device_info() -> MlxDeviceInfo:
    """Return the validated report-facing subset of ``mx.device_info()``."""
    if mx is None:
        return {}
    device_info_fn = getattr(mx, "device_info", None)
    if not callable(device_info_fn):
        return {}
    try:
        payload = _as_str_object_mapping(device_info_fn())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as err:
        logger.debug("Could not retrieve MLX device information: %s", err)
        return {}
    if payload is None:
        return {}

    info: MlxDeviceInfo = {}
    device_name = payload.get("device_name")
    if isinstance(device_name, str) and device_name.strip():
        info["device_name"] = device_name.strip()
    architecture = payload.get("architecture")
    if isinstance(architecture, str) and architecture.strip():
        info["architecture"] = architecture.strip()
    memory_size = payload.get("memory_size")
    if type(memory_size) is int and memory_size > 0:
        info["memory_size"] = memory_size
    recommended_size = payload.get("max_recommended_working_set_size")
    if type(recommended_size) is int and recommended_size > 0:
        info["max_recommended_working_set_size"] = recommended_size
    return info
```

- [ ] **Step 4: Run the focused MLX device tests**

Run the Step 2 command again.

Expected: PASS.

- [ ] **Step 5: Write failing tests for sysctl and total-memory fallback order**

Add tests that establish these exact contracts:

```python
def test_get_total_memory_bytes_prefers_psutil(
    mod: types.ModuleType,
) -> None:
    fake_psutil = MagicMock()
    fake_psutil.virtual_memory.return_value.total = 64 * 1024**3
    with (
        patch.object(mod, "psutil", fake_psutil),
        patch.object(mod, "_get_macos_sysctl_value") as sysctl_value,
        patch.object(mod, "_get_mlx_device_info") as mlx_info,
    ):
        assert mod._get_total_memory_bytes() == 64 * 1024**3
    sysctl_value.assert_not_called()
    mlx_info.assert_not_called()

def test_get_total_memory_bytes_falls_back_from_sysctl_to_mlx(
    mod: types.ModuleType,
) -> None:
    with (
        patch.object(mod, "psutil", None),
        patch.object(mod.platform, "system", return_value="Darwin"),
        patch.object(mod, "_get_macos_sysctl_value", return_value=None),
        patch.object(mod, "_get_mlx_device_info", return_value={"memory_size": 32 * 1024**3}),
    ):
        assert mod._get_total_memory_bytes() == 32 * 1024**3

def test_get_total_memory_bytes_returns_none_without_valid_source(
    mod: types.ModuleType,
) -> None:
    with (
        patch.object(mod, "psutil", None),
        patch.object(mod.platform, "system", return_value="Darwin"),
        patch.object(mod, "_get_macos_sysctl_value", return_value="0"),
        patch.object(mod, "_get_mlx_device_info", return_value={}),
    ):
        assert mod._get_total_memory_bytes() is None
```

Add a test for successful `hw.memsize`, one for a malformed integer, and one for
`_get_recommended_working_set_bytes()` returning only a positive MLX value.

- [ ] **Step 6: Run the fallback tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py -k "total_memory_bytes or recommended_working_set or macos_sysctl" -q'
```

Expected: FAIL because the fallback helpers do not exist.

- [ ] **Step 7: Implement bounded sysctl, total-memory, and working-set helpers**

Add:

```python
def _get_macos_sysctl_value(name: str) -> str | None:
    """Read one allowlisted macOS hardware sysctl through the bounded runner."""
    if platform.system() != "Darwin":
        return None
    if name not in {"hw.memsize", "machdep.cpu.brand_string"}:
        return None
    return _run_macos_toolchain_command(["/usr/sbin/sysctl", "-n", name])


def _positive_int_text(value: str | None) -> int | None:
    """Parse a strictly positive integer or return no value."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _get_total_memory_bytes() -> int | None:
    """Return detected unified memory without inventing a fallback amount."""
    if psutil is not None:
        try:
            total = psutil.virtual_memory().total
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            total = None
        if type(total) is int and total > 0:
            return total
    sysctl_total = _positive_int_text(_get_macos_sysctl_value("hw.memsize"))
    if sysctl_total is not None:
        return sysctl_total
    return _get_mlx_device_info().get("memory_size")


def _get_recommended_working_set_bytes() -> int | None:
    """Return Metal's positive recommended working-set ceiling when available."""
    return _get_mlx_device_info().get("max_recommended_working_set_size")
```

- [ ] **Step 8: Write failing tests for system-information output and chip fallback**

Test a fully populated MLX result and verify these rows:

```python
def test_get_system_characteristics_surfaces_mlx_hardware_facts(
    mod: types.ModuleType,
) -> None:
    with (
        patch.object(mod.platform, "system", return_value="Darwin"),
        patch.object(mod.platform, "machine", return_value="arm64"),
        patch.object(mod, "_get_macos_toolchain_info", return_value={}),
        patch.object(mod, "get_system_info", return_value=("arm64", None)),
        patch.object(mod, "_get_apple_silicon_info", return_value={}),
        patch.object(mod, "_get_macos_sysctl_value", return_value="Apple M5 Max"),
        patch.object(mod, "_get_total_memory_bytes", return_value=128 * 1024**3),
        patch.object(mod, "_get_mlx_device_info", return_value={
            "device_name": "Apple M5 Max",
            "architecture": "applegpu_g17s",
            "max_recommended_working_set_size": 96 * 1024**3,
        }),
        patch.object(mod, "_get_mlx_backend_artifact_info", return_value={}),
    ):
        info = mod.get_system_characteristics()

    assert info["GPU/Chip"] == "Apple M5 Max"
    assert info["MLX Device"] == "Apple M5 Max"
    assert info["GPU Architecture"] == "applegpu_g17s"
    assert info["RAM"] == "128.0 GB"
    assert info["Recommended Working Set"] == "96.0 GB"
```

Add separate tests proving that an existing `system_profiler` GPU name wins, a
valid `machdep.cpu.brand_string` fills a missing name, and MLX `device_name` is the
last fallback. Task 2 will extend these fixtures when it adds the fused-attention
row.

- [ ] **Step 9: Implement the additive system-information rows**

Refactor `get_system_characteristics()` to take one `device_info =
_get_mlx_device_info()` snapshot, preserve current platform/toolchain rows, and
apply this logic:

```python
_, profiler_gpu_name = get_system_info()
chip_name = profiler_gpu_name
if chip_name is None:
    chip_name = _get_macos_sysctl_value("machdep.cpu.brand_string")
if chip_name is None:
    chip_name = device_info.get("device_name")
if chip_name:
    info["GPU/Chip"] = chip_name

device_name = device_info.get("device_name")
if device_name:
    info["MLX Device"] = device_name
architecture = device_info.get("architecture")
if architecture:
    info["GPU Architecture"] = architecture

total_memory_bytes = _get_total_memory_bytes()
if total_memory_bytes is not None:
    info["RAM"] = f"{total_memory_bytes / (1024**3):.1f} GB"
recommended_bytes = _get_recommended_working_set_bytes()
if recommended_bytes is not None:
    info["Recommended Working Set"] = f"{recommended_bytes / (1024**3):.1f} GB"
```

Leave physical/logical core collection under the existing `psutil` guard. Do not
catch one broad exception around all probes if that would prevent later facts from
being collected after an earlier failure.

- [ ] **Step 10: Run all Task 1 tests and commit**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py -k "HardwareFacts or SystemProfilerParsing" -q'
```

Expected: PASS.

Commit:

```bash
git add src/check_models.py src/tests/test_pure_logic_functions.py
git commit -m "feat: add native MLX hardware facts"
```

---

### Task 2: Fused-Attention Capability State

**Files:**

- Modify: `src/check_models.py` at `RuntimeProbeResult`, `get_system_characteristics()`, and `collect_runtime_fingerprint()`
- Test: `src/tests/test_jsonl_output.py` in `TestRuntimeFingerprint`
- Test: `src/tests/test_pure_logic_functions.py` in `TestHardwareFacts`

**Interfaces:**

- Produces: `_probe_fused_attention() -> RuntimeProbeResult`
- Consumes: module-level `mx`; `RuntimeProbeResult`
- Changes fingerprint key set to: `metal_gpu`, `mlx_framework`, `mlx_vlm`, `gpu_memory`, `fused_attention`

- [ ] **Step 1: Write failing runtime-fingerprint tests for all fused-attention states**

Update the exact key assertion, then add:

```python
def test_collect_runtime_fingerprint_reports_fused_attention_available(self) -> None:
    runtime = SimpleNamespace(
        fast=SimpleNamespace(scaled_dot_product_attention=lambda: None),
    )
    with patch.object(check_models, "mx", runtime):
        fingerprint = check_models.collect_runtime_fingerprint()
    assert fingerprint["fused_attention"] == {"status": "ok"}

def test_collect_runtime_fingerprint_reports_fused_attention_unavailable(self) -> None:
    with patch.object(check_models, "mx", SimpleNamespace()):
        fingerprint = check_models.collect_runtime_fingerprint()
    assert fingerprint["fused_attention"]["status"] == "unavailable"

def test_probe_fused_attention_reports_attribute_error(self) -> None:
    class RaisingRuntime:
        @property
        def fast(self) -> object:
            raise RuntimeError("runtime unavailable")

    with patch.object(check_models, "mx", RaisingRuntime()):
        result = check_models._probe_fused_attention()
    assert result == {"status": "errored", "detail": "runtime unavailable"}
```

Keep existing probe tests isolated: fake MLX runtimes used by other tests must now
either tolerate missing `fast` as unavailable or define the required surface.
Import `SimpleNamespace` from `types` at the top of `src/tests/test_jsonl_output.py`.

- [ ] **Step 2: Run the runtime-fingerprint tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_jsonl_output.py::TestRuntimeFingerprint -q'
```

Expected: FAIL because the fingerprint lacks `fused_attention`.

- [ ] **Step 3: Implement the shared capability probe and fingerprint entry**

Add:

```python
def _probe_fused_attention() -> RuntimeProbeResult:
    """Report whether MLX exposes callable fused scaled-dot-product attention."""
    if mx is None:
        return RuntimeProbeResult(status="unavailable", detail="mlx not imported")
    try:
        fast = getattr(mx, "fast", None)
        attention = getattr(fast, "scaled_dot_product_attention", None)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return RuntimeProbeResult(status="errored", detail=str(exc)[:120])
    if callable(attention):
        return RuntimeProbeResult(status="ok")
    return RuntimeProbeResult(status="unavailable")
```

At the end of `collect_runtime_fingerprint()`, assign:

```python
probes["fused_attention"] = _probe_fused_attention()
```

- [ ] **Step 4: Add the human-readable system row from the same probe**

In `get_system_characteristics()`, map the structured state exactly once:

```python
fused_attention = _probe_fused_attention()
fused_attention_labels = {
    "ok": "Available",
    "unavailable": "Unavailable",
    "errored": "Error",
    "timed_out": "Timed out",
}
info["Fused Attention"] = fused_attention_labels[fused_attention["status"]]
```

Add a pure-logic test that patches `_probe_fused_attention()` to each status and
asserts the corresponding row. This makes human and structured outputs share one
source of truth.

- [ ] **Step 5: Run Task 2 tests and commit**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_jsonl_output.py::TestRuntimeFingerprint src/tests/test_pure_logic_functions.py -k "RuntimeFingerprint or fused_attention" -q'
```

Expected: PASS.

Commit:

```bash
git add src/check_models.py src/tests/test_jsonl_output.py src/tests/test_pure_logic_functions.py
git commit -m "feat: report fused attention capability"
```

---

### Task 3: Typed Working-Set Context and Structured Outputs

**Files:**

- Modify: `src/check_models.py` at `HistoryModelResultRecord`, `JsonlMetricsRecord`, `MachineArtifactFacts`, `ReportRenderContext`, `_build_report_render_context()`, `_machine_artifact_facts()`, `_populate_history_machine_facts()`, `_machine_artifact_tsv_cells()`, `generate_tsv_report()`, and `finalize_execution()`
- Test: `src/tests/test_jsonl_output.py`
- Test: `src/tests/test_tsv_output.py`
- Test: `src/tests/test_pure_logic_functions.py`

**Interfaces:**

- Produces: `_peak_memory_working_set_pct(peak_memory_gb: float | None, recommended_working_set_bytes: int | None) -> float | None`
- Adds: `ReportRenderContext.recommended_working_set_bytes: int | None`
- Adds: `MachineArtifactFacts.peak_memory_working_set_pct: float | None`
- Adds: `JsonlMetricsRecord.peak_memory_working_set_pct: float`
- Adds: `HistoryModelResultRecord.peak_memory_working_set_pct: NotRequired[float | None]`
- Adds TSV column: `peak_memory_working_set_pct`

- [ ] **Step 1: Write failing unit tests for exact ratio semantics**

Add parameterized tests near other numeric helper tests:

```python
@pytest.mark.parametrize(
    ("peak_gb", "working_set_bytes", "expected"),
    [
        (18.2, 96 * 1024**3, pytest.approx(17.6567025483)),
        (120.0, 96 * 1024**3, pytest.approx(116.4153218269)),
        (0.0, 96 * 1024**3, 0.0),
        (None, 96 * 1024**3, None),
        (float("nan"), 96 * 1024**3, None),
        (1.0, None, None),
        (1.0, 0, None),
        (-1.0, 96 * 1024**3, None),
    ],
)
def test_peak_memory_working_set_pct(
    mod: types.ModuleType,
    peak_gb: float | None,
    working_set_bytes: int | None,
    expected: object,
) -> None:
    assert mod._peak_memory_working_set_pct(peak_gb, working_set_bytes) == expected
```

- [ ] **Step 2: Run the ratio tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py -k "peak_memory_working_set_pct" -q'
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement the ratio helper and typed context fields**

Add:

```python
def _peak_memory_working_set_pct(
    peak_memory_gb: float | None,
    recommended_working_set_bytes: int | None,
) -> float | None:
    """Return peak allocator bytes as a percentage of Metal's recommendation."""
    if peak_memory_gb is None or not math.isfinite(peak_memory_gb) or peak_memory_gb < 0:
        return None
    if (
        type(recommended_working_set_bytes) is not int
        or recommended_working_set_bytes <= 0
    ):
        return None
    return peak_memory_gb * DECIMAL_GB / recommended_working_set_bytes * 100.0
```

Add the fields listed in **Interfaces**. Extend `_build_report_render_context()`
with:

```python
recommended_working_set_bytes: int | None = None,
```

Resolve it without parsing `system_info`:

```python
resolved_working_set_bytes = (
    recommended_working_set_bytes
    if recommended_working_set_bytes is not None
    else _get_recommended_working_set_bytes()
)
```

Store `resolved_working_set_bytes` in `ReportRenderContext`.

- [ ] **Step 4: Derive one canonical per-model machine fact**

Change `_machine_artifact_facts()` to accept:

```python
recommended_working_set_bytes: int | None,
```

and set:

```python
peak_memory_working_set_pct=_peak_memory_working_set_pct(
    view.peak_memory_gb,
    recommended_working_set_bytes,
),
```

Pass `context.recommended_working_set_bytes` from the sole call in
`_build_report_render_context()`. This is the canonical value consumed by all
machine outputs.

- [ ] **Step 5: Write failing JSONL and history propagation tests**

Build a `ReportRenderContext` with `recommended_working_set_bytes=2_000_000_000`
and a result with `peak_memory=1.0`. Assert:

```python
assert result_record["metrics"]["peak_memory_working_set_pct"] == 50.0
assert history_record["model_results"]["model-x"]["peak_memory_working_set_pct"] == 50.0
```

Add companion tests with `recommended_working_set_bytes=None` asserting the field
is absent from both formats.

- [ ] **Step 6: Run the structured-output tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_jsonl_output.py -k "working_set" -q'
```

Expected: FAIL because the new machine fact is not emitted.

- [ ] **Step 7: Emit the canonical fact in JSONL and history**

In `_populate_history_machine_facts()` add:

```python
if facts.peak_memory_working_set_pct is not None:
    record["peak_memory_working_set_pct"] = facts.peak_memory_working_set_pct
```

In the JSONL result loop, after resolving `facts`, add:

```python
if facts is not None and facts.peak_memory_working_set_pct is not None:
    record["metrics"]["peak_memory_working_set_pct"] = (
        facts.peak_memory_working_set_pct
    )
```

Do not recalculate the percentage in either output path.

- [ ] **Step 8: Write the failing TSV column test**

In `src/tests/test_tsv_output.py`, generate a report under `tmp_path` using a
context with a 2 GB recommendation and 1 GB peak. Parse the header and row and
assert:

```python
assert header_cells.count("peak_memory_working_set_pct") == 1
column = header_cells.index("peak_memory_working_set_pct")
assert float(row_cells[column]) == 50.0
```

Add a missing-denominator case asserting the cell is empty while the column remains
present.

- [ ] **Step 9: Run the TSV test and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_tsv_output.py -k "working_set" -q'
```

Expected: FAIL because the column is absent.

- [ ] **Step 10: Add the TSV header and cell**

Append `peak_memory_working_set_pct` to `canonical_headers`. Append this exact cell
to `_machine_artifact_tsv_cells()` in the same position:

```python
optional_score(facts.peak_memory_working_set_pct),
```

Keep the header/cell ordering aligned and covered by the test.

- [ ] **Step 11: Thread the detected denominator through finalization**

In `finalize_execution()` read the cached raw value once:

```python
recommended_working_set_bytes = _get_recommended_working_set_bytes()
```

Pass it to `_build_report_render_context()`:

```python
recommended_working_set_bytes=recommended_working_set_bytes,
```

Because JSONL, history, and TSV consume the same `ReportRenderContext`, do not add
separate denominator parameters to those public report functions.

- [ ] **Step 12: Run all Task 3 tests and commit**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py -k "working_set or RuntimeFingerprint or HardwareFacts" -q'
```

Expected: PASS.

Commit:

```bash
git add src/check_models.py src/tests/test_pure_logic_functions.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py
git commit -m "feat: persist working set memory context"
```

---

### Task 4: Human Peak-Memory Context in Console, HTML, and Markdown

**Files:**

- Modify: `src/check_models.py` at `_build_prepared_table_data()`, `_build_report_render_context()`, `_build_recommendation_summary_block()`, `_build_gallery_output_cost_summary_section()`, `_log_perf_block()`, `_log_compact_metrics()`, and current-run comparison logging
- Test: `src/tests/test_report_generation.py`
- Test: `src/tests/test_metrics_modes.py`

**Interfaces:**

- Produces: `_format_peak_memory_context(peak_memory_gb: float | None, recommended_working_set_bytes: int | None) -> str`
- Consumes: `_peak_memory_working_set_pct()`, `format_field_value("peak_memory", value)`
- Human format: `<peak> (<pct:.1f>% of <working-set GiB:.1f> GB recommended working set)`

- [ ] **Step 1: Write failing formatter tests**

Add exact assertions:

```python
def test_format_peak_memory_context_includes_recommended_working_set(
    mod: types.ModuleType,
) -> None:
    assert mod._format_peak_memory_context(18.2, 96 * 1024**3) == (
        "18 GB (17.7% of 96.0 GB recommended working set)"
    )

def test_format_peak_memory_context_preserves_bare_peak_without_denominator(
    mod: types.ModuleType,
) -> None:
    assert mod._format_peak_memory_context(18.2, None) == "18"
```

Also assert missing peak returns an empty string and a value above 100% is rendered
without clamping.

- [ ] **Step 2: Run the formatter tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "format_peak_memory_context" -q'
```

Expected: FAIL because the formatter does not exist.

- [ ] **Step 3: Implement the shared human formatter**

Add:

```python
def _format_peak_memory_context(
    peak_memory_gb: float | None,
    recommended_working_set_bytes: int | None,
) -> str:
    """Format peak memory with optional Metal working-set context."""
    if peak_memory_gb is None:
        return ""
    peak = format_field_value("peak_memory", peak_memory_gb)
    if not peak:
        return ""
    percentage = _peak_memory_working_set_pct(
        peak_memory_gb,
        recommended_working_set_bytes,
    )
    if percentage is None or recommended_working_set_bytes is None:
        return peak
    working_set_gb = recommended_working_set_bytes / (1024**3)
    return (
        f"{peak} GB ({percentage:.1f}% of {working_set_gb:.1f} GB "
        "recommended working set)"
    )
```

Do not change `format_field_value()`; the contextual formatter is intentionally
run-aware while the existing formatter remains a pure field formatter.

- [ ] **Step 4: Write failing HTML and Markdown report tests**

In `src/tests/test_report_generation.py`, build a cached context with a 2 GB raw
denominator and a 1 GB peak, then generate HTML and Markdown under `tmp_path`.
Assert both contain:

```text
1 GB (50.0% of 1.9 GB recommended working set)
```

Use the actual current `format_field_value()` result in the fixture if it renders
`1.0 GB`; the production expectation must be identical in both formats. Add a
missing-denominator case asserting the current bare value remains and the phrase
`recommended working set` is absent from the model row.

- [ ] **Step 5: Run the human report tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py -k "working_set and (html or markdown or peak)" -q'
```

Expected: FAIL because report model rows still contain bare peak memory.

- [ ] **Step 6: Use the formatter in cached report tables and summary blocks**

Extend `_format_table_field_value()` and `_build_prepared_table_data()` with the
keyword-only parameter `recommended_working_set_bytes: int | None = None`. In
`_format_table_field_value()`, handle `peak_memory` before generic formatting:

```python
if field_name == "peak_memory":
    return _format_peak_memory_context(
        _generation_float_metric(res.generation, "peak_memory"),
        recommended_working_set_bytes,
    )
```

Pass the parameter from `_build_prepared_table_data()` into every
`_format_table_field_value()` call.

In `_build_report_render_context()`, resolve the raw denominator before building
`table_data`, then pass it into `_build_prepared_table_data()`. Replace bare peak
formatting in these report-facing helpers:

```python
_build_recommendation_summary_block()
_build_gallery_output_cost_summary_section()
```

with:

```python
_format_peak_memory_context(
    view.peak_memory_gb,
    report_context.recommended_working_set_bytes,
) or "-"
```

Leave sorting and numeric scoring on `view.peak_memory_gb` unchanged.

- [ ] **Step 7: Write failing console tests for compact and detailed modes**

Patch `_get_recommended_working_set_bytes` to `2_000_000_000`, run the existing
compact/detailed metric logger fixtures with `peak_memory=1.0`, and assert captured
messages contain `50.0%` and `recommended working set`. Add a `None` patch case
proving the current bare peak output is retained.

- [ ] **Step 8: Run the console tests and verify failure**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_metrics_modes.py -k "working_set" -q'
```

Expected: FAIL because console paths still use `format_field_value()` directly.

- [ ] **Step 9: Apply the formatter to console memory surfaces**

Use one local cached denominator per logging function:

```python
recommended_working_set_bytes = _get_recommended_working_set_bytes()
```

Apply `_format_peak_memory_context()` to:

- `_log_perf_block()` for the `Peak:` row only;
- `_log_compact_metrics()` for `Memory: ... peak`;
- the `PeakGB` cell in the current-run comparison table; and
- `_log_performance_highlights()` for average peak memory when a positive average
  exists.

Do not contextualize active/cache deltas, because the denominator comparison is
specified only for peak memory. When the formatter returns a bare value because
the denominator is unavailable, retain each console path's existing `GB` suffix;
when it returns the parenthesized context, do not append a second unit.

- [ ] **Step 10: Run Task 4 tests and commit**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_metrics_modes.py -k "working_set or peak_memory_context" -q'
```

Expected: PASS.

Commit:

```bash
git add src/check_models.py src/tests/test_report_generation.py src/tests/test_metrics_modes.py
git commit -m "feat: show working set context in reports"
```

---

### Task 5: Output Reachability, Documentation, and Full Verification

**Files:**

- Modify: `src/tests/test_report_generation.py`
- Modify: `src/tests/test_jsonl_output.py`
- Modify: `CHANGELOG.md`
- Conditionally modify: `src/README.md`

**Interfaces:**

- Verifies: system/environment facts reach console-independent HTML, Markdown,
  diagnostics, JSONL metadata, history, and reproduction environment mappings
- Documents: MLX hardware facts, fused-attention state, and working-set percentage

- [ ] **Step 1: Add end-to-end output reachability tests**

Use representative system information:

```python
system_info = {
    "GPU/Chip": "Apple M5 Max",
    "MLX Device": "Apple M5 Max",
    "GPU Architecture": "applegpu_g17s",
    "RAM": "128.0 GB",
    "Recommended Working Set": "96.0 GB",
    "Fused Attention": "Available",
}
```

Extend existing report-generation tests to assert these keys/values appear in HTML
and Markdown system sections. Extend JSONL/history metadata tests to assert the
same mapping is preserved exactly. Use the existing diagnostics/reproduction test
helpers to assert at least `GPU Architecture`, `Recommended Working Set`, and
`Fused Attention` reach their environment sections.

Add `MLX Device`, `GPU Architecture`, `Recommended Working Set`, and
`Fused Attention` to `_DIAGNOSTICS_SYSTEM_KEYS`. Add the same keys plus `RAM` to
`GALLERY_STAMP_SYSTEM_KEYS`, preserving the existing key order before appending
new values.

- [ ] **Step 2: Run reachability tests and fix only missing propagation**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_report_generation.py src/tests/test_jsonl_output.py -k "system_info or environment or hardware or working_set or fused_attention" -q'
```

Expected: PASS. Every builder must receive the existing `system_info` or
`report_context`; do not introduce a second hardware probe in an output renderer.

- [ ] **Step 3: Update the changelog and inspect README schema documentation**

Under `CHANGELOG.md` `[Unreleased]`, add a feature bullet stating that reports now:

```markdown
- Add cached `mx.device_info()` hardware facts with `sysctl`/MLX memory fallbacks,
  fused-attention capability state, and peak-memory percentages against Metal's
  recommended working set across human and structured outputs.
```

Search `src/README.md`:

```bash
rg -n "runtime_fingerprint|peak_memory_gb|TSV|JSONL|System Information|Hardware" src/README.md
```

If an enumerated schema or sample lists the affected fields, add
`fused_attention` and `peak_memory_working_set_pct` there. If it does not enumerate
them, leave `src/README.md` unchanged and record that decision in the final handoff.

- [ ] **Step 4: Run focused suites without a keyword filter**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && PYTHONPATH=src pytest src/tests/test_pure_logic_functions.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_report_generation.py src/tests/test_metrics_modes.py -q'
```

Expected: all tests PASS.

- [ ] **Step 5: Apply formatting and clear lint findings**

Run in order:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make format'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make -C src lint-fix'
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make lint'
```

Expected: all three commands exit 0. Inspect `git diff` after auto-format/fix and
retain only task-related changes.

- [ ] **Step 6: Run commit hygiene**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && bash src/tools/run_commit_hygiene.sh'
```

Expected: exit 0 with no tracked generated-output changes.

- [ ] **Step 7: Run the full quality gate**

Run:

```bash
/bin/zsh -lc 'source /Users/jrp/miniconda3/etc/profile.d/conda.sh && conda activate mlx-vlm && make quality'
```

Expected: format, Ruff, typing, dead-code, security, pytest, shell, and Markdown
checks all pass.

- [ ] **Step 8: Review the final diff and commit documentation/verification changes**

Run:

```bash
git status --short
git diff --check
git diff --stat
```

Confirm the two pre-existing untracked suppression-audit files remain untouched.
Commit only files changed by this feature:

```bash
git add src/check_models.py src/tests/test_pure_logic_functions.py src/tests/test_jsonl_output.py src/tests/test_tsv_output.py src/tests/test_report_generation.py src/tests/test_metrics_modes.py CHANGELOG.md
git add src/README.md
git commit -m "feat: contextualize MLX benchmark memory"
```

Omit the `src/README.md` staging command when Step 3 determined no README change was
needed. If all production/test changes were already committed in Tasks 1-4, this
final commit should contain only `CHANGELOG.md` and any required README update.
