# double_buffered variant-windows slot-fit (#315) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per project convention, implementer subagents run on **Sonnet or weaker**; escalate to a stronger model only for a second-pass fix where the implementer critically failed.

**Goal:** Make `Dataset.to_dataloader(mode="double_buffered")` over flat `variant-windows` output fit its shared-memory slots on the real (Hartwig) corpus, and guarantee the slot-fit invariant by test so this class of estimator↔serializer drift can never silently recur.

**Architecture:** The double-buffer slot is sized from a per-instance byte *estimate* (`Dataset._output_bytes_per_instance`) that must upper-bound what the *serializer* (`write_chunk`) actually writes. The estimate is byte-exact for all synthetic records but diverges on a real-corpus/backend record class (a variant-count/selection mismatch, not a length-formula error). We (a) pin the divergence, (b) restore the estimate's upper-bound invariant for it, and harden the mechanism with (c) a schema-derived slot-overhead bound replacing the magic 4096, (d) a producer write-time guard that fails loud instead of cryptically, and (e) a property test asserting the invariant across a record-type × backend matrix.

**Tech Stack:** Python (numpy, seqpro.rag), Rust (PyO3 window kernel — read-only here), pixi (`-e dev`), pytest, maturin.

## Global Constraints

- Branch `fix/315-double-buffered-vw-slot` → **`main`** (released `Dataset.to_dataloader` double-buffer path; **not** StreamingDataset work — see PR #318). Do **not** target `streaming`.
- Environment: all commands via `pixi run -e dev <...>`. Platform linux-64.
- **Rebuild Rust before any pytest that imports the extension:** `pixi run -e dev maturin develop --release`. (No `src/` changes are planned here, so a one-time build at start suffices; `cargo test` compiles from source regardless.)
- Ruff: E501 (line length) ignored. New/edited docstrings in `python/genvarloader/` must be Google-style. Lint BOTH `python/ tests/`.
- Before pushing shared-code changes, run the **full tree**: `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`). `tests/data` fixtures require `pixi run -e dev gen` once.
- Conventional-commit messages (commitizen). End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Public-API surface is unchanged; no `api.md`/SKILL.md update expected (confirm in Task 6).

## Key facts the implementer needs (verified)

- Slot capacity today: `_double_buffered_loader.py:275` → `capacity = HEADER_RESERVED + peak + 4096`, `HEADER_RESERVED = 4096` (`_shm_layout.py:62`). `peak = ChunkPlanner.peak_chunk_bytes`, itself derived from `bpi = ds._output_bytes_per_instance(include_offsets=True)`.
- `write_chunk(buf, arrays, n_instances)` (`_shm_layout.py:400`) returns the final `cursor` (header+payload bytes). Real payload = `cursor - HEADER_RESERVED`. It raises the observed `ValueError: buffer is smaller than requested size` from an inner `np.frombuffer(buf, dtype, count, offset)` when `offset + count*itemsize > len(buf)`.
- Materialize a chunk exactly as the producer does: `chunk = ds[r_idx, s_idx]; arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]`.
- Per-chunk un-estimated overhead (measured): `8 × n_offset_arrays` (missing `+1` terminator per serialized offset array) + `≤7 B`/serialized array (`_align` to 8). For variant-windows a chunk serializes: per scalar `.field` → 1 offset array + 1 data array; per window slot (exactly 2: one ref-derived, one alt-derived) → 2 offset arrays (outer+inner) + 1 data array; per active track → 1 offset array + 1 data array.
- The estimate's variant-windows branch lives in `_dataset/_impl.py`, `seq_kind == "variant-windows"` (≈ lines 1564–1739); offsets accounting at ≈ 1677–1691.
- Existing byte-accounting tests: `tests/unit/test_output_bytes_variant_windows.py`, `tests/unit/dataset/test_output_bytes_per_instance.py`, `tests/unit/test_shm_layout.py`, `tests/unit/test_double_buffered_loader.py`. Backend fixtures: `tests/data/phased_dataset.{vcf,pgen,svar}.gvl` (rebuild via `pixi run -e dev gen`; open with `gvl.Dataset.open(path, reference=...)`).
- Investigation scripts (reference implementations for the comparisons): `/carter/users/dlaub/.claude/jobs/46f6d152/tmp/diag_315_realcorpus.py`, `diag_L128.py`.

## Task dependency / parallelism

- **Task 1 (Phase 0)** is a human-run investigation gate. It blocks only Task 5.
- **Task 2** and **Task 3** are independent → can run in parallel.
- **Task 4** depends on Task 2 (uses its `slot_overhead` helper).
- **Task 5** depends on Task 1 (pinned cause) + Task 4 (property-test harness).
- **Task 6** (docs/verify) is last.
- Suggested parallel dispatch: start {Task 1 (you), Task 2, Task 3} together; then Task 4; then Task 5; then Task 6.

---

### Task 1 (GATE — human-run): Phase 0, pin the divergence on the real corpus

**Files:**
- Create: `docs/superpowers/specs/2026-07-21-phase0-findings.md` (the recorded result)

**Interfaces:**
- Produces: the **pinned divergence** — which quantity the estimate mis-counts vs the writer, on which backend — and a **minimal reproducing fixture recipe** (dataset backend + a variant record pattern) that Task 5's property-test case will encode.

- [ ] **Step 1: Run the instrumentation against the real dataset**

```bash
pixi run -e dev python /carter/users/dlaub/.claude/jobs/46f6d152/tmp/diag_315_realcorpus.py \
    --dataset /path/to/hartwig.gvl --reference /path/to/ref.fa --flank 128
```

- [ ] **Step 2: Record the offending record class**

Capture the "Offending instances" dump (ilen / ALT strings / backend) into `docs/superpowers/specs/2026-07-21-phase0-findings.md`: the exact under-counted quantity (e.g. "estimate's `n_variants()` counts N variants for (r,s) but the writer's `to_packed()` emits N+k because <pattern>"; or "`_allele_bytes_sum` under-counts alt bytes for <backend/record>"), and the backend (VCF/PGEN/SVAR2).

- [ ] **Step 3: Derive a minimal reproducing fixture recipe**

Write, in the same findings file, the smallest synthetic dataset (backend + variant pattern) that reproduces `real write_chunk payload > sum(estimate)` for a single instance. If synthetic data cannot reproduce it, record instead the exact failing `(r_idx, s_idx)` indices plus the dataset/backend so Task 5 can use a real fixture. **This recipe is the required input to Task 5.**

- [ ] **Step 4: Commit the findings**

```bash
git add docs/superpowers/specs/2026-07-21-phase0-findings.md
git commit -m "docs(spec): #315 Phase 0 findings — pinned slot under-count on real corpus"
```

---

### Task 2: Layer 2c — schema-derived slot-overhead bound (drop the magic 4096)

**Files:**
- Create: `python/genvarloader/_slot_overhead.py`
- Modify: `python/genvarloader/_double_buffered_loader.py:275`
- Test: `tests/unit/test_slot_overhead.py`

**Interfaces:**
- Produces: `slot_overhead_bytes(dataset) -> int` — a per-chunk upper bound on serialization overhead not captured by `_output_bytes_per_instance` (offset-array terminators + `_align` padding), with a `4096` floor. Consumed by Task 4's property test and by the capacity calc.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_slot_overhead.py
"""slot_overhead_bytes must upper-bound the real per-chunk serialization overhead
(offset-array +1 terminators + _align padding) that _output_bytes_per_instance omits."""
import numpy as np
import seqpro as sp
import genvarloader as gvl
from genvarloader._slot_overhead import slot_overhead_bytes
from genvarloader._shm_layout import write_chunk, HEADER_RESERVED


def _vw(ds, L=8):
    opt = gvl.VarWindowOpt(flank_length=L, token_alphabet=sp.alphabets.DNA,
                           unknown_token=len(sp.alphabets.DNA), ref="window", alt="allele")
    return (ds.with_tracks(False).with_output_format("flat")
              .with_seqs("variant-windows", opt)
              .with_settings(unphased_union=True, jitter=0))


def test_overhead_covers_real_minus_estimate():
    ds = _vw(gvl.get_dummy_dataset())
    R, S = ds.shape[:2]
    rr, ss = np.meshgrid(np.arange(R), np.arange(S), indexing="ij")
    r, s = rr.reshape(-1), ss.reshape(-1)
    chunk = ds[r, s]
    arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]
    buf = memoryview(bytearray(64 * 1024 * 1024))
    real = write_chunk(buf, arrays, n_instances=len(r)) - HEADER_RESERVED
    est = int(np.asarray(ds._output_bytes_per_instance(r, s, include_offsets=True)).sum())
    overhead = slot_overhead_bytes(ds)
    assert real - est <= overhead, f"real-est={real-est} exceeds overhead={overhead}"
    assert overhead >= 4096  # floor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_slot_overhead.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'genvarloader._slot_overhead'`.

- [ ] **Step 3: Write minimal implementation**

```python
# python/genvarloader/_slot_overhead.py
"""Per-chunk serialization overhead not captured by _output_bytes_per_instance.

The byte estimate charges exact per-instance payload + per-instance offset entries,
but the serializer (_shm_layout.write_chunk) additionally writes, per serialized
offset array, one +1 terminator entry (8 bytes), and _align-pads (<=8 bytes) before
every serialized array. Those are per-chunk constants (independent of instance count)
that must be covered by the slot's fixed slack. This module derives a true upper
bound on them from the schema, replacing the historical magic 4096.
"""
from __future__ import annotations

_OFF = 8  # int64 offset entry / terminator
_ALIGN = 8  # max _align(8) padding per serialized array


def _array_counts(dataset) -> tuple[int, int]:
    """(n_offset_arrays, n_serialized_arrays) the serializer emits for one chunk
    under the dataset's active output schema. Upper bound; over-counting is safe."""
    seq = dataset.sequence_type
    n_off = 0
    n_arr = 0
    seqs = getattr(dataset, "_seqs", None)
    var_fields = list(getattr(seqs, "var_fields", []) or [])
    # scalar .fields: always-emitted "start" plus non-allele var_fields.
    scalars = {f for f in var_fields if f not in ("alt", "ref")}
    scalars.add("start")
    if seq == "variant-windows":
        n_scalar = len(scalars)
        n_window_slots = 2  # exactly one ref-derived + one alt-derived slot
        n_off += n_scalar * 1 + n_window_slots * 2  # scalars: outer; windows: outer+inner
        n_arr += n_scalar * 2 + n_window_slots * 3  # +1 data array each
    elif seq == "variants":
        n_scalar = len(scalars)
        n_allele = sum(1 for f in var_fields if f in ("alt", "ref"))
        n_off += n_scalar * 1 + n_allele * 2
        n_arr += n_scalar * 2 + n_allele * 3
    else:
        # reference / haplotypes / annotated / none: few arrays; the 4096 floor
        # dominates. Charge a generous constant so the floor is never exceeded.
        n_off += 8
        n_arr += 8
    # active tracks: each is a single-level ragged (1 offset + 1 data array).
    n_tracks = len(getattr(dataset, "active_tracks", None) or {})
    n_off += n_tracks
    n_arr += n_tracks * 2
    return n_off, n_arr


def slot_overhead_bytes(dataset) -> int:
    """Upper bound on per-chunk overhead beyond the per-instance byte estimate.

    Args:
        dataset: The gvl Dataset whose active output schema fixes the field/array
            structure serialized per chunk.

    Returns:
        Bytes to add to peak_chunk_bytes when sizing a double_buffered shm slot,
        floored at 4096 (covers the header-adjacent slack the format has always
        assumed).
    """
    n_off, n_arr = _array_counts(dataset)
    return max(4096, _OFF * n_off + _ALIGN * n_arr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_slot_overhead.py -v`
Expected: PASS.

- [ ] **Step 5: Wire it into the slot capacity**

In `python/genvarloader/_double_buffered_loader.py`, add near the top-level imports:

```python
from ._slot_overhead import slot_overhead_bytes
```

Replace line 275:

```python
        capacity = HEADER_RESERVED + peak + 4096
```

with:

```python
        capacity = HEADER_RESERVED + peak + slot_overhead_bytes(dataset)
```

- [ ] **Step 6: Run the double-buffered loader tests**

Run: `pixi run -e dev pytest tests/unit/test_double_buffered_loader.py tests/unit/test_slot_overhead.py -v`
Expected: PASS (no regression; slots are now sized with a derived, never-smaller overhead).

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_slot_overhead.py python/genvarloader/_double_buffered_loader.py tests/unit/test_slot_overhead.py
git commit -m "fix(dataloader): derive double_buffered slot overhead from schema, drop magic 4096

Replace the fixed 4096 slot slack with slot_overhead_bytes(dataset), an
upper bound on the per-chunk serialization overhead (+1 offset-array
terminators and _align padding) that _output_bytes_per_instance omits.
Floored at 4096 so it is never smaller than before.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Layer 2a — producer write-time slot-fit guard (fail loud, not cryptic)

**Files:**
- Modify: `python/genvarloader/_shm_layout.py` (`write_chunk`, ≈ lines 400–454)
- Test: `tests/unit/test_shm_layout.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `write_chunk` raises a `SlotOverflowError` (new, subclass of `ValueError`) with an actionable message when the serialized chunk exceeds `len(buf)`, instead of the raw inner `np.frombuffer` "buffer is smaller than requested size".

**Design note:** rather than duplicate every `_write_*` helper's cursor math in a parallel "dry run" (which would create exactly the estimator↔serializer drift this effort exists to kill), wrap the existing write loop and translate the numpy overflow into an actionable error. On overflow the chunk is abandoned (the producer never marks the slot `ready`), so a partial write into the scratch slot is harmless.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_shm_layout.py
import numpy as np
import pytest
from genvarloader._shm_layout import write_chunk, SlotOverflowError, HEADER_RESERVED


def test_write_chunk_raises_actionable_on_overflow():
    # A dense array whose payload exceeds a deliberately tiny slot.
    arr = np.arange(4096, dtype=np.int64)  # 32 KiB payload
    tiny = memoryview(bytearray(HEADER_RESERVED + 1024))  # far too small
    with pytest.raises(SlotOverflowError) as ei:
        write_chunk(tiny, [arr], n_instances=1)
    msg = str(ei.value)
    assert "slot" in msg.lower()
    assert "buffer_bytes" in msg or "batch_size" in msg  # actionable remedy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_shm_layout.py::test_write_chunk_raises_actionable_on_overflow -v`
Expected: FAIL with `ImportError: cannot import name 'SlotOverflowError'`.

- [ ] **Step 3: Add the exception class**

In `python/genvarloader/_shm_layout.py`, near the top (after imports, before `HEADER_RESERVED`):

```python
class SlotOverflowError(ValueError):
    """A serialized chunk does not fit its shared-memory slot.

    Raised by write_chunk in place of the raw numpy "buffer is smaller than
    requested size". Indicates the per-instance byte estimate under-sized the
    slot for this chunk: lower batch_size or raise buffer_bytes.
    """
```

- [ ] **Step 4: Translate the overflow in `write_chunk`**

Wrap the existing per-array write loop in `write_chunk` (the `for a in arrays: ... descriptors.append(desc)` block, ≈ lines 424–439) so a buffer-overflow `ValueError` from any inner `np.frombuffer` becomes a `SlotOverflowError`:

```python
    for a in arrays:
        try:
            if isinstance(a, _FlatVariantWindows):
                desc, cursor = _write_flat_variant_windows(buf, a, cursor)
            elif isinstance(a, _FlatVariants):
                desc, cursor = _write_flat_variants(buf, a, cursor)
            elif isinstance(a, RaggedVariants):
                desc, cursor = _write_rag_variants(buf, a, cursor)
            elif isinstance(a, (RaggedAnnotatedHaps, _FlatAnnotatedHaps)):
                desc, cursor = _write_rag_annotated(buf, a, cursor)
            elif isinstance(a, (Ragged, _Flat)):
                desc, cursor = _write_ragged(buf, a, cursor)
            elif isinstance(a, np.ndarray):
                desc, cursor = _write_dense(buf, a, cursor)
            else:
                raise TypeError(f"write_chunk: unsupported array type {type(a)}")
        except ValueError as e:
            # numpy raises "buffer is smaller than requested size" when a write
            # offset+extent exceeds the slot. Re-raise with actionable context;
            # let unrelated ValueErrors (e.g. the TypeError branch) propagate.
            if "buffer is smaller than requested size" not in str(e):
                raise
            raise SlotOverflowError(
                f"serialized chunk (n_instances={n_instances}) does not fit the "
                f"shared-memory slot of {len(buf)} bytes. The per-instance byte "
                f"estimate under-sized this slot. Lower batch_size or raise "
                f"buffer_bytes."
            ) from e
        descriptors.append(desc)
```

- [ ] **Step 5: Run the guard test**

Run: `pixi run -e dev pytest tests/unit/test_shm_layout.py::test_write_chunk_raises_actionable_on_overflow -v`
Expected: PASS.

- [ ] **Step 6: Run the full shm-layout + double-buffered suites (no regression)**

Run: `pixi run -e dev pytest tests/unit/test_shm_layout.py tests/unit/test_double_buffered_loader.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_shm_layout.py tests/unit/test_shm_layout.py
git commit -m "fix(dataloader): raise actionable SlotOverflowError on oversized chunk write

Translate numpy's cryptic 'buffer is smaller than requested size' (raised
when a chunk's serialized bytes exceed its shm slot) into a SlotOverflowError
naming n_instances, slot size, and the remedy (lower batch_size / raise
buffer_bytes). Wraps the existing write loop -- no parallel cursor math to
drift from the real writer.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Layer 2b — upper-bound property test (record-type × backend matrix)

**Files:**
- Create: `tests/unit/test_slot_fit_property.py`

**Interfaces:**
- Consumes: `slot_overhead_bytes` (Task 2), `write_chunk`/`HEADER_RESERVED` (core), `_chunk_serialized_nbytes` (Task 3).
- Produces: the standing invariant `sum(estimate) + slot_overhead_bytes(ds) >= real write_chunk payload` for every chunk in the matrix. This is the harness Task 5 extends with the Phase-0 case.

- [ ] **Step 1: Write the property test (passes today; proves no regression + guards the invariant)**

```python
# tests/unit/test_slot_fit_property.py
"""The double_buffered slot-fit invariant: the per-instance byte estimate plus the
schema-derived slot overhead must upper-bound the real serialized payload for every
chunk. This is the invariant #315 violated; it must hold across record types and
storage backends."""
import numpy as np
import pytest
import seqpro as sp
import genvarloader as gvl
from genvarloader._shm_layout import write_chunk, HEADER_RESERVED
from genvarloader._slot_overhead import slot_overhead_bytes


def _views(ds):
    DNA = sp.alphabets.DNA
    for ref, alt in [("window", "window"), ("window", "allele"), ("allele", "allele")]:
        if ref == "allele" and getattr(ds._seqs.variants, "ref", None) is None:
            continue
        for uu in (True, False):
            for L in (8, 128):
                opt = gvl.VarWindowOpt(flank_length=L, token_alphabet=DNA,
                                       unknown_token=len(DNA), ref=ref, alt=alt)
                yield (ds.with_tracks(False).with_output_format("flat")
                         .with_seqs("variant-windows", opt)
                         .with_settings(unphased_union=uu, jitter=0))


def _assert_upper_bound(view):
    R, S = view.shape[:2]
    rr, ss = np.meshgrid(np.arange(R), np.arange(S), indexing="ij")
    r, s = rr.reshape(-1), ss.reshape(-1)
    chunk = view[r, s]
    arrays = list(chunk) if isinstance(chunk, tuple) else [chunk]
    buf = memoryview(bytearray(64 * 1024 * 1024))
    real = write_chunk(buf, arrays, n_instances=len(r)) - HEADER_RESERVED
    est = int(np.asarray(view._output_bytes_per_instance(r, s, include_offsets=True)).sum())
    assert est + slot_overhead_bytes(view) >= real, (
        f"slot under-sized: est={est} overhead={slot_overhead_bytes(view)} real={real}")


def test_slot_fit_dummy_backend():
    for view in _views(gvl.get_dummy_dataset()):
        _assert_upper_bound(view)
```

- [ ] **Step 2: Run it**

Run: `pixi run -e dev pytest tests/unit/test_slot_fit_property.py -v`
Expected: PASS (the estimate is byte-exact for synthetic records; the overhead covers the per-chunk constant).

- [ ] **Step 3: Add file-backed backend coverage (VCF, PGEN, SVAR2)**

First ensure fixtures exist: `pixi run -e dev gen`. Then add, using the same conftest helpers the other `tests/dataset` tests use to open `tests/data/phased_dataset.{vcf,pgen,svar}.gvl` with a reference (grep `tests/dataset/conftest.py` for the reference-open pattern):

```python
@pytest.mark.parametrize("backend", ["vcf", "pgen", "svar"])
def test_slot_fit_file_backends(backend, request):
    # Open tests/data/phased_dataset.<backend>.gvl with the shared reference
    # fixture (mirror tests/dataset/conftest.py). Skip if the fixture/reference
    # is unavailable in this environment.
    ds = _open_phased_backend(request, backend)  # helper mirroring conftest
    for view in _views(ds):
        _assert_upper_bound(view)
```

Implement `_open_phased_backend` in the test module by copying the open pattern from `tests/dataset/conftest.py` (reference path + `gvl.Dataset.open`). If a backend fixture cannot be built in the environment, `pytest.skip(...)` with an explicit message (do not silently pass).

- [ ] **Step 4: Run the full property suite**

Run: `pixi run -e dev pytest tests/unit/test_slot_fit_property.py -v`
Expected: PASS for every available backend; explicit SKIP (never silent) for any unavailable one.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_slot_fit_property.py
git commit -m "test(dataloader): property test — estimate+slot_overhead upper-bounds real payload

Assert the double_buffered slot-fit invariant across ref/alt x unphased_union
x flank_length x {VCF,PGEN,SVAR2} backends. This is the invariant #315 broke;
it now guards against estimator<->serializer drift in CI.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5 (GATE on Task 1): Layer 1 — restore the estimate's upper-bound for the pinned case

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (`_output_bytes_per_instance`, `seq_kind == "variant-windows"` branch ≈ 1564–1739)
- Test: `tests/unit/test_slot_fit_property.py` (extend with the Phase-0 case)

**Interfaces:**
- Consumes: Task 1's pinned divergence + reproducing fixture recipe; Task 4's `_assert_upper_bound`.
- Produces: the estimate is a true upper bound for the pinned record class → the reported 40×7089 config no longer overflows.

- [ ] **Step 1: Encode the Phase-0 case as a failing test**

Using the fixture recipe from `docs/superpowers/specs/2026-07-21-phase0-findings.md`, add a parametrization/case to `tests/unit/test_slot_fit_property.py` that builds (or opens) the offending dataset+view and calls `_assert_upper_bound`. If the recipe is a real dataset + specific `(r_idx, s_idx)`, encode those indices directly.

- [ ] **Step 2: Run it to confirm it fails (reproduces #315 at the estimate level)**

Run: `pixi run -e dev pytest tests/unit/test_slot_fit_property.py -k phase0 -v`
Expected: FAIL with `slot under-sized: est=... real=...` (real exceeds est+overhead) — the estimate-level reproduction of the overflow.

- [ ] **Step 3: Fix the estimate to upper-bound the pinned quantity**

Per Task 1's finding, correct the `variant-windows` branch of `_output_bytes_per_instance` so the mis-counted quantity becomes a provable upper bound. The fix takes one of these shapes (Task 1 selects which):
- **Count/selection mismatch:** align the estimate's variant set with the writer's `genotypes.to_packed()` `v_idxs` for the failing backend (e.g. count the same records the writer emits, rather than `n_variants()` which diverges under the pinned pattern).
- **Backend allele-byte mismatch:** replace/augment the `_allele_bytes_sum` term with the backend-correct stored-ALT byte count (matching what the Rust `gather_alleles` reads).
Keep the change conservative (upper bound, never under-count) and localized to the variant-windows branch; add a short comment citing #315 and the pinned record class. Do **not** touch the writer.

- [ ] **Step 4: Run the Phase-0 case + the whole property suite**

Run: `pixi run -e dev pytest tests/unit/test_slot_fit_property.py -v`
Expected: PASS for the Phase-0 case and all pre-existing cases (fix must not break the byte-exact synthetic cases — it may only ever *raise* the estimate).

- [ ] **Step 5: End-to-end regression at the reported shape**

Add a smoke test that builds the offending view at `flank_length=128` and iterates one batch under `mode="double_buffered"` with a `buffer_bytes` small enough to force a multi-instance chunk, asserting no `SlotOverflowError` / overflow. (If only a real dataset reproduces it, mark `@pytest.mark.slow` and gate on the fixture being present.)

Run: `pixi run -e dev pytest tests/unit/test_slot_fit_property.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py tests/unit/test_slot_fit_property.py
git commit -m "fix(dataloader): variant-windows byte estimate upper-bounds writer for <pinned record class> (#315)

<one line naming the pinned divergence and backend from Phase 0>. Restores
the double_buffered slot-fit invariant so the reported 40x7089 config no
longer overflows; buffered/manual were already correct.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Verify, docs audit, finalize

**Files:**
- Modify (only if needed): `docs/source/dataset.md` / `docs/source/faq.md`
- Modify: `docs/superpowers/specs/2026-07-21-double-buffered-vw-slot-fit-design.md` (flip Status to Implemented; record Phase-0 result)

- [ ] **Step 1: Rebuild + full tree + cargo**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q
pixi run -e dev cargo test
```
Expected: green (record counts).

- [ ] **Step 2: Lint + typecheck**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
```
Expected: clean.

- [ ] **Step 3: Public-API / docs gate**

```bash
python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`. Confirm no user-facing default/flag changed; update `dataset.md`/`faq.md` only if `double_buffered`/`buffer_bytes` guidance changed. `SlotOverflowError` is internal (not in `__all__`) unless you deliberately export it — do not export without adding an `api.md` entry.

- [ ] **Step 4: Update the design spec status + push**

Flip the spec Status to `Implemented`, summarize the Phase-0 result, commit, and push the branch. The draft PR #320 updates automatically.

```bash
git add docs/superpowers/specs/2026-07-21-double-buffered-vw-slot-fit-design.md
git commit -m "docs(spec): mark #315 slot-fit design implemented; record Phase 0 result"
git push
```

- [ ] **Step 5: Comment on #315**

Post the corrected repro params (flank_length=128 / 2 GiB) + the confirmed root cause + the fix summary to issue #315, and open the separate **Bug A** issue (`realign_tracks` not propagated to the producer) cross-linked to #315.

---

## Out of scope (tracked separately)

- **Bug A — `realign_tracks` not propagated to the producer** (`_build_producer_schema` omits `realign_tracks`; producer defaults it to `True` and raises for variant-windows + tracks). Its own issue + PR: add `schema["realign_tracks"] = ds.realign_tracks` (schema dict at `_double_buffered_loader.py:85-92`) and thread it through `_producer._apply_schema`'s `settings_kwargs`, plus a regression test that `double_buffered` + variant-windows + an active track iterates.
- **Producer grow-or-split auto-recovery** — the escalation path if Phase 0 shows the estimate cannot be made a clean upper bound. Re-approve scope with the user before implementing.
