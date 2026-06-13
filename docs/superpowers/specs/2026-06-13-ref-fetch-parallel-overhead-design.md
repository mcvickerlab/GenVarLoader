# Reference.fetch parallel-overhead + fetch-fusion — design

**Date:** 2026-06-13
**Issue:** [#221](https://github.com/mcvickerlab/GenVarLoader/issues/221) (follow-up to #214)
**Status:** approved design, pre-implementation

## Problem

After the #214 flat `variant-windows` work landed, the awkward round-trip is gone and
`Reference.fetch` (`python/genvarloader/_dataset/_reference.py:117`) is the dominant
remaining cost of the variant-windows decode (~90% of it in isolation, ~11% of the full
production encode).

The cost is **not** the byte copy, allocation, or `Ragged` wrap. #221's decomposition on
captured production arg arrays shows ~101% of `fetch` time is in the `_fetch_impl` numba
kernel call while it copies almost nothing — it is the **fixed fork-join overhead of
`@nb.njit(parallel=True)`**. On a host where numba reports 208 threads, each `fetch` call
costs a flat **~37 ms regardless of window count**:

```
N windows     parallel      serial      speedup(ser/par)
       6      37,805 µs      1.1 µs        34,058×
     100      38,570 µs      9.0 µs         4,295×
   5,000      38,954 µs    374.6 µs           104×
  50,000      36,396 µs  3,513.2 µs            10×

threads (N=6):   1 → 1.7 µs    8 → 6.4 µs    208 → 38,374 µs
```

Serial wins across the **entire** realistic window-count range. Two compounding issues:

1. **`parallel=True` fork-join** — flat ~37 ms/call, scales with the thread count numba
   forks, independent of work done.
2. **3 redundant fetches per decode** — `compute_alt_window` fetches `f5 = [start−L, start)`
   and `f3 = [end, end+L)`, which are slices of the ref-window read `[start−L, end+L)`
   already done by `compute_ref_window`.

**Not pursued:** read-count dedup. #221 confirms ~98% of `(region, sample)` pairs are empty
(dummy window, no fetch) and the sparse real windows repeat only ~2% in the somatic
workload. Dedup buys ~2% and helps neither the germline nor somatic regime materially.

**Thread-detection caveat:** `numba.get_num_threads()` reports the host's logical CPUs, not
the cgroup allocation (208 reported vs. 52 actually allocated on the RunPod VM). numba forks
the misdetected count, so the parallel branch is 4× oversubscribed.

## Design

### 1. Thread resolution + cap (once, at import)

A small module (e.g. `python/genvarloader/_threads.py`) resolves a cgroup-aware worker count
and caps numba's parallel regions to it:

```python
def _resolve_num_threads() -> int:
    env = os.environ.get("GVL_NUM_THREADS")
    if env:
        return int(env)
    try:
        real = len(os.sched_getaffinity(0))   # respects cgroup cpuset (Linux)
    except AttributeError:
        real = os.cpu_count() or 1            # non-Linux fallback
    return min(numba.get_num_threads(), real)
```

`numba.set_num_threads(_resolve_num_threads())` is called **once**, early, from
`python/genvarloader/__init__.py`, so every gvl numba kernel stops oversubscribing.

- Caps **down** only (`min(...)`); never raises above what numba would launch.
- `GVL_NUM_THREADS` is the escape hatch for CFS-quota-only hosts that `sched_getaffinity`
  can't see, and for power users.
- After capping, `numba.get_num_threads()` is the single source of truth (=52 here) used by
  the dispatcher.
- **Rejected:** per-call `set_num_threads` save/restore. gvl batches with threads; racing on
  this process-global setter is unsafe. A one-time import-side cap avoids the race.
- Document interaction with numba's own `NUMBA_NUM_THREADS` (we only cap below it).

### 2. Shared thread-aware dispatch helper

A fixed byte threshold can't be right across hosts (4 MB should go parallel on 8 threads,
serial on 52). The threshold is **per-thread work**:

```
go parallel  iff  out_offsets[-1] >= numba.get_num_threads() * _MIN_BYTES_PER_THREAD
```

`_MIN_BYTES_PER_THREAD` is a tunable module constant (default ~1 MB). With the capped count
(52), parallel needs ≥~52 MB total before it is considered — so variant-windows reads
(KB–MB) always go serial, and only genuinely large reference-mode reads parallelize, at the
right width with no oversubscription.

Mechanism (no loop-body duplication):

```python
@nb.njit(nogil=True, cache=True, inline="always")
def _fetch_row(i, ...):
    # the existing padded_slice per-row body

@nb.njit(parallel=True, nogil=True, cache=True)
def _fetch_impl_par(...):
    for i in nb.prange(n): _fetch_row(i, ...)

@nb.njit(nogil=True, cache=True)
def _fetch_impl_ser(...):
    for i in range(n): _fetch_row(i, ...)
```

A tiny Python dispatcher picks the wrapper by the per-thread rule. **Both** `_fetch_impl`
(variant-windows flanks) and `get_reference` (reference/haplotype output mode) route through
the same dispatcher — `get_reference` has the identical latent 37 ms pathology on high-core
hosts.

### 3. Fetch fusion in `_flat_flanks.py` (3 → 1)

Read the ref window once and derive flanks by slicing instead of re-fetching. `ends =
starts − min(ilen, 0) + 1` is unchanged; `rw = fetch(v_contigs, starts − L, ends + L)`.
With fixed `L` and ragged row offsets `ro = rw.offsets`:

```
f5 = rw.data[ro[:-1, None] + arange(L)]      # (n, L) — each row's first L bytes
f3 = rw.data[ro[1:,  None] - L + arange(L)]  # (n, L) — each row's last L bytes
```

A vectorized numpy gather, no kernel.

**Byte-identical** to the current separate fetches: `padded_slice` pads OOB deterministically
by coordinate, so the first/last `L` bytes of the contiguous read equal the standalone flank
reads (including left/right contig-boundary padding). Rows are always `ref_len + 2L ≥ 2L + 1`
long (`ref_len = ends − starts ≥ 1`), so `f5`/`f3` never overlap.

Refactor the builders to take pre-fetched data instead of fetching:

- `compute_ref_window(rw, lut, row_offsets)` — tokens straight from `rw`.
- `compute_alt_window(f5, f3, alt_data, alt_seq_off, flank_len, lut, row_offsets)` —
  assemble `flank5·alt·flank3`.
- `compute_flank_tokens(f5, f3, lut, row_offsets)` — ride-along `[flank5|flank3]` (same 2→0
  fetch fix on the plain `variants` output).

The caller in `python/genvarloader/_dataset/_flat_variants.py` computes `rw` **once** iff
`opt.ref == "window" or opt.alt == "window"` (or the ride-along flank path is active). This
covers all four ref/alt combinations, including `ref=allele, alt=window` (the single `rw`
read still supplies the flanks). Tokenization/assembly stay as-is — profiling shows they are
~0%.

### 4. Testing

- **Byte-identical regression:** all four ref/alt combos × variants at contig start/end
  (left/right padding) × indels with varied `ilen` × ploidy > 1, asserting fused output
  equals the current separate-fetch output.
- **Fetch-count spy:** assert `reference.fetch` is called exactly **once** per window decode
  (directly verifies the 3→1 fusion without flaky wall-time assertions).
- **Dispatcher decision:** the serial/parallel choice function in isolation (small → serial,
  large → parallel at a mocked thread count), plus byte-identical output from both wrappers.
- **Thread resolution:** `GVL_NUM_THREADS` override honored; `sched_getaffinity` path and the
  `os.cpu_count()` fallback; cap is `min` (never raises).

### Scope

**In:** `_fetch_impl`, `get_reference`, the three `_flat_flanks` builders, the
`_flat_variants` caller, the new `_threads` module + `__init__` cap.

**Out:** read-count dedup (~2%); a full fetch+LUT+assemble fused kernel (eliminates passes
profiling shows are ~0%); the public `Reference.fetch` signature (unchanged).

## Acceptance criteria (from #221)

1. `Reference.fetch` no longer dominates the variant-windows decode; per-call cost scales
   with bytes copied, not a fixed ~37 ms.
2. Byte-identical output to the current `fetch` across a representative index set.
3. The trans production encode's `_reference.fetch` share drops from ~11% toward ~0.
