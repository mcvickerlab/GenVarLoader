# Round-3 Profiling Baseline

Captured 2026-06-25 on the Carter node.  
Build: `maturin develop --release`, corpus `tests/benchmarks/data/chr22_geuv.gvl`,
`with_len(16384)`, `BATCH=32`, `NUMBA_NUM_THREADS=1`.

---

## Starting Rust ÷ Numba Ratios

| Path | Metric | Rust | Numba | Rust ÷ Numba |
|------|--------|------|-------|--------------|
| tracks-only | pedantic min (ms/batch) | 1.091 | 1.121 | **0.97** |
| haplotypes | pedantic min (ms/batch) | 2.348 | 3.372 | **0.70** |
| variants | wall avg (ms/batch) | 2.293 | 2.859 | **0.80** |
| variant-windows | wall avg (ms/batch) | 2.117 | 3.773 | **0.56** |

All four paths are already faster in Rust than Numba, so these are the baselines
to beat, not ceilings. Ratios < 1.0 mean Rust is faster.

---

## Consolidated Flat Self-Time Table

Measured with `perf record -F 999 --no-children` over 12 000 batches per path (Rust only).
Rows = Rust kernel symbols appearing in any path's top self-time.
Columns = self-time % in that path (blank = not observed).
**Aggregate = sum of self-time % across all paths** — the descending sort of this
column is the tuning target order for all later round-3 tasks.

| Symbol | tracks | haplotypes | variants | variant-windows | **Aggregate** |
|--------|:------:|:----------:|:--------:|:---------------:|:-------------:|
| `genvarloader::intervals::intervals_to_tracks` | 26.08 | 16.64 | 17.60 | — | **60.32** |
| `genvarloader::variants::windows::tokenize` | — | — | — | 28.14 | **28.14** |
| `genvarloader::tracks::shift_and_realign_tracks_sparse` | — | 13.03 | 12.70 | — | **25.73** |
| `genvarloader::variants::windows::slice_flanks` | — | — | — | 20.14 | **20.14** |
| `genvarloader::variants::windows::assemble_alt_window` | — | — | — | 13.26 | **13.26** |
| `genvarloader::reverse::rc_flat_rows_inplace` | — | 9.31 | — | — | **9.31** |
| `genvarloader::ffi::intervals_and_realign_track_fused` | — | 4.54 | 4.43 | — | **8.97** |
| `genvarloader::reconstruct::reconstruct_haplotypes_from_sparse` | — | 4.47 | — | — | **4.47** |
| `ndarray::dimension::do_slice` | — | 1.92 | — | 0.64 | **2.56** |
| `ndarray::impl_methods::<impl ndarray::ArrayRef<A,D>>::slice_mut` | — | 1.89 | — | 0.61 | **2.50** |
| `genvarloader::reference::get_reference::{{closure}}` | — | — | — | 1.51 | **1.51** |
| `genvarloader::genotypes::get_diffs_sparse` | — | 0.81 | 0.44 | — | **1.25** |
| `genvarloader::variants::gather_alleles` | — | — | 0.54 | 0.55 | **1.09** |
| `genvarloader::variants::windows::fetch_windows` | — | — | — | 0.22 | **0.22** |
| `genvarloader::variants::windows::gather_starts_ilens` | — | — | — | 0.17 | **0.17** |
| `genvarloader::reference::get_reference` | — | — | — | 0.13 | **0.13** |
| `genvarloader::variants::gather_rows_i32` | — | — | — | 0.11 | **0.11** |

### Notes

- `__memset_avx2_unaligned_erms` (libc) appears at 12.89% in tracks and 3.89% in
  haplotypes as the second-largest entry — it is called from within
  `intervals_to_tracks` (zero-filling output buffers) and thus captured under the Rust
  symbol in any inlined build; it is not an independent target.
- `ndarray::dimension::do_slice` and `ndarray::impl_methods::slice_mut` are from the
  `ndarray` crate (not genvarloader-specific). They accumulate 2.56% and 2.50%
  aggregate respectively; addressable only by restructuring how outputs are sliced, not
  by rewriting a kernel.
- `genvarloader::ffi::intervals_and_realign_track_fused` (haplotypes 4.54%,
  variants 4.43%) is the combined FFI trampoline for intervals + track realignment;
  it likely contains overhead that belongs to either `intervals_to_tracks` or
  `shift_and_realign_tracks_sparse` when fused.

### Descending Target Order for Round-3 Tuning Tasks

1. `genvarloader::intervals::intervals_to_tracks` — Aggregate **60.32%** (shared: tracks + haps + variants)
2. `genvarloader::variants::windows::tokenize` — **28.14%** (variant-windows only)
3. `genvarloader::tracks::shift_and_realign_tracks_sparse` — **25.73%** (haps + variants)
4. `genvarloader::variants::windows::slice_flanks` — **20.14%** (variant-windows only)
5. `genvarloader::variants::windows::assemble_alt_window` — **13.26%** (variant-windows only)
6. `genvarloader::reverse::rc_flat_rows_inplace` — **9.31%** (haplotypes only)
7. `genvarloader::ffi::intervals_and_realign_track_fused` — **8.97%** (haps + variants)
8. `genvarloader::reconstruct::reconstruct_haplotypes_from_sparse` — **4.47%** (haplotypes only)
