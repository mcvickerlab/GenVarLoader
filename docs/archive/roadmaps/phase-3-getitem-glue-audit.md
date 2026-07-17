# Phase 3 `__getitem__` Glue Audit — Haps + Tracks Fusion Seams

**Purpose:** Task 12 of Phase 3 Rust migration (sub-unit 3d).  
Identifies every `np.ascontiguousarray` / boundary crossing / intermediate numpy
allocation on the two live read paths and proposes the minimal single-FFI-entry
fusion seams for Tasks 13 (fused haps) and 14 (fused tracks).

---

## 1. Haplotypes Path — Coercion / Crossing Inventory

Call chain:  
`Haps.__call__` → `Haps.get_haps_and_shifts` → `Haps._prepare_request` →  
`_haplotype_ilens` → `get_diffs_sparse` → (FFI #1)  
then back in `get_haps_and_shifts` → `_reconstruct_haplotypes` →  
`reconstruct_haplotypes_from_sparse` → (FFI #2)

### `_haplotype_ilens` / `_prepare_request`
(in `python/genvarloader/_dataset/_haps.py`)

| # | File:Line | Operation | Arrays coerced |
|---|-----------|-----------|----------------|
| H1 | `_haps.py:694` | `.astype(np.int32, copy=False)` on `regions` | `regions (b,3)` |

Note: `geno_offset_idx` is freshly computed (already `np.intp`) via
`np.ravel_multi_index` at `_haps.py:713–715`.  No allocation worth flagging —
it is required output.  `out_offsets = lengths_to_offsets(out_lengths)` at
`_haps.py:687` is also a required allocation (sizes the output buffer).

### `get_diffs_sparse` wrapper — FFI crossing #1
(in `python/genvarloader/_dataset/_genotypes.py`)

| # | File:Line | Operation | Arrays coerced |
|---|-----------|-----------|----------------|
| H2 | `_genotypes.py:149` | `np.ascontiguousarray(geno_offset_idx, np.int64)` | `(b,p)` |
| H3 | `_genotypes.py:150` | `np.ascontiguousarray(geno_v_idxs, np.int32)` | `(r*s*p*v)` — the full memmap |
| H4 | `_genotypes.py:151` | `_as_starts_stops(geno_offsets)` → `np.ascontiguousarray(np.stack([o[:-1], o[1:]]), np.int64)` | `(2, r*s*p)` — 2× alloc |
| H5 | `_genotypes.py:152` | `np.ascontiguousarray(ilens, np.int32)` | `(tot_v)` |
| H6 | `_genotypes.py:153` | `np.ascontiguousarray(keep, np.bool_)` (optional) | `(b*p*v)` |
| H7 | `_genotypes.py:154` | `np.ascontiguousarray(keep_offsets, np.int64)` (optional) | `(b*p+1)` |
| H8 | `_genotypes.py:155–157` | 3× `np.ascontiguousarray` for `q_starts`, `q_ends`, `v_starts` | `(b)`, `(b)`, `(tot_v)` |

**FFI crossing:** one Python→Rust boundary crossing into `_get_diffs_sparse_rust`.

Returns `diffs` shape `(b*p,)` — reshaped to `(b,p)` at `_haps.py:488` (view, no copy).

### `reconstruct_haplotypes_from_sparse` wrapper — FFI crossing #2
(in `python/genvarloader/_dataset/_genotypes.py`)

| # | File:Line | Operation | Arrays coerced |
|---|-----------|-----------|----------------|
| H9  | `_genotypes.py:316` | `np.ascontiguousarray(out_offsets, np.int64)` | `(b*p+1)` |
| H10 | `_genotypes.py:317` | `np.ascontiguousarray(regions, np.int32)` | `(b,3)` — already int32 from H1, still runs |
| H11 | `_genotypes.py:318` | `np.ascontiguousarray(shifts, np.int32)` | `(b,p)` |
| H12 | `_genotypes.py:319` | `np.ascontiguousarray(geno_offset_idx, np.int64)` | `(b,p)` — same array as H2 |
| H13 | `_genotypes.py:320` | `_as_starts_stops(geno_offsets)` again | `(2, r*s*p)` — **duplicate** of H4 |
| H14 | `_genotypes.py:321` | `np.ascontiguousarray(geno_v_idxs, np.int32)` | **duplicate** of H3 |
| H15 | `_genotypes.py:322` | `np.ascontiguousarray(v_starts, np.int32)` | **duplicate** of H8 |
| H16 | `_genotypes.py:323` | `np.ascontiguousarray(ilens, np.int32)` | **duplicate** of H5 |
| H17 | `_genotypes.py:324` | `np.ascontiguousarray(alt_alleles, np.uint8)` | `(tot_alt_bytes)` — memmap view |
| H18 | `_genotypes.py:325` | `np.ascontiguousarray(alt_offsets, np.int64)` | `(tot_v+1)` |
| H19 | `_genotypes.py:326` | `np.ascontiguousarray(ref, np.uint8)` | whole contig bytes — **large** |
| H20 | `_genotypes.py:327` | `np.ascontiguousarray(ref_offsets, np.int64)` | `(n_contigs+1)` |
| H21 | `_genotypes.py:329–330` | `None if keep is None else np.ascontiguousarray(keep, np.bool_)` | duplicate of H6 |
| H22 | `_genotypes.py:330` | same for `keep_offsets` | duplicate of H7 |

**Pre-kernel intermediate allocation:**  
`_haps.py:765`: `out_data = np.empty(req.out_offsets[-1], np.uint8)` — the output buffer.  
`_haps.py:766`: `out_offsets = np.asarray(req.out_offsets, np.int64)` — another dtype cast/view.

**FFI crossing:** one Python→Rust boundary crossing into `_reconstruct_haplotypes_from_sparse_rust`.

**Annotated haps path** adds two more pre-kernel allocations:  
`_haps.py:844`: `annot_v_data = np.empty(req.out_offsets[-1], V_IDX_TYPE)`  
`_haps.py:845`: `annot_pos_data = np.empty(req.out_offsets[-1], np.int32)`  
These are required outputs, not avoidable coercions.

### Summary — haplotypes path
- **2 FFI boundary crossings** (one per kernel)
- **~22 `np.ascontiguousarray` / `np.asarray` calls**, of which at least 8 are
  exact duplicates (H12–H16, H21–H22) because both wrapper functions independently
  normalize the same underlying arrays.
- **Key structural waste:** `_as_starts_stops(geno_offsets)` allocates a `(2, n)`
  int64 array twice — once per kernel crossing.  `geno_v_idxs`, `ilens`, `v_starts`,
  `keep`, `keep_offsets` are all re-coerced at the second crossing even though their
  dtypes are already correct after the first crossing.

---

## 2. Tracks Path — Coercion / Crossing Inventory

Call chain (HapsTracks mode, RaggedTracks output):  
`HapsTracks.__call__` → `get_haps_and_shifts` (same as above, 2 FFI crossings)  
then in the per-track loop:  
→ `intervals_to_tracks` → (FFI #3 per track)  
→ `_dispatch_get("shift_and_realign_tracks_sparse")` → (FFI #4 per track)

### Pre-loop allocations
(in `python/genvarloader/_dataset/_reconstruct.py`)

| # | File:Line | Operation |
|---|-----------|-----------|
| T1 | `_reconstruct.py:161` | `out = np.empty(n_tracks * n_per_track, np.float32)` — full fused output buffer |
| T2 | `_reconstruct.py:192` | `_tracks = np.empty(track_ofsts_per_t[-1], np.float32)` — **per-track intermediate** buffer, allocated inside the loop |

T2 is the key intermediate: it holds one track's reference-coordinate data before
realignment, then is discarded each iteration.  `n_tracks` loop iterations → `n_tracks`
temporary allocations + `n_tracks` FFI crossing pairs.

### `intervals_to_tracks` wrapper — FFI crossing #3 (×n_tracks)
(in `python/genvarloader/_dataset/_intervals.py`)

| # | File:Line | Operation | Arrays coerced |
|---|-----------|-----------|----------------|
| T3 | `_intervals.py:110` | `np.ascontiguousarray(offset_idxs, dtype=np.int64)` | `(b)` |
| T4 | `_intervals.py:111` | `np.ascontiguousarray(starts, dtype=np.int32)` | `(b)` |
| T5 | `_intervals.py:112` | `np.ascontiguousarray(itv_starts, dtype=np.int32)` | `(n_intervals)` — memmap |
| T6 | `_intervals.py:113` | `np.ascontiguousarray(itv_ends, dtype=np.int32)` | `(n_intervals)` — memmap |
| T7 | `_intervals.py:114` | `np.ascontiguousarray(itv_values, dtype=np.float32)` | `(n_intervals)` — memmap |
| T8 | `_intervals.py:115` | `np.ascontiguousarray(itv_offsets, dtype=np.int64)` | `(n_samples*n_regions+1)` |
| T9 | `_intervals.py:116` | `np.ascontiguousarray(out_offsets, dtype=np.int64)` | `(b+1)` |

**FFI crossing:** one Python→Rust boundary into `_intervals_to_tracks_rust`.  Writes
into `_tracks` (the per-track temp buffer).

### `shift_and_realign_tracks_sparse` wrapper — FFI crossing #4 (×n_tracks)
(in `python/genvarloader/_dataset/_tracks.py`)

| # | File:Line | Operation | Arrays coerced |
|---|-----------|-----------|----------------|
| T10 | `_tracks.py:433` | `_as_starts_stops(geno_offsets)` → `np.ascontiguousarray(np.stack(...), np.int64)` | `(2, r*s*p)` — duplicate of H4/H13, **again per track** |
| T11 | `_tracks.py:436` | `np.asarray(out_offsets, dtype=np.int64)` | `(b*p+1)` |
| T12 | `_tracks.py:437` | `np.asarray(regions, dtype=np.int32)` | `(b,3)` — already int32 |
| T13 | `_tracks.py:438` | `np.asarray(shifts, dtype=np.int32)` | `(b,p)` — already int32 |
| T14 | `_tracks.py:439` | `np.asarray(geno_offset_idx, dtype=np.int64)` | `(b,p)` |
| T15 | `_tracks.py:440` | `np.asarray(geno_v_idxs, dtype=np.int32)` | `(r*s*p*v)` — full memmap |
| T16 | `_tracks.py:442` | `np.asarray(v_starts, dtype=np.int32)` | `(tot_v)` |
| T17 | `_tracks.py:443` | `np.asarray(ilens, dtype=np.int32)` | `(tot_v)` |
| T18 | `_tracks.py:444` | `np.asarray(tracks, dtype=np.float32)` | `_tracks` intermediate |
| T19 | `_tracks.py:445` | `np.asarray(track_offsets, dtype=np.int64)` | `(b+1)` |
| T20 | `_tracks.py:446` | `np.asarray(params, dtype=np.float64)` | per-strategy params |
| T21 | `_tracks.py:448` | `np.asarray(keep_offsets, dtype=np.int64)` (optional) | `(b*p+1)` |

**FFI crossing:** one Python→Rust boundary into `_shift_and_realign_tracks_sparse_rust`.

### Summary — tracks path (HapsTracks, n_tracks tracks)
- **2 (haps) + 2×n_tracks (tracks)** FFI boundary crossings total per `__getitem__` call.
- **~22 (haps) + n_tracks × ~19 (tracks)** `np.ascontiguousarray`/`np.asarray` calls total.
- **Key structural waste:**
  - `_as_starts_stops(geno_offsets)` is re-executed **n_tracks+2 times** per call
    (once per haps kernel, once per track kernel pair). Each call allocates `(2, r*s*p)` int64.
  - `geno_v_idxs`, `v_starts`, `ilens` (full variant arrays, potentially large) are
    re-coerced **n_tracks+1 extra times** beyond the first.
  - `_tracks` intermediate buffer (T2, `np.empty`) is allocated **n_tracks times**;
    its data crosses the FFI twice (into `intervals_to_tracks` then read back by
    `shift_and_realign_tracks_sparse`) before being discarded.

---

## 3. Live Profiling

**Status: deferred.**

A profiling harness exists at `tests/benchmarks/profiling/profile.py` targeting
`tests/benchmarks/data/chr22_geuv.gvl`, and pre-existing speedscope profiles are
present at `tests/benchmarks/profiling/haps.speedscope.json` and
`tracks.speedscope.json`.  The chr22_geuv dataset and reference file are present
under `tests/benchmarks/data/`.

Live `cProfile` was not run during this audit because:
1. The static trace is complete and sufficient for identifying the fusion seams.
2. The pre-existing py-spy/memray profiles (generated before the Rust kernels were
   fully ported) reflect the old numba hot path and would need to be re-run with
   `GVL_BACKEND=rust` to measure the current Python glue share.
3. Running the dataset under `cProfile` (not py-spy) during a non-interactive session
   risks JIT warm-up noise and requires the pixi dev env.

**Recommendation for Task 13/14:** after implementing the fused entries, re-run
`pixi run -e dev profile-haps` and `profile-tracks` (py-spy) with `GVL_BACKEND=rust`
and compare the new profiles to confirm coercion overhead is gone.  The Phase 0 claim
(~62% glue) should be re-verified against the current Rust-kernel baseline.

---

## 4. Proposed Fused Entry Signatures

### 4a. Fused Haplotypes Entry (Task 13)

**Goal:** collapse FFI crossings H1 (get_diffs_sparse) and H2
(reconstruct_haplotypes_from_sparse) into a single Rust `#[pyfunction]` that:
1. Computes per-haplotype length diffs (`get_diffs_sparse` logic).
2. Allocates the output buffer and offset array in Rust.
3. Runs `reconstruct_haplotypes_from_sparse` logic.
4. Returns `(out_data: Array1<u8>, out_offsets: Array1<i64>)` — the raw ragged buffers.

The caller (Python `_reconstruct_haplotypes`) can then wrap them into a `_Flat`/`Ragged`
with zero further coercions.

```rust
/// Fused: compute diffs → out_offsets → reconstruct haplotypes.
/// Returns (out_data, out_offsets) as owned 1-D arrays.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_fused<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,          // (b, 3)
    geno_offset_idx: PyReadonlyArray2<i64>,  // (b, p)
    geno_offsets: PyReadonlyArray2<i64>,     // (2, r*s*p)
    geno_v_idxs: PyReadonlyArray1<i32>,      // (r*s*p*v) — full sparse store
    v_starts: PyReadonlyArray1<i32>,          // (tot_v)
    ilens: PyReadonlyArray1<i32>,             // (tot_v)
    alt_alleles: PyReadonlyArray1<u8>,        // (tot_alt_bytes)
    alt_offsets: PyReadonlyArray1<i64>,       // (tot_v + 1)
    ref_: PyReadonlyArray1<u8>,               // whole contig bytes
    ref_offsets: PyReadonlyArray1<i64>,       // (n_contigs + 1)
    pad_char: u8,
    output_length: i64,                       // -1 = ragged (hap length), else fixed
    keep: Option<PyReadonlyArray1<bool>>,     // (b*p*v) optional exonic mask
    keep_offsets: Option<PyReadonlyArray1<i64>>,  // (b*p + 1)
    // Optional annotation output buffers (annotated-haps mode).
    // When provided, filled in-place (caller pre-allocates based on returned out_offsets).
    // Task 13 may ship annotation support as a follow-on; initial version returns None.
    mut annot_v_idxs: Option<PyReadwriteArray1<i32>>,
    mut annot_ref_pos: Option<PyReadwriteArray1<i32>>,
) -> Bound<'py, PyTuple>   // (out_data: Array1<u8>, out_offsets: Array1<i64>)
```

**Rationale:**
- All arrays that were coerced twice (H2–H8 and H12–H22) are passed once.
- `_as_starts_stops` is done once in Rust (trivial row split of the `(2,n)` matrix).
- The Rust side owns the output buffer allocation — Python never calls `np.empty`.
- `output_length = -1` signals ragged mode; positive integer signals fixed-length
  (current Python: `np.full(..., output_length, np.int32)` is replaced by a Rust-side
  broadcast).
- Annotation buffers: for `_reconstruct_annotated_haplotypes`, the caller needs
  `out_offsets` before allocating them.  Two options: (a) two-call API (fused diffs +
  offsets in one call, then annotated reconstruct), or (b) pass pre-allocated buffers
  like the current Rust FFI does.  Option (b) is simpler and avoids a second crossing;
  the caller reads `out_offsets[-1]` from the first return to size the buffers if
  annotation is needed.

**Python-side after fusion (sketch):**
```python
out_data, out_offsets = gvl_rust.reconstruct_haplotypes_fused(
    regions=req.regions,
    geno_offset_idx=req.geno_offset_idx,
    geno_offsets=self.genotypes.offsets,   # already (2,n) or 1-D; Rust normalizes
    geno_v_idxs=self.genotypes.data,
    v_starts=self.variants.start,
    ilens=self.variants.ilen,
    alt_alleles=self.variants.alt.data.view(np.uint8),
    alt_offsets=self.variants.alt.offsets,
    ref_=self.reference.reference,
    ref_offsets=self.reference.offsets,
    pad_char=self.reference.pad_char,
    output_length=output_length if isinstance(output_length, int) else -1,
    keep=req.keep,
    keep_offsets=req.keep_offsets,
    annot_v_idxs=None,
    annot_ref_pos=None,
)
# out_data, out_offsets are fresh owned arrays — no further coercion needed
return _Flat.from_offsets(out_data, shape, out_offsets).view("S1")
```

**Risk — annotation path:** `_reconstruct_annotated_haplotypes` currently takes
in-place mutable annotation buffers whose sizes depend on `out_offsets[-1]`.  If
the fused entry returns `out_offsets` first and allocates buffers in a second step,
the annotation path gets a second Python call but still only ONE FFI crossing
(diffs+reconstruction in one shot).  Document this trade-off clearly in Task 13.

---

### 4b. Fused Tracks Entry (Task 14)

**Goal:** collapse FFI crossings T3+T4 (`intervals_to_tracks`) and the per-track
`shift_and_realign_tracks_sparse` crossing into a **single Rust entry per track** that:
1. Converts intervals → reference-coordinate tracks (inline, no intermediate Python buffer).
2. Shifts and realigns into the caller's pre-allocated `out` slice.

The outer Python loop over `n_tracks` stays — it is bounded by track count (small,
typically 1–10), not batch size — but each iteration drops from 2 FFI crossings + 1
intermediate allocation to 1 FFI crossing + 0 intermediate allocation.

```rust
/// Fused per-track: intervals → reference tracks → shift/realign into out.
/// Replaces the pair (intervals_to_tracks, shift_and_realign_tracks_sparse).
/// `out` is the per-track slice of the caller's pre-allocated output buffer.
/// `itv_offsets` is 1-D (n_samples*n_regions + 1) int64.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn intervals_and_realign_track_fused(
    mut out: PyReadwriteArray1<f32>,          // (b*p*l) — caller's pre-alloc slice
    out_offsets: PyReadonlyArray1<i64>,       // (b*p + 1)
    regions: PyReadonlyArray2<i32>,           // (b, 3)
    shifts: PyReadonlyArray2<i32>,            // (b, p)
    geno_offset_idx: PyReadonlyArray2<i64>,   // (b, p)
    geno_v_idxs: PyReadonlyArray1<i32>,       // (r*s*p*v)
    geno_offsets: PyReadonlyArray2<i64>,      // (2, r*s*p)
    v_starts: PyReadonlyArray1<i32>,           // (tot_v)
    ilens: PyReadonlyArray1<i32>,              // (tot_v)
    // intervals (reference-coordinate, for this track)
    offset_idxs: PyReadonlyArray1<i64>,       // (b) — per-query index into itv_offsets
    itv_starts: PyReadonlyArray1<i32>,         // (n_intervals)
    itv_ends: PyReadonlyArray1<i32>,           // (n_intervals)
    itv_values: PyReadonlyArray1<f32>,         // (n_intervals)
    itv_offsets: PyReadonlyArray1<i64>,        // (n_samples*n_regions + 1)
    // insertion-fill strategy
    params: PyReadonlyArray1<f64>,
    strategy_id: i64,
    base_seed: u64,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
) -> PyResult<()>
```

**Rust internals:** allocate a stack/thread-local scratch buffer of size
`max(track_lengths_for_batch)` instead of calling back to Python for the
intermediate `_tracks` buffer.  The `intervals_to_tracks` logic fills the scratch;
`shift_and_realign_track_sparse` reads from it and writes `out`.

**Rationale:**
- Removes the per-track `_tracks = np.empty(...)` intermediate allocation (T2).
- Removes 7 `np.ascontiguousarray` calls per track (T3–T9) for the
  `intervals_to_tracks` wrapper.
- Removes ~12 `np.asarray` calls per track (T10–T21) for the
  `shift_and_realign_tracks_sparse` wrapper.
- `_as_starts_stops(geno_offsets)` is done once in Rust per call, not per track.
- Net: from `2×n_tracks + 2` crossings to `n_tracks + 2` crossings per `__getitem__`.

**Python-side after fusion (sketch):**
```python
for track_ofst, (name, tracktype) in enumerate(self.tracks.active_tracks.items()):
    intervals = self.tracks.intervals[name]
    o_idx = idx if tracktype is TrackType.SAMPLE else r_idx
    _out = out[track_ofst * n_per_track : (track_ofst + 1) * n_per_track]
    gvl_rust.intervals_and_realign_track_fused(
        out=_out,
        out_offsets=out_ofsts_per_t,
        regions=regions,
        shifts=shifts,
        geno_offset_idx=geno_idx,
        geno_v_idxs=self.haps.genotypes.data,
        geno_offsets=self.haps.genotypes.offsets,
        v_starts=self.haps.variants.start,
        ilens=self.haps.variants.ilen,
        offset_idxs=o_idx,
        itv_starts=intervals.starts.data,
        itv_ends=intervals.ends.data,
        itv_values=intervals.values.data,
        itv_offsets=intervals.starts.offsets,
        params=strat_params[track_ofst],
        strategy_id=int(strat_ids[track_ofst]),
        base_seed=base_seed,
        keep=keep,
        keep_offsets=keep_offsets,
    )
```
No `np.ascontiguousarray` / `np.empty` inside the loop.

---

## 5. Risks and Notes

### 5a. Annotation buffers (haps path)

`_reconstruct_annotated_haplotypes` pre-allocates `annot_v_data` and
`annot_pos_data` at `_haps.py:844–845` **before** calling
`reconstruct_haplotypes_from_sparse`, because their sizes equal
`out_offsets[-1]` which is computed from `diffs`.  In the fused entry the caller
cannot know `out_offsets[-1]` until after Rust returns — unless the fused entry
accepts them as optional in/out parameters (like the existing FFI) or computes
diffs in a pre-flight call.

**Recommended approach for Task 13:** the fused entry accepts
`annot_v_idxs: Option<PyReadwriteArray1<i32>>` and
`annot_ref_pos: Option<PyReadwriteArray1<i32>>` as optional write buffers,
mirroring the current `reconstruct_haplotypes_from_sparse` FFI.  The Python
caller runs the non-annotated fused entry first when annotation is not needed
(the common path), and uses a two-step approach (get offsets, alloc, call annotated
variant) for the annotated path.  This keeps the common path at one crossing.

### 5b. `intervals_to_tracks` contract bug (tracks path)

**Filed bug mcvickerlab/GenVarLoader#242:**  
`intervals_to_tracks` assumes `itv.start >= query_start` (documented in the numba
source at `_intervals.py:73`).  For datasets with `max_jitter > 0`, jittered query
start positions can be less than the stored interval starts, violating this
contract. The numba backend silently returns wrong results; the Rust backend
panics.

**Task 14 scope:** the fused tracks entry REUSES the existing
`intervals_to_tracks` core logic as-is.  It does NOT fix this bug.  The fix is
deferred to a separate PR.

**Consequence for parity testing:** Task 14's parity tests MUST use `max_jitter=0`
datasets to stay within the contract.  This matches the current Task 11 parity test
setup.

### 5c. `_as_starts_stops` duplication

The `_as_starts_stops` helper (`_genotypes.py:119–125`) converts 1-D offset arrays
to `(2, n)` starts/stops.  It is called separately in:
- `get_diffs_sparse` wrapper (H4)
- `reconstruct_haplotypes_from_sparse` wrapper (H13)
- `_shift_and_realign_tracks_sparse_rust_wrapper` (T10) — once per track

After fusion, the Rust side can accept the offsets in either form and branch
internally (the `(2,n)` row-split is a view, not a copy).  Alternatively, the
Python caller can normalize once and pass the `(2,n)` array to all callers.

### 5d. Splice plan path

`_reconstruct_haplotypes` has a separate splice-plan branch
(`_haps.py:793–829`) that calls `_permute_request_for_splice` and invokes
`reconstruct_haplotypes_from_sparse` with reshuffled arrays.  The fused entry
should accept an optional `permutation` array and perform the permutation in Rust,
or alternatively the splice path can continue using the existing non-fused entry
(since spliced reconstruction is already uncommon and correct).  Task 13 should
explicitly decide this scope.

---

## 6. Files Affected by This Audit (no production changes)

| File | Role |
|------|------|
| `python/genvarloader/_dataset/_haps.py` | haps path — `_prepare_request`, `_reconstruct_haplotypes`, `_reconstruct_annotated_haplotypes` |
| `python/genvarloader/_dataset/_genotypes.py` | dispatch wrappers — `get_diffs_sparse`, `reconstruct_haplotypes_from_sparse` |
| `python/genvarloader/_dataset/_reconstruct.py` | compound reconstructor — `HapsTracks.__call__` |
| `python/genvarloader/_dataset/_tracks.py` | dispatch wrapper — `_shift_and_realign_tracks_sparse_rust_wrapper` |
| `python/genvarloader/_dataset/_intervals.py` | dispatch wrapper — `intervals_to_tracks` |
| `src/ffi/mod.rs` | current Rust `#[pyfunction]` entries (reference for Task 13/14 signatures) |
| `src/reconstruct/mod.rs` | Rust `reconstruct_haplotypes_from_sparse` core |
| `src/tracks/mod.rs` | Rust `shift_and_realign_tracks_sparse` core |
