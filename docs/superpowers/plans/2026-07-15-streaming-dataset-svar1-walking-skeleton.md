# StreamingDataset — SVAR1 → haplotypes walking skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A write-free `gvl.StreamingDataset(regions, reference, variants="x.svar").with_seqs("haplotypes")` that iterates haplotype batches in region-major order, byte-identical to `gvl.write()` + `Dataset.open()[r, s]` — using the simplest correct implementation (no double-buffer yet).

**Architecture:** A new Python `StreamingDataset` (torch `IterableDataset`) drives a region-major scheduler and yields index-carrying batches. Per batch it calls a new Rust pyclass `Svar1Store` that reads a region-window's sparse genotypes from a live `.svar` via genoray's (conversion-gated) `Svar1RecordSource`, assembles the SVAR1-style arrays (`geno_v_idxs`, `geno_offsets`, local variant table), and reconstructs via the **existing** `reconstruct_haplotypes_from_sparse` kernel. The static variant table (POS/REF/ALT/ILEN) is read **once at construction** from `genoray.SparseVar(path).index` (not per-batch — throughput-neutral); the per-batch sparse-genotype read is Rust.

**Tech Stack:** Rust (PyO3 0.29, ndarray, `genoray_core`+`svar2-codec` git deps), Python 3.10+ (numpy, polars, seqpro, torch optional), pixi (`-e dev`), maturin, pytest.

## Global Constraints

- **Byte-identical parity** with `gvl.write()` + `Dataset.open()[r, s]` (jitter=0) is the correctness gate — same contract as `docs/archive/roadmaps/rust-migration.md`.
- **Rebuild Rust before pytest:** `pixi run -e dev pytest` does **not** rebuild the extension. After any `src/` change run `pixi run -e dev maturin develop --release` first, or pytest imports the stale binary. (`cargo test` compiles from source.)
- **genoray release-gate (⛔ do not merge to `main`):** this work enables `genoray_core`'s `conversion` feature and depends on unreleased genoray; it is dev-wired to the local checkout. Ship only after genoray publishes (mirror the Phase 6a checklist in the archived roadmap).
- **Coordinates:** 0-based half-open `[start, end)` everywhere; `SparseVar.index.POS` is 1-based (subtract 1).
- **Variant precondition (unchanged from `gvl.write`):** normalized (left-aligned, biallelic, atomized), no symbolic/breakend ALTs — validated, not fixed up.
- **Scope of THIS plan:** SVAR1 source, `haplotypes` output, `jitter=0`, ragged output only. No double-buffer (separate optimization plan). No VCF/PGEN, no annotated/variants/tracks modes, no `with_len`, no `min_af`/`max_af` (later plans).
- **Commands:** `pixi run -e dev maturin develop --release`, `pixi run -e dev pytest <path> -q`, `pixi run -e dev cargo-test`, `pixi run -e dev ruff check python/ tests/`, `pixi run -e dev typecheck`. Commits may be slow (prek runs a full `pyrefly` on every commit).

---

### Task 1: Enable genoray's `conversion` feature + prove the build

**Files:**
- Modify: `Cargo.toml:26` (the `genoray_core` dependency line)

**Interfaces:**
- Consumes: nothing.
- Produces: a build in which `genoray_core::svar1_reader::Svar1RecordSource`, `genoray_core::record_source::{RecordSource, RawRecord}`, and the `VcfRecordSource`/`PgenRecordSource` types are compiled and importable from gvl's crate. This unblocks all RecordSource-based backends.

- [ ] **Step 1: Confirm the readers are gated out today**

Run: `grep -n 'cfg(feature = "conversion")' ~/.cargo/git/checkouts/genoray-*/66ba734/src/lib.rs`
Expected: lines showing `pub mod svar1_reader;` and `pub mod record_source;` are `#[cfg(feature = "conversion")]`.

- [ ] **Step 2: Add the `conversion` feature to gvl's `genoray_core` dep**

In `Cargo.toml`, change line 26 from:

```toml
genoray_core = { git = "https://github.com/d-laub/genoray.git", rev = "66ba734b85fcf1326008d66b33c052c7cf278a9f", package = "genoray", default-features = false }
```

to (add `features = ["conversion"]`, keep `default-features = false` so `extension-module` stays off in the rlib):

```toml
genoray_core = { git = "https://github.com/d-laub/genoray.git", rev = "66ba734b85fcf1326008d66b33c052c7cf278a9f", package = "genoray", default-features = false, features = ["conversion"] }
```

- [ ] **Step 3: Build and confirm htslib links**

Run: `pixi run -e dev cargo build --release 2>&1 | tail -20`
Expected: builds clean. If rust-htslib fails to find system htslib, note the error — this is the abi3-wheel risk the spec flags; capture the exact failure for the wheel-matrix decision before proceeding.

- [ ] **Step 4: Prove a probe import compiles (smoke)**

Add a temporary `src/svar1/mod.rs` with `#[allow(unused_imports)] use genoray_core::svar1_reader::Svar1RecordSource; use genoray_core::record_source::{RecordSource, RawRecord};` and `pub mod svar1;` in `src/lib.rs:9` area, then:
Run: `pixi run -e dev cargo build --release 2>&1 | tail -5`
Expected: compiles (the gated types are now visible). Keep `src/svar1/mod.rs` (Task 3 fills it); it currently just proves visibility.

- [ ] **Step 5: Wheel-matrix spike (build-risk gate)**

Run: `pixi run -e dev maturin build --release 2>&1 | tail -15`
Expected: a `cp310-abi3` wheel builds. **If it fails to build/link htslib**, STOP and report — the direct-read approach needs the fallback (optional-feature gating) from the spec before more work. Record wheel size delta.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/svar1/mod.rs src/lib.rs
git commit -m "build(streaming): enable genoray_core conversion feature (htslib) for RecordSource readers"
```

---

### Task 2: `StreamingDataset` Python skeleton — region-major scheduler + iteration, no backend yet

Prove the pure-Python iteration/scheduler/index contract with an injected stub reconstructor, before wiring Rust.

**Files:**
- Create: `python/genvarloader/_dataset/_streaming.py`
- Test: `tests/dataset/test_streaming_scheduler.py`

**Interfaces:**
- Consumes: `python/genvarloader/_utils.py::bed_to_regions` (existing; `(bed, ContigNormalizer) -> NDArray[int32] (n,3)`), `seqpro.bed.sort`.
- Produces:
  - `class StreamingDataset` with `__init__(self, regions, *, contigs: list[str], n_samples: int, ploidy: int, _reconstruct_window)`, where `_reconstruct_window(r_idx: NDArray[intp], s_idx: NDArray[intp]) -> object` is the per-batch backend callable (a real one arrives in Task 4; tests inject a stub).
  - `def __len__(self) -> int` → `n_regions * n_samples`.
  - `def __iter__(self) -> Iterator[tuple]` yielding `(data, region_idx, sample_idx)` batches of size `self._batch_size` in region-major order.
  - `def _plan(self) -> Iterator[tuple[NDArray[intp], NDArray[intp]]]` → region-major `(r_idx, s_idx)` batch index pairs.
  - `property shape -> tuple[int, int]` → `(n_regions, n_samples)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/dataset/test_streaming_scheduler.py
import numpy as np
import polars as pl
from genvarloader._dataset._streaming import StreamingDataset

def _bed(rows):
    return pl.DataFrame(rows, schema={"chrom": pl.Utf8, "chromStart": pl.Int64, "chromEnd": pl.Int64})

def test_plan_is_region_major_and_covers_grid():
    # 3 regions x 2 samples, batch_size 2 -> region-major flat order
    bed = _bed([{"chrom": "chr1", "chromStart": s, "chromEnd": s + 10} for s in (30, 10, 20)])
    seen = []
    def stub(r_idx, s_idx):
        seen.append((tuple(r_idx), tuple(s_idx)))
        return np.stack([r_idx, s_idx], axis=1)  # fake "data"
    sds = StreamingDataset(bed, contigs=["chr1"], n_samples=2, ploidy=2, _reconstruct_window=stub)
    sds = sds._with_batch_size(2)
    batches = list(sds)
    # region order is sorted by (contig,start): input starts 30,10,20 -> sorted r order 1,2,0
    # region-major flat index over (n_regions=3, n_samples=2): r sorted-inner sample
    flat_r = np.concatenate([b[1] for b in batches])
    flat_s = np.concatenate([b[2] for b in batches])
    # every (r,s) cell appears exactly once
    cells = set(zip(flat_r.tolist(), flat_s.tolist()))
    assert cells == {(r, s) for r in range(3) for s in range(2)}
    # region-major: sample index varies fastest within a region
    assert flat_r.tolist() == [1, 1, 2, 2, 0, 0]
    assert flat_s.tolist() == [0, 1, 0, 1, 0, 1]
    assert len(sds) == 6
    assert sds.shape == (3, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scheduler.py -q`
Expected: FAIL (`ModuleNotFoundError: genvarloader._dataset._streaming`).

- [ ] **Step 3: Write minimal implementation**

```python
# python/genvarloader/_dataset/_streaming.py
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import polars as pl
import seqpro as sp
from genoray._contigs import ContigNormalizer
from numpy.typing import NDArray

from .._utils import bed_to_regions  # if not present, see note below


@dataclass(frozen=True, slots=True)
class StreamingDataset:
    """Write-free, iterable-only dataset. Region-major iteration; no random access."""

    _bed: pl.DataFrame
    _regions: NDArray[np.int32]  # (n_regions, 3) sorted (contig_idx, start, end)
    _sort_order: NDArray[np.intp]  # maps sorted position -> original bed row
    contigs: list[str]
    n_samples: int
    ploidy: int
    _reconstruct_window: Callable[[NDArray[np.intp], NDArray[np.intp]], object]
    _batch_size: int = 1
    return_indices: bool = True

    def __init__(self, regions, *, contigs, n_samples, ploidy, _reconstruct_window):
        bed = regions if isinstance(regions, pl.DataFrame) else sp.bed.read(regions)
        sorted_bed = sp.bed.sort(bed)
        # record original-row order so emitted indices refer to the user's input order
        order = (
            bed.with_row_index("_r")
            .join(sorted_bed.with_row_index("_sorted"), on=list(bed.columns), how="right")
            .sort("_sorted")["_r"]
            .to_numpy()
            .astype(np.intp)
        )
        regs = bed_to_regions(sorted_bed, ContigNormalizer(contigs))
        object.__setattr__(self, "_bed", bed)
        object.__setattr__(self, "_regions", regs)
        object.__setattr__(self, "_sort_order", order)
        object.__setattr__(self, "contigs", list(contigs))
        object.__setattr__(self, "n_samples", int(n_samples))
        object.__setattr__(self, "ploidy", int(ploidy))
        object.__setattr__(self, "_reconstruct_window", _reconstruct_window)
        object.__setattr__(self, "_batch_size", 1)
        object.__setattr__(self, "return_indices", True)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._regions), self.n_samples)

    def __len__(self) -> int:
        return len(self._regions) * self.n_samples

    def _with_batch_size(self, batch_size: int) -> "StreamingDataset":
        return replace(self, _batch_size=int(batch_size))

    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        # region-major flat index over (n_regions, n_samples): sample varies fastest.
        n_regions, n_samples = self.shape
        flat = np.arange(n_regions * n_samples, dtype=np.intp)
        for start in range(0, flat.size, self._batch_size):
            chunk = flat[start : start + self._batch_size]
            r_idx, s_idx = np.unravel_index(chunk, (n_regions, n_samples))
            yield r_idx.astype(np.intp), s_idx.astype(np.intp)

    def __iter__(self) -> Iterator[tuple]:
        for r_idx, s_idx in self._plan():
            data = self._reconstruct_window(r_idx, s_idx)
            if self.return_indices:
                # map sorted region positions back to the user's original bed rows
                yield (data, self._sort_order[r_idx], s_idx)
            else:
                yield data
```

Note: if `bed_to_regions` is not importable from `.._utils`, import it from `._open` (`from ._open import bed_to_regions`) — confirm with `grep -rn "def bed_to_regions" python/genvarloader`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scheduler.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
pixi run -e dev ruff check python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_scheduler.py
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_scheduler.py
git commit -m "feat(streaming): StreamingDataset region-major scheduler + iteration skeleton"
```

---

### Task 3: `Svar1Store` Rust pyclass — open a live `.svar` per contig

Mirror `Svar2Store` (`src/svar2/store.rs`). This task exposes construction only (no reads yet), tested from `cargo test` + Python import.

**Files:**
- Create: `src/svar1/store.rs`
- Modify: `src/svar1/mod.rs` (replace the Task-1 probe with `pub mod store;`), `src/lib.rs` (register the pyclass)
- Test: `tests/dataset/test_svar1_store.py`

**Interfaces:**
- Consumes: `genoray_core::svar1_reader::Svar1RecordSource::new(svar1_dir, contig_start, n_local, num_samples, ploidy, pos, ref_bytes, ref_offsets, alt_bytes, alt_offsets, format_fields, format_src_dtypes) -> Result<Self, ConversionError>`; `genoray_core::record_source::{RecordSource, RawRecord}`.
- Produces: `#[pyclass] Svar1Store` with `#[new] fn new(store_path: &str, contigs: Vec<String>, n_samples: usize, ploidy: usize) -> PyResult<Self>` and `fn contigs(&self) -> Vec<String>`. Per-contig variant tables (`pos/ref/alt`) are supplied by Python at construction (Task 4 passes them); for this task `new` records `store_path/n_samples/ploidy/contigs` and validates the directory exists.

- [ ] **Step 1: Write the failing Rust test**

```rust
// bottom of src/svar1/store.rs
#[cfg(test)]
mod tests {
    #[test]
    fn store_path_roundtrips() {
        // Construction records metadata; a missing dir errors.
        let err = super::Svar1Store::open_meta("/no/such/svar", vec!["chr1".into()], 2, 2);
        assert!(err.is_err());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev cargo test --release svar1 2>&1 | tail -15`
Expected: FAIL (`Svar1Store` / `open_meta` undefined).

- [ ] **Step 3: Write minimal implementation**

```rust
// src/svar1/store.rs
use std::collections::HashMap;
use std::path::Path;

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

/// Per-contig static variant table + geometry supplied by Python at construction.
pub struct ContigTable {
    pub contig_start: usize, // index of this contig's first variant in the global table
    pub n_local: usize,      // variants on this contig
    pub pos: Vec<u32>,       // 0-based POS
    pub ref_bytes: Vec<u8>,
    pub ref_offsets: Vec<i64>,
    pub alt_bytes: Vec<u8>,
    pub alt_offsets: Vec<i64>,
}

#[pyclass]
pub struct Svar1Store {
    store_path: String,
    n_samples: usize,
    ploidy: usize,
    tables: HashMap<String, ContigTable>, // filled via `set_contig_table` in Task 4
}

impl Svar1Store {
    /// Metadata-only constructor (validates the store dir); used by tests + `#[new]`.
    pub fn open_meta(
        store_path: &str,
        contigs: Vec<String>,
        n_samples: usize,
        ploidy: usize,
    ) -> PyResult<Self> {
        if !Path::new(store_path).is_dir() {
            return Err(PyIOError::new_err(format!("svar store not found: {store_path}")));
        }
        let mut tables = HashMap::with_capacity(contigs.len());
        for c in &contigs {
            tables.insert(c.clone(), ContigTable {
                contig_start: 0, n_local: 0,
                pos: vec![], ref_bytes: vec![], ref_offsets: vec![0],
                alt_bytes: vec![], alt_offsets: vec![0],
            });
        }
        let _ = tables; // populated in Task 4
        Ok(Self { store_path: store_path.to_string(), n_samples, ploidy, tables: HashMap::new() })
    }

    pub fn store_path(&self) -> &str { &self.store_path }
    pub fn table(&self, contig: &str) -> Option<&ContigTable> { self.tables.get(contig) }
    pub fn n_samples(&self) -> usize { self.n_samples }
    pub fn ploidy(&self) -> usize { self.ploidy }
}

#[pymethods]
impl Svar1Store {
    #[new]
    fn new(store_path: &str, contigs: Vec<String>, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        Self::open_meta(store_path, contigs, n_samples, ploidy)
    }

    fn contigs(&self) -> Vec<String> { self.tables.keys().cloned().collect() }
}
```

```rust
// src/svar1/mod.rs  (replace the Task-1 probe)
pub mod store;
```

In `src/lib.rs`, register the class (after the `Svar2Store` line, ~`src/lib.rs:23`):

```rust
m.add_class::<svar1::store::Svar1Store>()?;
```

- [ ] **Step 4: Run to verify it passes + Python import works**

Run: `pixi run -e dev cargo test --release svar1 2>&1 | tail -8`
Expected: PASS.

```python
# tests/dataset/test_svar1_store.py
import pytest
from genvarloader.genvarloader import Svar1Store  # the compiled extension module

def test_missing_store_errors():
    with pytest.raises(Exception):
        Svar1Store("/no/such/svar", ["chr1"], 2, 2)
```

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_svar1_store.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/svar1/ src/lib.rs tests/dataset/test_svar1_store.py
git commit -m "feat(streaming): Svar1Store pyclass (metadata + per-contig table scaffold)"
```

---

### Task 4: SVAR1 window read → sparse arrays → `reconstruct_haplotypes_from_sparse`

The core: given a batch of `(r_idx, s_idx)`, read those regions' variants for those samples from the live `.svar`, build the SVAR1-style sparse arrays, and reconstruct haplotypes via the existing kernel. Static variant table read once at construction from `genoray.SparseVar(path).index`.

**Files:**
- Create: `src/ffi_svar1.rs` (or extend `src/ffi/mod.rs`) — the FFI reconstruct entry
- Modify: `src/svar1/store.rs` (add `set_contig_table` + a window-read method), `src/lib.rs` (register the function)
- Modify: `python/genvarloader/_dataset/_streaming.py` (a `_Svar1Backend` producing `_reconstruct_window`)
- Test: `tests/dataset/test_svar1_window.py`

**Interfaces:**
- Consumes: `reconstruct::reconstruct_haplotypes_from_sparse(out, out_offsets, regions:(b,3)i32, shifts:(b,ploidy)i32, geno_offset_idx:(b,ploidy)i64, geno_o_starts:i64, geno_o_stops:i64, geno_v_idxs:i32, v_starts:i32, ilens:i32, alt_alleles:u8, alt_offsets:i64, ref_:u8, ref_offsets:i64, pad_char:u8, keep:None, keep_offsets:None, annot_v_idxs:None, annot_ref_pos:None, parallel:bool)`; `genotypes::get_diffs_sparse(...)` for the diffs; `Svar1RecordSource::next_record() -> Option<RawRecord{pos,reference,alts,gt,...}>` (ILEN = `alts[0].len() as i32 - reference.len() as i32`); `genoray.SparseVar(path)` Python (`.index` polars DF with `POS`(1-based)/`REF`/`ALT`/`ILEN`, `.available_samples`, `.ploidy`, `.contigs`).
- Produces: FFI `reconstruct_haplotypes_svar1(store, contig, region_bounds:(b,2)i32, sample_idx:(b,)i64, ref_:u8, ref_offsets:i64, pad_char:u8, parallel:bool) -> (PyArray1<u8>, PyArray1<i64>)`; Python `_Svar1Backend.reconstruct_window(r_idx, s_idx) -> Ragged[S1]`.

- [ ] **Step 1: Write the failing parity-shaped unit test**

```python
# tests/dataset/test_svar1_window.py
import numpy as np
import genvarloader as gvl
from genvarloader._dataset._streaming import StreamingDataset, _Svar1Backend

def test_single_region_all_samples_matches_written(svar1_dataset_fixture):
    # fixture provides: svar_path, reference_path, bed (1 region), and a written gvl Dataset
    f = svar1_dataset_fixture
    backend = _Svar1Backend(f.svar_path, f.reference_path, f.contigs)
    sds = StreamingDataset(
        f.bed, contigs=f.contigs, n_samples=backend.n_samples, ploidy=backend.ploidy,
        _reconstruct_window=backend.reconstruct_window,
    )._with_batch_size(backend.n_samples)
    (data, r_idx, s_idx), = list(sds)  # one region, all samples in one batch
    written = f.dataset.with_seqs("haplotypes")
    for i, (r, s) in enumerate(zip(r_idx, s_idx)):
        np.testing.assert_array_equal(
            data[i].to_numpy(), written[int(r), int(s)].to_numpy()
        )
```

Add the fixture `svar1_dataset_fixture` to `tests/dataset/conftest.py`: build a tiny `.svar` (via `genoray.SparseVar.from_vcf` on a `vcfixture` VCF), a matching `gvl.write()` dataset from the same `.svar` + reference, and a 1-region BED. (Mirror the existing SVAR2 fixtures in `tests/dataset/test_svar2_dataset.py` / `conftest.py` — reuse their reference + vcfixture helpers.)

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_svar1_window.py -q`
Expected: FAIL (`_Svar1Backend` undefined / `reconstruct_haplotypes_svar1` missing).

- [ ] **Step 3a: Rust — `set_contig_table` + window read on `Svar1Store`**

In `src/svar1/store.rs`, add `#[pymethods]` `set_contig_table` (Python passes the per-contig static table read from `SparseVar.index`) and an internal window reader that, for a set of regions on one contig, instantiates a `Svar1RecordSource` for that contig and walks `next_record()` accumulating, per requested sample-hap, the local variant indices whose `pos ∈ [start,end)`:

```rust
// add to #[pymethods] impl Svar1Store
#[allow(clippy::too_many_arguments)]
fn set_contig_table(
    &mut self, contig: &str, contig_start: usize, n_local: usize,
    pos: Vec<u32>, ref_bytes: Vec<u8>, ref_offsets: Vec<i64>,
    alt_bytes: Vec<u8>, alt_offsets: Vec<i64>,
) {
    self.tables.insert(contig.to_string(), super::store::ContigTable {
        contig_start, n_local, pos, ref_bytes, ref_offsets, alt_bytes, alt_offsets,
    });
}
```

```rust
// internal (non-py) on impl Svar1Store: read one contig's window into CSR sparse arrays.
// Returns (geno_v_idxs, o_starts, o_stops, geno_offset_idx) for the (regions x samples x ploidy)
// grid, with v_idxs referencing the GLOBAL variant table (contig_start + local).
pub fn read_window(
    &self, contig: &str, region_bounds: &[(i32, i32)], samples: &[usize],
) -> anyhow::Result<crate::svar1::Sparse> {
    let t = self.table(contig).ok_or_else(|| anyhow::anyhow!("no contig {contig}"))?;
    let mut src = genoray_core::svar1_reader::Svar1RecordSource::new(
        self.store_path(), t.contig_start, t.n_local, self.n_samples(), self.ploidy(),
        t.pos.clone(), t.ref_bytes.clone(), t.ref_offsets.clone(),
        t.alt_bytes.clone(), t.alt_offsets.clone(), &[], &[],
    ).map_err(|e| anyhow::anyhow!("svar1 open: {e:?}"))?;
    // Accumulate per (region, sample, hap) sparse variant-index lists.
    // buckets[q][s][h] : Vec<i32> global variant indices; local_i tracks the record index.
    // ... walk src.next_record(): for each record at global v = contig_start + local_i,
    //     for each requested region q with pos in [lo,hi), for each requested sample s,
    //     for each hap h in 0..ploidy: if gt[s*ploidy+h] != 0 push v.
    // Then flatten to CSR (geno_v_idxs, o_starts, o_stops) row-major over (q, s, h),
    // and geno_offset_idx[(q*len(samples)+s), h] = row number of that (q,s,h).
    todo!("flatten buckets to Sparse — see struct below")
}
```

Add `pub struct Sparse { pub geno_v_idxs: Vec<i32>, pub o_starts: Vec<i64>, pub o_stops: Vec<i64>, pub geno_offset_idx: ndarray::Array2<i64> }` to `src/svar1/mod.rs`.

> Implementation note (no placeholder): the flatten is exactly the CSR gvl uses — one row per `(region, sample, hap)`; `o_starts[row]`/`o_stops[row]` bracket that row's slice in `geno_v_idxs`; `geno_offset_idx[(region*n_s+sample), hap] = row`. This mirrors `_get_geno_offset_idx` in `python/genvarloader/_dataset/_haps.py:753`.

- [ ] **Step 3b: Rust — FFI reconstruct entry**

In `src/ffi/mod.rs` (register in `src/lib.rs` alongside `reconstruct_haplotypes_fused`):

```rust
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_svar1<'py>(
    py: Python<'py>,
    store: PyRef<crate::svar1::store::Svar1Store>,
    contig: &str,
    region_bounds: PyReadonlyArray2<i32>,   // (b, 2) = (start, end), 0-based half-open
    sample_idx: PyReadonlyArray1<i64>,       // (b,) sample index per batch row
    v_starts: PyReadonlyArray1<i32>,         // GLOBAL static table (from SparseVar.index)
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)> {
    // 1. Build regions (b,3): contig_idx=0 (single-contig call), start, end.
    // 2. store.read_window(contig, bounds, samples) -> Sparse.
    // 3. shifts = zeros((b, ploidy)) i32  (jitter=0 in this plan).
    // 4. diffs = genotypes::get_diffs_sparse(geno_offset_idx, geno_v_idxs, o_starts, o_stops,
    //            ilens, None, None, Some(q_starts), Some(q_ends), Some(v_starts), parallel).
    // 5. out_offsets from (region_len + diff) per (b,ploidy) row (mirror ffi:714-735).
    // 6. out = uninit_output(total); reconstruct::reconstruct_haplotypes_from_sparse(...).
    // 7. return (out, out_offsets).
    todo!("assemble per fused wrapper at src/ffi/mod.rs:671-763, minus the RC/keep paths")
}
```

> Implementation note: this is the `reconstruct_haplotypes_fused` body (`src/ffi/mod.rs:671-763`) with `geno_*` sourced from `store.read_window` instead of Python arrays, `keep/keep_offsets/to_rc = None`, and `output_length = -1` (ragged). Every other step is identical; reuse the same helpers (`get_diffs_sparse`, `uninit_output`, the out_offsets prefix-sum).

- [ ] **Step 3c: Python — `_Svar1Backend`**

```python
# append to python/genvarloader/_dataset/_streaming.py
from seqpro.rag import Ragged
from ._utils import lengths_to_offsets  # confirm path via grep

class _Svar1Backend:
    def __init__(self, svar_path, reference_path, contigs):
        from genoray import SparseVar
        from ._reference import Reference
        from ..genvarloader import Svar1Store
        sv = SparseVar(str(svar_path))
        self.n_samples = len(sv.available_samples)
        self.ploidy = sv.ploidy
        self._ref = Reference.from_path(reference_path, contigs)
        idx = sv.index  # polars DF, POS 1-based
        # build GLOBAL static table + per-contig offsets from idx (POS->0-based)
        ... # v_starts=int32(POS-1), ilens=int32(ILEN), alt CSR from ALT, ref CSR from REF
        self._store = Svar1Store(str(svar_path), list(contigs), self.n_samples, self.ploidy)
        for c in contigs:
            self._store.set_contig_table(c, contig_start_c, n_local_c, pos_c, ref_bytes_c,
                                         ref_offsets_c, alt_bytes_c, alt_offsets_c)
        self._contigs = list(contigs)

    def reconstruct_window(self, r_idx, s_idx):
        # this plan: caller batches a single region's samples; group by that region's contig.
        from ..genvarloader import reconstruct_haplotypes_svar1
        ... # region_bounds from self._regions[r_idx], one contig per batch here
        data, offsets = reconstruct_haplotypes_svar1(
            self._store, contig, region_bounds, s_idx.astype("int64"),
            self._v_starts, self._ilens, self._alt_alleles, self._alt_offsets,
            self._ref_bytes, self._ref_offsets, ord("N"), True,
        )
        return Ragged.from_offsets(data.view("S1"), lengths_from_offsets(offsets))
```

> Implementation note: reuse `Reference.from_path` and the `SparseVar.index` → static-table conversion already done in `python/genvarloader/_dataset/_haps.py` (`_Variants.from_table` / `_HapsFfiStatic`); factor the shared conversion into a small helper rather than duplicating.

- [ ] **Step 4: Build + run to verify it passes**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_svar1_window.py -q`
Expected: PASS (byte-identical to the written dataset for the single region, all samples).

- [ ] **Step 5: Commit**

```bash
git add src/ python/genvarloader/_dataset/_streaming.py tests/dataset/test_svar1_window.py tests/dataset/conftest.py
git commit -m "feat(streaming): SVAR1 window read + reconstruct haplotypes via existing kernel"
```

---

### Task 5: Multi-region / multi-contig parity + `with_seqs` + `to_dataloader`

Promote the backend to the real `StreamingDataset` public shape and prove parity across many regions/contigs and via a torch `DataLoader`.

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (public constructor mirroring `gvl.write` params; `.with_seqs`; `.to_dataloader`; group plan batches by contig before the Rust call)
- Modify: `python/genvarloader/_torch.py` (add an `IterableDataset` wrapper mirroring `_buffered_loader.py`'s pattern, or reuse it)
- Test: `tests/dataset/test_streaming_parity.py`

**Interfaces:**
- Consumes: Task 4's `_Svar1Backend`; `python/genvarloader/_torch.py::{TORCH_AVAILABLE, requires_torch}`.
- Produces: `StreamingDataset(regions, reference=None, variants=<path>, jitter=0)` public constructor (classifies `.svar`); `.with_seqs("haplotypes") -> Self`; `.to_dataloader(batch_size=1, num_workers=0, return_indices=True, ...) -> td.DataLoader`; `.to_torch_dataset()` raises `TypeError`.

- [ ] **Step 1: Write the failing parity test (many regions, 2 contigs, batched)**

```python
# tests/dataset/test_streaming_parity.py
import numpy as np
import genvarloader as gvl

def test_streaming_matches_written_all_cells(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture  # >=2 contigs, >=10 regions, >=3 samples
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs("haplotypes")
    dl = sds.to_dataloader(batch_size=4, return_indices=True)
    seen = set()
    for data, r_idx, s_idx in dl:
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            np.testing.assert_array_equal(np.asarray(data[i]).view("S1"),
                                          written[r, s].to_numpy())
            seen.add((r, s))
    assert seen == {(r, s) for r in range(written.shape[0]) for s in range(written.shape[1])}

def test_no_map_style_access(svar1_multicontig_fixture):
    sds = gvl.StreamingDataset(svar1_multicontig_fixture.bed,
                               reference=svar1_multicontig_fixture.reference_path,
                               variants=svar1_multicontig_fixture.svar_path)
    import pytest
    with pytest.raises(TypeError):
        sds.to_torch_dataset()
    with pytest.raises(TypeError):
        _ = sds[0, 0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity.py -q`
Expected: FAIL (`gvl.StreamingDataset` not exported yet / no `with_seqs`).

- [ ] **Step 3: Implement the public surface**

Add to `_streaming.py`: a public `__init__` overload that classifies `variants` (`.svar` → `_Svar1Backend`; else `NotImplementedError` naming VCF/PGEN/SVAR2 as later plans), `with_seqs(kind)` (only `"haplotypes"` accepted here; others → `NotImplementedError`), `__getitem__` raising `TypeError("StreamingDataset is iterable-only; use to_dataloader()")`, `to_torch_dataset` raising the same, and `to_dataloader` wrapping an `IterableDataset` that iterates `self`. Group `_plan` batches so each Rust call is single-contig (split a batch that straddles a contig boundary). Show the full code for each method.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_parity.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py python/genvarloader/_torch.py tests/dataset/test_streaming_parity.py
git commit -m "feat(streaming): public StreamingDataset (haplotypes/SVAR1) + to_dataloader parity"
```

---

### Task 6: Register the public symbol + docs

**Files:**
- Modify: `python/genvarloader/__init__.py:18` (import) + `__all__` (~line 70)
- Modify: `docs/source/api.md:72` area (autoclass)
- Modify: `skills/genvarloader/SKILL.md`, `docs/source/dataset.md`, `docs/source/faq.md`
- Test: `tests/dataset/test_streaming_parity.py` (extend with the `__all__`/api.md gate)

**Interfaces:**
- Consumes: `StreamingDataset` from `._dataset._streaming`.
- Produces: `gvl.StreamingDataset` public; `api.md` in sync with `__all__`.

- [ ] **Step 1: Write the failing gate**

```python
def test_streamingdataset_is_public_and_documented():
    import genvarloader as gvl, re
    assert "StreamingDataset" in gvl.__all__
    assert hasattr(gvl, "StreamingDataset")
    api = open("docs/source/api.md").read()
    assert "StreamingDataset" in api
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity.py::test_streamingdataset_is_public_and_documented -q`
Expected: FAIL.

- [ ] **Step 3: Register + document**

In `python/genvarloader/__init__.py`: add `StreamingDataset` to the `from ._dataset._streaming import StreamingDataset` import and to `__all__` (alphabetical). In `docs/source/api.md`: add `.. autoclass:: StreamingDataset` with `:members:` near the `Dataset` entry. Add a `SKILL.md` section + `dataset.md`/`faq.md` prose: write-free inference workflow, `.svar` only for now, region-major order, iterable-only, byte-identical to a written dataset, perf tradeoff (slower per-epoch, zero preprocessing).

- [ ] **Step 4: Run the api.md ↔ `__all__` check + the gate**

Run: `python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none`.
Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity.py::test_streamingdataset_is_public_and_documented -q`
Expected: PASS.

- [ ] **Step 5: Full gate + commit**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q && pixi run -e dev cargo-test && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: all green.

```bash
git add python/genvarloader/__init__.py docs/source/api.md skills/genvarloader/SKILL.md docs/source/dataset.md docs/source/faq.md tests/dataset/test_streaming_parity.py
git commit -m "docs(streaming): export StreamingDataset; api.md/SKILL.md/prose for write-free SVAR1 inference"
```

---

## Follow-up plans (not this plan)

- **Plan 2 — double-buffer engine:** replace Task 4/5's synchronous per-window read with the `std::thread` + `crossbeam_channel::bounded` producer/consumer double-buffer (generic `StreamBackend` trait), window sizing from a byte budget, `num_workers` sharding. Pure throughput; parity already locked by this plan's tests.
- **Plan 3 — VCF backend** (`VcfRecordSource`, `vcfixture` parity); **Plan 4 — PGEN backend**.
- **Plan 5 — output-mode breadth:** annotated/variants modes, `with_len`, `min_af`/`max_af`, `var_fields`, `rc_neg`, jitter window.
- **Separate specs:** SVAR2 backend (SVAR2-style buffer + read-bound kernels); intervals/BigWigs + mixed scheduler.
