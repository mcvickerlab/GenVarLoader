# Single-pass streaming bigWig write path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the count-then-read double-decode bigWig write path with a single Rust entry point that decodes each interval once, streams memory-bounded batches to disk, and writes `intervals.npy`/`offsets.npy` directly — proven byte-identical to the current path before the old code is deleted.

**Architecture:** A new Rust function `bigwig::write_track` opens each bigWig once per worker thread (thread-local cache), parallelizes over regions, decodes each `(region, sample)` exactly once, and writes the raw on-disk byte layout directly. Python's `_write_track` and the bigWig branch of the annotation path dispatch into it behind an env switch during parity, then default to it and delete the legacy orchestration in this same PR.

**Tech Stack:** Rust (bigtools, ndarray, rayon, PyO3/numpy), Python (numpy, polars, pyBigWig), pixi, pytest, cargo test.

## Global Constraints

- **Byte-identical parity (hard gate):** new path must produce `intervals.npy` and `offsets.npy` byte-for-byte identical to the legacy path on identical inputs, across the py310–313 × linux/macOS matrix.
- **On-disk format (raw, header-less):** `intervals.npy` = packed `np.dtype([("start", i32), ("end", i32), ("value", f32)], align=True)` (12 bytes, no padding); `offsets.npy` = `i64`. `np.memmap` writes raw bytes — there is **no** `.npy` header.
- **Output ordering:** region-major, sample-minor. Offsets length = `n_regions * n_samples + 1` (per-sample) or `n_regions + 1` (annotation, `sample_less=True`).
- **Emit native interval coords:** write each interval's own `start`/`end`/`value` from bigtools `get_interval`; only the *query* range is clamped (`r_start = start.max(0)`, `r_end = end.min(max_len)`). Contig match: `name == contig || name == format!("chr{contig}")`. No NaN handling in the interval path.
- **Preserve the single hard memory limit:** a single region whose decoded intervals exceed `max_mem` raises (matches the existing `NotImplementedError`; per-sample chunking stays unimplemented).
- **Strangler-fig:** keep legacy path alive behind env switch `GVL_RUST_BIGWIG_WRITE` until parity is green in this PR; then flip the default and delete legacy in the same PR.
- **No public-API change:** `gvl.write` / `gvl.update` signatures and defaults are unchanged; `skills/genvarloader/SKILL.md` is NOT updated.
- **Out of scope:** Table/polars-bio annotation path; numba track realign (`_dataset/_tracks.py`); the rest of Phase 4's variant/genotype kernels.
- **Commands:** build/test via pixi: `pixi run -e dev test` (= `pytest tests && cargo test --release`); single test `pixi run -e dev pytest <path>::<name> -v`; Rust only `pixi run -e dev cargo test --release`; lint `pixi run -e dev ruff check python/ tests/` + `pixi run -e dev ruff format python/ tests/`. Use `rtk` prefix for git per CLAUDE.md.

---

### Task 1: Synthetic bigWig corpus helper

Reproducible synthetic bigWigs over chr21/chr22 used by the parity tests (small) and the bench (large). Deterministic (seeded), no external data.

**Files:**
- Create: `tests/_bigwig_corpus.py`
- Test: `tests/unit/test_bigwig_corpus.py`

**Interfaces:**
- Produces:
  - `make_synthetic_bigwigs(out_dir: Path, n_samples: int, *, contigs: dict[str, int] = {"chr21": 200_000, "chr22": 150_000}, density: float = 0.01, seed: int = 0) -> list[Path]` — writes `sample_{i}.bw` for `i in range(n_samples)`, returns their paths. `density` = fraction of base positions that start an interval. Deterministic given `seed`.
  - `make_regions(contigs: dict[str, int], n_per_contig: int, width: int, *, seed: int = 0) -> "pl.DataFrame"` — returns a polars DataFrame with columns `chrom, chromStart, chromEnd` (int), grouped by contig in `contigs` order (matches how gvl stores regions).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_bigwig_corpus.py
import numpy as np
import pyBigWig

from tests._bigwig_corpus import make_regions, make_synthetic_bigwigs


def test_make_synthetic_bigwigs_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    paths_a = make_synthetic_bigwigs(a, n_samples=2, seed=7)
    paths_b = make_synthetic_bigwigs(b, n_samples=2, seed=7)
    assert [p.name for p in paths_a] == ["sample_0.bw", "sample_1.bw"]
    # byte-identical given same seed
    assert paths_a[0].read_bytes() == paths_b[0].read_bytes()
    # has intervals on both contigs
    with pyBigWig.open(str(paths_a[0])) as bw:
        assert "chr21" in bw.chroms()
        assert len(bw.intervals("chr21")) > 0


def test_make_regions_grouped_in_contig_order(tmp_path):
    regions = make_regions({"chr21": 200_000, "chr22": 150_000}, n_per_contig=4, width=1000, seed=1)
    assert regions.columns == ["chrom", "chromStart", "chromEnd"]
    # contig-grouped in dict order (chr21 block then chr22 block)
    chroms = regions["chrom"].to_list()
    assert chroms == ["chr21"] * 4 + ["chr22"] * 4
    assert (regions["chromEnd"] - regions["chromStart"] == 1000).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_bigwig_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests._bigwig_corpus'`

- [ ] **Step 3: Write minimal implementation**

```python
# tests/_bigwig_corpus.py
"""Reproducible synthetic bigWig corpus for parity tests and benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig

DEFAULT_CONTIGS = {"chr21": 200_000, "chr22": 150_000}


def make_synthetic_bigwigs(
    out_dir: Path,
    n_samples: int,
    *,
    contigs: dict[str, int] = DEFAULT_CONTIGS,
    density: float = 0.01,
    seed: int = 0,
) -> list[Path]:
    """Write `sample_{i}.bw` for i in range(n_samples). Deterministic given `seed`.

    Each contig gets contiguous, non-overlapping intervals: starts are a sorted
    random subset of positions (~`density` fraction), each running to the next start.
    """
    out_dir = Path(out_dir)
    paths: list[Path] = []
    header = [(c, int(length)) for c, length in contigs.items()]
    for i in range(n_samples):
        rng = np.random.default_rng(seed + i)
        path = out_dir / f"sample_{i}.bw"
        with pyBigWig.open(str(path), "w") as bw:
            bw.addHeader(header, maxZooms=0)
            for contig, length in contigs.items():
                n = max(1, int(length * density))
                starts = np.unique(rng.integers(0, length - 1, size=n).astype(np.int64))
                starts.sort()
                ends = np.empty_like(starts)
                ends[:-1] = starts[1:]
                ends[-1] = min(int(starts[-1]) + 1, length)
                # drop any zero-width tail
                keep = ends > starts
                starts, ends = starts[keep], ends[keep]
                values = rng.standard_normal(len(starts)).astype(np.float32)
                bw.addEntries(
                    [contig] * len(starts),
                    [int(s) for s in starts],
                    ends=[int(e) for e in ends],
                    values=[float(v) for v in values],
                )
        paths.append(path)
    return paths


def make_regions(
    contigs: dict[str, int],
    n_per_contig: int,
    width: int,
    *,
    seed: int = 0,
) -> pl.DataFrame:
    """Contig-grouped regions DataFrame (chrom, chromStart, chromEnd)."""
    rng = np.random.default_rng(seed)
    chrom, start, end = [], [], []
    for contig, length in contigs.items():
        starts = rng.integers(0, max(1, length - width), size=n_per_contig)
        for s in starts:
            chrom.append(contig)
            start.append(int(s))
            end.append(int(s) + width)
    return pl.DataFrame(
        {"chrom": chrom, "chromStart": start, "chromEnd": end},
        schema={"chrom": pl.Utf8, "chromStart": pl.Int64, "chromEnd": pl.Int64},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_bigwig_corpus.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
rtk git add tests/_bigwig_corpus.py tests/unit/test_bigwig_corpus.py
rtk git commit -m "test: reproducible synthetic bigWig corpus helper"
```

---

### Task 2: Rust `bigwig::write_track` — single-pass streaming writer

The core: decode-once, region-parallel, thread-local readers, raw byte output. Validated in Rust against the existing `count_intervals`/`intervals` functions (same module) on the test fixtures.

**Files:**
- Modify: `src/bigwig.rs` (add `write_track` + a `#[cfg(test)]` module)

**Interfaces:**
- Consumes: existing `bigwig::count_intervals`, `bigwig::intervals` (for the Rust test oracle only).
- Produces:
  ```rust
  /// Write intervals.npy + offsets.npy directly. Output is region-major, sample-minor.
  /// `sample_less` collapses a single pseudo-sample (offsets length = n_regions + 1).
  pub fn write_track(
      paths: &[PathBuf],
      contigs: &[String],   // per-region normalized contig name, len = n_regions
      starts: ArrayView1<i32>,
      ends: ArrayView1<i32>,
      max_mem: usize,
      out_dir: &Path,
      sample_less: bool,
  ) -> Result<()>;
  ```
  Algorithm:
  - `n_regions = starts.len()`, `n_samples = paths.len()`, `region_bytes = 12` per interval.
  - Process regions in fixed-count batches (`const REGION_BATCH: usize = 512;`). For each batch, `par_iter` over region indices; each task gets a **thread-local** `HashMap<PathBuf, BigWigRead<...>>` (open on first use, reuse across regions/batches), and for that region decodes every sample once into a `Vec<Value>` per sample.
  - For each region, compute decoded bytes = `(sum of per-sample interval counts) * 12`. If a single region's bytes `> max_mem`, return `Err` ("region exceeds max_mem ...").
  - After a batch's parallel decode, write to the two output files **in order** (region-major, sample-minor): append packed `[start_i32_le, end_i32_le, value_f32_le]` per interval to `intervals.npy`; push the running offset for each `(region, sample)` to an in-memory `Vec<i64>`. Use `BufWriter<File>` opened once (create on first batch).
  - After all batches, append the final total offset and write the full `offsets.npy` (`Vec<i64>` of length `n_out_rows + 1`) as raw little-endian bytes. `n_out_rows = n_regions * n_samples` (or `n_regions` if `sample_less`).
  - Emit each interval's native `start`/`end`/`value`; clamp only the query: `r_start = s.max(0) as u32`, `r_end = (e as u32).min(max_len)`.

- [ ] **Step 1: Write the failing Rust test**

Add at the bottom of `src/bigwig.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::fs;

    fn fixture_paths() -> Vec<PathBuf> {
        // tests/data/bigwig/sample_{0,1}.bw (chr1 len 2000, chr2 len 1000)
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/bigwig");
        vec![base.join("sample_0.bw"), base.join("sample_1.bw")]
    }

    #[test]
    fn write_track_matches_count_and_intervals_oracle() {
        let paths = fixture_paths();
        let contigs = vec!["chr1".to_string(), "chr1".to_string()];
        let starts = array![0i32, 50];
        let ends = array![200i32, 110];
        let tmp = std::env::temp_dir().join("gvl_bw_write_test");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();

        write_track(
            &paths,
            &contigs,
            starts.view(),
            ends.view(),
            1 << 30,
            &tmp,
            false,
        )
        .unwrap();

        // Oracle: count_intervals (per contig) + intervals, replicating the Python path.
        // Region 0 and 1 are both on chr1; build expected offsets + packed bytes.
        let n0 = count_intervals(&paths, "chr1", array![0i32, 50].view(), array![200i32, 110].view())
            .unwrap(); // (regions, samples)
        let offsets: Vec<i64> = {
            let mut acc = 0i64;
            let mut v = vec![0i64];
            for r in 0..n0.nrows() {
                for s in 0..n0.ncols() {
                    acc += n0[[r, s]] as i64;
                    v.push(acc);
                }
            }
            v
        };
        let (coords, vals) = unsafe {
            intervals(
                &paths,
                "chr1",
                array![0i32, 50].view(),
                array![200i32, 110].view(),
                ndarray::aview1(&offsets),
            )
        }
        .unwrap();

        // Expected intervals.npy bytes: [i32 start, i32 end, f32 value] per row.
        let mut expected = Vec::new();
        for i in 0..vals.len() {
            expected.extend_from_slice(&(coords[[i, 0]] as i32).to_le_bytes());
            expected.extend_from_slice(&(coords[[i, 1]] as i32).to_le_bytes());
            expected.extend_from_slice(&vals[i].to_le_bytes());
        }
        let got = fs::read(tmp.join("intervals.npy")).unwrap();
        assert_eq!(got, expected, "intervals.npy bytes mismatch");

        // Expected offsets.npy bytes: i64 little-endian, full offsets vec.
        let mut expected_off = Vec::new();
        for o in &offsets {
            expected_off.extend_from_slice(&o.to_le_bytes());
        }
        let got_off = fs::read(tmp.join("offsets.npy")).unwrap();
        assert_eq!(got_off, expected_off, "offsets.npy bytes mismatch");
    }

    #[test]
    fn write_track_errors_when_region_exceeds_max_mem() {
        let paths = fixture_paths();
        let contigs = vec!["chr1".to_string()];
        let starts = array![0i32];
        let ends = array![2000i32];
        let tmp = std::env::temp_dir().join("gvl_bw_write_oom");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&tmp).unwrap();
        // max_mem = 1 byte -> any region with >=1 interval exceeds it
        let res = write_track(&paths, &contigs, starts.view(), ends.view(), 1, &tmp, false);
        assert!(res.is_err());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev cargo test --release write_track`
Expected: compile error (`write_track` not found).

- [ ] **Step 3: Write minimal implementation**

Add to the top of `src/bigwig.rs` (imports) and the function. Add imports:

```rust
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
```

Add the function. **Note on the reader type:** `BigWigRead::open_file` returns a concrete `BigWigRead<...>` whose generic parameter depends on the bigtools version (in the current pin it is `BigWigRead<bigtools::utils::reopen::ReopenableFile>`). Introduce a type alias and let the compiler confirm it — if the alias is wrong, `cargo build` prints the exact expected type in the mismatch error:

```rust
const REGION_BATCH: usize = 512;

// If this alias is wrong for the pinned bigtools, the build error names the real type.
type BwReader = BigWigRead<bigtools::utils::reopen::ReopenableFile>;

thread_local! {
    static READERS: RefCell<HashMap<PathBuf, BwReader>> =
        RefCell::new(HashMap::new());
}

/// Decoded intervals for one region across all samples: per-sample Vec<Value>.
type RegionDecoded = Vec<Vec<Value>>;

pub fn write_track(
    paths: &[PathBuf],
    contigs: &[String],
    starts: ArrayView1<i32>,
    ends: ArrayView1<i32>,
    max_mem: usize,
    out_dir: &Path,
    _sample_less: bool,
) -> Result<()> {
    let n_regions = starts.len();
    let n_samples = paths.len();
    let starts = starts.as_slice().expect("starts contiguous");
    let ends = ends.as_slice().expect("ends contiguous");

    let mut itv_writer = BufWriter::new(File::create(out_dir.join("intervals.npy"))?);
    // offsets accumulated in memory; region-major, sample-minor; final total appended.
    let mut offsets: Vec<i64> = Vec::with_capacity(n_regions * n_samples + 1);
    offsets.push(0);
    let mut acc: i64 = 0;

    let mut batch_start = 0usize;
    while batch_start < n_regions {
        let batch_end = (batch_start + REGION_BATCH).min(n_regions);
        let batch: Vec<usize> = (batch_start..batch_end).collect();

        // Parallel decode each region (all samples), preserving order via collect.
        let decoded: Vec<Result<RegionDecoded>> = batch
            .par_iter()
            .map(|&r| {
                READERS.with(|cell| {
                    let mut readers = cell.borrow_mut();
                    let contig = &contigs[r];
                    let mut per_sample: RegionDecoded = Vec::with_capacity(n_samples);
                    for path in paths.iter() {
                        let bw = readers
                            .entry(path.clone())
                            .or_insert_with(|| {
                                BigWigRead::open_file(path).expect("Error opening file")
                            });
                        let (max_len, name) = bw
                            .chroms()
                            .iter()
                            .filter_map(|c| {
                                if &c.name == contig || c.name == format!("chr{contig}") {
                                    Some((c.length, c.name.clone()))
                                } else {
                                    None
                                }
                            })
                            .exactly_one()
                            .expect("Contig not found or multiple contigs match");
                        let r_start = starts[r].max(0) as u32;
                        let r_end = (ends[r] as u32).min(max_len);
                        let vals: Vec<Value> = bw
                            .get_interval(name.as_str(), r_start, r_end)
                            .expect("Begin reading intervals")
                            .into_iter()
                            .map(|v| v.expect("Read interval"))
                            .collect();
                        per_sample.push(vals);
                    }
                    let region_bytes: usize =
                        per_sample.iter().map(|v| v.len()).sum::<usize>() * 12;
                    if region_bytes > max_mem {
                        anyhow::bail!(
                            "Memory usage per region exceeds max_mem ({} > {}).",
                            region_bytes,
                            max_mem
                        );
                    }
                    Ok(per_sample)
                })
            })
            .collect();

        for region in decoded {
            let per_sample = region?;
            for sample_vals in per_sample {
                for v in sample_vals {
                    itv_writer.write_all(&(v.start as i32).to_le_bytes())?;
                    itv_writer.write_all(&(v.end as i32).to_le_bytes())?;
                    itv_writer.write_all(&v.value.to_le_bytes())?;
                    acc += 1;
                }
                offsets.push(acc);
            }
        }
        batch_start = batch_end;
    }
    itv_writer.flush()?;

    let mut off_writer = BufWriter::new(File::create(out_dir.join("offsets.npy"))?);
    for o in &offsets {
        off_writer.write_all(&o.to_le_bytes())?;
    }
    off_writer.flush()?;
    Ok(())
}
```

Note: `_sample_less` is accepted for the binding's symmetry; when the caller passes a single pseudo-sample path (`n_samples == 1`) the offsets length is naturally `n_regions + 1`, matching the annotation layout. No separate branch needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev cargo test --release write_track`
Expected: PASS (`write_track_matches_count_and_intervals_oracle`, `write_track_errors_when_region_exceeds_max_mem`).

- [ ] **Step 5: Commit**

```bash
rtk git add src/bigwig.rs
rtk git commit -m "feat: rust single-pass streaming bigWig write_track"
```

---

### Task 3: PyO3 binding `bigwig_write_track`

**Files:**
- Modify: `src/lib.rs`
- Test: `tests/unit/test_bigwig_write_binding.py`

**Interfaces:**
- Consumes: `bigwig::write_track`.
- Produces (Python): `genvarloader.genvarloader.bigwig_write_track(paths: list[str], contigs: list[str], starts: NDArray[int32], ends: NDArray[int32], max_mem: int, out_dir: str, sample_less: bool) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_bigwig_write_binding.py
from pathlib import Path

import numpy as np

from genvarloader._ragged import INTERVAL_DTYPE
from genvarloader.genvarloader import bigwig_write_track


def test_bigwig_write_binding_roundtrip(tmp_path):
    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    paths = [str(data_dir / "sample_0.bw"), str(data_dir / "sample_1.bw")]
    contigs = ["chr1", "chr1"]
    starts = np.array([0, 50], dtype=np.int32)
    ends = np.array([200, 110], dtype=np.int32)
    out = tmp_path
    bigwig_write_track(paths, contigs, starts, ends, 1 << 30, str(out), False)

    itvs = np.memmap(out / "intervals.npy", dtype=INTERVAL_DTYPE, mode="r")
    offsets = np.memmap(out / "offsets.npy", dtype=np.int64, mode="r")
    # 2 regions x 2 samples -> offsets length 5
    assert len(offsets) == 2 * 2 + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(itvs)
    assert itvs.dtype == INTERVAL_DTYPE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_bigwig_write_binding.py -v`
Expected: FAIL with `ImportError: cannot import name 'bigwig_write_track'`

- [ ] **Step 3: Write minimal implementation**

In `src/lib.rs`, register the function in the `#[pymodule]`:

```rust
m.add_function(wrap_pyfunction!(bigwig_write_track, m)?)?;
```

And add (uses `PyReadonlyArray1`, `PathBuf` already imported; add `use std::path::Path;` if needed):

```rust
/// Write intervals.npy + offsets.npy for a bigWig track directly to `out_dir`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bigwig_write_track(
    paths: Vec<PathBuf>,
    contigs: Vec<String>,
    starts: PyReadonlyArray1<i32>,
    ends: PyReadonlyArray1<i32>,
    max_mem: usize,
    out_dir: PathBuf,
    sample_less: bool,
) -> PyResult<()> {
    bigwig::write_track(
        &paths,
        &contigs,
        starts.as_array(),
        ends.as_array(),
        max_mem,
        &out_dir,
        sample_less,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_bigwig_write_binding.py -v`
(The pixi test task rebuilds the Rust extension via maturin automatically.)
Expected: PASS

- [ ] **Step 5: Commit**

```bash
rtk git add src/lib.rs tests/unit/test_bigwig_write_binding.py
rtk git commit -m "feat: PyO3 binding for bigwig_write_track"
```

---

### Task 4: Route per-sample bigWig tracks through Rust (behind switch)

Refactor `_write_track` to dispatch: `BigWigs` + `GVL_RUST_BIGWIG_WRITE` truthy → new Rust writer; otherwise the existing legacy body. Non-`BigWigs` tracks (e.g. Table) always use legacy.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (rename current `_write_track` body to `_write_track_legacy`; add `_write_track_rust`; add a thin `_write_track` dispatcher; add `_rust_bigwig_write_enabled()` helper)
- Test: `tests/unit/dataset/test_write_track_dispatch.py`

**Interfaces:**
- Consumes: `genvarloader.genvarloader.bigwig_write_track` (Task 3); `BigWigs` (from `.._bigwig`); `normalize_contig_name` (already imported in `_write.py`).
- Produces:
  - `_write_track_legacy(out_dir, bed, track, samples, max_mem)` — the current implementation, unchanged.
  - `_write_track_rust(out_dir, bed: pl.DataFrame, track: "BigWigs", samples: list[str], max_mem: int) -> None` — builds ordered sample paths + per-region normalized contig list and calls the binding.
  - `_write_track(out_dir, bed, track, samples, max_mem)` — dispatcher.
  - `_rust_bigwig_write_enabled() -> bool` — reads `os.environ.get("GVL_RUST_BIGWIG_WRITE")` (truthy = `"1"`, `"true"`, `"yes"`, case-insensitive).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/dataset/test_write_track_dispatch.py
from pathlib import Path

import numpy as np

from genvarloader import BigWigs
from genvarloader._dataset import _write
from genvarloader._ragged import INTERVAL_DTYPE


def _track(data_dir: Path) -> BigWigs:
    return BigWigs(
        "signal",
        {
            "sample_0": str(data_dir / "sample_0.bw"),
            "sample_1": str(data_dir / "sample_1.bw"),
        },
    )


def test_write_track_rust_writes_files(tmp_path):
    import polars as pl

    data_dir = Path(__file__).parents[2] / "data" / "bigwig"
    track = _track(data_dir)
    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 50], "chromEnd": [200, 110]}
    )
    out = tmp_path / "signal"
    out.mkdir()
    _write._write_track_rust(out, bed, track, ["sample_0", "sample_1"], 1 << 30)
    itvs = np.memmap(out / "intervals.npy", dtype=INTERVAL_DTYPE, mode="r")
    offsets = np.memmap(out / "offsets.npy", dtype=np.int64, mode="r")
    assert len(offsets) == 2 * 2 + 1
    assert offsets[-1] == len(itvs)


def test_dispatch_env_off_uses_legacy(monkeypatch):
    monkeypatch.delenv("GVL_RUST_BIGWIG_WRITE", raising=False)
    assert _write._rust_bigwig_write_enabled() is False
    monkeypatch.setenv("GVL_RUST_BIGWIG_WRITE", "1")
    assert _write._rust_bigwig_write_enabled() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_write_track_dispatch.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_write_track_rust'`

- [ ] **Step 3: Write minimal implementation**

In `python/genvarloader/_dataset/_write.py`:

1. Add near the top imports: `import os` (if not present).
2. Rename the existing `def _write_track(` to `def _write_track_legacy(` (body unchanged).
3. Add the helper + rust writer + dispatcher:

```python
def _rust_bigwig_write_enabled() -> bool:
    return os.environ.get("GVL_RUST_BIGWIG_WRITE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _write_track_rust(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "BigWigs",
    samples: list[str],
    max_mem: int,
) -> None:
    from .genvarloader import bigwig_write_track

    out_dir.mkdir(parents=True, exist_ok=True)
    # ordered sample paths (dataset/sample order)
    paths = [track.paths[s] for s in samples]
    # per-region normalized contig name, in bed row order (bed is contig-grouped)
    contigs: list[str] = []
    starts_l: list[int] = []
    ends_l: list[int] = []
    for chrom, s, e in zip(
        bed["chrom"].to_list(),
        bed["chromStart"].to_list(),
        bed["chromEnd"].to_list(),
    ):
        norm = normalize_contig_name(chrom, track.contigs)
        if norm is None:
            raise ValueError(
                f"Contig {chrom!r} not found in bigWig track {track.name!r}."
            )
        contigs.append(norm)
        starts_l.append(int(s))
        ends_l.append(int(e))
    bigwig_write_track(
        paths,
        contigs,
        np.asarray(starts_l, dtype=np.int32),
        np.asarray(ends_l, dtype=np.int32),
        int(max_mem),
        str(out_dir),
        False,
    )


def _write_track(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "IntervalTrack",
    samples: list[str] | None,
    max_mem: int,
):
    from .._bigwig import BigWigs

    if isinstance(track, BigWigs) and _rust_bigwig_write_enabled():
        _samples = samples if samples is not None else track.samples
        if missing := (set(_samples) - set(track.samples)):
            raise ValueError(f"Samples {missing} not found in track.")
        return _write_track_rust(out_dir, bed, track, _samples, max_mem)
    return _write_track_legacy(out_dir, bed, track, samples, max_mem)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/test_write_track_dispatch.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/unit/dataset/test_write_track_dispatch.py
rtk git commit -m "feat: route per-sample bigWig writes through rust behind GVL_RUST_BIGWIG_WRITE"
```

---

### Task 5: Route annotation bigWig tracks through Rust (behind switch)

`_write_annot_track` currently calls `_annot_intervals` → (bigWig branch) `_annot_intervals_from_bigwig` → returns `RaggedIntervals` → `_write_ragged_intervals`. Add a Rust branch that writes files directly with `sample_less=True`.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (`_write_annot_track` gains a bigWig+switch branch)
- Test: `tests/unit/test_write_annot_bigwig.py` (add a case)

**Interfaces:**
- Consumes: `genvarloader.genvarloader.bigwig_write_track`; `_rust_bigwig_write_enabled` (Task 4); `BigWigs`.
- Produces: `_write_annot_track_rust(out_dir, regions: pl.DataFrame, path: Path, max_mem: int) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_write_annot_bigwig.py
import numpy as np
import polars as pl

from genvarloader._dataset import _write
from genvarloader._ragged import INTERVAL_DTYPE


def test_write_annot_track_rust_byte_matches_legacy(tmp_path):
    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    bw = data_dir / "sample_0.bw"
    regions = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 50], "chromEnd": [200, 110]}
    )

    legacy_dir = tmp_path / "legacy"
    rust_dir = tmp_path / "rust"
    legacy_dir.mkdir()
    rust_dir.mkdir()

    # legacy
    itvs = _write._annot_intervals(regions, bw, max_mem=2**30)
    _write._write_ragged_intervals(legacy_dir, itvs)
    # rust
    _write._write_annot_track_rust(rust_dir, regions, bw, max_mem=2**30)

    assert (legacy_dir / "intervals.npy").read_bytes() == (
        rust_dir / "intervals.npy"
    ).read_bytes()
    assert (legacy_dir / "offsets.npy").read_bytes() == (
        rust_dir / "offsets.npy"
    ).read_bytes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/test_write_annot_bigwig.py::test_write_annot_track_rust_byte_matches_legacy -v`
Expected: FAIL with `AttributeError: ... '_write_annot_track_rust'`

- [ ] **Step 3: Write minimal implementation**

In `python/genvarloader/_dataset/_write.py`, add the rust annot writer and branch `_write_annot_track`:

```python
def _write_annot_track_rust(
    out_dir: Path,
    regions: pl.DataFrame,
    path: Path,
    max_mem: int,
) -> None:
    from .._bigwig import BigWigs
    from .genvarloader import bigwig_write_track

    out_dir.mkdir(parents=True, exist_ok=True)
    bw = BigWigs(name="__annot__", paths={"__annot__": str(path)})
    contigs: list[str] = []
    starts_l: list[int] = []
    ends_l: list[int] = []
    for chrom, s, e in zip(
        regions["chrom"].to_list(),
        regions["chromStart"].to_list(),
        regions["chromEnd"].to_list(),
    ):
        norm = normalize_contig_name(chrom, bw.contigs)
        if norm is None:
            raise ValueError(f"Contig {chrom!r} not found in bigWig {path}.")
        contigs.append(norm)
        starts_l.append(int(s))
        ends_l.append(int(e))
    bigwig_write_track(
        [str(path)],
        contigs,
        np.asarray(starts_l, dtype=np.int32),
        np.asarray(ends_l, dtype=np.int32),
        int(max_mem),
        str(out_dir),
        True,
    )
```

Then in `_write_annot_track`, branch before the legacy body:

```python
def _write_annot_track(
    out_dir: Path,
    regions: pl.DataFrame,
    source: "str | Path | pl.DataFrame | pl.LazyFrame",
    max_mem: int,
) -> None:
    if (
        _rust_bigwig_write_enabled()
        and isinstance(source, (str, Path))
        and Path(source).suffix.lower() in (".bw", ".bigwig")
    ):
        return _write_annot_track_rust(out_dir, regions, Path(source), max_mem)
    itvs = _annot_intervals(regions, source, max_mem)
    _write_ragged_intervals(out_dir, itvs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/test_write_annot_bigwig.py -v`
Expected: PASS (existing + new test)

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/unit/test_write_annot_bigwig.py
rtk git commit -m "feat: route annotation bigWig writes through rust behind switch"
```

---

### Task 6: Differential parity test on the synthetic corpus

Byte-identical legacy-vs-rust over a multi-sample track AND an annotation track, on the chr21/chr22 synthetic corpus. This is the **landing gate**; it is removed in Task 8 when legacy is deleted.

**Files:**
- Create: `tests/integration/dataset/test_bigwig_write_parity.py`

**Interfaces:**
- Consumes: `tests._bigwig_corpus.make_synthetic_bigwigs`, `make_regions`; `_write._write_track_legacy`, `_write._write_track_rust`, `_write._annot_intervals`, `_write._write_ragged_intervals`, `_write._write_annot_track_rust`; `BigWigs`.

- [ ] **Step 1: Write the test (this is the parity gate; no separate "minimal impl")**

```python
# tests/integration/dataset/test_bigwig_write_parity.py
from pathlib import Path

import pytest

from genvarloader import BigWigs
from genvarloader._dataset import _write
from tests._bigwig_corpus import DEFAULT_CONTIGS, make_regions, make_synthetic_bigwigs


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("bw_corpus")
    paths = make_synthetic_bigwigs(d, n_samples=3, density=0.02, seed=11)
    regions = make_regions(DEFAULT_CONTIGS, n_per_contig=20, width=5000, seed=3)
    return paths, regions


def _assert_byte_identical(a: Path, b: Path):
    assert (a / "intervals.npy").read_bytes() == (b / "intervals.npy").read_bytes()
    assert (a / "offsets.npy").read_bytes() == (b / "offsets.npy").read_bytes()


def test_per_sample_parity(corpus, tmp_path):
    paths, regions = corpus
    samples = [f"sample_{i}" for i in range(len(paths))]
    track = BigWigs("signal", {s: str(p) for s, p in zip(samples, paths)})

    legacy = tmp_path / "legacy"
    rust = tmp_path / "rust"
    legacy.mkdir()
    rust.mkdir()
    _write._write_track_legacy(legacy, regions, track, samples, 1 << 30)
    _write._write_track_rust(rust, regions, track, samples, 1 << 30)
    _assert_byte_identical(legacy, rust)


def test_annotation_parity(corpus, tmp_path):
    paths, regions = corpus
    legacy = tmp_path / "legacy"
    rust = tmp_path / "rust"
    legacy.mkdir()
    rust.mkdir()
    itvs = _write._annot_intervals(regions, paths[0], max_mem=1 << 30)
    _write._write_ragged_intervals(legacy, itvs)
    _write._write_annot_track_rust(rust, regions, paths[0], 1 << 30)
    _assert_byte_identical(legacy, rust)
```

- [ ] **Step 2: Run the parity test**

Run: `pixi run -e dev pytest tests/integration/dataset/test_bigwig_write_parity.py -v`
Expected: PASS (both tests). If it fails, the new path diverges from legacy — debug with `superpowers:systematic-debugging` (compare `intervals.npy`/`offsets.npy` element-by-element) before proceeding.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/integration/dataset/test_bigwig_write_parity.py
rtk git commit -m "test: byte-identical legacy-vs-rust bigWig write parity gate"
```

---

### Task 7: Baselines, bench corpus, and profiling hooks

Capture `write()`/`update()` wall-clock + peak RSS (legacy vs rust) and wire a profiling driver. Provides the roadmap baseline numbers and the after-numbers.

**Files:**
- Create: `tests/benchmarks/data/build_bigwig_corpus.py` (large reproducible corpus)
- Create: `tests/benchmarks/profiling/profile_bigwig_write.py` (bench/profile driver)
- Create: `scripts/profile_bigwig_write.sh` (py-spy commands for David to run with sudo on macOS)
- Modify: `pixi.toml` (add `memray-bigwig-write` task; py-spy stays in the handed-off script)

**Interfaces:**
- Consumes: `tests._bigwig_corpus`; `genvarloader.write`.

- [ ] **Step 1: Write the bench corpus builder**

```python
# tests/benchmarks/data/build_bigwig_corpus.py
"""Build a large reproducible bigWig corpus for write/update benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

from tests._bigwig_corpus import DEFAULT_CONTIGS, make_synthetic_bigwigs

OUT = Path(__file__).resolve().parent / "bigwig_corpus"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--density", type=float, default=0.05)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = make_synthetic_bigwigs(
        OUT, n_samples=args.n_samples, contigs=DEFAULT_CONTIGS, density=args.density
    )
    print(f"wrote {len(paths)} bigWigs to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the bench/profile driver**

```python
# tests/benchmarks/profiling/profile_bigwig_write.py
"""Time + measure gvl.write() for a bigWig track (legacy vs rust).

  pixi run -e dev python tests/benchmarks/profiling/profile_bigwig_write.py --impl rust
  pixi run -e dev python tests/benchmarks/profiling/profile_bigwig_write.py --impl legacy

Set GVL_RUST_BIGWIG_WRITE via --impl. Reports wall-clock; run under memray for RSS.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "data" / "bigwig_corpus"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["legacy", "rust"], required=True)
    p.add_argument("--n-regions", type=int, default=2000)
    args = p.parse_args()
    os.environ["GVL_RUST_BIGWIG_WRITE"] = "1" if args.impl == "rust" else "0"

    if not CORPUS.exists():
        raise SystemExit(
            "Corpus missing. Run "
            "`pixi run -e dev python tests/benchmarks/data/build_bigwig_corpus.py`."
        )

    import tempfile

    import genvarloader as gvl
    from tests._bigwig_corpus import DEFAULT_CONTIGS, make_regions

    paths = sorted(CORPUS.glob("sample_*.bw"))
    samples = [p.stem for p in paths]
    track = gvl.BigWigs("signal", {s: str(p) for s, p in zip(samples, paths)})
    per_contig = max(1, args.n_regions // len(DEFAULT_CONTIGS))
    bed = make_regions(DEFAULT_CONTIGS, n_per_contig=per_contig, width=5000, seed=0)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "signal"
        out.mkdir()
        t0 = time.perf_counter()
        from genvarloader._dataset._write import _write_track

        _write_track(out, bed, track, samples, 4 << 30)
        dt = time.perf_counter() - t0
    print(f"impl={args.impl} regions={bed.height} samples={len(samples)} wall={dt:.3f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the py-spy handoff script**

```bash
# scripts/profile_bigwig_write.sh
#!/usr/bin/env bash
# py-spy needs sudo on macOS. Run this yourself; do not let the agent invoke py-spy.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=tests/benchmarks/profiling
pixi run -e dev python tests/benchmarks/data/build_bigwig_corpus.py
for impl in legacy rust; do
  sudo py-spy record -o "$OUT/bigwig_write_$impl.speedscope.json" -f speedscope -- \
    pixi run -e dev python "$OUT/profile_bigwig_write.py" --impl "$impl"
done
echo "wrote $OUT/bigwig_write_{legacy,rust}.speedscope.json"
```

Make it executable: `chmod +x scripts/profile_bigwig_write.sh`.

- [ ] **Step 4: Add the memray pixi task**

In `pixi.toml`, alongside the existing `memray-*` tasks, add:

```toml
memray-bigwig-write = { cmd = "memray run -fo tests/benchmarks/profiling/bigwig_write.memray.bin tests/benchmarks/profiling/profile_bigwig_write.py --impl rust" }
```

- [ ] **Step 5: Capture baselines and verify the driver runs**

Run:
```bash
pixi run -e dev python tests/benchmarks/data/build_bigwig_corpus.py
pixi run -e dev python tests/benchmarks/profiling/profile_bigwig_write.py --impl legacy
pixi run -e dev python tests/benchmarks/profiling/profile_bigwig_write.py --impl rust
```
Expected: two `wall=...s` lines. Record both numbers and the RSS from `pixi run -e dev memray-bigwig-write` (then `memray stats`). These are the baseline (legacy) + after (rust) figures for the roadmap. Hand `scripts/profile_bigwig_write.sh` to David for the py-spy flame graphs (macOS sudo).

- [ ] **Step 6: Commit**

```bash
rtk git add tests/benchmarks/data/build_bigwig_corpus.py tests/benchmarks/profiling/profile_bigwig_write.py scripts/profile_bigwig_write.sh pixi.toml
rtk git commit -m "bench: bigWig write corpus, profiling driver, py-spy handoff"
```

---

### Task 8: Flip default to Rust, delete legacy, update roadmap

Make Rust the unconditional bigWig write path, delete the legacy orchestration + the env switch + the transitional parity test, and update the migration roadmap. Existing e2e/snapshot tests (`tests/integration/dataset/test_write_tracks_e2e.py`, the #233 track snapshots, `tests/unit/test_write_annot_bigwig.py`) now exercise the Rust path and provide durable coverage.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (remove dispatch/legacy/switch for bigWig)
- Delete: `tests/integration/dataset/test_bigwig_write_parity.py`, `tests/unit/dataset/test_write_track_dispatch.py`
- Modify: `docs/roadmaps/rust-migration.md`

- [ ] **Step 1: Collapse the dispatcher to Rust-only for bigWig**

In `_write.py`:
- In `_write_track`, drop the `_rust_bigwig_write_enabled()` guard so `BigWigs` always uses `_write_track_rust`; keep `_write_track_legacy` ONLY if a non-bigWig `IntervalTrack` (e.g. Table) still routes through it — Table tracks DO, so keep `_write_track_legacy` for the non-`BigWigs` branch but remove the bigWig env gate:

```python
def _write_track(out_dir, bed, track, samples, max_mem):
    from .._bigwig import BigWigs

    if isinstance(track, BigWigs):
        _samples = samples if samples is not None else track.samples
        if missing := (set(_samples) - set(track.samples)):
            raise ValueError(f"Samples {missing} not found in track.")
        return _write_track_rust(out_dir, bed, track, _samples, max_mem)
    return _write_track_legacy(out_dir, bed, track, samples, max_mem)
```

- In `_write_annot_track`, drop the `_rust_bigwig_write_enabled()` guard (bigWig branch always rust); keep the Table/DataFrame branch on `_annot_intervals` + `_write_ragged_intervals`.
- Remove `_rust_bigwig_write_enabled` (now unused). Keep `_write_track_legacy` (still used by Table tracks).

- [ ] **Step 2: Delete the transitional tests**

```bash
rtk git rm tests/integration/dataset/test_bigwig_write_parity.py tests/unit/dataset/test_write_track_dispatch.py
```

Note: `test_bigwig_write_binding.py` (Task 3) and `test_write_annot_bigwig.py` (Task 5) stay — they don't depend on legacy.

- [ ] **Step 3: Run the full suite to confirm durable coverage holds**

Run: `pixi run -e dev pytest tests tests/unit -q` then `pixi run -e dev cargo test --release`
Expected: PASS. The e2e/snapshot track tests now run through Rust. (Full tree per CLAUDE.md since shared write code changed.)

- [ ] **Step 4: Lint/format**

Run: `pixi run -e dev ruff check python/ tests/` and `pixi run -e dev ruff format python/ tests/`
Expected: clean (format hook gate per `ruff_format_hook`).

- [ ] **Step 5: Update the roadmap**

In `docs/roadmaps/rust-migration.md`:
- Set Phase 4 marker to 🚧 with this PR link.
- Tick the bigWig-relevant Phase 4 item (interval extraction for the write path).
- Fill the baseline table rows for `gvl.write()` wall-clock + peak RSS (legacy = baseline; rust = after) from Task 7, noting the synthetic chr21/chr22 corpus and config (`n_samples`, `density`, `n_regions`).
- Add a dated entry to the "Notes & decisions log": single-pass streaming bigWig writer; rust-vs-legacy byte-identical; bench config + measured speedup/RSS.

- [ ] **Step 6: Commit**

```bash
rtk git add -A
rtk git commit -m "refactor: make rust the default bigWig write path; delete legacy + switch

Flips GVL_RUST_BIGWIG_WRITE default on and removes it. Legacy orchestration
retained only for non-bigWig (Table) tracks. Roadmap Phase 4 bigWig slice updated
with baseline + after numbers.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **If parity (Task 6) fails:** the most likely divergences are (a) interval *ordering* (must be region-major, sample-minor; sample order = the `samples` list order), (b) writing clamped query coords instead of the interval's native `start`/`end`, or (c) a contig-normalization mismatch between Python (`normalize_contig_name`) and the Rust `chr{contig}` match. Compare the two `intervals.npy` as structured arrays element-by-element to localize.
- **The pixi `test` task auto-rebuilds the Rust extension** (maturin). After editing `src/*.rs`, just re-run the relevant pytest/cargo task.
- **Do not invoke py-spy yourself** — hand `scripts/profile_bigwig_write.sh` to David (macOS sudo requirement).
- **`Dataset.__getitem__` is untouched** by this work; only `write`/`update` baseline rows are in scope.
