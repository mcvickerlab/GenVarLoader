# Zero-copy, scale-safe Rust read path (gvl format 2.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate per-batch materialization of per-sample-scale memmaps at the Python→Rust boundary, cache only the truly-static sub-linear arrays, and skip provably-unnecessary zero-init — all byte-identical to current output — gated behind a `format_version` 1.0.0 → 2.0.0 bump with an explicit `gvl.migrate`.

**Architecture:** One breaking on-disk change converts track-interval storage from array-of-structs (`INTERVAL_DTYPE`, itemsize 12, strided field views) to struct-of-arrays (three contiguous files `starts.npy`/`ends.npy`/`values.npy` sharing the existing `offsets.npy`). Contiguous memmaps then cross the FFI boundary zero-copy, replacing the `np.ascontiguousarray(...)` calls that copied the whole per-sample-scale interval store every batch. A loud boundary guard (`_ffi_array`) replaces silent materialization; sub-linear per-variant arrays are cached once per reconstructor; and fully-overwritten Rust output buffers drop their zero-init.

**Tech Stack:** Python 3.10+, NumPy, Polars, Rust (PyO3/ndarray/bigtools/coitrees), Maturin, pytest + cargo test, pixi.

## Global Constraints

- **Byte-identical parity is the landing gate.** Every change is layout/marshalling only; output bytes are unchanged. Verified across `GVL_BACKEND=rust` and `GVL_BACKEND=numba` via `tests/parity` plus the dataset/unit/integration suites.
- **Public API delta is exactly:** add `migrate` to `python/genvarloader/__init__.py` `__all__`; bump `DATASET_FORMAT_VERSION` to `2.0.0`. No other public signature changes. Per `CLAUDE.md`, this requires a `skills/genvarloader/SKILL.md` update (Task 7).
- **No new perf gate.** Throughput is recorded in the roadmap, not gated. The one hard new gate is the **scale-guard** test (Task 4): no memmap-materializing copy on the read path.
- **Commands run under pixi:** `pixi run -e dev <task>`. After any Rust change, rebuild the extension with `pixi run -e dev maturin develop --release` before running Python tests. Dataset/parity tests need `--basetemp=$(pwd)/.pytest_tmp` (Carter `os.link` Errno 18). Prefix shell commands with `rtk`.
- **Lint/format/typecheck scope:** `pixi run -e dev ruff check python/ tests/`, `pixi run -e dev ruff format python/ tests/`, `pixi run -e dev typecheck`. Rust: `pixi run -e dev cargo clippy`, `cargo test`.
- **Merge style:** merge commit, never squash. Work on branch `zero-copy-scale-safe-readpath` (off `rust-migration`, after #245/#246 closed out `phase-3-reconstruction`).
- **No committed `.gvl` fixtures exist** (verified: `git ls-files` shows only build scripts under `tests/benchmarks/data/`, no on-disk datasets). All test datasets are generated through `gvl.write`, so after Task 1 every freshly-built dataset is born 2.0.0/SoA — the version gate (Task 2) cannot break the committed suite. The migration test (Task 3) synthesizes its own 1.x AoS dataset.

---

## File-Touch Map

| File | Change | Task |
|---|---|---|
| `python/genvarloader/_dataset/_write.py` | `DATASET_FORMAT_VERSION` → 2.0.0; SoA writers (`_write_ragged_intervals`, `_write_track_legacy` chunked); `_check_dataset_format_version` helper | 1, 2 |
| `python/genvarloader/_dataset/_tracks.py` | `_open_intervals` memmaps three contiguous arrays; drop `INTERVAL_DTYPE` import | 1 |
| `src/bigwig.rs` | `write_track` emits SoA; update oracle byte test | 1 |
| `src/tables.rs` | `write_track_impl` emits SoA; update oracle byte test | 1 |
| `python/genvarloader/_dataset/_open.py` | call `_check_dataset_format_version` in `_load_metadata` | 2 |
| `python/genvarloader/_dataset/_migrate.py` (new) | `migrate()` streaming in-place AoS→SoA | 3 |
| `python/genvarloader/__init__.py` | export `migrate` in `__all__` | 3 |
| `python/genvarloader/_dataset/_utils.py` | `_ffi_array(arr, dtype, name)` boundary helper | 4 |
| `python/genvarloader/_dataset/_reconstruct.py` | drop `ascontiguousarray` on sample-scale args; apply `_ffi_array` | 4 |
| `python/genvarloader/_dataset/_haps.py` | same for fused haps/annotated/splice calls; cache sub-linear arrays (Task 5) | 4, 5 |
| `src/ffi/mod.rs` | uninitialized output allocation in the fused kernels | 6 |
| `tests/integration/conftest.py` (new) | `track_dataset_path` fixture | 1 |
| `tests/integration/test_format_2_soa.py` (new) | SoA round-trip | 1 |
| `tests/integration/test_format_version_gate.py` (new) | version gate | 2 |
| `tests/integration/test_migrate.py` (new) | migration round-trip / idempotency / interruption | 3 |
| `tests/integration/test_scale_guard.py` (new) | no-memmap-copy gate | 4 |
| `tests/unit/dataset/test_ffi_array.py` (new) | `_ffi_array` guard | 4 |
| `tests/unit/dataset/test_haps_ffi_cache.py` (new) | sub-linear cache | 5 |
| `skills/genvarloader/SKILL.md` | document `migrate` + format 2.0 open behavior | 7 |
| `docs/roadmaps/rust-migration.md` | mark targets addressed; record throughput | 7 |

---

## Background facts the implementer needs

- **`.npy` files here are headerless raw little-endian bytes.** The writers stream raw `to_le_bytes()` / `np.memmap`; the reader memmaps with an explicit `dtype`. There is no numpy `.npy` magic header. SoA = three raw files of the same length (number of intervals), all 4 bytes per element (`int32`, `int32`, `float32`), sharing one `int64` `offsets.npy`.
- **`INTERVAL_DTYPE`** (`python/genvarloader/_ragged.py:26`) `= np.dtype([("start", i4), ("end", i4), ("value", f4)], align=True)`, itemsize 12. After Task 1 it is no longer on the read or born-write path; it survives only for the migration reader (Task 3) and any in-memory record construction. (A second, unused copy exists at `python/genvarloader/_types.py:18`; it is not imported anywhere — leave it untouched, out of scope.)
- **Four interval writers feed the same on-disk layout:** `_write_ragged_intervals` (Python, annotation/table single-chunk), `_write_track_legacy` (Python, chunked sample tracks), `bigwig.rs::write_track` (Rust, BigWig tracks via `_write_track_rust`), `tables.rs::write_track_impl` (Rust, table tracks via `_write_track_table`). **All four** must emit SoA in Task 1, or datasets written by the path you missed will be unreadable by the new reader.
- **`_as_starts_stops`** (`_genotypes.py:119`) builds a fresh contiguous `(2, n)` array via `np.stack`; its output `.base` is not a memmap, so it never trips the scale-guard. Leave it and the `_geno_offsets_2d` precompute (`_reconstruct.py:198`) unchanged.

---

## Task 1: AoS → SoA interval storage + `format_version` 2.0.0 (Component A)

The single breaking change. Flips all four writers and the one reader together (a partial flip is not independently green) and bumps the format version. Atomic deliverable: a freshly-written dataset stores SoA and reads back byte-identically.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (`DATASET_FORMAT_VERSION` `:44`; `_write_ragged_intervals` `:1085-1108`; `_write_track_legacy` chunked block `:1322-1334`)
- Modify: `python/genvarloader/_dataset/_tracks.py` (`_open_intervals` `:706-725`; `INTERVAL_DTYPE` import `:18`)
- Modify: `src/bigwig.rs` (`write_track` `:26-126`; oracle test `:319-335`)
- Modify: `src/tables.rs` (`write_track_impl` `:161-224`; oracle test `:453-467`)
- Create: `tests/integration/conftest.py`
- Create: `tests/integration/test_format_2_soa.py`

**Interfaces:**
- Produces (on-disk, per track dir under `intervals/<track>/` and `annot_intervals/<track>/`):
  - `starts.npy` — raw `int32`, contiguous, length = total intervals
  - `ends.npy` — raw `int32`, contiguous
  - `values.npy` — raw `float32`, contiguous
  - `offsets.npy` — raw `int64`, **unchanged** (length n+1)
- Produces: `DATASET_FORMAT_VERSION == SemanticVersion.parse("2.0.0")`
- Produces (test): `track_dataset_path` fixture → `Path` to a freshly-written 2.0 dataset with a phased VCF + one BigWig `"cov"` track.
- Consumes: existing `RaggedIntervals` (`_ragged.py:31`) and `Ragged.from_offsets`.

- [ ] **Step 1: Write the failing round-trip test + fixture**

Create `tests/integration/conftest.py`:

```python
"""Shared fixtures for tests/integration/."""

from __future__ import annotations

from pathlib import Path

import pyBigWig
import pytest

import genvarloader as gvl


@pytest.fixture
def track_dataset_path(source_bed, vcf_dir, tmp_path) -> Path:
    """A freshly-written 2.0 dataset (phased VCF + one BigWig 'cov' track),
    yielded as a writable path so tests may downgrade/migrate it in place.

    Mirrors tests/dataset/conftest.py::snap_dataset but yields a path (not an
    opened Dataset) and is function-scoped so each test gets a mutable copy.
    """
    from genoray import VCF

    samples = ["s0", "s1", "s2"]
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, s in enumerate(samples):
        p = tmp_path / f"{s}.bw"
        with pyBigWig.open(str(p), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            v = float(i + 1)
            bw.addEntries(
                ["chr1", "chr1", "chr2", "chr2"],
                [499_990, 1_010_686, 17_320, 1_234_560],
                ends=[500_030, 1_010_706, 17_340, 1_234_580],
                values=[v, v, v, v],
            )
        bw_paths[s] = str(p)
    out = tmp_path / "ds.gvl"
    gvl.write(
        path=out,
        bed=source_bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        tracks=gvl.BigWigs("cov", bw_paths),
        max_jitter=2,
    )
    return out
```

Create `tests/integration/test_format_2_soa.py`:

```python
"""Format 2.0 stores track intervals as struct-of-arrays (Task 1)."""

from __future__ import annotations

import json

import numpy as np

import genvarloader as gvl
from genvarloader._dataset._write import DATASET_FORMAT_VERSION


def test_dataset_version_is_2(track_dataset_path):
    assert str(DATASET_FORMAT_VERSION) == "2.0.0"
    meta = json.loads((track_dataset_path / "metadata.json").read_text())
    assert meta["format_version"] == "2.0.0"


def test_soa_files_present_and_aos_absent(track_dataset_path):
    track_dir = track_dataset_path / "intervals" / "cov"
    assert (track_dir / "starts.npy").exists()
    assert (track_dir / "ends.npy").exists()
    assert (track_dir / "values.npy").exists()
    assert (track_dir / "offsets.npy").exists()
    assert not (track_dir / "intervals.npy").exists()


def test_soa_files_contiguous_and_typed(track_dataset_path):
    track_dir = track_dataset_path / "intervals" / "cov"
    starts = np.memmap(track_dir / "starts.npy", dtype=np.int32, mode="r")
    ends = np.memmap(track_dir / "ends.npy", dtype=np.int32, mode="r")
    values = np.memmap(track_dir / "values.npy", dtype=np.float32, mode="r")
    assert starts.flags["C_CONTIGUOUS"]
    assert ends.flags["C_CONTIGUOUS"]
    assert values.flags["C_CONTIGUOUS"]
    assert len(starts) == len(ends) == len(values)


def test_reads_back(track_dataset_path, reference):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_tracks("cov")
    out = ds[0, 0]
    assert out is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/integration/test_format_2_soa.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — `test_dataset_version_is_2` fails (`"1.0.0" != "2.0.0"`) and `test_soa_files_present_and_aos_absent` fails (`intervals.npy` still present, `starts.npy` absent).

- [ ] **Step 3: Bump the format version**

In `python/genvarloader/_dataset/_write.py:44` change:

```python
DATASET_FORMAT_VERSION = SemanticVersion.parse("1.0.0")
```

to:

```python
DATASET_FORMAT_VERSION = SemanticVersion.parse("2.0.0")
```

- [ ] **Step 4: Convert the Python single-chunk writer to SoA**

In `python/genvarloader/_dataset/_write.py`, replace `_write_ragged_intervals` (`:1085-1108`) body. New version:

```python
def _write_ragged_intervals(out_dir: Path, itvs: "RaggedIntervals") -> None:
    """Write a RaggedIntervals (values/starts/ends share offsets) to out_dir as
    struct-of-arrays: starts/ends/values.npy + offsets.npy. Single-chunk writer
    used for annotation tracks (format 2.0)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data, dt in (
        ("starts", itvs.starts.data, np.int32),
        ("ends", itvs.ends.data, np.int32),
        ("values", itvs.values.data, np.float32),
    ):
        out = np.memmap(out_dir / f"{name}.npy", dtype=dt, mode="w+", shape=data.shape)
        out[:] = data
        out.flush()

    offsets = itvs.values.offsets
    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=offsets.dtype,
        mode="w+",
        shape=len(offsets),
    )
    out[:] = offsets
    out.flush()
```

- [ ] **Step 5: Convert the Python chunked writer to SoA**

In `python/genvarloader/_dataset/_write.py`, the chunked sample-track writer (`_write_track_legacy`) currently writes one AoS memmap at `:1322-1334`:

```python
        pbar.set_description(f"Writing intervals for {part.height} regions on {contig}")
        out = np.memmap(
            out_dir / "intervals.npy",
            dtype=INTERVAL_DTYPE,
            mode="w+" if interval_offset == 0 else "r+",
            shape=intervals.values.data.shape,
            offset=interval_offset,
        )
        out["start"] = intervals.starts.data
        out["end"] = intervals.ends.data
        out["value"] = intervals.values.data
        out.flush()
        interval_offset += out.nbytes
```

Replace with three SoA memmaps. `interval_offset` becomes an **element** counter (all three dtypes are 4 bytes, so each file's byte offset is `interval_offset * itemsize`):

```python
        pbar.set_description(f"Writing intervals for {part.height} regions on {contig}")
        n = intervals.values.data.shape[0]
        for name, data, dt in (
            ("starts", intervals.starts.data, np.int32),
            ("ends", intervals.ends.data, np.int32),
            ("values", intervals.values.data, np.float32),
        ):
            out = np.memmap(
                out_dir / f"{name}.npy",
                dtype=dt,
                mode="w+" if interval_offset == 0 else "r+",
                shape=n,
                offset=interval_offset * np.dtype(dt).itemsize,
            )
            out[:] = data
            out.flush()
        interval_offset += n
```

(`interval_offset` is initialized to `0` at `:1304`; it previously counted bytes, now counts elements — both start at 0 so the `mode="w+" if interval_offset == 0` guard is unchanged in meaning.) Leave the `INTERVAL_DTYPE` import at `:37` in place — Task 3's migration reader still needs it, and `_write.py` is not on the hot read path.

- [ ] **Step 6: Convert the reader to SoA**

In `python/genvarloader/_dataset/_tracks.py`, replace `_open_intervals` (`:706-725`):

```python
    @staticmethod
    def _open_intervals(path: Path, n_regions: int, n_samples: int) -> RaggedIntervals:
        if n_samples == 0:
            shape = (n_regions, None)
        else:
            shape = (n_regions, n_samples, None)
        starts_data = np.memmap(path / "starts.npy", dtype=np.int32, mode="r")
        ends_data = np.memmap(path / "ends.npy", dtype=np.int32, mode="r")
        values_data = np.memmap(path / "values.npy", dtype=np.float32, mode="r")
        offsets = np.memmap(path / "offsets.npy", dtype=np.int64, mode="r")
        starts = Ragged.from_offsets(starts_data, shape, offsets)
        ends = Ragged.from_offsets(ends_data, shape, offsets)
        values = Ragged.from_offsets(values_data, shape, offsets)
        return RaggedIntervals(starts, ends, values)
```

Then drop `INTERVAL_DTYPE` from the import at `_tracks.py:18`:

```python
from .._ragged import FlatIntervals, RaggedIntervals, RaggedTracks
```

(was `from .._ragged import INTERVAL_DTYPE, FlatIntervals, RaggedIntervals, RaggedTracks`).

- [ ] **Step 7: Convert the Rust BigWig writer to SoA**

In `src/bigwig.rs::write_track`, replace the single `itv_writer` with three writers. At `:40`:

```rust
    let mut itv_writer = BufWriter::new(File::create(out_dir.join("intervals.npy"))?);
```

becomes:

```rust
    let mut starts_writer = BufWriter::new(File::create(out_dir.join("starts.npy"))?);
    let mut ends_writer = BufWriter::new(File::create(out_dir.join("ends.npy"))?);
    let mut values_writer = BufWriter::new(File::create(out_dir.join("values.npy"))?);
```

At the write loop (`:106-114`):

```rust
            for sample_vals in per_sample {
                for v in sample_vals {
                    itv_writer.write_all(&(v.start as i32).to_le_bytes())?;
                    itv_writer.write_all(&(v.end as i32).to_le_bytes())?;
                    itv_writer.write_all(&v.value.to_le_bytes())?;
                    acc += 1;
                }
                offsets.push(acc);
            }
```

becomes:

```rust
            for sample_vals in per_sample {
                for v in sample_vals {
                    starts_writer.write_all(&(v.start as i32).to_le_bytes())?;
                    ends_writer.write_all(&(v.end as i32).to_le_bytes())?;
                    values_writer.write_all(&v.value.to_le_bytes())?;
                    acc += 1;
                }
                offsets.push(acc);
            }
```

And the flush (`:118`):

```rust
    itv_writer.flush()?;
```

becomes:

```rust
    starts_writer.flush()?;
    ends_writer.flush()?;
    values_writer.flush()?;
```

- [ ] **Step 8: Update the Rust BigWig oracle byte test**

In `src/bigwig.rs`, the oracle test currently builds one interleaved `expected` and reads `intervals.npy` (`:319-327`):

```rust
        // Expected intervals.npy bytes: [i32 start, i32 end, f32 value] per row.
        let mut expected = Vec::new();
        for i in 0..vals.len() {
            expected.extend_from_slice(&(coords[[i, 0]] as i32).to_le_bytes());
            expected.extend_from_slice(&(coords[[i, 1]] as i32).to_le_bytes());
            expected.extend_from_slice(&vals[i].to_le_bytes());
        }
        let got = fs::read(tmp.join("intervals.npy")).unwrap();
        assert_eq!(got, expected, "intervals.npy bytes mismatch");
```

Replace with three SoA expectations:

```rust
        // Expected SoA bytes: separate i32 starts, i32 ends, f32 values.
        let mut exp_starts = Vec::new();
        let mut exp_ends = Vec::new();
        let mut exp_values = Vec::new();
        for i in 0..vals.len() {
            exp_starts.extend_from_slice(&(coords[[i, 0]] as i32).to_le_bytes());
            exp_ends.extend_from_slice(&(coords[[i, 1]] as i32).to_le_bytes());
            exp_values.extend_from_slice(&vals[i].to_le_bytes());
        }
        assert_eq!(fs::read(tmp.join("starts.npy")).unwrap(), exp_starts, "starts mismatch");
        assert_eq!(fs::read(tmp.join("ends.npy")).unwrap(), exp_ends, "ends mismatch");
        assert_eq!(fs::read(tmp.join("values.npy")).unwrap(), exp_values, "values mismatch");
```

(The `offsets.npy` assertion below it is unchanged.)

- [ ] **Step 9: Convert the Rust table writer to SoA**

In `src/tables.rs::write_track_impl`, at `:161`:

```rust
        let mut itv_w = BufWriter::new(File::create(out_dir.join("intervals.npy"))?);
```

becomes:

```rust
        let mut starts_w = BufWriter::new(File::create(out_dir.join("starts.npy"))?);
        let mut ends_w = BufWriter::new(File::create(out_dir.join("ends.npy"))?);
        let mut values_w = BufWriter::new(File::create(out_dir.join("values.npy"))?);
```

The row-write loop (`:211-215`):

```rust
            for (s, e, v) in &region_rows {
                itv_w.write_all(&s.to_le_bytes())?;
                itv_w.write_all(&e.to_le_bytes())?;
                itv_w.write_all(&v.to_le_bytes())?;
            }
```

becomes:

```rust
            for (s, e, v) in &region_rows {
                starts_w.write_all(&s.to_le_bytes())?;
                ends_w.write_all(&e.to_le_bytes())?;
                values_w.write_all(&v.to_le_bytes())?;
            }
```

The flush (`:222`):

```rust
        itv_w.flush()?;
```

becomes:

```rust
        starts_w.flush()?;
        ends_w.flush()?;
        values_w.flush()?;
```

- [ ] **Step 10: Update the Rust table oracle byte test**

In `src/tables.rs`, the oracle test (`:453-466`) builds `exp_itv` interleaved and reads `intervals.npy`:

```rust
            for i in 0..vals.len() {
                exp_itv.extend_from_slice(&coords[[i, 0]].to_le_bytes());
                exp_itv.extend_from_slice(&coords[[i, 1]].to_le_bytes());
                exp_itv.extend_from_slice(&vals[i].to_le_bytes());
            }
```

Replace the `exp_itv` declaration and this loop with three vectors. Find the `let mut exp_itv = Vec::new();` declaration near the top of the test and replace it plus the loop and the final read/assert (`:464-467`):

```rust
        let mut exp_starts: Vec<u8> = Vec::new();
        let mut exp_ends: Vec<u8> = Vec::new();
        let mut exp_values: Vec<u8> = Vec::new();
```

loop body:

```rust
            for i in 0..vals.len() {
                exp_starts.extend_from_slice(&coords[[i, 0]].to_le_bytes());
                exp_ends.extend_from_slice(&coords[[i, 1]].to_le_bytes());
                exp_values.extend_from_slice(&vals[i].to_le_bytes());
            }
```

final assertions (replacing the `intervals.npy` read at `:464,466`):

```rust
        assert_eq!(std::fs::read(tmp.join("starts.npy")).unwrap(), exp_starts, "starts mismatch");
        assert_eq!(std::fs::read(tmp.join("ends.npy")).unwrap(), exp_ends, "ends mismatch");
        assert_eq!(std::fs::read(tmp.join("values.npy")).unwrap(), exp_values, "values mismatch");
```

(The `got_off`/`exp_off` offsets assertion is unchanged.)

- [ ] **Step 11: Rebuild the extension and run cargo tests**

Run: `pixi run -e dev maturin develop --release`
Expected: builds clean.

Run: `pixi run -e dev cargo test`
Expected: PASS, including `bigwig::tests::write_track_matches_count_and_intervals_oracle` and `tables::tests::write_track_matches_oracle_bytes`.

- [ ] **Step 12: Run the Task 1 round-trip test**

Run: `pixi run -e dev pytest tests/integration/test_format_2_soa.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (4 tests).

- [ ] **Step 13: Run the full parity + dataset suites on both backends**

Run: `pixi run -e dev pytest tests/parity tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (byte-identical on the numba backend too).

- [ ] **Step 14: Lint, format, typecheck, commit**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck && pixi run -e dev cargo clippy`
Expected: clean.

```bash
rtk git add python/genvarloader/_dataset/_write.py python/genvarloader/_dataset/_tracks.py src/bigwig.rs src/tables.rs tests/integration/conftest.py tests/integration/test_format_2_soa.py
rtk git commit -m "feat(format)!: store track intervals as struct-of-arrays (gvl 2.0)

Convert AoS INTERVAL_DTYPE (itemsize 12, strided field views) to three
contiguous files starts/ends/values.npy sharing offsets.npy, across all
four writers (Python single-chunk + chunked, Rust bigwig + table) and the
reader. Bump DATASET_FORMAT_VERSION to 2.0.0. Byte-identical output.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Version gate on open (Component B)

Reject a 1.x (or `None`) dataset at open with a clear `gvl.migrate` hint; reject a future-major dataset with an upgrade error.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (add `_check_dataset_format_version` near `DATASET_FORMAT_VERSION` `:44`)
- Modify: `python/genvarloader/_dataset/_open.py` (`_load_metadata` `:103-107`)
- Create: `tests/integration/test_format_version_gate.py`

**Interfaces:**
- Consumes: `Metadata` (`_write.py:65`, has `format_version: SemanticVersion | None`), `DATASET_FORMAT_VERSION` (now `2.0.0`).
- Produces: `_check_dataset_format_version(meta: Metadata, path: Path) -> None` — raises `ValueError` on `format_version is None` or `major < 2` (migrate hint) and on `major > 2` (upgrade hint); returns `None` when `major == 2`.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_format_version_gate.py`:

```python
"""Open-time format_version gate (Task 2)."""

from __future__ import annotations

import json
import shutil

import pytest

import genvarloader as gvl


def _set_version(path, version):
    meta_path = path / "metadata.json"
    raw = json.loads(meta_path.read_text())
    raw["format_version"] = version
    meta_path.write_text(json.dumps(raw))


def test_old_major_raises_migrate_hint(track_dataset_path, reference):
    _set_version(track_dataset_path, "1.0.0")
    with pytest.raises(ValueError, match="migrate"):
        gvl.Dataset.open(track_dataset_path, reference=reference)


def test_none_version_raises_migrate_hint(track_dataset_path, reference, tmp_path):
    dst = tmp_path / "noneversion.gvl"
    shutil.copytree(track_dataset_path, dst)
    meta_path = dst / "metadata.json"
    raw = json.loads(meta_path.read_text())
    raw["format_version"] = None
    meta_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="migrate"):
        gvl.Dataset.open(dst, reference=reference)


def test_future_major_raises_upgrade_hint(track_dataset_path, reference):
    _set_version(track_dataset_path, "3.0.0")
    with pytest.raises(ValueError, match="[Uu]pgrade"):
        gvl.Dataset.open(track_dataset_path, reference=reference)


def test_current_major_opens(track_dataset_path, reference):
    # written fresh at 2.0.0 by the fixture
    ds = gvl.Dataset.open(track_dataset_path, reference=reference)
    assert ds is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/integration/test_format_version_gate.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — `test_old_major_raises_migrate_hint` and the others that expect a raise do not raise (no gate yet).

- [ ] **Step 3: Add the gate helper**

In `python/genvarloader/_dataset/_write.py`, immediately after the `DATASET_FORMAT_VERSION` definition (`:44-46`), add:

```python
def _check_dataset_format_version(meta: "Metadata", path: Path) -> None:
    """Validate a dataset's on-disk format version against the supported major.

    Pre-versioning datasets (``format_version is None``) and any older major are
    treated as needing migration. A newer major means the reader is too old.
    """
    fv = meta.format_version
    current = DATASET_FORMAT_VERSION
    if fv is None or fv.major < current.major:
        raise ValueError(
            f"Dataset at {path} uses format version {fv} but this genvarloader "
            f"expects {current}. Run `genvarloader.migrate({str(path)!r})` to "
            f"upgrade it in place."
        )
    if fv.major > current.major:
        raise ValueError(
            f"Dataset at {path} was written by a newer genvarloader (format "
            f"version {fv} > supported {current}). Upgrade genvarloader."
        )
```

(`Metadata` is defined later in the file at `:65`; the forward reference in the annotation string is fine.)

- [ ] **Step 4: Wire the gate into open**

In `python/genvarloader/_dataset/_open.py`, update the import at `:27`:

```python
from ._write import Metadata, _check_dataset_format_version
```

and `_load_metadata` (`:103-107`):

```python
    def _load_metadata(self) -> Metadata:
        with _py_open(self.path / "metadata.json") as f:
            metadata = Metadata.model_validate_json(f.read())
        _check_dataset_format_version(metadata, self.path)
        validate_dataset(metadata, self.path)
        return metadata
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/integration/test_format_version_gate.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (4 tests).

- [ ] **Step 6: Confirm no regression in the open path**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (all fixtures are born 2.0.0, so the gate is a no-op for them).

- [ ] **Step 7: Lint, format, typecheck, commit**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

```bash
rtk git add python/genvarloader/_dataset/_write.py python/genvarloader/_dataset/_open.py tests/integration/test_format_version_gate.py
rtk git commit -m "feat(open): gate dataset open on format_version major

Reject pre-2.0 (or unversioned) datasets with a gvl.migrate hint and
future-major datasets with an upgrade error.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `gvl.migrate(path)` — streaming in-place AoS → SoA (Component C)

In-place, streaming, idempotent, crash-safe rewrite of a 1.x AoS dataset to 2.0 SoA.

**Files:**
- Create: `python/genvarloader/_dataset/_migrate.py`
- Modify: `python/genvarloader/__init__.py` (import + `__all__`)
- Create: `tests/integration/test_migrate.py`

**Interfaces:**
- Consumes: `INTERVAL_DTYPE` (`_ragged.py:26`), `DATASET_FORMAT_VERSION` (`_write.py:44`), `SemanticVersion`.
- Produces: `migrate(path: str | Path) -> None` — exported in `genvarloader.__all__`. Converts every `intervals/<track>/intervals.npy` and `annot_intervals/<track>/intervals.npy` to SoA, bumps `metadata.json` `format_version` to `2.0.0` (durable, after all SoA written), then deletes the AoS files. No-op (with leftover-AoS cleanup) on an already-2.0 dataset.
- Produces (test helper, local to the test module): `_downgrade_to_aos(path)` — inverse for synthesizing a 1.x fixture from a fresh 2.0 dataset.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_migrate.py`:

```python
"""gvl.migrate: 1.x AoS -> 2.0 SoA round-trip, idempotency, crash-safety (Task 3)."""

from __future__ import annotations

import json

import numpy as np

import genvarloader as gvl
from genvarloader._ragged import INTERVAL_DTYPE


def _track_dirs(path):
    for base in ("intervals", "annot_intervals"):
        d = path / base
        if d.is_dir():
            for child in sorted(d.iterdir()):
                if child.is_dir():
                    yield child


def _downgrade_to_aos(path):
    """Rewrite a fresh 2.0 SoA dataset back to a 1.x AoS dataset in place."""
    for d in _track_dirs(path):
        starts = np.memmap(d / "starts.npy", dtype=np.int32, mode="r")
        ends = np.memmap(d / "ends.npy", dtype=np.int32, mode="r")
        values = np.memmap(d / "values.npy", dtype=np.float32, mode="r")
        rec = np.empty(len(starts), dtype=INTERVAL_DTYPE)
        rec["start"] = starts
        rec["end"] = ends
        rec["value"] = values
        out = np.memmap(d / "intervals.npy", dtype=INTERVAL_DTYPE, mode="w+", shape=rec.shape)
        out[:] = rec
        out.flush()
        del starts, ends, values, out
        (d / "starts.npy").unlink()
        (d / "ends.npy").unlink()
        (d / "values.npy").unlink()
    meta_path = path / "metadata.json"
    raw = json.loads(meta_path.read_text())
    raw["format_version"] = "1.0.0"
    meta_path.write_text(json.dumps(raw))


def test_round_trip_byte_identical(track_dataset_path, reference):
    before = gvl.Dataset.open(track_dataset_path, reference=reference).with_tracks("cov")[0, 0]
    before = np.asarray(before).copy()

    _downgrade_to_aos(track_dataset_path)
    gvl.migrate(track_dataset_path)

    track_dir = track_dataset_path / "intervals" / "cov"
    assert (track_dir / "starts.npy").exists()
    assert (track_dir / "ends.npy").exists()
    assert (track_dir / "values.npy").exists()
    assert not (track_dir / "intervals.npy").exists()
    assert json.loads((track_dataset_path / "metadata.json").read_text())["format_version"] == "2.0.0"

    after = gvl.Dataset.open(track_dataset_path, reference=reference).with_tracks("cov")[0, 0]
    np.testing.assert_array_equal(np.asarray(after), before)


def test_idempotent(track_dataset_path):
    _downgrade_to_aos(track_dataset_path)
    gvl.migrate(track_dataset_path)
    gvl.migrate(track_dataset_path)  # second run is a no-op, must not raise
    track_dir = track_dataset_path / "intervals" / "cov"
    assert not (track_dir / "intervals.npy").exists()


def test_resumable_after_interrupt_before_metadata_bump(track_dataset_path):
    """Crash after SoA written but before metadata bump: still 1.x, re-runnable."""
    _downgrade_to_aos(track_dataset_path)
    # Simulate partial migration: write SoA, leave AoS + 1.x metadata.
    from genvarloader._dataset._migrate import _migrate_track

    for d in _track_dirs(track_dataset_path):
        _migrate_track(d)
    meta = json.loads((track_dataset_path / "metadata.json").read_text())
    assert meta["format_version"] == "1.0.0"  # not bumped yet
    track_dir = track_dataset_path / "intervals" / "cov"
    assert (track_dir / "intervals.npy").exists()  # AoS still present

    gvl.migrate(track_dataset_path)  # completes the migration
    assert json.loads((track_dataset_path / "metadata.json").read_text())["format_version"] == "2.0.0"
    assert not (track_dir / "intervals.npy").exists()


def test_cleans_leftover_aos_after_interrupt_before_delete(track_dataset_path):
    """Crash after metadata bump but before AoS delete: re-run removes AoS."""
    _downgrade_to_aos(track_dataset_path)
    gvl.migrate(track_dataset_path)  # full migration -> SoA + 2.0 metadata
    track_dir = track_dataset_path / "intervals" / "cov"
    # Re-introduce a leftover AoS file (as if delete was interrupted).
    starts = np.memmap(track_dir / "starts.npy", dtype=np.int32, mode="r")
    rec = np.zeros(len(starts), dtype=INTERVAL_DTYPE)
    out = np.memmap(track_dir / "intervals.npy", dtype=INTERVAL_DTYPE, mode="w+", shape=rec.shape)
    out[:] = rec
    out.flush()
    del starts, out

    gvl.migrate(track_dataset_path)  # idempotent cleanup
    assert not (track_dir / "intervals.npy").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/integration/test_migrate.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — `ImportError`/`AttributeError`: `genvarloader` has no attribute `migrate`.

- [ ] **Step 3: Implement the migration module**

Create `python/genvarloader/_dataset/_migrate.py`:

```python
"""In-place, streaming, idempotent migration of a 1.x AoS dataset to 2.0 SoA.

Per track under ``intervals/<track>/`` and ``annot_intervals/<track>/``:
stream ``intervals.npy`` (INTERVAL_DTYPE) in record chunks into three contiguous
``starts/ends/values.npy`` files. Only after every track's SoA is durable do we
bump ``metadata.json`` (last durable write); then delete the AoS files.

Crash-safety by ordering: an interruption before the metadata bump leaves the
dataset still-1.x (old AoS intact, re-runnable); an interruption after the bump
but before deletion leaves both layouts, and a re-run completes the cleanup.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic_extra_types.semantic_version import SemanticVersion

from .._ragged import INTERVAL_DTYPE
from ._write import DATASET_FORMAT_VERSION

_CHUNK = 1_000_000  # records per streamed block


def _track_dirs(path: Path) -> Iterator[Path]:
    for base in ("intervals", "annot_intervals"):
        d = path / base
        if d.is_dir():
            for child in sorted(d.iterdir()):
                if child.is_dir():
                    yield child


def _migrate_track(track_dir: Path) -> None:
    """Stream one track's AoS intervals.npy into SoA starts/ends/values.npy.

    No-op if intervals.npy is absent (already migrated or never AoS). Leaves the
    AoS file in place; the caller deletes it only after metadata is bumped.
    """
    aos = track_dir / "intervals.npy"
    if not aos.exists():
        return
    src = np.memmap(aos, dtype=INTERVAL_DTYPE, mode="r")
    n = int(src.shape[0])
    starts = np.memmap(track_dir / "starts.npy", dtype=np.int32, mode="w+", shape=n)
    ends = np.memmap(track_dir / "ends.npy", dtype=np.int32, mode="w+", shape=n)
    values = np.memmap(track_dir / "values.npy", dtype=np.float32, mode="w+", shape=n)
    for i in range(0, n, _CHUNK):
        j = min(i + _CHUNK, n)
        block = src[i:j]
        starts[i:j] = block["start"]
        ends[i:j] = block["end"]
        values[i:j] = block["value"]
    for m in (starts, ends, values):
        m.flush()
    logger.info(f"Migrated {n} intervals in {track_dir} to SoA.")
    del src, starts, ends, values


def migrate(path: str | Path) -> None:
    """Migrate a GVL dataset's track intervals from format 1.x (array-of-structs)
    to format 2.0 (struct-of-arrays), in place.

    Streaming and crash-safe: peak extra disk is one track's interval store.
    Genotypes, regions, and reference are untouched. Idempotent — a no-op (with
    leftover-AoS cleanup) on a dataset that is already 2.0.

    Parameters
    ----------
    path
        Path to the GVL dataset directory.
    """
    path = Path(path)
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json at {meta_path}")
    raw = json.loads(meta_path.read_text())
    fv = raw.get("format_version")
    already_v2 = (
        fv is not None
        and SemanticVersion.parse(fv).major >= DATASET_FORMAT_VERSION.major
    )
    track_dirs = list(_track_dirs(path))

    if already_v2:
        # Idempotent cleanup: remove leftover AoS from an interrupted delete.
        for d in track_dirs:
            aos = d / "intervals.npy"
            if aos.exists() and (d / "starts.npy").exists():
                aos.unlink()
        return

    # 1. Convert every track to SoA (AoS left in place).
    for d in track_dirs:
        _migrate_track(d)

    # 2. Durably bump metadata LAST (atomic replace).
    raw["format_version"] = str(DATASET_FORMAT_VERSION)
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw))
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, meta_path)

    # 3. Delete AoS files.
    for d in track_dirs:
        aos = d / "intervals.npy"
        if aos.exists():
            aos.unlink()
    logger.info(f"Migrated dataset {path} to format {DATASET_FORMAT_VERSION}.")
```

- [ ] **Step 4: Export `migrate`**

In `python/genvarloader/__init__.py`, add the import (after the `_svar_link` import at `:29`):

```python
from ._dataset._migrate import migrate
```

and insert `"migrate"` into `__all__` (alphabetically, between `"get_splice_bed"` and `"migrate_svar_link"`):

```python
    "get_splice_bed",
    "migrate",
    "migrate_svar_link",
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/integration/test_migrate.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (4 tests).

- [ ] **Step 6: Lint, format, typecheck, commit**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

```bash
rtk git add python/genvarloader/_dataset/_migrate.py python/genvarloader/__init__.py tests/integration/test_migrate.py
rtk git commit -m "feat(migrate): add gvl.migrate for 1.x AoS -> 2.0 SoA

Streaming, idempotent, crash-safe in-place rewrite of track intervals.
Metadata is bumped only after all SoA files are durable, then AoS deleted.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Zero-copy FFI contract + loud boundary guard (Component D)

Drop `np.ascontiguousarray(...)` on per-sample-scale memmapped args (now contiguous after Task 1, or already contiguous for genotypes), replacing it with `_ffi_array` — which crosses zero-copy or raises a precise error. The scale-guard test locks the defect closed.

**Files:**
- Modify: `python/genvarloader/_dataset/_utils.py` (add `_ffi_array`)
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`:232-250` track-fused args)
- Modify: `python/genvarloader/_dataset/_haps.py` (`:796`, `:869`, `:958` — `geno_v_idxs` in the three fused calls)
- Create: `tests/unit/dataset/test_ffi_array.py`
- Create: `tests/integration/test_scale_guard.py`

**Interfaces:**
- Produces: `_ffi_array(arr: np.ndarray, dtype, name: str) -> np.ndarray` in `_dataset/_utils.py` — returns `arr` unchanged if C-contiguous and exact dtype; else raises `ValueError` naming `name`.
- Consumes: SoA interval memmaps (Task 1), `self.haps.genotypes.data` / `self.genotypes.data` (already contiguous `int32` memmaps).
- **Scope:** the guard applies ONLY to per-sample-scale memmap args. Batch-bounded freshly-constructed arrays (`req.regions`, `req.shifts`, `req.geno_offset_idx`, `req.keep`, `req.keep_offsets`, the `_reconstruct.py` `o_idx`/`out_ofsts_per_t`/etc.) keep `np.ascontiguousarray` (cheap). The sub-linear per-variant args (`v_starts`, `ilens`, `alt`, `ref`, ...) are handled by Task 5 — leave them as `np.ascontiguousarray(...)` in this task.

- [ ] **Step 1: Write the failing FFI-guard unit test**

Create `tests/unit/dataset/test_ffi_array.py`:

```python
"""_ffi_array boundary guard (Task 4)."""

from __future__ import annotations

import numpy as np
import pytest

from genvarloader._dataset._utils import _ffi_array


def test_passes_contiguous_correct_dtype():
    arr = np.arange(10, dtype=np.int32)
    out = _ffi_array(arr, np.int32, "geno_v_idxs")
    assert out is arr  # zero-copy: same object


def test_raises_on_non_contiguous():
    base = np.zeros((10, 3), dtype=np.int32)
    strided = base[:, 1]  # non-contiguous column view
    assert not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="geno_v_idxs"):
        _ffi_array(strided, np.int32, "geno_v_idxs")


def test_raises_on_wrong_dtype():
    arr = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="itv_starts"):
        _ffi_array(arr, np.int32, "itv_starts")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_ffi_array.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — `ImportError: cannot import name '_ffi_array'`.

- [ ] **Step 3: Implement `_ffi_array`**

In `python/genvarloader/_dataset/_utils.py`, add (the file already imports `numpy as np`):

```python
def _ffi_array(arr: np.ndarray, dtype, name: str) -> np.ndarray:
    """Assert a per-sample-scale FFI argument crosses zero-copy.

    Returns ``arr`` unchanged iff it is C-contiguous with exactly ``dtype``;
    otherwise raises a precise ``ValueError`` naming ``name``. This replaces a
    silent ``np.ascontiguousarray`` that would copy the whole per-sample-scale
    memmap (GB-scale at the >1M-sample design target). Use it ONLY for
    sample-scale memmap args; batch-bounded arrays may keep coercing.
    """
    dt = np.dtype(dtype)
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError(
            f"FFI argument {name!r} must be C-contiguous to cross zero-copy; got "
            f"a non-contiguous array (coercing would force a sample-scale copy)."
        )
    if arr.dtype != dt:
        raise ValueError(
            f"FFI argument {name!r} must have dtype {dt}; got {arr.dtype} "
            f"(coercing would force a sample-scale cast/copy)."
        )
    return arr
```

- [ ] **Step 4: Run the FFI-guard test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/test_ffi_array.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (3 tests).

- [ ] **Step 5: Apply the guard in the track-fused path**

In `python/genvarloader/_dataset/_reconstruct.py`, add the import near the top (it already imports from `._utils`; if not, add `from ._utils import _ffi_array`). Then in the `intervals_and_realign_track_fused(...)` call (`:232-250`), replace the sample-scale args:

`geno_v_idxs` (`:232-234`):

```python
                        geno_v_idxs=_ffi_array(
                            self.haps.genotypes.data, np.int32, "geno_v_idxs"
                        ),
```

`itv_starts` / `itv_ends` / `itv_values` / `itv_offsets` (`:241-250`):

```python
                        itv_starts=_ffi_array(
                            intervals.starts.data, np.int32, "itv_starts"
                        ),
                        itv_ends=_ffi_array(intervals.ends.data, np.int32, "itv_ends"),
                        itv_values=_ffi_array(
                            intervals.values.data, np.float32, "itv_values"
                        ),
                        itv_offsets=_ffi_array(
                            intervals.starts.offsets, np.int64, "itv_offsets"
                        ),
```

Leave `v_starts` and `ilens` (`:236-239`) as `np.ascontiguousarray(...)` — Task 5 converts those to the cached arrays. Leave `o_idx`, `out_ofsts_per_t`, `regions`, `shifts`, `geno_idx`, `track_ofsts_per_t`, `params`, `keep`, `keep_offsets` as `np.ascontiguousarray(...)` (batch-bounded).

- [ ] **Step 6: Apply the guard to the fused haps/annotated/splice calls**

In `python/genvarloader/_dataset/_haps.py`, add `from ._utils import _ffi_array` to the imports if not already present. Then replace `geno_v_idxs` in all three fused calls:

`:796` (plain `reconstruct_haplotypes_fused`):

```python
                    geno_v_idxs=_ffi_array(self.genotypes.data, np.int32, "geno_v_idxs"),
```

`:869` (`reconstruct_haplotypes_spliced_fused`):

```python
                geno_v_idxs=_ffi_array(self.genotypes.data, np.int32, "geno_v_idxs"),
```

`:958` (`reconstruct_annotated_haplotypes_fused`):

```python
                        geno_v_idxs=_ffi_array(self.genotypes.data, np.int32, "geno_v_idxs"),
```

Leave the sub-linear args (`v_starts`, `ilens`, `alt_alleles`, `alt_offsets`, `ref_`, `ref_offsets`) as `np.ascontiguousarray(...)` for now — Task 5. Leave `regions`, `shifts`, `geno_offset_idx`, `keep`, `keep_offsets`, `permuted_regions`, `flat_shifts`, `flat_geno_offset_idx`, `out_offsets` as `np.ascontiguousarray(...)` (batch-bounded). Leave `_as_starts_stops(self.genotypes.offsets)` untouched.

- [ ] **Step 7: Write the failing scale-guard test**

Create `tests/integration/test_scale_guard.py`:

```python
"""Scale-guard: no per-batch copy materializes a memmap on the read path (Task 4).

Mirrors the py-spy diagnostic that found the defect: monkeypatch
np.ascontiguousarray over one ds[r, s] and assert zero copies whose source
.base is an np.memmap.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl


@pytest.fixture
def _no_memmap_copies(monkeypatch):
    real = np.ascontiguousarray
    offenders: list[str] = []

    def spy(a, dtype=None, *args, **kwargs):
        arr = np.asarray(a)
        base = getattr(arr, "base", None)
        if isinstance(base, np.memmap) or isinstance(arr, np.memmap):
            # A copy would be forced iff non-contiguous or dtype-mismatched.
            would_copy = (not arr.flags["C_CONTIGUOUS"]) or (
                dtype is not None and arr.dtype != np.dtype(dtype)
            )
            if would_copy:
                offenders.append(f"{getattr(arr, 'shape', None)} {arr.dtype}->{dtype}")
        return real(a, dtype, *args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", spy)
    return offenders


def test_tracks_only_no_memmap_copy(track_dataset_path, reference, _no_memmap_copies):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_tracks("cov")
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"


def test_haps_no_memmap_copy(track_dataset_path, reference, _no_memmap_copies):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_seqs("haplotypes")
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"


def test_annotated_no_memmap_copy(track_dataset_path, reference, _no_memmap_copies):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_seqs("annotated")
    _ = ds[0, 0]
    assert _no_memmap_copies == [], f"sample-scale memmap copies: {_no_memmap_copies}"
```

- [ ] **Step 8: Run the scale-guard test**

Run: `pixi run -e dev pytest tests/integration/test_scale_guard.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS. (After Task 1 the interval memmaps are contiguous and the guard replaced their `ascontiguousarray`; `genotypes.data`/`offsets` and the reference/variant memmaps are contiguous so no copy is forced. If any test fails, the offender list names the shape/dtype — that is a real sample-scale copy to eliminate, not a test to relax.)

- [ ] **Step 9: Run parity on both backends**

Run: `pixi run -e dev pytest tests/parity tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

- [ ] **Step 10: Lint, format, typecheck, commit**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

```bash
rtk git add python/genvarloader/_dataset/_utils.py python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_haps.py tests/unit/dataset/test_ffi_array.py tests/integration/test_scale_guard.py
rtk git commit -m "feat(ffi): zero-copy boundary guard for sample-scale memmaps

Replace silent np.ascontiguousarray on per-sample-scale interval/genotype
memmaps with _ffi_array (cross zero-copy or raise). Scale-guard test asserts
no memmap-materializing copy on the read path.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: RAM-cache the sub-linear static arrays (Component E)

Cache, once per `Haps` reconstructor, the typed-contiguous per-variant/reference arrays the kernels consume, dropping their per-batch `np.ascontiguousarray` (chiefly the `int64`→`int32` recast of `v_starts`).

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (add `_HapsFfiStatic` dataclass + `_ffi_static` field + `ffi_static` property on `Haps` `:238-280`; replace sub-linear args at the fused calls `:797-806`, `:870-877`, `:959-970`)
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`v_starts`/`ilens` in the track-fused call `:236-239`)
- Create: `tests/unit/dataset/test_haps_ffi_cache.py`

**Interfaces:**
- Produces: `Haps.ffi_static -> _HapsFfiStatic` (cached) with fields:
  - `v_starts: NDArray[np.int32]` (from `variants.start`, int64→int32)
  - `ilens: NDArray[np.int32]` (from `variants.ilen`)
  - `alt_alleles: NDArray[np.uint8]` (from `variants.alt.data.view(np.uint8)`)
  - `alt_offsets: NDArray[np.int64]` (from `variants.alt.offsets`)
  - `ref: NDArray[np.uint8] | None` (from `reference.reference`; `None` if no reference)
  - `ref_offsets: NDArray[np.int64] | None` (from `reference.offsets`; `None` if no reference)
- Consumes: `self.variants` (`_Variants`), `self.reference` (`Reference | None`).
- **Excluded from caching:** per-sample-scale arrays (genotypes) — those are governed by Task 4.

- [ ] **Step 1: Write the failing cache test**

Create `tests/unit/dataset/test_haps_ffi_cache.py`:

```python
"""Haps caches FFI-ready sub-linear arrays once (Task 5)."""

from __future__ import annotations

import numpy as np

import genvarloader as gvl
from genvarloader._dataset._haps import Haps


def _haps(track_dataset_path, reference) -> Haps:
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_seqs("haplotypes")
    seqs = ds._seqs
    assert isinstance(seqs, Haps)
    return seqs


def test_ffi_static_cached(track_dataset_path, reference):
    haps = _haps(track_dataset_path, reference)
    first = haps.ffi_static
    second = haps.ffi_static
    assert first is second  # cached, computed once


def test_ffi_static_contiguous_and_typed(track_dataset_path, reference):
    s = _haps(track_dataset_path, reference).ffi_static
    assert s.v_starts.dtype == np.int32 and s.v_starts.flags["C_CONTIGUOUS"]
    assert s.ilens.dtype == np.int32 and s.ilens.flags["C_CONTIGUOUS"]
    assert s.alt_alleles.dtype == np.uint8 and s.alt_alleles.flags["C_CONTIGUOUS"]
    assert s.alt_offsets.dtype == np.int64 and s.alt_offsets.flags["C_CONTIGUOUS"]
    assert s.ref is not None and s.ref.dtype == np.uint8 and s.ref.flags["C_CONTIGUOUS"]
    assert s.ref_offsets is not None and s.ref_offsets.dtype == np.int64


def test_ffi_static_v_starts_matches_source(track_dataset_path, reference):
    haps = _haps(track_dataset_path, reference)
    np.testing.assert_array_equal(
        haps.ffi_static.v_starts, np.asarray(haps.variants.start, np.int32)
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_haps_ffi_cache.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — `AttributeError: 'Haps' object has no attribute 'ffi_static'` (and `_HapsFfiStatic` import would fail if referenced).

- [ ] **Step 3: Add the cache dataclass and property**

In `python/genvarloader/_dataset/_haps.py`, add a small dataclass above `class Haps` (near the existing `@dataclass(slots=True)` at `:238`):

```python
@dataclass(slots=True)
class _HapsFfiStatic:
    """FFI-ready, contiguous, correctly-typed sub-linear arrays consumed by the
    fused kernels. Grows only with the variant/reference count (sub-linear in
    samples), so it is cached for the lifetime of the Haps reconstructor."""

    v_starts: NDArray[np.int32]
    ilens: NDArray[np.int32]
    alt_alleles: NDArray[np.uint8]
    alt_offsets: NDArray[np.int64]
    ref: "NDArray[np.uint8] | None"
    ref_offsets: "NDArray[np.int64] | None"
```

On the `Haps` dataclass, add a private cache field. Place it among the other `field(init=False)` declarations (e.g. after `available_var_fields: list[str] = field(init=False)` at `:262`):

```python
    _ffi_static: "_HapsFfiStatic | None" = field(default=None, init=False)
```

And add the property (anywhere in the `Haps` class body, e.g. after `__post_init__`):

```python
    @property
    def ffi_static(self) -> _HapsFfiStatic:
        """Lazily-computed, cached FFI-ready sub-linear arrays (see _HapsFfiStatic)."""
        if self._ffi_static is None:
            ref = self.reference
            self._ffi_static = _HapsFfiStatic(
                v_starts=np.ascontiguousarray(self.variants.start, np.int32),
                ilens=np.ascontiguousarray(self.variants.ilen, np.int32),
                alt_alleles=np.ascontiguousarray(
                    self.variants.alt.data.view(np.uint8), np.uint8
                ),
                alt_offsets=np.ascontiguousarray(self.variants.alt.offsets, np.int64),
                ref=None if ref is None else np.ascontiguousarray(ref.reference, np.uint8),
                ref_offsets=None
                if ref is None
                else np.ascontiguousarray(ref.offsets, np.int64),
            )
        return self._ffi_static
```

(`Haps` is `@dataclass(slots=True)` but not frozen, so assigning `self._ffi_static` is allowed; `NDArray` is already imported in `_haps.py`.)

- [ ] **Step 4: Use the cache in the fused haps/annotated/splice calls**

In `python/genvarloader/_dataset/_haps.py`, at the plain fused call (`:797-806`) replace:

```python
                    v_starts=np.ascontiguousarray(self.variants.start, np.int32),
                    ilens=np.ascontiguousarray(self.variants.ilen, np.int32),
                    alt_alleles=np.ascontiguousarray(
                        self.variants.alt.data.view(np.uint8), np.uint8
                    ),
                    alt_offsets=np.ascontiguousarray(
                        self.variants.alt.offsets, np.int64
                    ),
                    ref_=np.ascontiguousarray(self.reference.reference, np.uint8),
                    ref_offsets=np.ascontiguousarray(self.reference.offsets, np.int64),
```

with:

```python
                    v_starts=self.ffi_static.v_starts,
                    ilens=self.ffi_static.ilens,
                    alt_alleles=self.ffi_static.alt_alleles,
                    alt_offsets=self.ffi_static.alt_offsets,
                    ref_=self.ffi_static.ref,
                    ref_offsets=self.ffi_static.ref_offsets,
```

Apply the identical replacement at the spliced fused call (`:870-877`) and the annotated fused call (`:959-970`), matching each call's indentation. (Each of those three sites asserts `self.reference is not None` upstream, so `ffi_static.ref`/`ref_offsets` are non-`None` there.)

- [ ] **Step 5: Use the cache in the track-fused call**

In `python/genvarloader/_dataset/_reconstruct.py`, at the `intervals_and_realign_track_fused(...)` call (`:236-239`) replace:

```python
                        v_starts=np.ascontiguousarray(
                            self.haps.variants.start, np.int32
                        ),
                        ilens=np.ascontiguousarray(self.haps.variants.ilen, np.int32),
```

with:

```python
                        v_starts=self.haps.ffi_static.v_starts,
                        ilens=self.haps.ffi_static.ilens,
```

- [ ] **Step 6: Run the cache test**

Run: `pixi run -e dev pytest tests/unit/dataset/test_haps_ffi_cache.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (3 tests).

- [ ] **Step 7: Run parity + scale-guard on both backends**

Run: `pixi run -e dev pytest tests/parity tests/dataset tests/unit tests/integration -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (scale-guard still green — `v_starts` is no longer recast from a memmap per batch).

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

- [ ] **Step 8: Lint, format, typecheck, commit**

Run: `pixi run -e dev ruff format python/ tests/ && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

```bash
rtk git add python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_reconstruct.py tests/unit/dataset/test_haps_ffi_cache.py
rtk git commit -m "perf(haps): cache FFI-ready sub-linear per-variant arrays

Compute v_starts(int32)/ilens/alt/ref once per reconstructor instead of
re-coercing every batch (chiefly the int64->int32 v_starts recast).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Skip zero-initialization where provably full-write (Component F)

Replace `Array1::zeros(total)` with uninitialized allocation in the fused kernels, **only** for buffers the reconstruct/track core overwrites at every position. Isolated in its own commit so it can be reverted independently — this is the one component where parity could regress if the full-write invariant is wrong.

**Files:**
- Modify: `src/ffi/mod.rs` (add `uninit_output` helper; apply at the data-buffer allocations `:453`, `:530`, `:669`, `:670`, `:671`; conditionally `:867`)

**Interfaces:**
- Produces: `fn uninit_output<T: Copy>(len: usize) -> Array1<T>` — an uninitialized owned buffer; safe only when every element is written before any read.
- **Do NOT touch** the `out_offsets_vec` allocations (`:432`, `:648`) — those are read during incremental accumulation.

- [ ] **Step 1: Establish the parity baseline (both backends)**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev cargo test`
Expected: PASS (clean starting point before the risky change).

Run: `pixi run -e dev pytest tests/parity/test_reconstruct_haplotypes_parity.py tests/parity/test_fused_haps_parity.py tests/parity/test_fused_tracks_parity.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

- [ ] **Step 2: Add the uninitialized-allocation helper**

In `src/ffi/mod.rs`, add near the top of the module (after the imports, before the first `#[pyfunction]`):

```rust
/// Allocate an output buffer of `len` elements WITHOUT zero-initialization.
///
/// SAFETY/INVARIANT: every element is fully overwritten by the reconstruct/track
/// core before it is read. For in-contract inputs the core writes every output
/// position; out-of-contract inputs (e.g. a deletion driving `ref_idx` past the
/// contig end) are already undefined and excluded from the parity oracle by the
/// overshoot/double-init guards in
/// tests/parity/test_reconstruct_haplotypes_parity.py, so skipping the zero-init
/// adds no new observable exposure. `T` is a plain numeric type (u8/i32/f32) with
/// no invalid bit patterns.
#[allow(clippy::uninit_vec)]
fn uninit_output<T: Copy>(len: usize) -> Array1<T> {
    let mut v: Vec<T> = Vec::with_capacity(len);
    // SAFETY: see function-level invariant — every element is written before read.
    unsafe {
        v.set_len(len);
    }
    Array1::from_vec(v)
}
```

- [ ] **Step 3: Apply to the plain fused haplotype buffer**

In `src/ffi/mod.rs:453` replace:

```rust
    let mut out_data: Array1<u8> = Array1::zeros(total);
```

with:

```rust
    let mut out_data: Array1<u8> = uninit_output(total);
```

- [ ] **Step 4: Apply to the spliced fused haplotype buffer**

In `src/ffi/mod.rs:530` replace the same `Array1::zeros(total)` for `out_data` with `uninit_output(total)`.

- [ ] **Step 5: Apply to the annotated fused buffers**

In `src/ffi/mod.rs:669-671` replace:

```rust
    let mut out_data: Array1<u8> = Array1::zeros(total);
    let mut annot_v: Array1<i32> = Array1::zeros(total);
    let mut annot_pos: Array1<i32> = Array1::zeros(total);
```

with:

```rust
    let mut out_data: Array1<u8> = uninit_output(total);
    let mut annot_v: Array1<i32> = uninit_output(total);
    let mut annot_pos: Array1<i32> = uninit_output(total);
```

- [ ] **Step 6: Verify the tracks scratch buffer is full-write before converting**

The tracks-fused scratch (`src/ffi/mod.rs:867`, `Array1::<f32>::zeros(scratch_len)`) is filled by `intervals::intervals_to_tracks` and then read by `shift_and_realign_tracks_sparse`. Read `intervals_to_tracks` (in `src/intervals.rs` or wherever the core lives — find with `grep -rn "fn intervals_to_tracks" src/`) and confirm it writes **every** position of the scratch slice for in-contract inputs. If any scratch position can be left unwritten (a gap defaulting to 0 that the downstream read relies on), **leave `:867` as `Array1::zeros`** and add a one-line comment explaining why it must stay zero-initialized. If it is provably full-write, replace `:867`:

```rust
    let mut scratch = uninit_output::<f32>(scratch_len);
```

Record your determination in the commit message.

- [ ] **Step 7: Rebuild and run cargo tests + clippy**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev cargo test && pixi run -e dev cargo clippy`
Expected: PASS, clippy clean (the `#[allow(clippy::uninit_vec)]` is scoped to the helper).

- [ ] **Step 8: Run the reconstruct/track parity suites on both backends**

Run: `pixi run -e dev pytest tests/parity/test_reconstruct_haplotypes_parity.py tests/parity/test_fused_haps_parity.py tests/parity/test_fused_tracks_parity.py tests/parity/test_spliced_haplotypes_parity.py -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS. (If any parity test now fails, the full-write invariant was wrong for that buffer — revert the offending `uninit_output` line back to `Array1::zeros` and re-run.)

- [ ] **Step 9: Full suite + commit**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

```bash
rtk git add src/ffi/mod.rs
rtk git commit -m "perf(ffi): skip zero-init of fully-overwritten fused output buffers

Allocate out_data/annot_v/annot_pos (and scratch where verified full-write)
uninitialized; the reconstruct/track core writes every in-contract position.
Out-of-contract inputs are already excluded from the parity oracle. Isolated
for independent revert.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Documentation — SKILL.md + roadmap

Per `CLAUDE.md`, the new public symbol (`migrate`) and the on-disk format bump require a `skills/genvarloader/SKILL.md` update; the roadmap is the source of truth for the migration targets.

**Files:**
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `docs/roadmaps/rust-migration.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Read the current skill and roadmap sections**

Run: `rtk read skills/genvarloader/SKILL.md`
Read the "open a dataset" workflow section and the "Common gotchas" / "Where to look next" pointer table.

Run: `rtk read docs/roadmaps/rust-migration.md`
Find the Phase 3 optimization targets (targets 1–2 and the zero-init part of target 3) referenced by the spec.

- [ ] **Step 2: Update SKILL.md**

In `skills/genvarloader/SKILL.md`:
- In the open-a-dataset workflow, add a note that datasets written by genvarloader < 2.0 must be upgraded once with `genvarloader.migrate(path)` (in place, streaming, idempotent, crash-safe), and that opening a pre-2.0 dataset raises a `ValueError` with that hint.
- Add `migrate(path)` to the public-API surface listing (it is now in `__all__`).
- Note that format 2.0 stores track intervals as struct-of-arrays (`starts/ends/values.npy`) rather than the 1.x `intervals.npy` record array — relevant to anyone inspecting a dataset directory on disk.
- Re-check the "Common gotchas" and "Where to look next" pointer table for accuracy against this change.

- [ ] **Step 3: Update the roadmap**

In `docs/roadmaps/rust-migration.md`:
- Tick the optimization targets addressed: the track-interval AoS→SoA copy (target 1), the genotype `ascontiguousarray` footgun + sub-linear caching (target 2), and the zero-init skip portion of target 3.
- Record throughput: re-run `pixi run -e dev pytest tests/benchmarks/test_e2e.py -q --basetemp=$(pwd)/.pytest_tmp` on both `GVL_BACKEND=rust` and `GVL_BACKEND=numba` and note the rust tracks/annotated numbers (expected to close further on numba now the per-batch interval copy is gone). Recorded, not gated.
- Set the relevant phase status marker (⬜/🚧/✅) and link this PR.

- [ ] **Step 4: Commit**

```bash
rtk git add skills/genvarloader/SKILL.md docs/roadmaps/rust-migration.md
rtk git commit -m "docs: document gvl.migrate + format 2.0 SoA; record throughput

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Final full-tree verification before integration**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (whole tree, both dataset and unit).

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

Run: `pixi run -e dev cargo test && pixi run -e dev cargo clippy && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: all clean.

---

## Self-Review

**Spec coverage:**
- Component A (AoS→SoA + version bump) → Task 1, incl. the **two Rust writers** (`bigwig.rs`, `tables.rs`) the spec's "no Rust change" note missed, plus their oracle byte tests, and all four Python/Rust writers + the reader.
- Component B (version gate) → Task 2.
- Component C (`gvl.migrate`) → Task 3.
- Component D (zero-copy FFI + `_ffi_array` guard) → Task 4, incl. the scale-guard gate.
- Component E (cache sub-linear arrays) → Task 5.
- Component F (skip zero-init) → Task 6, with the scratch-buffer full-write verification the spec flagged as the one parity-risk site.
- Testing & parity (round-trip, version gate, scale-guard, FFI-guard) → Tasks 1–5 tests; both-backend parity runs in every task.
- SKILL.md + roadmap → Task 7.

**Placeholder scan:** every code step shows complete code; every run step shows the exact command and expected result. The one deliberately conditional step (Task 6 Step 6, scratch buffer) gives an explicit decision rule and both outcomes, because correctness there depends on a fact (`intervals_to_tracks` full-write) that must be verified in-repo, not assumed.

**Type/name consistency:** `_ffi_array(arr, dtype, name)` (Task 4) is consumed unchanged in Task 4 call sites. `_HapsFfiStatic` field names (`v_starts`, `ilens`, `alt_alleles`, `alt_offsets`, `ref`, `ref_offsets`) (Task 5) match the kernel kwargs (`v_starts`, `ilens`, `alt_alleles`, `alt_offsets`, `ref_`, `ref_offsets`) — note the kernel kwarg is `ref_` but the cache field is `ref`; the call sites map `ref_=self.ffi_static.ref`. `track_dataset_path` fixture (Task 1) is reused by Tasks 2–5. `DATASET_FORMAT_VERSION` and `_check_dataset_format_version` (Tasks 1–2) are imported consistently. `uninit_output<T>` (Task 6) is applied only to data buffers, never to `out_offsets_vec`.

**Notes carried forward for the implementer:**
- The second, unused `INTERVAL_DTYPE` at `_types.py:18` is intentionally left untouched (not on any path).
- `_as_starts_stops` / `_geno_offsets_2d` are intentionally unchanged (output base is not a memmap → never trips the scale-guard).
- After Rust edits, always `maturin develop --release` before Python tests.
