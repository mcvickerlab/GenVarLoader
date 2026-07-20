# StreamingDataset VCF/PGEN Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce per-epoch wall-clock of `StreamingDataset.to_iter()` for the VCF and PGEN backends at cohort scale (1k→50k samples), gated by deterministic counters, and add a permanent `gvl.Dataset`-vs-streaming benchmark comparison — all in PR #299.

**Architecture:** Measurement-first (`skills/performant-py-rust`). First land two deterministic decode counters + a `--compare-dataset` harness arm and capture a same-session baseline. Then apply two counter-gated Rust optimizations — narrow PGEN's per-window variant range (`src/record_stream/pgen.rs`) and rewrite the dense→sparse transpose to a word-level two-pass counting sort (`src/record_stream/transpose.rs`) — each guarded by byte-identical parity. A third lever (reader reuse / genoray `PvarReader` seek) is conditional on the profile.

**Tech Stack:** Rust (PyO3, `genoray_core`), Python (`genvarloader`, `polars`, `numpy`), `pixi` envs, `pytest` + `cargo test`, `perf`/`perf stat` for profiling.

## Global Constraints

- **Byte-identical parity is non-negotiable.** Every optimization must keep `tests/dataset/test_streaming_vcf_parity.py` and `tests/dataset/test_streaming_pgen_parity.py` (streamed haplotypes == `Dataset.open(...)[r,s]`, `jitter=0`, haplotypes-only) green, and the Rust filler-equivalence tests in `src/record_stream/pgen.rs` + `transpose.rs`.
- **Rebuild before Python tests:** run `pixi run -e dev maturin develop --release` after ANY `src/` edit before running pytest — pytest imports the stale `.so` otherwise. `cargo test` compiles from source and is exempt.
- **Rust test command:** `pixi run -e dev cargo-test` (sets the env `cargo test --release` needs; the bare `cargo test` binary can fail to load libpython without `LD_LIBRARY_PATH=.pixi/envs/dev/lib`).
- **Perf gate = deterministic counters + same-session before/after.** Wall-clock on this shared node is secondary color only (best/median-of-N), NEVER a pass/fail gate (`gvl-rust-perf-gate-shared-node-noise`).
- **Sweep points:** `n_samples ∈ {1000, 10000, 50000}` at ~200 regions/contig.
- **Scope guard (YAGNI):** no N-slot ring, no multiallelic PGEN, no `.pvar.zst`, no output-mode breadth (#277). Throughput of the existing haplotypes-only / `jitter=0` path only.
- **Branch:** `spec/276-vcf-pgen` (PR #299). Conventional-commit messages (commitizen-checked by prek).

---

## File Structure

- `src/record_stream/transpose.rs` — **modify.** Add `TRANSPOSE_WORD_READS` counter (Task 1); rewrite `fill_decoded_window` to word-level two-pass counting sort (Task 5).
- `src/record_stream/pgen.rs` — **modify.** Add `PGEN_VARIANTS_DECODED` counter (Task 1); retain per-contig `POS`/`max_ref_len` at construction and narrow `var_start`/`var_end` per window (Task 4).
- `src/ffi/mod.rs` — **modify.** FFI wrappers `transpose_word_reads` / `transpose_word_reads_reset` and `pgen_variants_decoded` / `pgen_variants_decoded_reset` (Task 1).
- `src/lib.rs` — **modify.** Register the four new pyfunctions (Task 1).
- `benchmarking/streaming/bench_streaming.py` — **modify.** `--compare-dataset` arm + `bytes_emitted` cross-driver assertion (Task 2); print the two counters (Task 3).
- `tests/dataset/test_streaming_scale.py` — **modify/create sections.** Counter scale gates for PGEN narrowing (Task 4) and transpose (Task 5).
- `docs/roadmaps/streaming-dataset.md`, `src/record_stream/pgen.rs` doc, `bench_streaming.py` docstring — **modify** (Task 7).

---

## Task 1: Deterministic decode counters (Rust foundation)

Two process-wide `AtomicUsize` counters mirroring `src/svar1/store.rs::CSR_ENTRIES_TOUCHED`, incremented on the producer thread, read from Python. This ships first: baseline + every optimization gate depends on it. No algorithm changes here — the current coarse/strided code is instrumented as-is to establish the "before" numbers.

**Files:**
- Modify: `src/record_stream/transpose.rs` (top of file + inside `fill_decoded_window`)
- Modify: `src/record_stream/pgen.rs` (top of file + inside `PgenWindowFiller::fill`)
- Modify: `src/ffi/mod.rs` (near the existing `svar1_csr_entries_touched`, ~line 1152)
- Modify: `src/lib.rs` (near line 57)

**Interfaces:**
- Produces (Rust, `crate::record_stream::transpose`): `pub fn word_reads() -> usize`, `pub fn word_reads_reset()`.
- Produces (Rust, `crate::record_stream::pgen`): `pub fn variants_decoded() -> usize`, `pub fn variants_decoded_reset()`.
- Produces (Python, `genvarloader._core` / the extension module): `transpose_word_reads() -> int`, `transpose_word_reads_reset() -> None`, `pgen_variants_decoded() -> int`, `pgen_variants_decoded_reset() -> None`.

- [ ] **Step 1: Write the failing Rust test for the transpose counter**

In `src/record_stream/transpose.rs`, inside `mod tests`, add:

```rust
#[test]
fn word_reads_counter_tracks_transpose_work() {
    word_reads_reset();
    assert_eq!(word_reads(), 0);
    let chunk = dense_fixture(); // 2 var, 2 samples, ploidy 2
    let mut slot = DecodedWindow::default();
    fill_decoded_window(&chunk, 2, 2, &mut slot);
    // Current (pre-rewrite) impl calls get_bit once per (v,s,p) cell = 2*2*2 = 8.
    assert_eq!(word_reads(), 8);
}
```

- [ ] **Step 2: Write the failing Rust test for the PGEN counter**

In `src/record_stream/pgen.rs`, inside `mod tests`, add:

```rust
#[test]
fn variants_decoded_counter_tracks_range_width() {
    variants_decoded_reset();
    let filler = PgenWindowFiller::new(&pgen_fixture_path(), &["s1", "s2"]).unwrap();
    let job = RecordJob { contig_idx: 0, regions: vec![(0, 100)], s_lo: 0, s_hi: 2 };
    let contig = ContigRef { name: "chr1".into(), ref_bytes: vec![b'A'; 100] };
    let mut slot = DecodedWindow::default();
    filler.fill(&job, &contig, &mut slot).unwrap();
    // The two_var fixture has one contig with 2 variants; coarse var_start decodes
    // the whole contig range [0, 2) = 2 variants.
    assert_eq!(variants_decoded(), 2);
}
```

- [ ] **Step 3: Run both tests to verify they fail**

Run: `pixi run -e dev cargo-test -- word_reads_counter_tracks_transpose_work variants_decoded_counter_tracks_range_width`
Expected: FAIL — `word_reads`, `word_reads_reset`, `variants_decoded`, `variants_decoded_reset` are not defined.

- [ ] **Step 4: Add the transpose counter + instrument the current loop**

At the top of `src/record_stream/transpose.rs` (after the `use` line):

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// Process-wide count of genotype-grid word reads performed by the transpose
/// (see `crate::svar1::store::csr_entries_touched` for the pattern). Deterministic
/// given the input grid; the perf gate on this noisy node, not wall-clock.
static WORD_READS: AtomicUsize = AtomicUsize::new(0);

pub fn word_reads() -> usize {
    WORD_READS.load(Ordering::Relaxed)
}
pub fn word_reads_reset() {
    WORD_READS.store(0, Ordering::Relaxed);
}
```

Inside the existing `fill_decoded_window` hap-major loop, add one increment per `get_bit` call:

```rust
    for s in 0..n_samples {
        for p in 0..ploidy {
            for v in 0..n_var {
                let flat_idx = v * n_samples * ploidy + s * ploidy + p;
                WORD_READS.fetch_add(1, Ordering::Relaxed);
                if chunk.genos.get_bit(flat_idx) {
                    slot.geno_v_idxs.push(v as i32);
                }
            }
            slot.geno_offsets.push(slot.geno_v_idxs.len() as i64);
        }
    }
```

- [ ] **Step 5: Add the PGEN counter + instrument the current range**

At the top of `src/record_stream/pgen.rs` (after the `use` block):

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

/// Process-wide count of variants pgenlib is asked to decode across all windows
/// (`var_end - var_start` per `fill`). With the coarse per-contig range this is
/// `contig_prefix * n_windows`; Task 4's narrowing drops it toward `Σ window
/// variants`. Deterministic; the PGEN perf gate.
static VARIANTS_DECODED: AtomicUsize = AtomicUsize::new(0);

pub fn variants_decoded() -> usize {
    VARIANTS_DECODED.load(Ordering::Relaxed)
}
pub fn variants_decoded_reset() {
    VARIANTS_DECODED.store(0, Ordering::Relaxed);
}
```

In `PgenWindowFiller::fill`, immediately after `let (var_start, var_end) = ...`:

```rust
        VARIANTS_DECODED.fetch_add(var_end.saturating_sub(var_start), Ordering::Relaxed);
```

- [ ] **Step 6: Add the four FFI wrappers**

In `src/ffi/mod.rs`, next to `svar1_csr_entries_touched` (~line 1152):

```rust
/// Deterministic transpose work counter — see `crate::record_stream::transpose::word_reads`.
#[pyfunction]
pub fn transpose_word_reads() -> usize {
    crate::record_stream::transpose::word_reads()
}

#[pyfunction]
pub fn transpose_word_reads_reset() {
    crate::record_stream::transpose::word_reads_reset()
}

/// Deterministic PGEN decode-width counter — see `crate::record_stream::pgen::variants_decoded`.
#[pyfunction]
pub fn pgen_variants_decoded() -> usize {
    crate::record_stream::pgen::variants_decoded()
}

#[pyfunction]
pub fn pgen_variants_decoded_reset() {
    crate::record_stream::pgen::variants_decoded_reset()
}
```

(If `word_reads`/`variants_decoded` are not visible, add `pub` re-exports in `src/record_stream/mod.rs` — check how `fill_decoded_window` is re-exported there and mirror it.)

- [ ] **Step 7: Register the pyfunctions**

In `src/lib.rs`, next to the `svar1_csr_entries_touched` registration (~line 57):

```rust
    m.add_function(wrap_pyfunction!(ffi::transpose_word_reads, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::transpose_word_reads_reset, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::pgen_variants_decoded, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::pgen_variants_decoded_reset, m)?)?;
```

- [ ] **Step 8: Run the Rust tests to verify they pass**

Run: `pixi run -e dev cargo-test -- word_reads_counter_tracks_transpose_work variants_decoded_counter_tracks_range_width`
Expected: PASS (both).

- [ ] **Step 9: Verify the Python-visible symbols exist**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev python -c "import genvarloader as g; from genvarloader import _core as c; c.transpose_word_reads_reset(); c.pgen_variants_decoded_reset(); print(c.transpose_word_reads(), c.pgen_variants_decoded())"`
Expected: prints `0 0`. (If `_core` is not the module name, find the extension module — `grep -rn 'from .* import' python/genvarloader/__init__.py` — and use that; the four names must resolve.)

- [ ] **Step 10: Commit**

```bash
git add src/record_stream/transpose.rs src/record_stream/pgen.rs src/ffi/mod.rs src/lib.rs
git commit -m "perf(streaming): deterministic transpose + PGEN decode counters"
```

---

## Task 2: `--compare-dataset` harness arm (Python)

Add a third driver to `bench_streaming.py` that writes a `gvl.Dataset` once (preprocessing cost reported separately) and iterates it over the SAME (region, sample) plan order as the streaming run, then asserts `bytes_emitted` matches across drivers. Independent of Task 1 (different file) — runs in parallel.

**Files:**
- Modify: `benchmarking/streaming/bench_streaming.py`

**Interfaces:**
- Consumes: `gvl.write`, `gvl.Dataset`, `StreamingDataset._plan`, `StreamingDataset._regions`, the existing `RunResult` dataclass and `_prepare_fixture`/`FixturePaths`.
- Produces: `_drive_dataset(sds, fp, batch_size) -> tuple[RunResult, float, int]` (result, write_time_s, dataset_bytes); a `"dataset"` entry in `_DRIVERS`-style dispatch; a `--compare-dataset` CLI flag.

- [ ] **Step 1: Add the `--compare-dataset` flag**

In `main()`'s argparse block, after `--strategy`:

```python
    p.add_argument(
        "--compare-dataset",
        action="store_true",
        help="also write a gvl.Dataset once (write time + on-disk size reported "
        "SEPARATELY as preprocessing cost) and time a full region-major "
        "Dataset[r,s] sweep over the identical (region, sample) cells, for a "
        "streaming-vs-written throughput comparison (byte-identical, so bytes_emitted "
        "is cross-checked equal across drivers)",
    )
```

- [ ] **Step 2: Write the dataset driver**

Add near `_drive_sync` (needs `import tempfile`, `import shutil` — already imported):

```python
def _drive_dataset(
    sds: "gvl.StreamingDataset", fp: "FixturePaths", batch_size: int
) -> tuple[RunResult, float, int]:
    """Write a gvl.Dataset once from the SAME variants+reference+bed, then iterate
    it over the identical region-major (region, sample) plan order the streaming
    run uses (haplotypes-only, jitter=0). Returns (sweep RunResult, write_time_s,
    dataset_bytes). Write cost is preprocessing, reported separately."""
    ds_dir = Path(tempfile.mkdtemp(prefix="gvl_bench_ds_"))
    try:
        t_w = time.perf_counter()
        gvl.write(
            path=ds_dir / "ds",
            bed=sds._regions_bed if hasattr(sds, "_regions_bed") else fp.bed,
            variants=fp.variants,
            reference=fp.reference,
            max_jitter=0,
        )
        write_time = time.perf_counter() - t_w
        ds_bytes = sum(f.stat().st_size for f in (ds_dir / "ds").rglob("*") if f.is_file())

        ds = gvl.Dataset.open(ds_dir / "ds", reference=fp.reference).with_seqs("haplotypes")

        t0 = time.perf_counter()
        n_batches = n_rows = n_bytes = 0
        for r_idx, s_idx in sds._plan():
            haps = ds[r_idx, s_idx]  # region-major, same plan cells
            data = haps.data if hasattr(haps, "data") else haps
            n_batches += 1
            n_rows += len(r_idx) * len(s_idx)
            n_bytes += np.asarray(data).nbytes
        elapsed = time.perf_counter() - t0
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (
            RunResult(elapsed, n_batches, n_batches, n_rows, n_bytes, peak_rss),
            write_time,
            ds_bytes,
        )
    finally:
        shutil.rmtree(ds_dir, ignore_errors=True)
```

**Interface note for the implementer:** confirm the exact `gvl.write` signature and `Dataset.__getitem__` return shape against `skills/genvarloader/SKILL.md` / `python/genvarloader/_dataset/_impl.py` before running — the parameter names (`path`/`output_path`, `bed`/`regions`) and whether the BED must be passed as a path vs DataFrame may differ; adjust the call, keep the semantics (same variants/reference/bed, `max_jitter=0`, haplotypes). The `bytes_emitted` cross-check in Step 4 will catch a wrong access pattern.

- [ ] **Step 3: Wire the driver into `bench_one`**

In `bench_one`, after the engine/sync loop, gate on the flag (thread `compare_dataset: bool` through `bench_one`'s signature and the `main()` call site):

```python
    if compare_dataset:
        sds = gvl.StreamingDataset(
            fp.bed, reference=fp.reference, variants=fp.variants
        ).with_seqs("haplotypes")
        ds_result, write_time, ds_bytes = _drive_dataset(sds, fp, batch_size)
        last_result["dataset"] = ds_result
        print(
            f"[dataset] sweep {ds_result.elapsed_s:.3f}s  rows={ds_result.n_rows} "
            f"bytes={ds_result.bytes_emitted} peak_rss_kb={ds_result.peak_rss_kb}  "
            f"| preprocessing: write={write_time:.3f}s disk={ds_bytes} bytes"
        )
```

- [ ] **Step 4: Add the cross-driver `bytes_emitted` assertion**

After the print loop in `bench_one`, add:

```python
    emitted = {k: r.bytes_emitted for k, r in last_result.items()}
    if len(set(emitted.values())) > 1:
        raise RuntimeError(
            f"{kind}: bytes_emitted diverged across drivers {emitted} — streaming "
            "and written Dataset must be byte-identical; a driver is not exercising "
            "the same cells"
        )
```

- [ ] **Step 5: Smoke-test the arm at tiny scale**

Generate a tiny fixture if none exists, then run (VCF only, 1 sample point):

Run: `N=1000 SIZE=5MB CONTIGS=chr1 pixi run -e dev gen-bench-vcf && pixi run -e dev python benchmarking/streaming/bench_streaming.py --n 1000 --backend vcf --strategy engine --repeats 1 --compare-dataset`
Expected: prints an `[dataset] sweep ...` line with a `preprocessing:` suffix; no `bytes_emitted diverged` error; the run completes.

- [ ] **Step 6: Commit**

```bash
git add benchmarking/streaming/bench_streaming.py
git commit -m "test(streaming): --compare-dataset arm (streaming vs written Dataset)"
```

---

## Task 3: Baseline capture + profile (measurement gate)

No code. Establish the same-session "before" numbers every later task is measured against, and run the profile that decides whether Task 6 is warranted. Depends on Tasks 1 + 2 merged and rebuilt.

**Files:** none (produces a report note under `.superpowers/sdd/` or the PR description).

- [ ] **Step 1: Rebuild and generate the sweep fixtures**

Run:
```bash
pixi run -e dev maturin develop --release
for N in 1000 10000 50000; do N=$N SIZE=50MB CONTIGS=chr1 pixi run -e dev gen-bench-vcf; N=$N pixi run -e dev gen-bench-pgen; done
```
Expected: `benchmarking/streaming/bench_{1000,10000,50000}.{bcf,pgen}` exist.

- [ ] **Step 2: Capture the baseline counters + wall-clock (same session)**

Run: `pixi run -e dev python benchmarking/streaming/bench_streaming.py --n 1000 --n 10000 --n 50000 --backend both --strategy both --repeats 3 --compare-dataset`
Record per (backend, N): `n_windows`, `pgen_variants_decoded` (call `c.pgen_variants_decoded()` before/after — add a print in the harness if not yet wired, else read via a one-liner), `transpose_word_reads`, best/median wall-clock, `peak_rss_kb`, and the `dataset` sweep + preprocessing line. Save to a report note.

- [ ] **Step 3: Profile VCF and PGEN at 10k**

Run (per backend):
```bash
pixi run -e dev perf record -g --call-graph=dwarf -o /carter/users/dlaub/.claude/jobs/2b4cf8e8/tmp/perf_vcf.data -- python benchmarking/streaming/bench_streaming.py --n 10000 --backend vcf --strategy engine --repeats 1
pixi run -e dev perf report -i /carter/users/dlaub/.claude/jobs/2b4cf8e8/tmp/perf_vcf.data --stdio | head -60
```
Also capture `perf stat -e instructions,LLC-load-misses` for the same command. Fill the Phase-1 dimensions table in the spec with measured per-stage shares (decode vs transpose vs reconstruct vs reader-open).

- [ ] **Step 4: Decide Task 6**

If the profile shows reader-open (`VcfRecordSource::new` / `PgenRecordSource::new`) or the PGEN `.pvar` text re-skip is a **material** share (rule of thumb: >10% of producer time at 10k), Task 6 proceeds. Otherwise mark Task 6 skipped in the plan and the report with the measured share as justification. Record the decision.

- [ ] **Step 5: Commit the baseline note**

```bash
git add .superpowers/sdd/optimization-baseline.md
git commit -m "docs(streaming): optimization baseline counters + profile (#276)"
```

---

## Task 4: Narrow PGEN `var_start`/`var_end` per window

Retain per-contig `POS` + `max_ref_len` from the construction-time `.pvar` scan, then binary-search each window's variant range instead of decoding the whole contig prefix. gvl-only. Depends on Task 1 (the counter). Parallel with Task 5 (different file).

**Files:**
- Modify: `src/record_stream/pgen.rs`
- Modify: `tests/dataset/test_streaming_scale.py`

**Interfaces:**
- Consumes: `crate::record_stream::pgen::variants_decoded` (Task 1); `RecordJob { regions: Vec<(u32,u32)>, .. }`, `ContigRef { name, .. }`.
- Changes `contig_var_ranges` to also return, per contig, the sorted `Vec<u32>` of POS (local order) and the `max_ref_len: u32`. Stored on `PgenWindowFiller` as `contig_pos: HashMap<String, Vec<u32>>` and `contig_max_ref_len: HashMap<String, u32>`.

- [ ] **Step 1: Write the failing Rust narrowing test**

In `src/record_stream/pgen.rs` `mod tests`, add (uses the existing `vcf_snp_ins_del_multi` fixture, which has variants spread across a contig so a narrow window must decode fewer than the whole prefix):

```rust
#[test]
fn narrowed_var_range_decodes_fewer_than_whole_contig() {
    // A window over a late, narrow region must NOT decode the whole contig prefix.
    let pgen = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/streaming/vcf_snp_ins_del_multi.pgen")
        .to_str().unwrap().to_string();
    // Derive the full-contig width via a whole-contig window.
    let filler = PgenWindowFiller::new(&pgen, &["s1", "s2"]).unwrap();
    let contig = ContigRef { name: "chr1".into(), ref_bytes: vec![b'A'; 100_000] };

    variants_decoded_reset();
    let wide = RecordJob { contig_idx: 0, regions: vec![(0, 100_000)], s_lo: 0, s_hi: 2 };
    let mut slot = DecodedWindow::default();
    filler.fill(&wide, &contig, &mut slot).unwrap();
    let wide_decoded = variants_decoded();

    variants_decoded_reset();
    // A 1bp window near the contig start: must decode far fewer than the whole contig.
    let narrow = RecordJob { contig_idx: 0, regions: vec![(0, 2)], s_lo: 0, s_hi: 2 };
    let mut slot2 = DecodedWindow::default();
    filler.fill(&narrow, &contig, &mut slot2).unwrap();
    let narrow_decoded = variants_decoded();

    assert!(
        narrow_decoded < wide_decoded,
        "narrow window decoded {narrow_decoded}, whole-contig {wide_decoded} — \
         var_start/var_end were not narrowed"
    );
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run -e dev cargo-test -- narrowed_var_range_decodes_fewer_than_whole_contig`
Expected: FAIL — coarse range makes `narrow_decoded == wide_decoded`.

- [ ] **Step 3: Extend `contig_var_ranges` to return POS + max_ref_len**

Change its signature and body to also collect, per contig, the local POS list and the max REF length. Return type becomes:

```rust
struct ContigIndex {
    ranges: HashMap<String, (usize, usize)>,
    pos: HashMap<String, Vec<u32>>,       // local-order POS (0-based), per contig
    max_ref_len: HashMap<String, u32>,    // max REF length seen, per contig (the pad)
}
```

In the row loop, after parsing `chrom`/`pos`/`alt`, also parse `REF` (locate the `REF` column alongside `POS`/`ALT`), convert POS `1-based → 0-based`, and:

```rust
    let ref_len = ref_field.len() as u32;
    index.pos.entry(chrom.clone()).or_default().push(pos0);
    let e = index.max_ref_len.entry(chrom.clone()).or_insert(0);
    *e = (*e).max(ref_len);
```

(The `pos` vector is naturally ascending because plink2 writes `.pvar` position-sorted within a contig — assert monotonic non-decreasing while building; a non-sorted `.pvar` is an error, matching the existing contiguity check.)

- [ ] **Step 4: Store the index on the struct and narrow in `fill`**

Add fields `contig_pos: HashMap<String, Vec<u32>>` and `contig_max_ref_len: HashMap<String, u32>` to `PgenWindowFiller`, populated in `new`. Replace the coarse range lookup in `fill`:

```rust
        let (contig_lo, contig_hi) = self
            .contig_ranges
            .get(&contig.name)
            .copied()
            .unwrap_or((0, 0));
        let (var_start, var_end) = match self.contig_pos.get(&contig.name) {
            Some(pos) if !pos.is_empty() => {
                let win_start = job.regions.iter().map(|r| r.0).min().unwrap_or(0);
                let win_end = job.regions.iter().map(|r| r.1).max().unwrap_or(0);
                let pad = self.contig_max_ref_len.get(&contig.name).copied().unwrap_or(0);
                // Lower bound: earliest variant whose extent (POS + ref_len) could reach
                // win_start. POS + ref_len > win_start  ==>  POS > win_start - ref_len.
                // Use the contig's max ref_len as a safe (over-inclusive) pad; the
                // OverlapMode::Variant filter inside PgenRecordSource still narrows the
                // OUTPUT exactly, so over-inclusion is harmless, under-inclusion is a bug.
                let lo_pos = win_start.saturating_sub(pad);
                let start_local = pos.partition_point(|&p| p < lo_pos);
                // Upper bound: first variant starting at/after win_end can't overlap.
                let end_local = pos.partition_point(|&p| p < win_end);
                (contig_lo + start_local, contig_lo + end_local)
            }
            _ => (contig_lo, contig_hi),
        };
        debug_assert!(var_start >= contig_lo && var_end <= contig_hi);
        VARIANTS_DECODED.fetch_add(var_end.saturating_sub(var_start), Ordering::Relaxed);
```

(Remove the old `VARIANTS_DECODED.fetch_add` line added in Task 1 — the increment now lives here after narrowing. Keep exactly one increment.)

- [ ] **Step 5: Run the narrowing test + all PGEN Rust tests**

Run: `pixi run -e dev cargo-test -- pgen`
Expected: PASS, including `narrowed_var_range_decodes_fewer_than_whole_contig` and every existing `pgen_filler_matches_vcf_filler_*` equivalence test (parity preserved).

- [ ] **Step 6: Rebuild and confirm end-to-end PGEN parity**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_pgen_parity.py -q`
Expected: PASS (byte-identical). This is the correctness oracle for the narrowing.

- [ ] **Step 7: Add a Python counter scale gate**

In `tests/dataset/test_streaming_scale.py`, add a test asserting `pgen_variants_decoded()` over a full multi-window sweep is far below `n_windows * pvar_variants` (i.e. the coarse worst case). Use the `vcf_snp_ins_del_multi` PGEN fixture and a multi-region bed forcing ≥2 windows:

```python
def test_pgen_variants_decoded_scales_with_window_not_contig(streaming_pgen_fixture):
    from genvarloader import _core as c
    sds = ...  # StreamingDataset over the pgen fixture, small window_regions to force >1 window
    n_windows = sum(1 for _ in sds._plan())
    pvar_variants = ...  # non-header line count of the sibling .pvar
    c.pgen_variants_decoded_reset()
    for _ in sds.to_iter(batch_size=4):
        pass
    decoded = c.pgen_variants_decoded()
    assert decoded < n_windows * pvar_variants, (
        f"decoded {decoded} ≈ coarse worst case {n_windows * pvar_variants}; "
        "var_start narrowing regressed"
    )
```

(Fill the `...` from the existing PGEN fixture setup in `tests/dataset/conftest.py` / `test_streaming_pgen*.py`.)

- [ ] **Step 8: Run the scale gate**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scale.py::test_pgen_variants_decoded_scales_with_window_not_contig -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/record_stream/pgen.rs tests/dataset/test_streaming_scale.py
git commit -m "perf(streaming): narrow PGEN per-window var_start/var_end via pvar POS index"
```

---

## Task 5: Word-level two-pass counting-sort transpose

Rewrite `fill_decoded_window` to iterate the `BitGrid3.words` (public `Vec<u64>`) in v-major flat order — sequential (cache-friendly) and skipping empty words via `trailing_zeros` (genotypes are sparse). Two passes: count per hap → prefix-sum offsets → fill. Depends on Task 1. Parallel with Task 4.

**Files:**
- Modify: `src/record_stream/transpose.rs`
- Modify: `tests/dataset/test_streaming_scale.py`

**Interfaces:**
- Consumes: `genoray_core::types::BitGrid3 { pub words: Vec<u64>, .. }` (flat index `v*S*P + s*P + p`, word-packed LSB-first, v slowest); `crate::record_stream::transpose::WORD_READS`.
- Produces: byte-identical `DecodedWindow` CSR (`geno_offsets`, `geno_v_idxs`) to the current implementation — the existing `transpose::tests` are the equivalence oracle.

- [ ] **Step 1: Add a larger sparse equivalence fixture + test**

In `transpose.rs` `mod tests`, add a test that builds a sparse grid two ways and checks the CSR matches a reference (hap-major) computation:

```rust
#[test]
fn word_transpose_matches_naive_on_sparse_grid() {
    let (n_var, n_samples, ploidy) = (37usize, 11usize, 2usize);
    let mut genos = BitGrid3::zeros(n_var, n_samples, ploidy);
    // Deterministic sparse pattern (~5% density).
    for v in 0..n_var {
        for s in 0..n_samples {
            for p in 0..ploidy {
                if (v * 7 + s * 3 + p) % 19 == 0 {
                    genos.or_bit(v * n_samples * ploidy + s * ploidy + p, true);
                }
            }
        }
    }
    // Reference: the old hap-major scan.
    let mut ref_idxs: Vec<i32> = Vec::new();
    let mut ref_offsets: Vec<i64> = vec![0];
    for s in 0..n_samples {
        for p in 0..ploidy {
            for v in 0..n_var {
                if genos.get_bit(v * n_samples * ploidy + s * ploidy + p) {
                    ref_idxs.push(v as i32);
                }
            }
            ref_offsets.push(ref_idxs.len() as i64);
        }
    }
    let chunk = DenseChunk {
        chunk_id: 0,
        pos: (0..n_var as u32).collect(),
        ilens: vec![0; n_var],
        alt: vec![b'A'; n_var],
        alt_offsets: (0..=n_var as i64).collect(),
        genos,
        info_staged: Vec::new(),
        format_staged: Vec::new(),
    };
    let mut slot = DecodedWindow::default();
    fill_decoded_window(&chunk, n_samples, ploidy, &mut slot);
    assert_eq!(slot.geno_offsets, ref_offsets);
    assert_eq!(slot.geno_v_idxs, ref_idxs);
}
```

- [ ] **Step 2: Run it to verify it passes on the CURRENT impl (guard against a bad oracle)**

Run: `pixi run -e dev cargo-test -- word_transpose_matches_naive_on_sparse_grid`
Expected: PASS — the current implementation already produces this CSR; the test locks in the exact expected output so the rewrite can't drift.

- [ ] **Step 3: Rewrite `fill_decoded_window`'s genotype transpose**

Replace the static-table copy prologue AS-IS, then replace the hap-major loop with:

```rust
    let plane = n_samples * ploidy; // hap stride; hap = s*ploidy + p = flat % plane
    let n_haps = plane;
    let total_bits = n_var * plane;
    let words = &chunk.genos.words;

    // Pass 1: count set bits per hap (offsets[hap+1] holds the count, then prefix-sum).
    slot.geno_offsets.clear();
    slot.geno_offsets.resize(n_haps + 1, 0);
    for (w_idx, &word) in words.iter().enumerate() {
        WORD_READS.fetch_add(1, Ordering::Relaxed);
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let flat = w_idx * 64 + b;
            if flat >= total_bits {
                continue; // padding bit in the final word (genoray leaves these 0, defensive)
            }
            let hap = flat % plane;
            slot.geno_offsets[hap + 1] += 1;
        }
    }
    for h in 0..n_haps {
        slot.geno_offsets[h + 1] += slot.geno_offsets[h];
    }
    let total = slot.geno_offsets[n_haps] as usize;

    // Pass 2: fill geno_v_idxs at per-hap cursors. v is the slowest-varying flat axis,
    // so iterating words in order yields ascending v per hap — the CSR contract.
    slot.geno_v_idxs.clear();
    slot.geno_v_idxs.resize(total, 0);
    let mut cursor: Vec<usize> = slot.geno_offsets[..n_haps]
        .iter()
        .map(|&o| o as usize)
        .collect();
    for (w_idx, &word) in words.iter().enumerate() {
        WORD_READS.fetch_add(1, Ordering::Relaxed);
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let flat = w_idx * 64 + b;
            if flat >= total_bits {
                continue;
            }
            let v = flat / plane;
            let hap = flat % plane;
            slot.geno_v_idxs[cursor[hap]] = v as i32;
            cursor[hap] += 1;
        }
    }
```

Remove the old per-cell `WORD_READS.fetch_add` (Task 1) — the counter now counts word reads (`2 * n_words`), the deterministic before/after signal.

- [ ] **Step 4: Run the full transpose test module**

Run: `pixi run -e dev cargo-test -- transpose`
Expected: PASS — `transpose_emits_variant_indices_hap_major`, `transpose_empty_chunk_is_all_zero_csr`, `transpose_monomorphic_hap_all_bits_on_one_variant`, and `word_transpose_matches_naive_on_sparse_grid` all green. `word_reads_counter_tracks_transpose_work` from Task 1 will now report `2 * n_words` instead of `V*S*P` — **update that test's expected value** to `2 * chunk.genos.words.len()` and note the metric change in a comment.

- [ ] **Step 5: Rebuild and confirm end-to-end parity, both backends**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_vcf_parity.py tests/dataset/test_streaming_pgen_parity.py -q`
Expected: PASS (byte-identical for both — the transpose is shared).

- [ ] **Step 6: Add a Python transpose word-reads scale gate**

In `tests/dataset/test_streaming_scale.py`, assert `transpose_word_reads()` over a sweep is `≈ 2 * ceil(total_bits/64)`, far below `total_bits` (`V*S*P`), on a cohort-scale fixture:

```python
def test_transpose_word_reads_below_cell_count(streaming_vcf_scale_fixture):
    from genvarloader import _core as c
    sds = ...  # StreamingDataset over a scale fixture (many samples)
    c.transpose_word_reads_reset()
    total_cells = 0
    for _ in sds.to_iter(batch_size=8):
        pass
    reads = c.transpose_word_reads()
    # word reads must be ~1/32 of the naive per-cell count (2 passes over words vs
    # one get_bit per cell); assert a comfortable margin.
    assert reads > 0
    # (compute total_cells = Σ window V*S*P if the fixture exposes it, else assert
    #  reads is small relative to n_samples * n_variants * ploidy)
```

(Fill `...`/`total_cells` from the scale-fixture helpers already in the test module; the essential assertion is `reads ≪ V*S*P`.)

- [ ] **Step 7: Run the gate**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scale.py::test_transpose_word_reads_below_cell_count -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/record_stream/transpose.rs tests/dataset/test_streaming_scale.py
git commit -m "perf(streaming): word-level two-pass counting-sort transpose"
```

---

## Task 6 (CONDITIONAL — only if Task 3 Step 4 says material): reader reuse / genoray PvarReader seek

Do this ONLY if the profile flagged reader-open or the PGEN `.pvar` text re-skip as a material producer-time share. Otherwise mark skipped with the measured share as justification and go to Task 7.

**Files:**
- Modify: `src/record_stream/vcf.rs` and/or `src/record_stream/pgen.rs`
- Possibly a separate genoray PR (`PvarReader::seek`) consumed via a `Cargo.toml` `rev` bump (both `genoray_core` + `svar2-codec` to the same rev).

**Interfaces:**
- Consumes: whichever genoray reader API the profile implicates. A genoray seek is Rust-side; merging it to genoray `main` + bumping the `rev` unblocks it (no release gate).

- [ ] **Step 1: Confirm the target from the profile**

State which reader (`VcfRecordSource::new` htslib open, or `PgenRecordSource::new`'s `.pvar` text re-skip) the profile named and its measured share. If neither is material, STOP — mark Task 6 skipped in the plan + report and proceed to Task 7.

- [ ] **Step 2: Write the failing open-count / reuse test**

Add a deterministic counter (mirror Task 1) for reader constructions, and a test asserting one full sweep constructs the reader once per contig, not once per window. (Exact code depends on the target chosen in Step 1 — write it against that reader.)

- [ ] **Step 3: Implement reader persistence (gvl-side) and/or bump the genoray rev for `PvarReader::seek`**

Persist the reader on the producer side (confined to the single producer thread — `engine.rs` guarantees strictly-sequential `fill`), or wire the genoray seek so `PgenRecordSource::new` seeks instead of text-skipping from byte 0.

- [ ] **Step 4: Rebuild + full parity, both backends**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_vcf_parity.py tests/dataset/test_streaming_pgen_parity.py -q`
Expected: PASS (byte-identical).

- [ ] **Step 5: Commit (+ note the genoray rev bump if any)**

```bash
git add -A
git commit -m "perf(streaming): reuse per-contig reader across windows"
```

---

## Task 7: Docs, roadmap, and full-tree gate

Record the measured numbers, mark the coarse-`var_start` limitation resolved, and run the full test tree (renames/shared-code touched → CI-parity).

**Files:**
- Modify: `docs/roadmaps/streaming-dataset.md`
- Modify: `src/record_stream/pgen.rs` (the "Coarse `var_start`" doc section)
- Modify: `benchmarking/streaming/bench_streaming.py` (module docstring)

- [ ] **Step 1: Update the roadmap**

In `docs/roadmaps/streaming-dataset.md`, under the #276 task block, add an "optimization pass" bullet recording: the measured streaming-vs-`Dataset` throughput ratio + peak-RSS (from Task 3), the PGEN `pgen_variants_decoded` before/after, and the transpose `transpose_word_reads` before/after. Point to this plan + the spec.

- [ ] **Step 2: Mark the pgen.rs coarse-var_start doc resolved**

Edit the "Coarse `var_start`: correct, but a known v1 perf limitation" section in `src/record_stream/pgen.rs` to describe the POS-index narrowing (with the `max_ref_len` pad) as the shipped behavior, keeping the correctness rationale for the pad.

- [ ] **Step 3: Update the bench_streaming.py docstring**

Replace the "KNOWN PGEN LIMITATION" analytic-multiplier paragraph with the measured `pgen_variants_decoded` counter story (narrowed to window), and document the new `--compare-dataset` arm.

- [ ] **Step 4: Run the full test tree (CI parity)**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests -q && pixi run -e dev cargo-test`
Expected: PASS across `tests/dataset` + `tests/unit` and all cargo tests (no stale references; renames/shared-code caught here, not in CI).

- [ ] **Step 5: Typecheck + lint**

Run: `pixi run -e dev typecheck && pixi run -e dev ruff check python/ tests/ benchmarking/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add docs/roadmaps/streaming-dataset.md src/record_stream/pgen.rs benchmarking/streaming/bench_streaming.py
git commit -m "docs(streaming): record VCF/PGEN optimization results + resolve coarse var_start (#276)"
```

---

## Self-Review notes (author)

- **Spec coverage:** harness arm (Task 2) + counters (Task 1) = Phase 1a/1b; profile/baseline (Task 3) = Phase 1c/3; PGEN narrowing (Task 4) = lever 1; transpose (Task 5) = lever 2; reader reuse/genoray seek (Task 6) = lever 3, correctly gated on the profile; docs (Task 7) = deliverable 4. All spec sections map to a task.
- **Parallelism (per user's SDD preference):** Wave 1 = {Task 1 (Rust counters), Task 2 (Python harness)} in parallel (disjoint files). Wave 2 = Task 3 (needs 1+2). Wave 3 = {Task 4 (pgen.rs), Task 5 (transpose.rs)} in parallel (disjoint files, both depend only on Task 1). Wave 4 = Task 6 (conditional). Wave 5 = Task 7. Tasks 4 and 5 both touch `tests/dataset/test_streaming_scale.py` — sequence their scale-test steps or have the second rebase that one file to avoid a conflict.
- **Counter-metric change is explicit:** Task 1 counts per-cell `get_bit` (= `V*S*P`); Task 5 redefines the same `WORD_READS` counter to per-word reads (= `2*n_words`) and updates the Task-1 unit test's expected value in the same commit. Same unit (word reads), so before/after is comparable.
