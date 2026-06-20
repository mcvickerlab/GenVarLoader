# Design: Single-pass streaming bigWig write path (Rust)

**Date:** 2026-06-19
**Status:** Approved — ready for implementation planning
**Roadmap:** bigWig slice of Phase 4 in `docs/roadmaps/rust-migration.md`

## 1. Goal & framing

Make `gvl.write()` / `update()` faster for bigWig tracks by eliminating the
redundant work in the current path:

- **2× decode.** `count_intervals` (`src/bigwig.rs`) fully decodes every interval
  via `get_interval(...).into_iter().count()` just to count them; `intervals`
  then re-decodes the same data to extract it. Every region is decoded twice.
- **Repeated file opens.** `BigWigRead::open_file` runs inside every parallel
  closure, so the header + chrom R-tree is re-parsed once per contig-partition /
  per chunk / per sample.
- **Sample-only parallelism.** `paths.par_iter()` parallelizes over samples only;
  annotation tracks (a single pseudo-sample) get no parallelism.
- **Annotation Python loop.** `_annot_intervals_from_bigwig` builds per-region
  Python lists + `np.asarray` calls with no rayon parallelism.

This lands as the **bigWig slice of Phase 4**, behind a byte-identical parity gate,
with a bootstrapped (bigWig-scoped) differential harness and recorded baselines.
`main` stays shippable; the old write path is deleted in the same bundled PR once
parity holds.

Note: unlike the rest of Phase 4 (numba kernels to *port*), bigWig extraction is
*already* Rust, so this slice is "optimize existing Rust + prove parity" and does
not structurally depend on the Phase 1–3 ragged/genotype/reconstruct beachhead.

## 2. Architecture & seam

A new Rust entry point owns the whole per-track loop and writes the dataset files
directly:

```
bigwig_write_track(paths, contig_partitions{contig, starts, ends}, max_mem, out_dir, sample_less: bool)
    → writes out_dir/intervals.npy + out_dir/offsets.npy
```

- Lives in `src/bigwig.rs` (new write submodule); PyO3 binding added alongside the
  existing `intervals` / `count_intervals` bindings.
- Python `_write_track` and the bigWig branch of `_annot_intervals_from_bigwig`
  collapse to: normalize / validate inputs → hand regions + paths to Rust. The
  `Table` / polars-bio annotation branch is **untouched**.
- Output is raw, header-less bytes. `np.memmap` already writes raw bytes (no `.npy`
  header), so:
  - `intervals.npy` = packed `align=True` struct `{i32 start, i32 end, f32 value}`
    (12 B), `INTERVAL_DTYPE` in `python/genvarloader/_ragged.py`.
  - `offsets.npy` = `i64`.
  Rust writes little-endian structs via a `BufWriter`; this matches numpy
  byte-for-byte.
- `update()` adds a track to an existing dataset via the same `_write_track` /
  `_write_annot_track` path, so it is covered automatically.

## 3. The streaming writer algorithm

- **Decode once.** Each `(region, sample)` work item calls `get_interval` exactly
  once. Open each bigWig **once per worker** (cached reader, reused across all
  regions for that worker).
- **Ordering (parity-critical).** Data and offsets are emitted **region-major,
  sample-minor** — identical to today's `n_per_query.ravel()` layout and the
  `offsets[r_idx * n_samples + s_idx]` indexing in `src/bigwig.rs`.
- **Memory-bounded streaming.** Accumulate decoded intervals into a chunk buffer;
  flush to disk before a region would push the buffer past `max_mem`, then reset.
  Parallel-decode within a chunk (rayon over region×sample work items), ordered
  write. **Chunk boundaries are internal** — the final concatenated files are
  independent of how we chunk, which frees us to chunk differently from today
  without breaking parity.
- **Preserved exactly (parity-relevant behavior):**
  - Contig matching: `chrom.name == contig || chrom.name == format!("chr{contig}")`.
  - Clamping: `r_start = start.max(0)`, `r_end = end.min(max_len)`.
  - The "single region exceeds `max_mem`" → error behavior (existing
    `NotImplementedError`; per-sample chunking remains unimplemented).
  - No NaN handling in the interval path — values pass straight through from
    bigtools (this matches the current `intervals`, and is distinct from `read()`
    which calls `np.nan_to_num`).
- `offsets.npy` length stays `n_regions * n_samples + 1` (all prefix offsets plus
  the final total). For the annotation path (`sample_less=True`) the sample axis is
  a single pseudo-sample collapsed away → length `n_regions + 1`.

## 4. Parity harness + synthetic corpus (bootstrapped Phase 0, bigWig-scoped)

- **Corpus generator:** reproducible synthetic bigWigs over chr21/chr22 (same
  contigs as the vcfixture tier), parameterized by `n_samples` × interval
  density/resolution. A small tier for CI parity, a larger tier for the bench.
- **Differential test:** run the **old path** (current count + read orchestration)
  vs the **new path** on identical inputs; assert `intervals.npy` and `offsets.npy`
  are **byte-identical**. Runs across the py310–313 × linux/macOS matrix. The unit
  only lands when parity holds.
- Built so the general Phase 0 harness can later generalize from it.

## 5. Baselines & profiling

- Capture `write()` and `update()` **wall-clock + peak RSS** on the synthetic bench
  corpus **before** optimizing (fills the roadmap baseline table, currently TBD),
  and the after-numbers under the Phase 4 checkpoint.
- Confirm decode actually dominates before committing hard to the decode-cost
  story: hand David a **bash script** of py-spy commands (macOS py-spy needs sudo;
  do not invoke py-spy directly). `memray` for RSS can be run directly.
- `Dataset.__getitem__` throughput is **not** affected by this slice (read path
  unchanged); only the write/update rows of the baseline table are in scope here.

## 6. Switch & landing

- Env-var switch (e.g. `GVL_RUST_BIGWIG_WRITE`) selects new vs old during parity.
- Keep the old Rust `count_intervals` / `intervals` + Python orchestration alive
  until parity is green, then **flip the default and delete the old bigWig write
  orchestration in the same bundled PR**. `count_intervals` / `intervals`
  themselves stay — they are still used by `Dataset` reads.
- Update `docs/roadmaps/rust-migration.md`: tick the bigWig items, record baseline
  + after numbers, set the Phase 4 marker 🚧 + PR link.

## 7. Scope / non-goals (YAGNI)

**In scope:**
- bigWig write/update path (per-sample `_write_track` + annotation
  `_annot_intervals_from_bigwig` bigWig branch).
- Bigwig-scoped differential parity harness + synthetic corpus generator.
- Baselines (write/update wall-clock + RSS) + profiling script.

**Out of scope:**
- Table / polars-bio annotation path.
- The numba track *realign* (`_dataset/_tracks.py`) and the rest of Phase 4's
  variant/genotype write kernels.
- Per-sample chunking when a single region exceeds `max_mem` (keep the existing
  `NotImplementedError`).
- `skills/genvarloader/SKILL.md` update — no public-API change (`gvl.write` /
  `update` signatures and defaults are unchanged).

## 8. Risks & mitigations

- **Byte-identical f32:** values pass straight through bigtools → bytes with no
  arithmetic, so bit-identity should hold; the differential harness is the guard.
- **Reader caching across threads:** `BigWigRead` is not suitable for shared use
  across threads → cache is per-worker / thread-local, not a shared handle.
- **Decode may not dominate:** if profiling shows the path is disk/IO bound, the
  rewrite still helps (removes the second decode + repeated opens) but the headline
  speedup shrinks. Report honestly against baseline either way.
