# Design: Port `gvl.Table` overlap to Rust (COITrees)

**Date:** 2026-06-19
**Status:** Approved (design); pending implementation plan
**Roadmap:** Phase 4 continuation of `docs/roadmaps/rust-migration.md`

## Problem

`gvl.Table` is a long-form interval track keyed by `(sample_id, chrom, start, end,
value)`. Its overlap queries run through **polars-bio** (`pb.overlap`). This causes two
problems:

1. **`max_mem` is wildly disrespected during `gvl.write()` / `update()`.** The write path
   (`_write_track_legacy`) budgets memory as `24 bytes × n_overlapping_intervals` per
   region, then chunks regions so each chunk's materialized output is `<= max_mem`. But:
   - polars-bio / DataFusion's internal join memory has little to do with that 24-byte
     output estimate, so actual peak RSS can vastly exceed `max_mem`.
   - `count_intervals` itself runs a **full-contig overlap before any chunking**, an
     entirely unbudgeted spike.
2. **polars-bio segfaults non-deterministically** (CPython 3.12/3.13, upstream
   biodatageeks/polars-bio#395). This is why `Table` is gated to
   `genvarloader.experimental`, excluded from CI, and shipped behind a `[table]` extra.

## Goal

Port the entire `Table` implementation (plus the sample-less `annot_overlap` path) to a
self-contained Rust module backed by **COITrees**, eliminating polars-bio. This:

- Makes `max_mem` meaningful: COITrees counting is exact, so the output-size budget is the
  true materialized size with no opaque query-engine overhead.
- Removes the segfault source, the `[table]` extra, and the experimental gating.
- Promotes `Table` to the public API, finally CI-covered.
- Advances the Rust-migration roadmap (Phase 4 left "legacy Python orchestration retained
  only for non-BigWigs IntervalTracks (e.g. Table)").

## Decisions (from brainstorming)

| Question | Decision |
|---|---|
| Port scope | **Full Table object in Rust** — Rust owns interval store, overlap queries, budgeted streaming write. Python is a thin constructor/validation shim. |
| Overlap crate | **COITrees** — cache-oblivious interval tree, build-once/query-many, has `query` + count. |
| Parity oracle | **Brute-force numpy oracle**, property-tested (hypothesis/vcfixture-style). No polars-bio in tests. |
| `max_mem` scope | **Accurate output budget + lazy per-contig store** — trees built one contig at a time. |
| polars-bio removal | **Full removal** — port `Table` + `annot_overlap`, drop `[table]` extra, de-experimentalize, enable in CI. |
| API move | **Hard move** out of `genvarloader.experimental` (delete the subpackage). |
| `ExperimentalWarning` | **Delete** (Table was its only user). |
| Contig normalization | Use **`ContigNormalizer.c_idxs`** (genoray) vectorized API, not per-row `normalize_contig_name`. |

## Architecture

### FFI boundary

Python keeps all file I/O and dataframe work (polars' strength); Rust owns the hot,
memory-sensitive overlap + streaming (the broken part).

- **New Rust module `src/tables/`** (alongside `src/bigwig.rs`; fits the roadmap's eventual
  `tracks/` umbrella). Owns three things:
  - an immutable interval **store** (compact, per-contig, per-sample sorted column slices),
  - a COITrees-backed **overlap engine**,
  - a budgeted **streaming writer** (mirrors the existing `bigwig_write_track`).
- **FFI surface (`src/ffi/`, via the existing `pymodule`):**
  - PyO3 class `RustTable`, constructed from canonical columns as arrays: `sample_codes:
    i32[]`, `chrom_codes: i32[]`, `starts: i64[]`, `ends: i64[]`, `values: f32[]`, plus the
    sample-name and contig-name vocabularies. Python factor-encodes the frame and hands
    Rust the five columns — no re-porting CSV/parquet/arrow readers.
  - `table_write_track(out_dir, chrom_codes, starts, ends, max_mem, sample_codes)` — runs
    the count → budget → stream-to-`intervals.npy`/`offsets.npy` loop **inside Rust**,
    exactly like `bigwig_write_track`.
  - `annot_write_track(...)` — the sample-less variant for `annot_overlap`, sharing the
    same engine with a single pseudo-sample collapsed away.

### Overlap engine & memory model (the core fix)

- **Store:** intervals grouped by `chrom_code`, partitioned within a contig by
  `sample_code`, stored as compact sorted column slices (Python pre-sorts by `chrom,
  sample_id, start`). COITrees are **not** built at construction — only vocabularies +
  sorted slices are resident.
- **Lazy per-contig trees:** the writer processes one contig at a time (bed is
  contig-grouped). For the current contig, build one `COITree` per sample on demand, query
  all regions, drop the trees before the next contig. Resident tree memory = one contig's
  intervals, not the whole Table.
- **The memory fix, concretely:**
  1. **Count pass** — `coitree.query_count(start, end)` per (region, sample). Pure counter,
     ~zero allocation. Yields the *exact* output interval count, so `24 bytes × count` is
     the true materialized size, not an estimate.
  2. **Budget** — reuse `splits_sum_le_value(mem_per_region, max_mem)` logic (ported to
     Rust so the whole loop is one FFI call). Chunk split points must match the Python
     version so the offsets layout is identical.
  3. **Materialize pass** — per chunk, gather overlaps into pre-sized `start/end/value`
     buffers (size known exactly from step 1), emit in `(region, sample, start)` order to
     match BigWigs, `memmap`-stream to disk, free, next chunk.
- **Peak working set** is bounded by `max_mem` (output buffers) + one contig's resident
  intervals + one contig's COITrees — all predictable, none tied to a query engine's opaque
  internals.
- **Ordering contract:** within each (region, sample) cell, sort overlaps by `start` —
  byte-for-byte the same order the current `np.lexsort((start, sample, region))` produces.

### Python shim & API changes

- **`python/genvarloader/_table.py` shrinks to a constructor + validator.** `Table.__init__`
  keeps the polars work it does today (`_normalize_input`, `column_map`, cast to canonical
  dtypes, sort by `(chrom, sample_id, start)`), then factor-encodes `sample_id` →
  `sample_codes`, builds a `ContigNormalizer(self.contigs)`, encodes `chrom` →
  `chrom_codes` via `c_idxs`, and constructs a `RustTable`. `from_path`, `_read_path`,
  `_normalize_input`, `_apply_column_map`, `_resolve_samples` stay in Python.
- **Delete** `ExperimentalWarning`, `_TABLE_EXPERIMENTAL_MSG`, `_POLARS_BIO_MISSING_MSG`,
  `_import_polars_bio`.
- `count_intervals` and `_intervals_from_offsets` become thin wrappers delegating to
  `RustTable` (kept so the `_write_track_legacy` contract still holds for any path that
  uses them).
- `annot_overlap` becomes a thin wrapper over `annot_write_track` / a sample-less
  `RustTable` query.
- **Write-path wiring (`_dataset/_write.py`):** `_write_track` gains a `Table →
  table_write_track` branch (alongside `BigWigs → _write_track_rust`), so Table gets the
  single-FFI-call streaming writer. `_write_track_legacy` then only serves any remaining
  non-BigWigs / non-Table `IntervalTrack`. `_annot_intervals` routes BED-like sources to
  the Rust annot path. Also switch the per-row `normalize_contig_name` loop in
  `_write_track_rust` to `ContigNormalizer.c_idxs` (vectorized).
- **Public API promotion (hard move):** move `Table` into the main package, export it in
  `genvarloader/__init__.py` `__all__`, and **delete the `genvarloader.experimental`
  subpackage** (its only exports were `Table` + `ExperimentalWarning`). Grep for and fix
  any stray references.
- **`pyproject.toml`:** delete the `table = ['polars-bio']` extra and its comment.
- **`skills/genvarloader/SKILL.md`** (mandatory per CLAUDE.md): Table moves from
  experimental to core — no `[table]` extra, no polars-bio install step.

## Testing & parity

- **Brute-force numpy oracle:** for each region × sample, scan that sample's intervals on
  the contig, keep those overlapping `[start, end)` (half-open, zero-based — matching the
  polars-bio options set today), sort by `start`, emit `RaggedIntervals`. ~10 lines,
  obviously correct.
- **Property tests (hypothesis + vcfixture-style generators):** random tables (varying
  sample counts, contig sets, interval density incl. nested/touching/zero-overlap, empty
  contigs, regions off the end) → assert Rust `count_intervals`, `_intervals_from_offsets`,
  and `annot_overlap` are **byte-identical** to the oracle (starts/ends/values arrays *and*
  offsets).
- **Memory regression test:** build a Table whose full-contig overlap is large but set a
  small `max_mem`; assert peak RSS stays within a multiplier of `max_mem` (the test that
  would have caught the polars-bio blow-up). Measure via memray / a child-process RSS probe.
- **CI enablement:** with polars-bio gone, these run in the normal matrix (py310–313 ×
  linux/macOS) — Table is finally CI-covered.
- **Rust units (`cargo test`):** store/engine cases — empty contig, single interval, dense
  overlaps, ordering, out-of-vocab contig.

## Roadmap, PR strategy, risks

- **Roadmap:** add a Phase 4 sub-task for the Table/annot overlap port; record memory +
  wall-clock results under the Phase 4 checkpoint table; add a dated decisions-log entry.
- **PR strategy:** one bundled PR (solo-maintainer preference): Rust module + FFI + Python
  shim + write-path wiring + experimental-subpackage deletion + polars-bio/extra removal +
  tests + roadmap + SKILL.md. `main` stays shippable.
- **Dependency:** add `coitrees` to `Cargo.toml`; remove the `[table]` extra from
  `pyproject.toml`.

### Risks / watch-items

1. **Half-open / zero-based semantics** — current code sets polars-bio to zero-based +
   disables coordinate checks. The COITrees engine must replicate `[start, end)` overlap
   exactly; the oracle pins this.
2. **Contig-name normalization** — use `ContigNormalizer.c_idxs` for the vectorized happy
   path, but still detect out-of-vocab contigs (those `norm` returns `None` for) and handle
   them as today (zero output for that region) rather than mapping to a bogus index.
3. **Float32 value round-trip** — values cast to f32 in Python today; keep identical so
   byte-parity holds.
4. **Stale memory** — `project_polars_bio_segfault` note says "transitive via genoray";
   verified nothing in the env requires polars-bio except gvl's extra. Update that memory
   after landing.
5. **`splits_sum_le_value` parity** — if chunk-splitting moves to Rust it must match the
   Python split points so chunk boundaries (and offsets layout) are identical; the
   differential test covers the final arrays regardless.
