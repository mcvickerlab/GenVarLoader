# Design: `StreamingDataset` — SVAR1 window reads + double-buffer engine

**Date:** 2026-07-16
**Status:** design (pending spec review)
**Roadmap:** `docs/roadmaps/streaming-dataset.md`
**Issue:** [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275) (Plan 2)
**Follows:** walking skeleton [#274](https://github.com/mcvickerlab/GenVarLoader/pull/274)
**Supersedes parts of:** `docs/superpowers/specs/2026-07-15-streaming-dataset-vcf-pgen-svar1-design.md` ("spec A")

## Summary

Replace the walking skeleton's synchronous, per-batch SVAR1 read path with **window-granular
reads over a new ungated genoray SVAR1 query API**, consumed by a **crossbeam double-buffer
engine** behind a generic `StreamBackend` trait. Also settles `StreamingDataset`'s public
surface on a single iteration entry point (`to_iter`).

Two PRs, two repos:

1. **genoray** — add an ungated `svar1_query` module (`Svar1Reader` + `var_ranges` +
   cartesian `find_ranges`). SVAR1's query path exists today only in Python/numba; SVAR2 has an
   ungated Rust one. That asymmetry is the root cause of every defect below.
2. **gvl** — bump the genoray `rev`, rewrite `Svar1Store` on the new API, land the engine,
   settle the API surface.

Byte-identical parity with `gvl.write()` + `Dataset.open()[r, s]` (jitter=0) is preserved
throughout and is the control on the rewrite.

## Root cause

The skeleton reads SVAR1 through `genoray_core::svar1_reader::Svar1RecordSource` — a
**conversion-pipeline record producer**, not a query API. It was reached for because genoray
exposes an ungated Rust query surface for SVAR2 (`genoray_core::query::ContigReader`) and
**none for SVAR1**.

Consequences, all downstream of that one choice:

- `Svar1RecordSource::new` is **O(all CSR entries)**, not O(1): `build_variant_major`
  (`svar1_reader.rs:17-38`) eagerly inverts the contig's entire hap-major CSR into a
  variant-major `Vec<Vec<(u32,u64)>>`. The skeleton calls it **once per batch**.
- It is **forward-only** (`cursor` is private, no seek; `RecordSource` is a one-method trait),
  so a long-lived cursor cannot serve a region-major sweep that skips.
- It takes `pos`/`ref`/`alt` **by value**, forcing the per-batch `t.pos.clone()` et al.
- Its per-contig table must be pushed in from Python, hence `set_contig_table` + `.tolist()`.
- The whole walk runs **under the GIL** (`src/ffi/mod.rs:826`, before the `py.detach` at `:850`).

**Therefore #275's four "debt" items are not four fixes.** They are artifacts of one wrong
dependency, and the correct architecture deletes all four rather than optimizing any:

| #275 says | Reality |
|---|---|
| Static table crosses FFI as lists — "Fix with `PyReadonlyArray1`" | The table exists **only** to feed `Svar1RecordSource`. Delete the reader → the table, `set_contig_table`, and `.tolist()` all disappear. |
| `read_window` clones the contig table per batch — "Fix with slices / `Arc`" | The clone exists only because the constructor takes vectors by value. Gone with the constructor. |
| Re-opens the source and walks the whole contig per batch | Becomes a binary search. |
| GIL held during the record walk | Fixed by the `Svar2Store` template: borrow the reader into `py.detach`. |

SVAR1's on-disk genotype layout is **already hap-major sparse CSR of sorted global variant IDs**
— exactly what `reconstruct_haplotypes_from_sparse` consumes. There is no decode. The query
output *is* the kernel input.

## Supersession of spec A

Spec A remains the framework record. These claims are corrected here:

| Spec A claim | Correction |
|---|---|
| SVAR1 producer drives `Svar1RecordSource`; "window population is a bounded slice + local-table rebuild" | SVAR1 uses the new ungated `svar1_query`. There is **no local-table rebuild and no materialized buffer** — `geno_v_idxs` is borrowed from the `variant_idxs` mmap. |
| Buffer holds "bulk-decoded variants"; each "compressed block / mmap page is touched once" | For SVAR1 there is **nothing to decode**. The engine's win is **hiding I/O (page-fault) latency**, not decode. The premise still holds for VCF/PGEN. |
| Slot recycling "mirrors genoray's `orchestrator.rs`" | It does not. genoray allocates each chunk fresh, moves it through a bounded channel, and drops it — **no return channel, no slot ring**. Recycling is net-new design. The *thread/channel/shutdown* pattern is a faithful template; the recycling is not. |
| "genoray release-gate… cannot ship until genoray publishes" | **False.** genoray is not published to crates.io; it is a git `rev` pin. Bumping the rev is the supported mechanism (CLAUDE.md, "Development Notes"; issue #278). |
| A bespoke ungated SVAR1 reader is "extra work justified only if the wheel matrix forces it" | The **performance architecture** forces it. htslib-free SVAR1 is a side benefit, not the motive. |
| `StreamingDataset` **is** a torch `IterableDataset` | It is a plain frozen dataclass. `to_iter` is the entry point; the torch handle is built on demand. |

Unchanged from spec A: region-major order for variants, reuse of reconstruction kernels and
output types, byte-identical parity as the oracle, `std::thread` + `crossbeam_channel::bounded`
(not tokio), two buffer styles with no cross-conversion.

## Iteration model

`StreamingDataset` has **no ad-hoc queries and no random access**. It is a fixed **cartesian**
sweep of BED × samples, traversed in an order chosen from the data layout (region-major for
variants). This is what makes the double buffer unconditionally safe: the next window is always
known, so prefetch is never speculative.

**Window ≫ batch.**

- **Window** = R regions × all samples × ploidy, cartesian. **One** `find_ranges` call per
  window. The window is the read granularity.
- **Batch** = a *slice* of the reconstructed window. Never its own read.

The skeleton conflated the two (one Rust call per batch). That conflation is the source of both
the per-batch contig walk and the apparent need for "pairwise" `(region, sample)` reads —
pairwise is a map-style artifact of `gvl.Dataset.__getitem__` and has no place here.

## Python API surface

One and only one way to iterate:

```python
sds = gvl.StreamingDataset(regions, reference=..., variants="x.svar").with_seqs("haplotypes")

for data, region_idxs, sample_idxs in sds.to_iter(batch_size=32):
    ...
```

- `StreamingDataset` — plain frozen dataclass owning traversal config. **Not** a torch
  `IterableDataset`.
- **`to_iter(batch_size=1, return_indices=True, ...)`** — the real work. Drives the engine and
  yields batches. Everything else wraps it.
- `to_torch_dataset()` — thin wrapper returning a torch `IterableDataset` over `to_iter()`.
  Named to match `gvl.Dataset.to_torch_dataset()` — same concept, same name.
- `to_dataloader(...)` — thin wrapper over `to_torch_dataset()`, `batch_size=None` (the dataset
  yields pre-assembled batches).
- **`__iter__` is removed.** Users are directed to `to_iter`.
- `__getitem__` continues to raise `TypeError`, pointing at `to_iter`.

Torch stays an optional import behind the accessors (`requires_torch`), as today.

Batches remain index-carrying: `(data, region_idx, sample_idx)`, `return_indices=True` by
default — traversal order is not the user's bed-row order, so indices are essential.

## genoray: ungated `svar1_query`

New module, **ungated** (memmap2 + bytemuck + `search.rs`; zero htslib/zstd):

```rust
// src/svar1_query.rs
pub struct Svar1Reader {
    n_samples: usize,
    ploidy: usize,
    variant_idxs: Option<Mmap>,   // raw headerless i32 — sorted global variant ids
    offsets: Vec<i64>,            // resident; num_haps + 1 — small
}
impl Svar1Reader {
    pub fn open(svar1_dir: &str, n_samples: usize, ploidy: usize) -> std::io::Result<Self>;
    pub fn n_samples(&self) -> usize;
    pub fn ploidy(&self) -> usize;
    pub fn variant_idxs(&self) -> &[i32];   // enables a zero-copy kernel input downstream
}

/// Stage A — pure, no reader. POS ranges -> global variant-id ranges.
/// Thin wrapper over the EXISTING `search::overlap_range`.
pub fn var_ranges(
    v_starts: &[u32], v_ends: &[u32], max_v_len: u32,
    contig_start: u32, regions: &[(u32, u32)],
) -> Vec<Range<u32>>;

/// Stage B — variant-id ranges -> ABSOLUTE CSR index pairs, cartesian (2, r, s, p).
pub fn find_ranges(
    reader: &Svar1Reader, ranges: &[Range<u32>], samples: Option<&[usize]>,
) -> Svar1RangesBundle;
```

Plus `src/py_svar1_query.rs` (`PySvar1Reader`) in its own `#[pymethods]` block
(pyo3 `multiple-pymethods`, the established genoray convention), registered ungated next to
`m.add_class::<py_query::PyContigReader>()` (`lib.rs:1068`).

**Stage A is nearly free.** `src/search.rs::overlap_range` (ungated, `lib.rs:78`) is already a
port of the Python `var_ranges` algorithm — its module doc says it "mirrors the SVAR 1.0
`var_ranges` shape". Nobody wired it to SVAR1.

**Stage B is ~10 lines.** Each hap's CSR run holds sorted global variant IDs, so a `[v_lo, v_hi)`
range maps to a sub-slice via two `partition_point` calls — the idiom already at
`svar1_reader.rs:30-31`.

Deliberate divergences from the SVAR2 template, each with a reason:

- **`open` takes no `chrom`** — SVAR1 is one flat store with contigs contiguous in global-ID
  space, unlike SVAR2's `{out}/{chrom}/` layout.
- **Stage A takes `v_starts`/`v_ends` as arguments** rather than owning the index — matches the
  existing SVAR1↔Rust boundary (Python already reads `index.arrow` via polars), keeps Stage A
  pure and testable, and keeps Arrow out of the query core.
- **Cartesian `find_ranges`, matching `_find_starts_ends`'s `(2, r, s, p)` contract** — one
  shape, reusable by genoray's own Python later. Not pairwise: `StreamingDataset` traverses a
  cartesian product.
- **No `gather_ranges`.** SVAR2 needs one because it merges two channels and decodes keys.
  SVAR1 has neither — `read_ranges` hands the `(2, N)` pairs straight to `Ragged.from_offsets`
  over the mmap. The Rust API stops at index pairs.
- **`max_v_len` is a caller-computed scalar**, not a per-hap `max_del.npy` sidecar — SVAR1
  stores no such file.

Follow the repo convention of a `// NOTE:` on the `pub mod` justifying why it is ungated despite
`svar1_reader` being gated (cf. `lib.rs:32-38`, `:63-67`, `:84-88`). The gating criterion is
purely "does it pull rust-htslib/zstd" — this does not.

**Out of scope:** `_find_starts_ends_with_length` (a much larger port: `_length_walk_n_keep`,
biallelic-only). Migrating genoray's Python off numba is a separate concern.

### Known traps (pin with tests before building on them)

1. **`max_v_len` off-by-one.** Python's `max_v_len = (v_ends - v_starts).max()` is `1` for a
   SNP-only contig; `search.rs::overlap_range`'s `max_region_length` contract wants `0`. There is
   **no existing Rust↔Python differential test** for `overlap_range` vs `var_ranges` despite the
   "mirrors" claim. This is the highest-value test in the effort and is cheap — it needs only the
   index table, not genotypes.
2. **Empty ranges must be in-bounds zero-length, never a sentinel.** Python's `var_ranges`
   returns `INT32_MAX` for no-overlap; `_kernels.py:239-243` documents that an out-of-range
   offset is poison — seqpro's `Ragged.to_packed` multiplies by element size and overflows int64,
   even for an empty row. Rust's `(ub, ub)` is already correct; the danger is a shim
   reintroducing the sentinel for legacy compatibility.
3. **`SearchTree` reserves `u32::MAX`** as its padding sentinel (`search.rs:13`) — positions must
   be `< u32::MAX`.
4. **`svar2_view` is gated** but `svar1_reader.rs:43` imports `OverlapMode`/`keeps` from it. If
   overlap-mode semantics are to be shared with the ungated path, check whether `svar2_view`
   (147 lines) has an htslib dep and split/ungate as needed.
5. **`variant_idxs` must stay mmap'd; `offsets` may be resident.** `offsets` is `num_haps + 1`
   (cheap); `variant_idxs` is one entry per non-ref call. Never materialize it.
   Note SVAR1's files are **headerless raw buffers despite the `.npy` extension** — do **not**
   copy SVAR2's `ndarray_npy::read_npy` loader, whose files are real `.npy`.

## gvl: `Svar1Store` + window reads

```rust
#[pyclass]
pub struct Svar1Store {
    reader: Svar1Reader,                    // ONE reader — flat store
    contigs: HashMap<String, ContigMeta>,   // { contig_start, n_local, max_v_len } — scalars
}
```

Converges on the `Svar2Store` shape and ends up smaller. **Nothing sample-scale becomes
Rust-resident**: `v_starts`/`v_ends` stay as the numpy arrays gvl already loads at construction
and cross per-call as zero-copy `PyReadonlyArray1`. Only three scalars per contig live in the
pyclass. (`v_ends` is derived once in Python: `v_end = v_start - min(ilen, 0)`.)

Deleted: `set_contig_table`, `ContigTable`, every `.tolist()`, every per-batch clone, and the
`Svar1RecordSource` dependency.

The window read, entirely inside `py.detach` (reader borrowed across it, per the `Svar2Store`
template):

1. `var_ranges(v_starts, v_ends, max_v_len, contig_start, window_regions)` → variant-ID ranges.
2. `find_ranges(reader, ranges, samples)` → `(2, r, s, p)` absolute CSR index pairs.
3. Kernel inputs: `geno_v_idxs = reader.variant_idxs()` — **the mmap slice itself, zero copy** —
   with the pairs as `geno_o_starts`/`geno_o_stops` and `geno_offset_idx` derived from the
   cartesian window shape.
4. `reconstruct_haplotypes_from_sparse` unchanged.

The global static table (`v_starts`/`ilens`/`alt_alleles`/`alt_offsets`) is genuinely needed by
the kernel, is loaded once at construction, and already crosses as `PyReadonlyArray1`. It stays.

`conversion` **remains enabled** even though nothing in gvl will use it until Plan 3 (VCF/PGEN).
Rationale: it already builds green with static htslib, and Plans 3/4 need it — dropping and
re-adding would churn `Cargo.toml` + `pixi.toml`'s `LIBCLANG_PATH` twice. Recorded here so a
future reader does not mistake it for an oversight.

## The engine

Generic over `StreamBackend` (per spec A), `std::thread` + `crossbeam_channel::bounded`, rayon
inside stages. This introduces gvl's **first threading primitive** — there is currently zero
`std::thread`/crossbeam in `src/` — so it carries the most risk in this design and is where
review attention belongs.

**What is overlapped: I/O latency, not decode.** The producer faults in `variant_idxs` mmap
pages and runs binary searches for window N+1 while the consumer reconstructs batches from
window N. The page cache does not prefetch on an application's access pattern; a producer
thread running a known traversal does. For a store that does not fit in page cache this is the
whole win. Because the traversal is fixed and fully known, the prefetch is speculation-free.

**Buffer shape is backend-specific** (`type Buffer`), and SVAR1's is **degenerate**:

| Backend | Window buffer |
|---|---|
| SVAR1 | Offsets only. `geno_v_idxs` borrowed from the `variant_idxs` mmap. Nothing materialized. |
| VCF / PGEN (#276) | Owned: decoded local static table + per-hap sparse indices. |

This is why #276's "both feed the SVAR1-style window buffer" needs correcting — VCF/PGEN feed an
SVAR1-*style* owned buffer; SVAR1 itself materializes no table.

**From genoray's `orchestrator.rs`, copy:** named OS threads per stage, `bounded` channels,
close-by-`Sender`-drop shutdown, and **join-everything-then-classify-panics** (its comment is
explicit that early-returning on a producer error leaves a consumer blocked on `recv()` forever).
**Do not copy** slot recycling — it does not exist there.

Follow spec A's deferred decisions: start with 2 slots (ping-pong); promote to an N-slot ring only
if profiling shows one producer cannot stay ahead. Single producer + rayon-within-fill first.

Window sizing (`window_regions` / `buffer_bytes`) ships with a measured default, not a guess.
`num_workers` sharding of the window plan is **deferred** (the skeleton's `num_workers>0` guard
stays), consistent with gvl's established "the loader is the concurrency strategy" stance.

## Testing

**Parity is the control and must stay green untouched** — `tests/dataset/test_streaming_parity.py`
asserts every cell byte-matches `Dataset.open(...)[r, s]` across an unsorted, interleaved
multi-contig bed through a real `DataLoader`. It is the oracle for the entire rewrite. It will
need a mechanical update for the `__iter__` → `to_iter` surface change and nothing else; if it
needs more, that is a signal the rewrite changed behavior.

**genoray:**
- **Stage A differential vs Python `var_ranges`** (highest value — pins trap 1).
- Stage B vs `_find_starts_ends` (cartesian, so shapes line up directly).
- Raw-fixture unit tests — trivial, since SVAR1 files are headerless: `write_raw` +
  `tempfile::tempdir()` (prefer `tempfile` over `svar1_reader.rs`'s older manual temp-dir).
- **Add the new symbols to `tests/test_query_only_build.rs`.** Non-optional: genoray's
  `test-rust` task always runs with `conversion` on, so the ungated guarantee is enforced only by
  `check-core` (cargo check) plus that compile guard.

**gvl:**
- Existing parity, unchanged.
- **Scale fixture** — the current 40bp toy references cannot observe an O(all-entries)-per-batch
  bug. Needed to make the measurement meaningful.
- **Scale guard** — assert no per-window materialization of a sample-scale array (mirrors the
  rust-migration defense; see also the standing rule against `ascontiguousarray` on
  sample-scale memmaps).
- Engine unit tests (Rust): buffer/slot handoff, window vs batch boundary, bounded-channel
  backpressure caps memory, producer-error propagation does not hang the consumer.

## Measurement

Per the established convention on this hardware, **do not gate on absolute wall-clock** — the
node is too noisy. The gate is a **deterministic counter: CSR entries touched per window**, going
from `O(all entries on the contig)` to `O(log n + variants in window)`. That is a flat-vs-linear
curve and is noise-immune. Same-session before/after wall-clock is reported as secondary color.

Engine benefit is measured separately (producer/consumer overlap on a cold page cache), since the
asymptotic fix and the prefetch win are independent effects and must not be conflated in one
number.

## Docs

The docs-audit gate **fires** — the public API changes (`__iter__` removed; `to_iter` /
`to_torch_dataset` added):

- `docs/source/api.md` — must stay in sync with `__all__`.
- `skills/genvarloader/SKILL.md` — mandatory for any public-API change.
- `docs/source/dataset.md` — the `StreamingDataset` section's example uses `for batch in sds`
  and must move to `to_iter`; its limitations list needs refreshing.
- `docs/source/faq.md` — check the streaming entries.

## Roadmap / issue updates

- **`docs/roadmaps/streaming-dataset.md`** (this commit): add this spec's row; re-scope Plan 2 to
  window reads + engine; record the genoray `svar1_query` prerequisite (a cross-repo dependency
  currently tracked nowhere); correct the "Key design decisions" bullets that claim
  orchestrator-mirrored slot recycling and decode-amortization for SVAR1; strike the stale
  release-gate framing; rewrite the inherited-debt bullet to say the items are deleted, not fixed.
- **genoray** (after approval): new issue for ungated `svar1_query`.
- **#275** (after approval): rewrite the body — window/batch separation, record-source deletion,
  genoray prerequisite, `to_iter` surface, I/O-latency rationale for the engine.
- **#276** (after approval): correct the "SVAR1-style window buffer" claim.
- **Walking-skeleton plan**: its "⛔ do not merge — ship only after genoray publishes" constraint
  is stale (CLAUDE.md now states the opposite).

## Implementation chunks

1. **genoray `svar1_query`** — `Svar1Reader` + `var_ranges` + `find_ranges`; py bindings;
   compile-guard; **differential tests first** (traps 1–2). Merge to genoray `main`.
2. **gvl pin bump** — both crates to the new rev.
3. **`Svar1Store` rewrite** — window reads on the new API; delete `set_contig_table`/`ContigTable`/
   `.tolist()`/clones/`Svar1RecordSource`. Parity green. Scale fixture + counter measurement.
4. **API surface** — `to_iter` as the core; `to_torch_dataset`/`to_dataloader` as wrappers;
   remove `__iter__`; docs + skill + api.md.
5. **Engine** — `StreamBackend` trait; crossbeam producer/consumer; window sizing default;
   engine unit tests; overlap measurement.

Chunks 1–2 gate everything. Chunks 3 and 4 are independent of each other. Chunk 5 lands last, on
a foundation whose parity and asymptotics are already locked.

## Deferred / open questions

- **Window sizing defaults** — measure; ship a default; expose the knob.
- **N-slot ring vs 2-slot ping-pong** — start with 2; promote only on profiling evidence.
- **`num_workers` sharding** — deferred; the skeleton's guard stays.
- **genoray Python off numba** — out of scope. The new Rust API makes it possible later; this
  spec does not attempt it.
- **`_find_starts_ends_with_length`** — out of scope for the ungated port.
