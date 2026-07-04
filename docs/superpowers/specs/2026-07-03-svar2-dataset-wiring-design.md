# SVAR2 Dataset Wiring — Design Spec

> **Purpose:** wire the SVAR2 format into the gvl `Dataset` the way SVAR1 already is —
> by caching genoray's interval-search result at `gvl.write` time and replaying it at read
> time — so SVAR2 haplotype/track reads stop paying a per-query search-tree rebuild. This is
> the deferred **Task B** (`TODO(svar2-dataset-dispatch)` in `_svar2_source.py`), now
> justified and shaped by the E1 profiling result below.

**Date:** 2026-07-03 · **Depends on:** the M6b SVAR2 kernels (`reconstruct_haplotypes_from_svar2`,
`shift_and_realign_tracks_from_svar2`, already built + parity-validated vs the genoray `decode`
oracle), the `SparseVar2Source` adapter (`python/genvarloader/_dataset/_svar2_source.py`), and the
E1 profiling result (`docs/superpowers/specs/2026-07-03-svar2-profiling-followup.md`). **Two
deliverables:** a genoray PR (search/gather split) then a gvl PR (write + read wiring).

## Problem & evidence (E1)

The benchmarked SVAR2 path is the **raw `SparseVar2Source` adapter**, which calls genoray
`SparseVar2.overlap_batch` **live on every query**. E1 profiled it with `perf --call-graph fp`
(py-spy is unusable on Carter compute nodes — `ptrace_scope=2`), attributing time by DSO on the
3-region × all-samples chr21 workload:

| path | native genoray `_core.so` | gvl kernel | numpy conv | python interp |
| --- | --- | --- | --- | --- |
| **svar2 germline** (3202) | **88.3%** (68.6% `SearchTree::build` + 14.3% `overlap_batch`) | 1.1% | 0.1% | 3.9% |
| **svar2 somatic** (16007) | **78.5%** (60.3% `SearchTree::build`) | 0.7% | 0.2% | 4.8% |
| svar1 germline | 0% | 56.9% (`get_diffs_sparse` + `reconstruct…_from_sparse`) | 20.4% | 7.3% |
| svar1 somatic | 0% | 10.2% (rest: genotype IO / ZSTD decompress + page faults) | 24.2% | 12.1% |

**Conclusion:** SVAR2 adapter latency is **neither Python-adapter overhead (≤5%) nor the gvl
reconstruct kernel (~1%) nor numpy conversions (~0.1%)** — it is genoray **rebuilding interval
search trees on every `overlap_batch` call**. In genoray's source: `DenseUnion::overlap` builds a
fresh `SearchTree::new(&self.positions)` **per region** (`src/query.rs:288`, comment "one per region
in a batch") over the contig-wide dense union, and `vk_slice → spine::gather_keys` builds one **per
hap** (`src/spine.rs:48`). SVAR1 does not pay this: `gvl.write` calls genoray's **search-only**
`_find_starts_ends` once and caches a compact `(2, R, S, P)` offsets memmap
(`_write_from_svar` in `python/genvarloader/_dataset/_write.py:961`); reads then **slice the shared
`.svar` store directly** by those offsets, with no search. **This spec gives SVAR2 the same
write-time cache** — the fix E1 points to. (The earlier "dense-gather layout" hypothesis, E3, is
**moot**: the gvl kernel where that gather lives is ~1% of time; the cost is the tree build.)

## Approach — mirror SVAR1

Split genoray's fused `overlap_batch` into a **search-only** step and a **tree-free gather** step,
mirroring SparseVar's `_find_starts_ends` / `read_ranges` pair. `gvl.write` runs the search once and
caches the compact result into `.gvl`; `Dataset` reads replay the cache through the gather step and
the existing SVAR2 kernels. The bulk variant data stays in the compressed `.svar2`, referenced by a
`Svar2Link` — so the cache is O(offsets) and **SVAR2's on-disk size advantage (1.46–5.67× smaller
than `.svar`, established in the MVP benchmark) is preserved**, not materialized away.

Unlike SVAR1 (whose per-hap variants are a contiguous slice, so reads slice the shared store with no
genoray call), SVAR2's dense-presence bits are **computed** from the class tables (the `carried`
tests), not sliceable. So SVAR2's read path makes one genoray gather call per read — but with **no
interval search**, which is the entire cost E1 identified.

## Component A — genoray API (dependency; separate genoray PR)

Refactor `overlap_batch` into three public methods on `SparseVar2`, mirroring SparseVar's naming and
all accepting a `samples=` subset (like every SparseVar range method). Signatures follow SparseVar's
`_find_starts_ends` / `read_ranges` exactly where analogous:

- **`find_ranges(contig, starts, ends, samples=None, out=None) → ranges`** — *search-only*, the
  analog of `SparseVar._find_starts_ends`. Builds each search tree **≤ once per contig** and returns
  the compact bundle needed to gather later: per-region `dense_range` `(R, 2)` and per-hap var_key
  column offsets (no gathered `KeyRef`s, no presence bits). `samples=` restricts which samples'
  offsets are computed; `out=` writes into a preallocated memmap so `gvl.write` streams the cache
  straight to disk (exactly as `_find_starts_ends(..., out=out)` does for SVAR1).

- **`gather_ranges(contig, ranges, samples=None) → payload`** — the *gather* step. Consumes a
  `find_ranges` bundle (the **cached ranges**, not `starts/ends`) and therefore performs **no
  interval search**: it slices var_key from the shared `.svar2` store and computes dense-presence
  bits tree-free, returning the full reconstruct payload dict (`vk_pos`, `vk_key`, `vk_off`,
  `dense_pos`, `dense_key`, `dense_range`, `dense_present`, `dense_present_off`, `lut_bytes`,
  `lut_off`) — the exact input shape the SVAR2 kernels already accept. `samples=` restricts which
  samples are materialized (cache all at write, read any subset).

- **`read_ranges(contig, starts, ends, samples=None) → payload`** — the public **fused** API, exactly
  like `SparseVar.read_ranges`. A thin wrapper:
  `gather_ranges(contig, find_ranges(contig, starts, ends, samples), samples)`. It replaces
  `overlap_batch`'s role for live/uncached queries and serves as the parity oracle. (`overlap_batch`
  may be kept as a deprecated alias or removed — genoray maintainer's call.)

**Contract (byte-identical parity):** for any `contig, starts, ends, samples`,
`reconstruct(read_ranges(...))` ≡ `reconstruct(gather_ranges(find_ranges(...)))` ≡
`reconstruct(overlap_batch(...))` ≡ genoray `decode` oracle, byte-for-byte. gvl treats the
`find_ranges` bundle as an **opaque** array bundle — it persists and replays it, with no gvl coupling
to genoray's internal layout (the same way the adapter treats the `overlap_batch` dict today).

## Component B — gvl write (`_write.py`, new `_write_from_svar2`, new `_svar2_link.py`)

Mirror the SVAR1 write path:

- **Detect `.svar2`.** In `write()` variant-source dispatch (`_write.py:~225`, alongside the existing
  `.svar` branch), a directory with suffix `.svar2` → `SparseVar2(dir)`. Add a `SparseVar2` branch to
  the genotype-writing dispatch (`_write.py:~325`, next to `isinstance(variants, SparseVar)`).

- **`_write_from_svar2(path, bed, svar2, samples, extend_to_length)`** mirrors `_write_from_svar`
  (`_write.py:961`): allocate a memmap in `<path>/genotypes/`, and per contig call
  `svar2.find_ranges(c, df["chromStart"], df["chromEnd"], samples=samples, out=out)` to stream the
  compact ranges cache to disk. Write `<path>/genotypes/svar2_meta.json` (bundle shapes/dtypes) and
  a `Svar2Link` into `metadata.json`. Set `metadata["ploidy"] = svar2.ploidy`.

- **`_svar2_link.py`** mirrors `_svar_link.py`: a `Svar2Link` pydantic model
  (`relative_path`, `absolute_path`, `fingerprint`) + `_resolve_svar2(gvl_path, link, override)` and
  `_verify_fingerprint(...)`. Fingerprint the `.svar2` on its stable identity (e.g. `n_variants` from
  its index + a byte count of a canonical store file), analogous to `SvarFingerprint`.

- **Reject unsupported variants** (symbolic/breakend ALTs) as SVAR1 does; upstream normalization
  (genoray roadmap M13, `-V other,bnd`) already filters these in the MVP build scripts.

## Component C — gvl read (`_open.py`, `_haps.py`, `_reconstruct.py`, `_tracks.py`; refactor `_svar2_source.py`)

- **`Dataset.open`** resolves + fingerprints the `Svar2Link` (mirroring `_resolve_svar` /
  `_verify_fingerprint`, with a `svar2=` override on `open` paralleling `svar=`), and holds a
  `SparseVar2` handle plus the memmapped cached-ranges bundle.

- **Haplotypes.** The `Haps` reconstruction path routes SVAR2 datasets to: load the cached ranges for
  the requested `(region, sample)` block → `svar2.gather_ranges(contig, ranges_block, samples=block)`
  → `reconstruct_haplotypes_from_svar2(...)` → `Ragged` haps. This retires
  `TODO(svar2-dataset-dispatch)`. A dataset carries a source discriminant (svar2_link present) that
  selects the SVAR2 reconstructor over the SVAR1 one.

- **Tracks.** `with_tracks` / the track re-aligner routes to `shift_and_realign_tracks_from_svar2`
  via the **same** cached ranges + `gather_ranges` — the cache is written once and serves both
  haplotype and track reconstruction.

- **Refactor `_svar2_source.py`.** Replace the adapter's live `overlap_batch` call with the
  cached-ranges + `gather_ranges` path (or fold its kernel-marshalling into `Haps`). The
  region→(R·S) expansion and `ascontiguousarray` marshalling already there stay valid; only the data
  source changes from live query to cache+gather.

## Cache format (`.gvl/genotypes/`)

- The compact `find_ranges` bundle, region/sample-sharded to match the dataset layout (per-region
  `dense_range` + per-hap var_key offsets), stored as memmaps written in place via `out=`.
  `svar2_meta.json` records shapes/dtypes (mirrors SVAR1's `svar_meta.json`). `Svar2Link` →
  shared `.svar2`.
- **Size:** O(offsets) — same order as SVAR1's `offsets.npy`, **not** the ~1.8 MB/3-region
  `overlap_batch` payload. The bulk (var_key positions/keys, dense class tables, LUT) stays in the
  compressed `.svar2`.

## Data flow

- **WRITE:** `bed + .svar2` → per contig `find_ranges(..., out=memmap)` (search once) → compact ranges
  cache in `.gvl/genotypes/` + `Svar2Link` + `svar2_meta.json`.
- **READ:** `dataset[regions, samples]` → load cached ranges block → `gather_ranges` (tree-free) →
  `reconstruct_haplotypes_from_svar2` / `shift_and_realign_tracks_from_svar2` → haps / tracks. **No
  interval search at read.**

## Parity & testing (the contract)

- **Byte-identical:** cached-path reconstruct ≡ live `read_ranges`/`overlap_batch` reconstruct ≡
  `decode` oracle, on the M6b matrix (SNP / INS / DEL × samples × ploids) + real chr21 germline &
  somatic stores. Track re-alignment matched the same way.
- **Additive guarantee:** the SVAR1 path is byte-unchanged — full SVAR1 regression suite green
  (`pixi run -e dev pytest tests -q`; and `cargo test` for the kernels). Follows the rust-migration
  byte-identical parity contract and the numba-oracle-bug policy (if the cached path and a numba
  oracle disagree, check whether numba is the buggy one before "fixing" the new path).
- **Perf verification (same-session, shared-node caveat):** a warm SVAR2 `Dataset` read no longer
  shows `SearchTree::build` — the perf DSO split flips from ~80% genoray to gvl-kernel-bound, like
  SVAR1. Report as a relative before/after within one allocation (absolute wall-clock is not
  comparable across allocations on the shared Carter nodes).
- **Docs/roadmaps:** update the genoray roadmap (search/gather split + read-bound conversion open
  question) and `docs/roadmaps/rust-migration.md`; update user-facing docs for `.svar2` as a `write`
  variant source — `skills/genvarloader/SKILL.md`, `docs/source/{api.md,write.md,format.md,faq.md}`,
  `README.md` (per the repo's docs-audit + skill-maintenance gates), and keep `api.md` in sync with
  any new `__all__` symbols.

## Out of scope

- SVAR2 **variants** and **annotated** output modes (this spec covers haplotypes + tracks only; the
  same cache extends to them later).
- The remaining profiling experiments **E2** (same-cohort sample-count sweep) and **E4** (conversion
  build-thread allocation) — independent of this fix; parked, resumable from the profiling plan.
  **E3** (dense-access layout probe) is **dropped** — E1 established the gvl dense-gather is ~1% of
  time, not the hot path.
- Any change to the on-disk `.svar2` format itself (that is genoray's; this spec only adds the
  search/gather split and the gvl-side cache).

## Deliverables & sequencing

1. **genoray PR** — `find_ranges` / `gather_ranges` / `read_ranges` split with `samples=`, parity
   tests vs the current `overlap_batch`, release a wheel. (Crate/wheel release gate per the existing
   SVAR2 dev-wiring notes.)
2. **gvl PR** — `_write_from_svar2` + `_svar2_link.py` (write) and the `_open.py` / `_haps.py` /
   `_reconstruct.py` / `_tracks.py` dispatch + `_svar2_source.py` refactor (read), targeting the
   genoray API from (1); byte-identical parity + docs/roadmap updates.

## Open questions / risks

- **`find_ranges` bundle contents:** the exact compact fields (are per-hap var_key offsets one range
  per hap, or separate SNP + indel channel ranges?) are a genoray-internal detail settled in the
  genoray PR; the gvl side stays agnostic (opaque persist/replay), so this does not block the gvl
  design.
- **`gather_ranges` presence-bit cost at read:** the per-hap `carried` tests over `dense[ds..de]`
  remain at read (they are cheap, no tree) — confirm in the perf verification they don't become the
  new hot path (E1 already shows this work is a small fraction of the gvl-side once the tree is gone).
- **Format version:** adding `svar2_link` to `metadata.json` is additive; confirm
  `_check_dataset_format_version` tolerates it and that a bump (if any) is handled by `_migrate.py`.
