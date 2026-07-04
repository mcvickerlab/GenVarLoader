# SVAR2 gvl Read-Bound Dataset Wiring — Design Spec

> **Purpose:** wire the SVAR2 format into `gvl.write()` and `gvl.Dataset.__getitem__` so all
> four output modes (haplotypes, tracks, variants, variant-windows) reconstruct **in Rust** off a
> write-time offsets cache, with **no interval-search tree rebuild and no dense-union rebuild at
> read**. This supersedes the gvl-side (Components B/C) of
> `2026-07-03-svar2-dataset-wiring-design.md`, now that genoray's search/gather split has shipped,
> and elects the deferred "fully read-bound" follow-up.

**Date:** 2026-07-04 · **Status:** design, not yet implemented · **Ship:** **local-only, not
shipped** (genoray path-dep + local wheel; no crates.io / PyPI release in this spec).

## Background & what changed

The prior spec (`2026-07-03-svar2-dataset-wiring-design.md`) split genoray's fused `overlap_batch`
into search-only `find_ranges` + tree-free `gather_ranges` + fused `read_ranges`. **That split is
shipped** (merged to genoray `svar-2`): the interval-search trees are built once at write and not
rebuilt at read. Two things drive this follow-up spec:

1. **The read path should be all-Rust.** `gvl.Dataset.__getitem__`'s hot loop is a single Rust FFI
   call for SVAR1. SVAR2 reconstruction must match that — reconstruct in Rust, **not** via the
   Python `SparseVar2.gather_ranges` API, and with no per-read numpy round-trip.
2. **The shipped `gather_ranges` still rebuilds a per-read dense union.** It merges the two on-disk
   dense class tables (`dense/snp` 2-bit, `dense/indel` u32) into a transient position-sorted
   `DenseUnion` **over the entire contig on every read** (genoray `src/query.rs` `dense_union()`) —
   the O(N_contig) residual the prior spec flagged as "presence-bit cost at read." At 16k-sample
   scale this is real. This spec eliminates it with a **per-class read-bound gather**.

The `.svar2` dense store is **already** split and 2-bit-packed on disk (`src/dense.rs`
`DENSE_REGISTRY`: `dense/snp` `pack_snp: true`, `dense/indel` u32) — so **no genoray writer/format
change is needed**. The change is a new *gather* that slices those split tables' per-region windows
directly instead of unioning them.

## Global constraints

- **Byte-identical parity contract.** For any `contig, starts, ends, samples`:
  `reconstruct(read-bound BatchResult)` ≡ `reconstruct(gather_ranges(find_ranges(...)))` (the shipped
  union path) ≡ `reconstruct(overlap_batch(...))` ≡ genoray `decode` oracle — field-for-field /
  byte-for-byte, for every output mode.
- **Additive.** The shipped union-based `find_ranges`/`gather_ranges`/`read_ranges`/`overlap_batch`
  stay byte-unchanged and serve as the parity oracle and live-query API. The SVAR1 gvl path is
  byte-unchanged (full SVAR1 regression green). Follows the rust-migration byte-identical parity
  contract and the numba-oracle-bug policy.
- **Write caches only the dataset's samples.** `gvl.write` already selects which samples enter a
  dataset (requested samples, plus any track-overlap resolution); the cache is sized to that
  selection `S'`, not the full `.svar2` cohort — mirroring `_write_from_svar`.
- **Local-only.** gvl links `genoray_core` as a **path** dependency (`default-features = false`);
  the Python genoray dep stays a **local wheel**. No crates.io/PyPI release. The path-dep and the
  wheel MUST be built from the same genoray commit (the `RangesBundle` field layout is the contract).

## Architecture — two deliverables

**(A) genoray** — a second genoray PR, additive to the shipped split:
1. A **`conversion` cargo feature** (default-on) gating the htslib-tainted modules so the query core
   builds htslib-free.
2. A **per-class read-bound path**: `find_ranges` emits per-class dense ranges; a read-bound gather
   returns a **split-dense `BatchResult`** without building the contig-wide union.

**(B) gvl** — the write cache + all-Rust read wiring targeting (A).

Linkage: gvl adds `genoray_core = { path = "../genoray", default-features = false }` to `Cargo.toml`
(alongside the existing `svar2-codec` path-dep). Only `src/vcf_reader.rs` references `rust-htslib`,
so `default-features = false` yields a query-only, htslib-free core — gvl's build stays
toolchain-light. gvl's Rust opens `genoray_core::query::ContigReader` and calls the read-bound
gather directly; the Python `gather_ranges` API is **not** on gvl's read path.

## Component A — genoray query-only feature + read-bound gather (dependency PR)

### A1. `conversion` cargo feature (query-only build)
- `Cargo.toml`: make `rust-htslib` **optional**; add `[features] conversion = ["rust-htslib", ...]`,
  `default = ["conversion", "extension-module"]` (keep `extension-module` behavior for the wheel).
- `src/lib.rs`: `#[cfg(feature = "conversion")]` on `vcf_reader` and its conversion-only dependents
  (`writer`, `orchestrator`, `normalize` pipeline, and the `py_*` conversion entry points — the set
  that transitively needs htslib). The query core (`query`, `search`, `spine`, `bits`, `nrvk`,
  `rvk`, `layout`, `dense`, `types`, `py_query*`) builds with `default-features = false`.
- **Verify** the query core compiles with `--no-default-features` and that the Python wheel (built
  with defaults) is byte-behavior-unchanged. The full genoray test suite stays green.

### A2. Per-class read-bound `find_ranges` + gather + split-dense `BatchResult`
- **`find_ranges`** additionally emits **`dense_snp_range (R,2)`** and **`dense_indel_range (R,2)`**
  — each a per-region `[ds,de]` into the corresponding on-disk dense table, computed by a per-class
  overlap search (a `SearchTree` per class per region, at **write**, cold). These join the existing
  `vk_snp_range`, `vk_indel_range` (per-hap) — the **4 offset arrays**. (The shipped union
  `dense_range` may remain for the oracle path; the read-bound path uses the two per-class ranges.)
- **Read-bound gather** (`gather_ranges_readbound` or a bundle mode — genoray maintainer's naming):
  slices `dense/snp[ds..de]` (2-bit) and `dense/indel[ds..de]` (u32) directly, applies the
  per-element `q_start < v_end` left-overlap re-check, and computes **per-class presence bits**
  (`DenseView::carried(hap, col)` over each class's window). Returns a **read-bound `BatchResult`**:
  the var_key channel as today (snp+indel merged per hap via `merge_keys`, cheap over a small window)
  **plus split `dense_snp` and `dense_indel` channels** (positions, keys, presence bits, and range),
  **never** the contig-wide union. Exact channel factoring is a genoray-internal detail; gvl consumes
  whatever the read-bound `BatchResult` exposes.
- **Parity:** a genoray test asserts `read-bound BatchResult` reconstructs identically to the shipped
  union `BatchResult` and to the `decode` oracle (SNP/INS/DEL × samples × ploids, plus real chr21).

## Component B — gvl write (`_write.py`, `_write_from_svar2`, `_svar2_link.py`)

Mirror the SVAR1 write path (`_write_from_svar`, `_write.py:961`):
- **Detect `.svar2`.** Add a `.svar2` arm to variant-source path coercion (`_write.py:~217`) and a
  `SparseVar2` arm to genotype-writing dispatch (`_write.py:~315`), alongside the `.svar`/`SparseVar`
  branches.
- **`_write_from_svar2(path, bed, svar2, samples, extend_to_length)`.** Per contig, call
  `svar2.find_ranges(c, df["chromStart"], df["chromEnd"], samples=samples, out=memmaps)` — for the
  **dataset's samples only** — streaming the 4 offset arrays plus `region_starts`/`sample_cols` into
  memmaps under `<path>/genotypes/svar2_ranges/`. Write `svar2_meta.json` (shapes/dtypes) and a
  `Svar2Link` into `metadata.json`; set `metadata["ploidy"] = svar2.ploidy`.
- **`_svar2_link.py`** mirrors `_svar_link.py`: a `Svar2Link` pydantic model
  (`relative_path`, `absolute_path`, `fingerprint`) + `_resolve_svar2(gvl_path, link, override)` +
  `_verify_fingerprint`. Fingerprint the `.svar2` on stable identity (n_variants from its index + a
  canonical store-file byte count), analogous to `SvarFingerprint`.
- **Reject unsupported variants** (symbolic/breakend) as SVAR1 does; upstream `-V other,bnd` /
  genoray M13 already filters these in the MVP build scripts.

## Component C — gvl read (all Rust; `_open.py`, `_impl.py`/`_query.py`, `_haps.py`, kernels)

- **`Svar2Store` (gvl pyclass).** Wraps a `genoray_core::query::ContigReader` per contig, opened
  **once** at `Dataset.open` from the resolved `.svar2` path (the analog of SVAR1's once-built
  `ffi_static` global table). Holds the memmapped cached 4-array ranges bundle. `Dataset.open`
  resolves + fingerprints the `Svar2Link` (with a `svar2=` override paralleling `svar=`).
- **One FFI call per read.** `Dataset.__getitem__` → `_query.py` → a gvl Rust pyfunction that, for the
  requested `(region, sample)` block: reconstructs a `genoray_core::RangesBundle` from the cached
  memmap slice → calls the genoray **read-bound gather** → gets the split-dense `BatchResult` →
  feeds the reconstruct core **in Rust** (no numpy round-trip, no Python `gather_ranges`).
- **Reconstruct kernel (read-bound variant).** A read-bound form of
  `reconstruct_haplotypes_from_svar2` consumes the split-dense `BatchResult` and **merges
  `var_key ⋈ dense_snp ⋈ dense_indel` by position** during assembly (extending the current
  two-source splice), decoding keys inline via `svar2-codec`. `LongAlleleReader` lookups resolve
  through the `ContigReader`'s LUT (no `lut_bytes`/`lut_off` marshalling). Tracks route
  `shift_and_realign_tracks_from_svar2` off the **same** `BatchResult`.
- **Dispatch discriminant.** A dataset carries a source discriminant (`svar2_link` present) selecting
  the SVAR2 reconstructor over the SVAR1 one; retires `TODO(svar2-dataset-dispatch)` in
  `_svar2_source.py` (its live `overlap_batch` marshalling is removed for the cached path).

## Output modes — all four, all Rust (Phase 1)

All four modes read the **same** cached ranges and the **same** read-bound `BatchResult`; they differ
only in the final assembly kernel:
- **Haplotypes** (`with_seqs("haplotypes")` / default) — `reconstruct_haplotypes_from_svar2`
  (read-bound).
- **Tracks** (`with_tracks`) — `shift_and_realign_tracks_from_svar2` (needs only `ilen`/`deletion_len`,
  no alleles).
- **Variants** and **variant-windows** (`with_seqs("variants"/"variant-windows")`) — a gvl Rust
  kernel decodes the per-hap overlapping keys from the read-bound `BatchResult` into `RaggedVariants`
  (via `svar2-codec` `decode_key`: `Inline`/`PureDel`/`Lookup`, LUT via the `ContigReader`), mirroring
  genoray's `decode_hap`. **No** Python `gather_ranges`/`decode` path. Same static-table Rust route
  gvl already uses for SVAR1 variants.

## Cache format (`.gvl/genotypes/svar2_ranges/`)

The 4 offset arrays + 2 index vectors from `find_ranges(samples=dataset)`, as in-place memmaps
(`out=`), region/sample-sharded to the dataset layout; `svar2_meta.json` records shapes/dtypes
(mirrors SVAR1's `svar_meta.json`); `Svar2Link` → shared `.svar2`.

```
vk_snp_range      (R, S', P, 2)   -> vk_snp packed positions/keys (2-bit)
vk_indel_range    (R, S', P, 2)   -> vk_indel packed positions/keys (u32)
dense_snp_range   (R, 2)          -> dense/snp  window (2-bit)
dense_indel_range (R, 2)          -> dense/indel window (u32)
region_starts     (R,)            -> q_start per region (left-overlap re-check)
sample_cols       (S',)           -> selected slot -> original sample index
```
**Size:** O(offsets) — same order as SVAR1's `offsets.npy`. The bulk (var_key + dense positions/keys,
genotype bitmatrices, LUT) stays in the compressed `.svar2`; SVAR2's on-disk size advantage
(1.46–5.67× smaller than `.svar`, per the MVP benchmark) is preserved.

## Data flow

- **WRITE:** `bed + .svar2` → per contig `find_ranges(..., samples=dataset, out=memmaps)` (search
  once, per-class) → 4-array ranges cache + `Svar2Link` + `svar2_meta.json`.
- **READ:** `dataset[regions, samples]` → slice cached 4-array block → **read-bound gather**
  (tree-free, union-free, per-class windows only) → split-dense `BatchResult` → Rust reconstruct
  (haps / tracks / variants / variant-windows). **No interval search and no contig-wide union at
  read.**

## Parity & testing

- **Byte-identical:** read-bound reconstruct ≡ shipped union reconstruct ≡ `decode` oracle, for all
  four output modes, on the M6b matrix (SNP/INS/DEL × samples × ploids) + real chr21 germline &
  somatic stores. Track re-alignment matched the same way.
- **Additive guarantee:** SVAR1 path byte-unchanged — `pixi run -e dev pytest tests -q` +
  `cargo test`. genoray: query-only build compiles; full genoray suite green; the shipped union path
  unchanged.
- **Perf verification (same-session, shared-node caveat):** a warm SVAR2 `Dataset` read shows
  **neither** `SearchTree::build` **nor** the dense-union rebuild — the perf DSO split flips from
  ~80% genoray to gvl-kernel-bound, like SVAR1. Report as a relative before/after within one
  allocation (absolute wall-clock is not comparable across allocations on shared Carter nodes).
- **Docs/roadmaps:** update the genoray roadmap (read-bound per-class gather; conversion feature) and
  `docs/roadmaps/rust-migration.md`; update user-facing docs for `.svar2` as a `write` variant source
  — `skills/genvarloader/SKILL.md`, `docs/source/{api.md,write.md,format.md,faq.md}`, `README.md`
  (docs-audit + skill-maintenance gates), and keep `api.md` in sync with any new `__all__` symbols.

## Benchmark — relocate & re-run

- **Relocate:** `mv /carter/users/dlaub/repos/for_loukik/svar2_mvp /carter/users/dlaub/projects/svar2_mvp`
  (4.1 GB, not a git repo — plain move; update any absolute paths in `build_source.sh` /
  benchmark driver).
- **Re-run** the SVAR1-vs-SVAR2 `gvl.Dataset.__getitem__` benchmark on chr21 germline (3202) +
  somatic (16007) after wiring: latency (same-session before/after) + store size. **Success:** the
  SVAR2 read is gvl-kernel-bound (no `SearchTree::build`, no union rebuild), and SVAR2's store-size
  advantage holds.

## Out of scope

- **Annotated** output mode (extends from the same cache later).
- Shipping / release (crates.io / PyPI) — local-only path-dep + wheel; the release gate is a later,
  separate step.
- Any change to the on-disk `.svar2` format itself (already split + 2-bit-packed; genoray-owned).
- Profiling experiments **E2** (same-cohort sample sweep) and **E4** (conversion build-thread
  allocation) — parked; **E3** (dense-access probe) dropped (gvl dense-gather ≈1%).

## Deliverables & sequencing

1. **genoray PR** — `conversion` query-only feature + per-class read-bound `find_ranges`/gather +
   split-dense `BatchResult` + parity (vs union & `decode`); build local wheel + crate.
2. **gvl PR** — write: `_write_from_svar2` + `_svar2_link.py` (4-array cache, dataset samples);
   read: `Svar2Store` + read-bound reconstruct kernels for all four output modes +
   `Dataset.open`/`__getitem__` dispatch + `_svar2_source.py` retirement; `genoray_core` path-dep.
3. **Relocate + re-run** the MVP benchmark.
4. **Docs/roadmap** updates.

## Open questions / risks

- **read-bound `BatchResult` channel factoring** (keep var_key merged + dense split, vs. fully
  4-channel unmerged) is a genoray-internal detail settled in the genoray PR; gvl consumes the
  payload the read-bound gather exposes, staying agnostic.
- **wheel ↔ path-dep sync:** the Python wheel that writes the cache and the Rust path-dep that reads
  it must be the same genoray commit; document the local-dev sync discipline (a fingerprint/version
  check at `Dataset.open` is a possible guard).
- **format version:** adding `svar2_link` to `metadata.json` is additive; confirm
  `_check_dataset_format_version` tolerates it and any bump is handled by `_migrate.py`.
- **`conversion` feature partition:** confirm the exact module set that transitively needs
  `rust-htslib` (only `vcf_reader.rs` references it directly; verify `writer`/`orchestrator`/
  `normalize` and their py-entry points are the complete gated set).
