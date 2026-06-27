# Zero-copy, scale-safe Rust read path (gvl format 2.0) — Design

**Status:** approved design, ready for implementation planning
**Date:** 2026-06-25
**Author:** brainstormed with the maintainer (david@standardmodel.bio)
**Related:** `docs/roadmaps/rust-migration.md` (Phase 3 throughput → optimization targets); memory `rust-memmap-ascontiguous-scalability`.

## Problem

The rust read path materializes **per-sample-scale memmapped arrays into RAM on every `ds[r, s]`**, which OOMs at gvl's >1M-sample design target. Confirmed via py-spy (`--native`, 43k samples: the hottest self-time leaf is numpy's `_aligned_strided_to_contig_size4` at ~20%) plus a per-batch copy trace (monkeypatched `np.ascontiguousarray` over one `ds[r, s]`):

- **The defect (rust-only):** track intervals are stored **array-of-structs** — `INTERVAL_DTYPE = [(start, i4), (end, i4), (value, f4)]`, itemsize 12 (`_ragged.py:26`). So `RaggedIntervals.{starts,ends,values}.data` are **strided field views** (stride 12, non-contiguous). The fused-rust track branch (`_reconstruct.py:241-250`) wraps each in `np.ascontiguousarray(..., i4/f4)`, copying the **entire per-sample-scale interval record store** into RAM every batch (3 × 3.6 MB on the toy corpus; GB-scale → OOM at 1M samples). The **numba** branch (`_reconstruct.py:271-274`) passes the same strided views directly with no copy, so this is a rust-path regression, not a pre-existing cost.
- **Same footgun, currently benign:** the fused kernels also wrap the full `genotypes.data`/`offsets` memmap in `np.ascontiguousarray`. Today that is a no-op (contiguous `int32`/`int64`) — but any future non-contiguous/mistyped genotype view would silently copy the whole sample-scale store.
- **Minor, sub-linear:** `variants.start` is stored `int64` and re-cast to `int32` every batch.
- **Unrelated avoidable work:** the fused kernels `Array1::zeros(total)` output buffers they then fully overwrite (`__memset` ~7.6% with 3 buffers on the annotated path).

## Goal

Eliminate per-batch materialization of per-sample-scale memmaps at the Python→Rust boundary; cache only the truly-static **sub-linear** arrays; skip provably-unnecessary zero-init — all **byte-identical** to current output. One breaking on-disk change (AoS → SoA intervals), gated behind a `format_version` major bump and an explicit migration.

## Global constraints

- **Byte-identical parity is the landing gate.** Every change here is layout/marshalling only; output bytes are unchanged. Verified across `GVL_BACKEND=rust` and `GVL_BACKEND=numba` via the existing `tests/parity` suites.
- **Public API change is limited and intentional:** add `gvl.migrate` to `python/genvarloader/__init__.py` `__all__`, and bump `DATASET_FORMAT_VERSION` to `2.0.0`. Per `CLAUDE.md`, the new public symbol + changed on-disk format **requires a `skills/genvarloader/SKILL.md` update** (open-a-dataset workflow + the migration note). No other public signatures change.
- **No new perf gate.** Throughput is recorded, not gated (consistent with the migration roadmap). The hard new gate is the **scale-guard test** (no memmap-materializing copy on the read path).
- **Commands under pixi:** `pixi run -e dev <task>`; build the ext with `pixi run -e dev maturin develop --release` after Rust changes. Dataset/parity tests need `--basetemp=$(pwd)/.pytest_tmp` (Carter os.link Errno 18). Prefix shell with `rtk`. Lint/format/typecheck scope: `ruff check python/ tests/`, `ruff format python/ tests/`, `pixi run -e dev typecheck`.
- **Merge style:** merge commit, never squash.

---

## Components

### A. On-disk intervals: AoS → SoA (`format_version` 1.0.0 → 2.0.0)

The single biggest change and the only breaking one.

- **Constant:** `DATASET_FORMAT_VERSION` (`_write.py:44`) → `2.0.0`. Its doc comment already says "Bump MAJOR only when an existing dataset can no longer be read correctly by new code" — this qualifies.
- **Write** (`_write.py`, the two `dtype=INTERVAL_DTYPE` allocation/serialization sites near `:1091` and `:1325`, plus the per-track writer that emits `intervals/<track>/intervals.npy`): emit **three contiguous arrays** per track instead of one record array:
  - `intervals/<track>/starts.npy` — `int32`, contiguous
  - `intervals/<track>/ends.npy` — `int32`, contiguous
  - `intervals/<track>/values.npy` — `float32`, contiguous
  - `intervals/<track>/offsets.npy` — **unchanged** (the ragged grouping is identical; only the data layout changes).
- **Read** (`_tracks.py::_open_intervals`, `:707-722`): memmap the three contiguous arrays directly and build `RaggedIntervals` from them, so `.starts/.ends/.values.data` are C-contiguous memmaps (no field-view stride).
- `INTERVAL_DTYPE` (`_ragged.py:26`) is **removed from the on-disk format and the read path**. It may remain for (a) one-time in-memory record construction during `gvl.write` (the write path is not the hot per-batch path, so a copy there is harmless) and (b) the migration reader (Component C). The binding requirement is that **`_open_intervals` no longer produces strided field views** — what the writer does in memory before serializing three contiguous files is an implementation detail.
- New `gvl.write` datasets are born `2.0.0` / SoA.
- **No Rust-kernel change.** The Rust entries (`intervals_to_tracks`, `intervals_and_realign_track_fused`) already take `itv_starts`/`itv_ends`/`itv_values` as three separate arrays; SoA storage simply makes the arrays Python hands them contiguous.

### B. Version gate on open (new)

The dataset open path does **not** currently validate `format_version` (only `_fasta_cache.py:175 _check_format_version` does, for the FASTA cache). Add the equivalent for datasets:

- A `_check_dataset_format_version(meta, path)` helper invoked where `_open.py` loads `metadata.json` into the `Metadata` model (`format_version` field at `_write.py:72`).
- `meta.format_version.major < DATASET_FORMAT_VERSION.major` → raise a clear error instructing the user to run `gvl.migrate(path)`.
- `meta.format_version.major > DATASET_FORMAT_VERSION.major` → raise "dataset written by a newer gvl; upgrade genvarloader".
- Equal major → proceed.
- Datasets with `format_version is None` (pre-versioning) are treated as the oldest major → migrate path. The committed test datasets must be brought to 2.0.0 so the suite runs: regenerate the toy fixtures via `pixi run -e dev gen`, and bring the benchmark corpus (`tests/benchmarks/data/chr22_geuv.gvl`, built by `build_realistic.py` rather than `gen`) to 2.0.0 by running the new `gvl.migrate` on it — which also dogfoods the migration. Confirm which committed datasets are `None` vs `1.0.0` during implementation.

### C. `gvl.migrate(path)` — new public API

In-place, streaming, idempotent rewrite of a 1.x AoS dataset to 2.0 SoA.

- **Signature:** `gvl.migrate(path: str | Path) -> None` (added to `__init__.py __all__`). Lives in a new module, e.g. `python/genvarloader/_dataset/_migrate.py`.
- **Algorithm, per track under `intervals/<track>/`:**
  1. Open `intervals.npy` as an `INTERVAL_DTYPE` memmap (read-only); stream it in fixed-size record chunks (never load the whole store into RAM).
  2. Write `starts.npy`, `ends.npy`, `values.npy` by appending each chunk's `["start"]/["end"]/["value"]` fields to the three contiguous output files; `flush`/`fsync` each.
  3. After **all** tracks' SoA files are written and fsynced, update `metadata.json` `format_version` → `2.0.0` (**last** durable write).
  4. Then delete each `intervals.npy`.
- **Idempotency / crash-safety by ordering:** metadata is bumped only after SoA is durable, so an interruption leaves the dataset still-1.x (old `intervals.npy` intact, re-runnable). If interrupted after the metadata bump but before deletion, both layouts coexist harmlessly; a re-run completes the cleanup. `migrate` on an already-2.0 dataset is a no-op (idempotent check on `format_version`).
- **Disk:** peak extra ≈ one track's interval store (transient), never the whole dataset. Genotypes/regions/reference are untouched.
- Emit progress logging (per-track, record counts) consistent with the existing writer's logging.

### D. Zero-copy FFI contract + loud boundary guard

Establish one rule for **all per-sample-scale FFI args**: cross zero-copy, or fail loudly — never silently materialize.

- **Drop `np.ascontiguousarray(...)`** on per-sample-scale memmapped args at the call sites:
  - `_reconstruct.py:241-250` — the SoA interval fields (now contiguous → drop is safe and the copy is gone).
  - `_reconstruct.py:232-234` and the `_haps.py` fused calls (plain `~789-813`, annotated `~917`, splice `~859`) — `genotypes.data`, `genotypes.offsets` / `_as_starts_stops(...)` inputs derived from them.
- **Add a shared boundary helper**, e.g. `_ffi_array(arr, dtype, name) -> np.ndarray` in a small util, that asserts `arr.flags["C_CONTIGUOUS"]` and `arr.dtype == dtype` and raises a precise `ValueError` naming the arg if violated (so a future non-contiguous/mistyped per-sample-scale array fails at the call site with an intelligible message instead of a silent GB copy or an opaque PyO3 error). Apply it to the per-sample-scale args in place of the dropped `ascontiguousarray`.
- Per-batch-sized arrays that are genuinely freshly constructed and may be non-contiguous (e.g. a strided column slice like `regions[:, 1]`, `flat_shifts.reshape(...)`) are **batch-bounded**, not sample-scale; keep coercing those (cheap) — the guard is specifically for the sample-scale memmaps. Document this distinction at the call sites.

### E. RAM-cache the sub-linear static arrays

- Cache, once per reconstructor (lazy, lifetime = the `Haps`/reconstructor object), the typed-contiguous per-variant/reference arrays the kernels consume: chiefly `v_starts` (`variants.start`, `int64`→`int32` recast today); `ilens`, `alt.data`, `alt.offsets`, `reference`, `ref_offsets` are already no-ops but get cached for uniformity and to drop their per-batch `ascontiguousarray` calls.
- **No memory knob** (YAGNI): these grow only with the variant count (≲ a few billion germline variants even at 1M samples → fits ≥64 GB RAM, per the maintainer's sizing). Per-sample-scale arrays are explicitly **excluded** from caching (Component D governs them).
- Implementation seam: a cached property / precomputed dataclass field on the reconstructor holding the FFI-ready arrays; computed on first `ds[r, s]` (or at reconstructor construction).

### F. Skip zero-initialization where provably full-write

- Replace `Array1::zeros(total)` with uninitialized allocation in the fused kernels (`src/ffi/mod.rs`): `out_data` in `reconstruct_haplotypes_fused`, `reconstruct_annotated_haplotypes_fused` (+ its `annot_v`/`annot_pos`), `reconstruct_haplotypes_spliced_fused`, and the fused tracks kernel's scratch/output buffer — **only** where the reconstruct/track core writes **every** output position for in-contract inputs.
- **Safety argument (documented at each site):** out-of-contract inputs (a deletion driving `ref_idx` past the contig end) are **already** undefined and excluded from the parity oracle by the existing overshoot/double-init guards (`tests/parity/test_reconstruct_haplotypes_parity.py`). So uninitialized allocation adds no new observable exposure: in-contract → fully written; out-of-contract → already undefined. Use a safe-Rust uninitialized pattern (e.g. `Array1::uninit` + assume-init only after the full-write, or `Vec::with_capacity` + set_len behind a clearly-documented invariant). Prefer the least-`unsafe` construction that compiles clean under clippy.
- This is the one component where parity could regress if the full-write invariant is wrong; gate it behind the existing reconstruct/track parity suites on both backends and keep the change isolated (own commit) so it can be reverted independently.

### Out of scope (deferred)

- **Reverse-complement fusion** into the kernel (the strand RC numpy post-pass, ~9% inclusive). Noted by the maintainer for future planning; not part of this spec.
- The Phase 5 "single big `__getitem__` kernel" rewrite — targets D–F are complementary to it but do not depend on it.

---

## Testing & parity

- **Byte-identical parity (gate):** run `GVL_BACKEND=rust` and `GVL_BACKEND=numba` over `tests/parity` (and the dataset/unit/integration suites) — output unchanged by every component.
- **New tests:**
  1. **Migration round-trip:** write a small 1.x AoS dataset (or fixture), run `gvl.migrate`, assert (a) the three SoA files exist and `intervals.npy` is gone, (b) `metadata.json` `format_version == 2.0.0`, (c) `ds[r, s]` is byte-identical to the pre-migration read. Also assert `migrate` is idempotent (second run is a no-op) and re-runnable after a simulated mid-write interruption.
  2. **Version gate:** opening a 1.x dataset raises with the `gvl.migrate` hint; opening a synthesized "future major" raises the upgrade error.
  3. **Scale-guard (the hard new gate):** monkeypatch `np.ascontiguousarray` over one `ds[r, s]` (haps, annotated, tracks-only) and assert **zero** copies whose source `.base` is an `np.memmap` — locks the defect closed and prevents regressions. (Mirrors the diagnostic used to find the bug.)
  4. **FFI guard:** feed a deliberately non-contiguous per-sample-scale array to the boundary helper and assert it raises the precise error (never a silent copy).
- **Build/CI:** `maturin develop --release`, `cargo test`, `ruff check/format`, `typecheck`, abi3 wheel build. Regenerate committed test datasets to 2.0.0 (`pixi run -e dev gen`) so the suite runs against the new format.
- **Throughput (recorded, not gated):** re-run `tests/benchmarks/test_e2e.py` on both backends; expect the rust tracks/annotated paths to close further on numba once the per-batch interval copy is gone. Record in the roadmap.

## File-touch map

| File | Change | Component |
|---|---|---|
| `python/genvarloader/_dataset/_write.py` | `DATASET_FORMAT_VERSION` → 2.0.0; write SoA `starts/ends/values.npy` per track | A |
| `python/genvarloader/_ragged.py` | retire `INTERVAL_DTYPE` from read/write (keep for migration only) | A |
| `python/genvarloader/_dataset/_tracks.py` | `_open_intervals` memmaps three contiguous arrays | A |
| `python/genvarloader/_dataset/_open.py` | call `_check_dataset_format_version` on load | B |
| `python/genvarloader/_dataset/_migrate.py` (new) | `migrate()` streaming in-place AoS→SoA | C |
| `python/genvarloader/__init__.py` | export `migrate` in `__all__` | C |
| `python/genvarloader/_dataset/_reconstruct.py` | drop `ascontiguousarray` on sample-scale args; apply `_ffi_array` guard | D |
| `python/genvarloader/_dataset/_haps.py` | same for the fused haps/annotated/splice calls | D |
| `python/genvarloader/_dataset/_utils.py` (or new util) | `_ffi_array(arr, dtype, name)` boundary helper | D |
| reconstructor (`_haps.py` / `_reconstruct.py`) | cache FFI-ready sub-linear arrays | E |
| `src/ffi/mod.rs` | uninitialized output allocation in the four fused kernels | F |
| `skills/genvarloader/SKILL.md` | document `gvl.migrate` + format 2.0 open behavior | A/C |
| `tests/parity/`, `tests/unit/`, `tests/integration/` | migration round-trip, version gate, scale-guard, FFI-guard tests | all |
| `docs/roadmaps/rust-migration.md` | mark targets 1–2 (and the zero-init part of 3) addressed; record throughput | all |

## Risks & mitigations

- **Parity regression from skip-zero-init (F)** — isolate in its own commit; gate on reconstruct/track parity both backends; revertable independently.
- **Committed test datasets are 1.x** — bring to 2.0.0 as part of the work (toy fixtures via `gen`; benchmark corpus via `gvl.migrate`), else the version gate fails the whole suite. Verify the `gen` task and every committed `.gvl` fixture.
- **Hidden interval readers** — audit for any consumer of `intervals.npy` / `INTERVAL_DTYPE` beyond `_open_intervals` and the writer (e.g. tooling, `_table.py`) before retiring the AoS read path.
- **`format_version is None` datasets** — treat as oldest-major (migrate); confirm behavior on a synthesized `None` metadata.
- **Migration interruption** — ordering (SoA durable → metadata bump → delete AoS) makes it re-runnable; the round-trip test exercises an interrupted-then-resumed run.
