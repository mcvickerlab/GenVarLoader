# Streaming variants output — `min_af` / `max_af` (Wave B PR-B2)

**Issue:** [#317](https://github.com/mcvickerlab/GenVarLoader/issues/317) (split out of
[#304](https://github.com/mcvickerlab/GenVarLoader/issues/304), Wave B of
[#277](https://github.com/mcvickerlab/GenVarLoader/issues/277)).
**Follows:** PR-B0 + PR-B1 (streaming `with_seqs("variants")`), landed in
[#316](https://github.com/mcvickerlab/GenVarLoader/pull/316).
**Follow-up filed:** [#319](https://github.com/mcvickerlab/GenVarLoader/issues/319)
(VCF live-INFO AF via the record-stream `FieldSpec` path — explicitly out of scope here).
**Parent design:** `docs/superpowers/specs/2026-07-20-streaming-variants-output-wave-b-design.md` (§PR-B2).
**Target branch:** `streaming` (streaming-coordination rules — see `CLAUDE.md`).

## Goal

Add `min_af` / `max_af` allele-frequency filtering to **streaming**
`StreamingDataset.with_seqs("variants")`, byte-identical to the written `Dataset`
variants path at `jitter=0`. The deliverable is **full support on the SVAR1
backend** plus **guard-parity on VCF/PGEN**, so streaming and written remain
interchangeable everywhere.

## Key feasibility finding (why this scope)

Investigating the code (not just the parent spec) showed the parent spec's
"VCF/PGEN: request the AF INFO column from genoray" assumption is **wrong**. AF
filtering is **effectively SVAR-only** on the written path today:

- **`gvl.write` never computes `AF`.** VCF: `_write_gvi_index()` is called with no
  `info=` (`_write.py:248`), so the written `variants.arrow` has no `AF` column
  unless a `.gvi` was pre-built externally with `info=["AF"]`. PGEN: the index
  parser (`genoray/_pgen.py:_load_index`) only reads INFO for SVLEN/END/ILEN —
  **no numeric AF path at all**. So `Dataset.with_settings(min_af=...)` on a
  VCF/PGEN-sourced dataset **raises** the AF-missing `RuntimeError`
  (`_haps.py:334-340`) today.
- **PGEN record streams cannot supply AF regardless** — genotype-only, no INFO;
  `PgenWindowFiller` has no `fields` and `DenseChunk.info_staged` is always empty
  for PGEN.
- **SVAR is the coherent case.** `genoray.SparseVar.cache_afs()`
  (`_svar/_annotate.py:556-561`) computes AF from genotypes and writes an `AF`
  column into the `.svar` index. The written oracle reads it via
  `variants.info["AF"]` (`_haps.py:145-179`, a plain numeric-column read of the
  linked index) and streaming can read the **same** `sv.index["AF"]` at backend
  construction (`_Svar1Backend.__init__`, `_streaming.py:1148-1293`).

Reading AF from *live* VCF INFO on the streaming side (`FieldSpec` →
`DenseChunk.info_staged`) would give streaming a capability the written path
lacks by default, **breaking the streaming ⟺ written parity contract**. That
path is therefore deferred to **#319**, where it is fixed on *both* sides
together.

## Background — the written oracle

`min_af`/`max_af` are `Dataset.with_settings` params (`_impl.py:97-98`, threaded
into `Haps`). The filter lives in the variants assembly path `get_variants_flat`
(`_flat_variants.py:811-817`):

```python
if haps.min_af is not None or haps.max_af is not None:
    geno_afs = np.asarray(haps.variants.info["AF"])[v_idxs]
    keep = np.full(len(v_idxs), True, np.bool_)
    if haps.min_af is not None:
        keep &= geno_afs >= haps.min_af
    if haps.max_af is not None:
        keep &= geno_afs <= haps.max_af
```

`keep` is combined with the #202 region-overlap clip and applied via
`_compact_keep` before allele assembly. Semantics PR-B2 must reproduce exactly:

- **Inclusive bounds** on both sides (`>=` / `<=`).
- **AF gathered per variant** (`AF[v_idxs]`) — a **pure per-variant predicate**
  (AF does not vary by row/sample), so it can be folded into the per-variant
  inline keep.
- **AF-missing guard** (`_haps.py:334-340`): if `min_af`/`max_af` is set and the
  dataset has no cached `AF` column, raise `RuntimeError` with the message
  `"Either this dataset is not backed by an SVAR file, or the SVAR file has not
  had AFs cached yet."`
- **Haplotype/annotated output** raises `NotImplementedError` when AF filtering
  is requested (`_haps.py:707-709`) — AF filtering is variants-only.

## Streaming baseline (post-PR-B1)

- `StreamingDataset.with_settings` (`_streaming.py:1009`) exposes only
  `jitter`/`rng`/`deterministic` — **no** `min_af`/`max_af`.
- SVAR1 backend `Svar1Backend` (`src/ffi/stream_engine.rs:109-141`) holds global
  variant-scale tables `v_starts`/`ilens`/`alt_alleles`/`alt_offsets` (`:118-121`);
  `generate_variants` (`stream_engine.rs:257-315`) walks the region-expanded CSR
  and applies the region-overlap clip **inline** before pushing each global
  `gvi`, then calls the shared `assemble_variants_window` (`src/variants/mod.rs:105`).
- SVAR1 arrays are built in `_Svar1Backend.__init__` from `sv.index` via
  `_variant_arrays_from_table` (`_haps.py:119-133`) — POS/ILEN/REF/ALT only, **no
  AF** — and passed to the engine `#[new]` constructor
  (`_streaming.py:1387-1390`; constructor `stream_engine.rs:394-432`).
- Record backend (`RecordBackend`, `src/record_stream/engine.rs`; `DecodedWindow`,
  `transpose.rs`) carries no AF and — per the finding above — **needs none** for
  PR-B2.

## Design

### Approach: SVAR1 inline-fold; VCF/PGEN guard-only

**SVAR1 (full support):** extend `Svar1Backend`'s existing per-variant inline
keep with an AF term (user-approved inline-fold, byte-identical to
`region_keep & af_keep` + compact because AF keep is a pure per-variant
predicate). No separate compaction pass; no `DecodedWindow` touch (SVAR1 uses a
global backend table, not a per-window channel).

**VCF/PGEN (guard-parity):** no Rust changes. Reproduce the written
`RuntimeError` at the Python surface when `min_af`/`max_af` is set on a source
with no cached `AF`. This matches the written path exactly (which raises there
today) and keeps the two interchangeable.

### 1. Python surface (`python/genvarloader/_dataset/_streaming.py`)

- Add `min_af: float | None = None` / `max_af: float | None = None` keyword
  params to `StreamingDataset.with_settings` (mirroring `Dataset.with_settings`).
  Store as `_min_af` / `_max_af` on the copied instance (same
  `object.__setattr__` pattern as `_jitter`); initialize both to `None` in
  `__init__`/field defaults.
- **Output-mode guard:** AF filtering with a non-`variants` output raises
  `NotImplementedError`, mirroring `_haps.py:707-709`. Fire at `to_iter`/config
  time (where `_variants = self._seq_kind is RaggedVariants` is already computed,
  `_streaming.py:464`), before any batch is produced.
- **AF-missing guard:** when `min_af`/`max_af` is set, check whether the source
  exposes a cached `AF` column; if not, raise the **same** `RuntimeError`
  (same message) as `_haps.py:334-340`. Availability check per backend:
  - **SVAR1:** `"AF" in sv.index.collect_schema().names()` (the `.svar` index the
    backend already opens). Computed in `_Svar1Backend` and surfaced to the
    `StreamingDataset` guard (e.g. a `has_cached_af: bool` property on the
    backend, analogous to how `_variants`/`_annotated` capabilities are checked).
  - **VCF/PGEN:** `False` unless the genoray index already exposes an `AF` numeric
    INFO column (`available_info_fields`-style check). In practice this raises for
    PGEN always and for VCF unless a `.gvi` was pre-built with `info=["AF"]` —
    exactly matching the written path.
- Thread `_min_af`/`_max_af` into `build_engine` → the engine constructor (below).

### 2. `build_engine` plumbing (`_streaming.py`, all three backends)

Extend the shared `build_engine` signature (identical across
`_Svar1Backend`/`_VcfBackend`/`_PgenBackend`, at `_streaming.py:1295`/`1858`/`1999`):

```python
def build_engine(self, jobs, batch_size, output_length,
                 annotated=False, variants=False,
                 min_af=None, max_af=None):
```

- `_Svar1Backend.build_engine`: load the global AF array when present
  (`afs = self._afs`, an `NDArray[np.float32]` gathered in `__init__` from
  `sv.index["AF"]` parallel to `_v_starts`; `None`/empty when uncached), and pass
  `afs`, `min_af`, `max_af` to the `Svar1StreamEngine` constructor.
- `_VcfBackend.build_engine` / `_PgenBackend.build_engine`: accept and **ignore**
  `min_af`/`max_af` (the Python guard already rejected any case where they are
  meaningfully set — a set value here only survives if AF was somehow available,
  which the record engines do not support in PR-B2; the guard prevents reaching
  here with AF filtering active). Signature parity keeps `_iter_batches`'
  single `build_engine(... , min_af=, max_af=)` call site uniform.

### 3. SVAR1 AF array (`_Svar1Backend.__init__`, `_streaming.py`)

- After reading `idx = sv.index.sort("index")` (`_streaming.py:1200`), if the
  schema has `AF`, extract `self._afs = idx["AF"].to_numpy().astype(np.float32)`
  (parallel to `_v_starts`, global variant order). Else `self._afs = None`.
- Expose `has_cached_af` (True iff `_afs is not None`) for the Python guard.
- `_variant_arrays_from_table` is left unchanged (POS/ILEN/REF/ALT); AF is read
  separately since it is optional and only SVAR1 uses it.

### 4. Rust: `afs` on `Svar1Backend` + keep-fold (`src/ffi/stream_engine.rs`)

- Add `afs: Option<Array1<f32>>` and `min_af: Option<f32>`, `max_af: Option<f32>`
  fields to `Svar1Backend` (`:109-141`), populated via the shared constructor
  `Svar1StreamEngine::build` (`:331-368`) and the `#[new]` pyclass constructor
  (`:394-432`). Add trailing pyO3 params `afs: Option<PyReadonlyArray1<f32>>`,
  `min_af: Option<f32>`, `max_af: Option<f32>` (all with `= None` in the
  `#[pyo3(signature = ...)]`), so existing callers are unaffected.
- In `Svar1Backend::generate_variants` (`:257-315`), extend the inline keep:

  ```rust
  let v_start = self.v_starts[gvi as usize] as i64;
  let v_ilen  = self.ilens[gvi as usize] as i64;
  let v_end   = v_start - v_ilen.min(0) + 1;
  let region_keep = v_start < r_e as i64 && v_end > r_s as i64;
  let af_keep = match &self.afs {
      Some(afs) => {
          let af = afs[gvi as usize];
          self.min_af.map_or(true, |m| af >= m) && self.max_af.map_or(true, |m| af <= m)
      }
      None => true, // no filtering requested (or no AF); guard handled in Python
  };
  if region_keep && af_keep { kept.push(gvi); }
  ```

  Everything downstream (`row_offsets`, `assemble_variants_window`, FFI
  marshaling, Python `RaggedVariants` packing) is unchanged.

### 5. Tests

**SVAR1 parity** (`tests/dataset/test_streaming_variants_parity.py` — extend, or
a sibling test module):

- Build an **AF-cached SVAR1 case**: copy the svar source into `tmp`, call
  `genoray.SparseVar(copy).cache_afs()` (writes `AF` into the `.svar` index),
  then `gvl.write` a fresh oracle dataset from it and point `StreamingDataset` at
  the same svar. This gives both paths a real `AF` column. (New fixture/helper;
  the existing `svar1_multicontig_fixture` carries no AF.)
- Parametrize over `min_af` / `max_af` / a combined band; apply
  `.with_settings(min_af=..., max_af=...)` on **both** the written oracle and the
  `StreamingDataset`; assert **byte-identical** `RaggedVariants` cell-by-cell
  (reuse `_assert_variants_cell_matches`).
- **Non-vacuity:** assert the filtered total is **strictly less** than the
  unfiltered total for at least one case (proving the filter removes variants),
  keeping the existing `total_variants > 0` guard.

**AF-missing guard parity** (all backends):

- SVAR1 **without** `cache_afs` + `min_af` → `RuntimeError` (same message) on
  both streaming and written.
- VCF + `min_af` → `RuntimeError` on both (source has no AF index).
- PGEN + `min_af` → `RuntimeError` on both.
- Non-`variants` output (e.g. `with_seqs("haplotypes")`) + `min_af` on streaming
  → `NotImplementedError`, matching `_haps.py:707-709`.

**Rust unit** (`src/ffi/stream_engine.rs` tests): `Svar1Backend::generate_variants`
with a small hand-built `afs` — a variant kept by the region clip but dropped by
AF, and one inside the AF band but outside the region — proving the two keep
terms compose as **AND**.

## Decisions

- **SVAR1-only AF filtering; VCF/PGEN guard-parity.** Matches the written path's
  actual capabilities (AF is SVAR-only today) and preserves streaming ⟺ written
  interchangeability. VCF live-INFO AF deferred to #319 (a written-path gap
  first).
- **Inline-fold over a separate compact pass** — byte-identical (pure
  per-variant predicate), no extra allocation.
- **SVAR1 AF as an optional global backend table** (`Option<Array1<f32>>`), not a
  per-window channel — matches SVAR1's global-id model; no `DecodedWindow` /
  record-backend change.
- **Guard + output-mode errors mirror the written path exactly** (same exception
  types and messages).
- **`build_engine` gains `min_af`/`max_af` on all three backends** for call-site
  uniformity; the record backends accept-and-ignore them (the Python guard
  prevents them from being meaningfully set for VCF/PGEN).

## Scope / non-goals

- **No VCF/PGEN AF *filtering***; only the guard (raise). VCF live-INFO AF →
  **#319**.
- No `var_fields` (dosage / custom FORMAT/INFO) — **PR-B3**.
- No REF bytes / `variant-windows` — **PR-B4**.
- No AF filtering for haplotype / annotated / `variant-windows` output (raises,
  same as written).
- `.svar2` remains haplotypes-only (unchanged).
- No new on-disk format. The only public-surface change is two new keyword args
  on `StreamingDataset.with_settings` (matching `Dataset.with_settings`) — **no
  new exported symbol**, so no `api.md`/`__all__` change.

## Docs to update when PR-B2 lands

Per `CLAUDE.md` (skill + docs-audit gates):

- `skills/genvarloader/SKILL.md` — `StreamingDataset.with_settings` now accepts
  `min_af`/`max_af` (SVAR1 variants output; VCF/PGEN raise the same guard as the
  written path).
- `docs/source/dataset.md` / `docs/source/faq.md` — streaming AF-filtering note +
  the SVAR-only caveat and `cache_afs` requirement.
- `docs/roadmaps/streaming-dataset.md` — tick PR-B2, update the Wave B status
  line, link this design + #317 + #319.
- No `api.md` / `__all__` change.

## Testing plan (summary)

| Test | Backend(s) | Asserts |
|---|---|---|
| AF-cached SVAR1 parity | svar1 | byte-identical `RaggedVariants` vs written oracle under `min_af`, `max_af`, combined band; filtered total < unfiltered for ≥1 case |
| AF-missing guard | svar1 (uncached), vcf, pgen | same `RuntimeError` + message as written when `min_af`/`max_af` set without cached AF |
| Output-mode guard | svar1 | `NotImplementedError` for `min_af` + non-`variants` output, matching `_haps.py:707-709` |
| Rust unit (svar1 engine) | svar1 kernel | region-kept-but-AF-dropped and AF-kept-but-region-dropped compose as AND |

**Rebuild reminder:** Rust changes require `pixi run -e dev maturin develop
--release` before the Python parity tests import the extension (per `CLAUDE.md`)
— else pytest silently tests the stale binary.
