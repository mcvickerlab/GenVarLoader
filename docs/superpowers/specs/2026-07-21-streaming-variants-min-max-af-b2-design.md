# Streaming variants output — `min_af` / `max_af` (Wave B PR-B2)

**Issue:** [#317](https://github.com/mcvickerlab/GenVarLoader/issues/317) (split out of
[#304](https://github.com/mcvickerlab/GenVarLoader/issues/304), Wave B of
[#277](https://github.com/mcvickerlab/GenVarLoader/issues/277)).
**Also closes:** [#319](https://github.com/mcvickerlab/GenVarLoader/issues/319) — VCF/BCF
live-INFO AF via the record-stream `FieldSpec` path. **Folded into this PR** (was previously
deferred; see the 2026-07-21 scope change below).
**Follows:** PR-B0 + PR-B1 (streaming `with_seqs("variants")`), landed in
[#316](https://github.com/mcvickerlab/GenVarLoader/pull/316).
**Parent design:** `docs/superpowers/specs/2026-07-20-streaming-variants-output-wave-b-design.md` (§PR-B2).
**Target branch:** `streaming` (streaming-coordination rules — see `CLAUDE.md`).

## Goal

Add `min_af` / `max_af` allele-frequency filtering to **streaming**
`StreamingDataset.with_seqs("variants")`, **byte-identical** to the written `Dataset`
variants path at `jitter=0`. The deliverable is:

- **SVAR1** — full support: AF from the `.svar` index (`SparseVar.cache_afs()`).
- **VCF/BCF** — full support: AF read from the **VCF `INFO/AF`** field, on **both** the
  streaming side (live, via genoray's htslib record stream + a `FieldSpec`) **and** the
  written side (via a `.gvi` index built with `info=["AF"]`), so the two stay interchangeable.
- **PGEN** — guard-parity only: a PGEN record stream carries no INFO, and the written PGEN
  index has no AF, so **both** paths raise the same AF-missing `RuntimeError`.

## Scope change (2026-07-21): fold #319 into B2

An earlier revision of this spec scoped AF as **SVAR-only** and deferred VCF/BCF live-INFO
AF to #319, on the reasoning that the *written* path cannot filter VCF by AF today (so
streaming filtering VCF would break the streaming ⟺ written parity contract). That reasoning
is correct **only while the written path stays as-is**. The decision now is to **fix both
sides together in this PR** (exactly the "both sides together" #319 called for):

- The written VCF path **gains** AF by building its `.gvi` index with `info=["AF"]` when the
  VCF header declares an `INFO/AF` field. `gvl.Dataset` then reads it via the existing
  `variants.info["AF"]` numeric-column path — no new written-path filter code.
- The streaming VCF path reads the **same** `INFO/AF` field live via a genoray `FieldSpec`.

Because **both sides read the same `INFO/AF` field, ALT-resolved by the same genoray
`resolve_scalar(source_alt_index, ...)`**, parity holds by construction. This closes #319 and
removes the SVAR-only limitation.

### Feasibility (verified against the pinned genoray rev)

**No genoray change is required.** Everything is on the gvl side (rev `73d25cb`,
`Cargo.toml:25-26`):

- **Streaming AF (live INFO).** genoray's `FieldSpec` (`genoray/src/field.rs:102-108`;
  `FieldCategory::Info`, `HtslibType::Float`, `StorageDtype::F32`) is already accepted by
  `ChunkAssembler::new(&[FieldSpec])` and by `VcfRecordSource::with_sample_indices(&[FieldSpec], …)`.
  When an `AF` INFO spec is requested, the assembler stages **one Float per variant** into
  `DenseChunk.info_staged` (`genoray/src/types.rs:137-180`, `StagedColumn::Float`), ALT-resolved
  per atom via `resolve_scalar(atom.source_alt_index, spec)` in
  `chunk_assembler.rs:decompose_raw_record`. gvl's `VcfWindowFiller` currently passes
  `fields: Vec::new()` (`src/record_stream/vcf.rs:127`) so `info_staged` is always empty today —
  that is the one line to change.
- **Written AF (index INFO).** genoray's `_write_gvi_index(fields=None, info=None, …)`
  (`genoray/python/genoray/_vcf.py:943`) already accepts `info=["AF"]` and writes an `AF` column
  into the `.gvi`. gvl calls it with no `info=` today (`_write.py:248`).

## Background — the written oracle

`min_af`/`max_af` are `Dataset.with_settings` params (`_impl.py:97-98`, threaded into `Haps`).
The filter lives in the variants assembly path `get_variants_flat` (`_flat_variants.py:810-817`):

```python
keep = None
if haps.min_af is not None or haps.max_af is not None:
    geno_afs = np.asarray(haps.variants.info["AF"])[v_idxs]
    keep = np.full(len(v_idxs), True, np.bool_)
    if haps.min_af is not None:
        keep &= geno_afs >= haps.min_af
    if haps.max_af is not None:
        keep &= geno_afs <= haps.max_af
```

`keep` is AND-combined with the #202 region-extent clip (`_flat_variants.py:851-852`) and applied
via `_compact_keep` (`_flat_variants.py:855-858`) before allele assembly. Semantics PR-B2 must
reproduce exactly:

- **Inclusive bounds** on both sides (`>=` / `<=`).
- **AF gathered per variant** (`AF[v_idxs]`) — a **pure per-variant predicate** (AF does not
  vary by row/sample), so it can be folded into the per-variant keep.
- **AF availability drives the guard** (`_haps.py:333-340`): AF filtering with no available `AF`
  column raises `RuntimeError("Either this dataset is not backed by an SVAR file, or the SVAR
  file has not had AFs cached yet." …)`. Availability is discovered by scanning the index schema
  for a numeric `AF` column (`_haps.available_info_fields`, `_haps.py:184-195`, `313-332`) — so
  **once `gvl.write` writes an `AF` column, the written VCF guard passes automatically** and the
  filter runs.
- **Haplotype/annotated output** raises `NotImplementedError` when AF filtering is requested
  (`_haps.py:707-710`) — AF filtering is variants-only.

## Streaming baseline (post-PR-B1)

- `StreamingDataset.with_settings` (`_streaming.py:1009`) exposes only `jitter`/`rng`/`deterministic`.
- SVAR1 backend `Svar1Backend` (`src/ffi/stream_engine.rs:109-141`) holds global variant-scale
  tables `v_starts`/`ilens`/`alt_alleles`/`alt_offsets`; `generate_variants`
  (`stream_engine.rs:257-315`) walks the region-expanded CSR and applies the region-overlap clip
  **inline** before pushing each global `gvi`, then calls `assemble_variants_window`.
- Record backends (`RecordBackend`, `src/record_stream/engine.rs`; `DecodedWindow`,
  `src/record_stream/transpose.rs:29-49`) carry **no AF channel**. `fill_decoded_window`
  (`transpose.rs:50-`) copies `pos/ilens/alt/alt_offsets/global_idx` + the genotype transpose but
  **does not read `chunk.info_staged`**. The record variants assembly
  (`src/variants/windows.rs:assemble_variants_mode`, `generate_batch_core` in `src/ffi/mod.rs`)
  gathers alleles by window-local `v_idxs`; **confirm where PR-B1 applies the record-path region
  clip** — the AF keep folds in at the same site.

## Design

### A. SVAR1 (full support — inline fold)

Extend `Svar1Backend`'s existing per-variant inline keep with an AF term. Byte-identical to
`region_keep & af_keep` + compact because AF keep is a pure per-variant predicate. No separate
compaction pass; no `DecodedWindow` touch (SVAR1 uses a global backend table, not a per-window
channel).

- `Svar1Backend` gains `afs: Option<Array1<f32>>`, `min_af: Option<f32>`, `max_af: Option<f32>`
  (`stream_engine.rs:109-141`), threaded via `Svar1StreamEngine::build` (`:331-368`) and the
  `#[new]` pyclass constructor (`:394-432`, trailing pyO3 params defaulted `None`).
- In `generate_variants` (`:257-315`) the inner keep becomes
  `if region_keep && af_keep { kept.push(gvi) }`, with `af_keep` reading `self.afs[gvi]`.
- Python `_Svar1Backend.__init__` loads `_afs` from `sv.index["AF"]` (global order, `float32`)
  when the schema has `AF`; exposes `has_cached_af`.

### B. VCF/BCF (full support — write INFO/AF on both sides)

**B.1 Written path — `.gvi` with `info=["AF"]`.** In `gvl.write` (`_write.py:244-249`), when the
source is a VCF **whose header declares an `INFO/AF` field**, build the index with
`variants._write_gvi_index(info=["AF"])` instead of `_write_gvi_index()`. Then
`Dataset.open(...)` surfaces `variants.info["AF"]` through the existing numeric-column path and
the written AF filter runs unchanged.

- **Conditional on header presence.** Check the VCF header for an `INFO/AF` definition (via
  genoray's VCF metadata surface — confirm the exact API during implementation, e.g. an
  `available_info_fields`/header-inspection call on the `VCF` object). If `AF` is **absent**, keep
  the current `_write_gvi_index()` (no AF column) so AF-less VCFs are unaffected and both paths
  raise the AF-missing guard (parity preserved).
- **Benign default-behavior change.** VCFs that carry `INFO/AF` now get an `AF` column in the
  written index (one `f32`/variant — negligible) and `"AF"` becomes an available var_field. Flag
  in docs. *(Open a follow-up if a user later wants to opt out; not expected.)*

**B.2 Streaming path — live INFO/AF via `FieldSpec`.**

1. `VcfWindowFiller` requests an `AF` INFO `FieldSpec`
   (`FieldSpec { name: "AF", category: Info, htype: Float, dtype: F32, default: None }`,
   struct literal per `genoray/src/field.rs:248-255` — there is no `FieldSpec::new`) instead of
   `fields: Vec::new()` (`src/record_stream/vcf.rs:127`). It flows unchanged into both
   `VcfRecordSource::with_sample_indices` (`vcf.rs:166`) and `ChunkAssembler::new` (`vcf.rs:170-179`),
   so genoray populates `DenseChunk.info_staged[0]` as a per-variant `StagedColumn::Float`.
   - Request AF **only when AF filtering is active** for this job (keeps the no-filter path a pure
     genotype read — no INFO decode cost). Thread a "want AF" flag from Python into the filler
     constructor.
2. `DecodedWindow` gains `afs: Vec<f32>` (`transpose.rs:29-49`); `fill_decoded_window`
   (`transpose.rs:50-`) copies from `chunk.info_staged[af_col]` when present, else leaves it empty
   (`afs.is_empty()` ⇒ no AF filter, matching the SVAR1 `afs: None` no-op).
3. The record variants assembly folds an AF keep-term into the per-`(region, sample, ploid)`
   variant list, AND-combined with the region clip, then compacts — mirroring the written
   `keep`/`_compact_keep` (`_flat_variants.py:810-858`). **Locate PR-B1's record-path region-clip
   first** (`generate_batch_core`/`assemble_variants_mode`); if PR-B1 relies on the window already
   being region-restricted (genoray `skip_out_of_scope` + overlap), add the AF keep + compaction as
   the record path's first per-variant keep mask. The parity test is the gate.

**Parity invariant:** streaming `afs[i]` (from `info_staged`, ALT-resolved by
`resolve_scalar(source_alt_index)`) must equal written `variants.info["AF"][global_i]` (from the
`.gvi`, same genoray resolution). Multiallelic sites atomized to biallelic must map each atom to
its source-ALT's AF on **both** sides — this is the primary parity risk; the interior-exclusion /
multiallelic parity fixture is the gate.

### C. PGEN (guard-parity only)

PGEN record streams are genotype-only (no INFO; `PgenWindowFiller` has no `fields`,
`DenseChunk.info_staged` always empty), and the written PGEN index parser has no AF path. Both
paths raise the AF-missing `RuntimeError`. `_PgenBackend.has_cached_af` returns `False`.
*(pvar `INFO/AF` support on both sides is a future extension — out of scope; both raising keeps
parity.)*

### D. Guard reconciliation (all backends)

- **Output-mode guard:** AF filtering with a non-`variants` output raises `NotImplementedError`
  at `to_iter`/config time (mirrors `_haps.py:707-710`).
- **AF-missing guard:** when `min_af`/`max_af` is set, each backend reports `has_cached_af`; if
  `False`, raise the same `RuntimeError` (same message) as `_haps.py:333-340`:
  - **SVAR1:** `"AF" in sv.index` schema.
  - **VCF/BCF:** the VCF header declares an `INFO/AF` field (the same condition that made B.1
    write the column, so streaming ⟺ written agree: AF present ⇒ both filter; absent ⇒ both raise).
  - **PGEN:** always `False`.

### Python surface (`_streaming.py`)

- `StreamingDataset.with_settings` gains `min_af: float | None = None`, `max_af: float | None = None`
  (mirroring `Dataset.with_settings`); stored as `_min_af`/`_max_af`, propagated through the
  copy/subset path.
- `build_engine` (shared across `_Svar1Backend`/`_VcfBackend`/`_PgenBackend`) gains
  `min_af=None, max_af=None`. SVAR1 passes `afs`; VCF passes the "want AF" flag into its filler;
  PGEN ignores (guarded out upstream).

## Testing / parity plan

Byte-identical vs `gvl.write()` + `Dataset[r, s]` under matching `.with_settings(min_af=, max_af=)`,
`jitter=0`, reusing `_assert_variants_cell_matches`.

| Test | Backend(s) | Asserts |
|---|---|---|
| AF-cached SVAR1 parity | svar1 | byte-identical `RaggedVariants` under `min_af`/`max_af`/band; filtered total < unfiltered for ≥1 case |
| **VCF/BCF INFO-AF parity** | vcf, bcf | fixture VCF **with `INFO/AF`**; `gvl.write(info=["AF"])` oracle vs streaming; byte-identical under `min_af`/`max_af`/band; **multiallelic/atomized ALT→AF mapping** matches on both sides; filtered total < unfiltered |
| AF-missing guard | svar1 (uncached), vcf (no `INFO/AF`), pgen | same `RuntimeError` + message as written when AF unavailable |
| Output-mode guard | svar1 | `NotImplementedError` for `min_af` + non-`variants` output |
| Rust unit (svar1) | svar1 kernel | region-kept-but-AF-dropped and AF-kept-but-region-dropped compose as **AND** |
| Rust unit (record afs) | vcf filler | `fill_decoded_window` copies `info_staged → afs`; AF keep + region clip compose as AND in the record assembly |

- **No fixture with `INFO/AF`?** Add one (a small multiallelic VCF with an `AF` INFO line), or
  synthesize AF into an existing streaming VCF fixture.
- **libdeflate:** VCF/BCF AF decode rides genoray's htslib reader, built with the `libdeflate`
  feature (`genoray/Cargo.toml:33`, `rust-htslib { features = ["libdeflate"] }`; `libdeflate-sys`
  present in gvl `Cargo.lock`). Add a build-time assertion/guard so this cannot silently regress
  (e.g. a `cargo test` that fails if the non-libdeflate bgzf path is linked, or at minimum a
  documented check in the plan's verification task).
- **Rebuild:** `pixi run -e dev maturin develop --release` before pytest after any `src/` change.
  Run the full tree before pushing (`CLAUDE.md`).

## Scope / non-goals

- **No PGEN AF** (guard only). pvar `INFO/AF` is a future extension.
- No `var_fields` (dosage / custom FORMAT/INFO) — **PR-B3**.
- No REF bytes / `variant-windows` — **PR-B4**.
- No AF filtering for haplotype / annotated / `variant-windows` output (raises, same as written).
- `.svar2` remains haplotypes-only.
- No new on-disk format for streaming. The written change is additive (an optional `AF` column in
  the VCF `.gvi` when the header has it). The only public-surface change is two new keyword args on
  `StreamingDataset.with_settings` (matching `Dataset.with_settings`) — **no new exported symbol**,
  so no `api.md`/`__all__` change.

## Decisions

- **Fold #319 into B2; fix both sides (chosen 2026-07-21).** VCF/BCF AF read from `INFO/AF` on both
  the written (`.gvi info=["AF"]`) and streaming (`FieldSpec`) paths, so parity holds by
  construction. Supersedes the earlier SVAR-only scope. PGEN stays guard-only.
- **No genoray change.** Both the `FieldSpec` staging and `_write_gvi_index(info=…)` paths exist at
  the pinned rev; verified. No rev bump.
- **Written `.gvi info=["AF"]` gated on header presence.** AF-less VCFs are untouched (both paths
  raise). AF-bearing VCFs "just work" on both paths.
- **Stay on rust-htslib + libdeflate** for the VCF/BCF record read (faster than noodles; user
  decision). Add a guard so libdeflate cannot silently regress.
- **Inline-fold over a separate compact pass for SVAR1** (byte-identical, no extra allocation); the
  record path uses a keep-mask + compaction mirroring the written `_compact_keep`.
- **SVAR1 AF as an optional global backend table** (`Option<Array1<f32>>`); VCF AF as a per-window
  `DecodedWindow.afs` channel — each matches its backend's id model.
- **Guard + output-mode errors mirror the written path exactly** (same exception types/messages).

## Docs to update when PR-B2 lands

Per `CLAUDE.md` (skill + docs-audit + roadmap gates):

- `skills/genvarloader/SKILL.md` — `StreamingDataset.with_settings` accepts `min_af`/`max_af`
  (variants output); SVAR1 needs `cache_afs()`, VCF/BCF need an `INFO/AF` field; PGEN raises.
- `docs/source/dataset.md` / `docs/source/faq.md` — streaming AF-filtering note + the per-backend
  AF-source table (SVAR `cache_afs`, VCF `INFO/AF`, PGEN unsupported) and that `gvl.write` now
  caches `AF` from VCF `INFO/AF`.
- `docs/source/write.md` — `gvl.write` writes an `AF` column for VCFs with `INFO/AF`.
- `docs/roadmaps/streaming-dataset.md` — tick PR-B2, update the Wave B status line, link this
  design + #317 + #319 (now closed here). Update the StreamingDataset project board (move #319 to
  the B2 PR; note the fold-in).
- No `api.md` / `__all__` change.
