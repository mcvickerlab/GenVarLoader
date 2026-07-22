# Streaming variants output — `min_af` / `max_af` (Wave B PR-B2)

**Issue:** [#317](https://github.com/mcvickerlab/GenVarLoader/issues/317) (split out of
[#304](https://github.com/mcvickerlab/GenVarLoader/issues/304), Wave B of
[#277](https://github.com/mcvickerlab/GenVarLoader/issues/277)).
**Follows:** PR-B0 + PR-B1 (streaming `with_seqs("variants")`), landed in
[#316](https://github.com/mcvickerlab/GenVarLoader/pull/316).
**Parent design:** `docs/superpowers/specs/2026-07-20-streaming-variants-output-wave-b-design.md` (§PR-B2).
**Target branch:** `streaming` (streaming-coordination rules — see `CLAUDE.md`).

## Goal

Add `min_af` / `max_af` allele-frequency filtering to **streaming**
`StreamingDataset.with_seqs("variants")`, byte-identical to the written
`Dataset` variants path at `jitter=0`, on the **SVAR1, VCF, and PGEN** backends.
`.svar2` stays haplotypes-only (unchanged). No other output mode gains AF
filtering — this mirrors the written path, where AF filtering is implemented
**only** for `variants` output.

## Background — the written oracle

On the written `Dataset`, `min_af`/`max_af` are `with_settings` parameters
(`_impl.py:97-98`, threaded into `Haps`). The filter lives in the variants
assembly path `get_variants_flat` (`_flat_variants.py:811-817`):

```python
if haps.min_af is not None or haps.max_af is not None:
    geno_afs = np.asarray(haps.variants.info["AF"])[v_idxs]
    keep = np.full(len(v_idxs), True, np.bool_)
    if haps.min_af is not None:
        keep &= geno_afs >= haps.min_af
    if haps.max_af is not None:
        keep &= geno_afs <= haps.max_af
```

`keep` is then combined with the #202 region-overlap clip and applied via
`_compact_keep` before allele assembly. Semantics that PR-B2 must reproduce
exactly:

- **Inclusive bounds** on both sides (`>=` / `<=`).
- **AF gathered per window-local variant** (`AF[v_idxs]`), so the same variant
  can be kept for one region/sample and dropped for another only if the AF
  itself differed — it does not; AF is per-variant, so the keep is a pure
  per-variant predicate independent of row. This is what lets PR-B2 fold it
  into the per-variant inline keep (below).
- **AF-missing guard** (`_haps.py:335-338`): if `min_af`/`max_af` is set and the
  dataset has no cached `AF` INFO column, raise `RuntimeError` with the message
  "Either this dataset is not backed by an SVAR file, or the SVAR file has not
  had AFs cached yet."
- **Haplotype/annotated output** raises `NotImplementedError` when AF filtering
  is requested (`_haps.py:707-709`) — AF filtering is a variants-only feature.

## Streaming baseline (post-PR-B1)

- `StreamingDataset.with_settings` (`_streaming.py:1009`) exposes **only**
  `jitter` / `rng` / `deterministic` — **no** `min_af` / `max_af` yet.
- Rust record backend `generate_variants` (`record_stream/engine.rs:242-305`)
  walks the per-hap CSR and applies the **region-overlap clip inline** before
  pushing each `vidx`:

  ```rust
  for &vidx in &slot.geno_v_idxs[csr_lo..csr_hi] {
      let v_start = slot.v_starts[vidx as usize] as i64;
      let v_ilen  = slot.ilens[vidx as usize] as i64;
      let v_end   = v_start - v_ilen.min(0) + 1;
      let keep = v_start < r_e as i64 && v_end > r_s as i64;
      if keep { v_idxs.push(vidx); }
  }
  row_offsets.push(v_idxs.len() as i64);
  ```

  Kept `v_idxs` / `row_offsets` then flow through the shared
  `assemble_variants_window` (`src/variants/mod.rs:105`).
- `DecodedWindow` (`record_stream/transpose.rs:30`) holds the per-window static
  columns (`v_starts`, `ilens`, `alt_alleles`, `alt_offsets`, `global_v_idxs`),
  filled from the genoray `DenseChunk` in `fill_decoded_window`.
- SVAR1 `Svar1Backend::generate_variants` carries dataset-**global** variant
  ids and reuses the same `assemble_variants_window` against a global variant
  table on the backend.

## Design — fold the AF keep into the inline predicate

**Chosen approach (user-approved):** extend the existing per-variant inline keep
in `generate_variants` with an AF term, rather than adding a separate
build-`keep`-then-`compact` pass. Because the AF keep is a pure per-variant
predicate (AF does not vary by row/sample), folding it into the same loop that
already does the region clip yields a kept set and order **byte-identical** to
the written path's `region_keep & af_keep` + compact, with no extra allocation
or compaction pass.

Five layers, one per backend seam.

### 1. Python surface (`python/genvarloader/_dataset/_streaming.py`)

- Add `min_af: float | None = None` and `max_af: float | None = None` keyword
  params to `StreamingDataset.with_settings` (mirroring `Dataset.with_settings`,
  same names/types). Store as `_min_af` / `_max_af` on the copied instance
  (same `object.__setattr__` pattern as `_jitter`).
- **AF-missing guard, same surface as written:** when `min_af`/`max_af` is set,
  validate that the source exposes a cached `AF` INFO column and raise the same
  `RuntimeError` (same message) if not. Placed at the earliest point the source
  schema is known — at `with_settings` time if the schema is already loaded,
  else at `to_iter`/engine-build time (whichever the existing streaming code
  makes natural; guard must fire before any batch is produced, matching the
  written path's construction-time raise).
- **Output-mode consistency:** AF filtering combined with a non-`variants`
  output mode raises `NotImplementedError`, mirroring `_haps.py:707-709`
  (haplotype/annotated). Keep the error text aligned with the written path.
- Thread `_min_af` / `_max_af` into engine construction (below).

### 2. AF channel into the window — record backend (VCF/PGEN)

- Add `afs: Vec<f32>` to `DecodedWindow` (`record_stream/transpose.rs`),
  **parallel to `v_starts`** (one AF per window-local variant column).
- Fill it in `fill_decoded_window` from an `AF` column on the `DenseChunk`,
  cleared+extended every call like the other channels (so a recycled slot never
  leaks a stale value — same discipline as `global_v_idxs`).
- The filler (`VcfWindowFiller` / `PgenWindowFiller`) requests the `AF` INFO
  column (`info_staged`, `StagedColumn::Float`) from genoray **only when AF
  filtering is active** (an `af_filter: bool` construction flag). When inactive,
  `afs` is left empty and never read — zero cost on the common path, matching
  the written path's "read AF only when filtering."

### 3. AF table — SVAR1 backend

- Load the global per-variant `AF` array from the `.svar` store at
  `Svar1Backend` construction (indexed by the dataset-global `vidx` SVAR1
  already carries), stored as `afs: Vec<f32>` on the backend. No `DecodedWindow`
  change for SVAR1.
- **Verification point (implementation-time):** confirm the streaming SVAR1
  backend can read cached `AF` from the `.svar` store the same way the written
  path reads `variants.info["AF"]`. The parent spec asserts "SVAR1: read AF from
  its store." If genoray does not expose `AF` at that seam, that becomes a small
  cross-repo prerequisite (bump genoray `rev`), handled like PR-B1's clip
  predicate — flagged here, resolved in the plan's first task.

### 4. Apply the filter — Rust `generate_variants` (both backends)

- Thread `min_af: Option<f32>` / `max_af: Option<f32>` onto the engine at
  construction (record + SVAR1), forwarded from the Python `_min_af`/`_max_af`.
- In the inline per-variant loop, extend the keep predicate:

  ```rust
  let af = afs[vidx as usize];               // DecodedWindow.afs (record) / backend.afs (svar1)
  let region_keep = v_start < r_e as i64 && v_end > r_s as i64;
  let af_keep = min_af.map_or(true, |m| af >= m)
             && max_af.map_or(true, |m| af <= m);
  if region_keep && af_keep { v_idxs.push(vidx); }
  ```

- Everything downstream (`row_offsets`, `assemble_variants_window`, the FFI
  marshaling, the Python `RaggedVariants` packing) is **unchanged** — the filter
  only narrows the kept `v_idxs`.

### 5. Parity & unit tests

- **Python parity** (`tests/dataset/test_streaming_variants_parity.py`): add
  `min_af` / `max_af` cases (and a combined `min_af`+`max_af` band) on
  **svar1 + vcf + pgen**, asserting **byte-identical** `RaggedVariants` vs the
  written oracle at `jitter=0`. Keep the existing non-vacuous-pass guard
  (`total_variants > 0` before filtering) so a filter that drops everything
  can't pass vacuously — and additionally assert the filtered total is **less
  than** the unfiltered total for at least one case, so the filter is proven to
  actually remove variants.
- **AF-missing guard test:** on a source with no cached `AF`, setting
  `min_af`/`max_af` raises the same `RuntimeError` (same message) as the written
  path.
- **Rust unit tests** (`record_stream/engine.rs`): a variant kept by the region
  clip but dropped by AF, and a variant inside the AF band but outside the
  region — proving the two keep terms compose correctly (AND), not that either
  alone decides.

## Decisions

- **Inline-fold over separate `afs`-channel-then-compact.** Byte-identical
  because AF keep is a pure per-variant predicate; avoids an extra allocation
  and compaction pass. (Deviates from the parent spec's literal "add an `afs`
  channel and compact" wording; the `afs` channel still exists on
  `DecodedWindow` for the record backend — it's the *application* that folds
  inline rather than compacting.)
- **AF staged only when filtering** (record backend `af_filter` flag) — no cost
  on the default `with_seqs("variants")` path.
- **SVAR1 AF as a global backend table**, not a per-window channel — matches
  SVAR1's existing global-id model; no `DecodedWindow` touch.
- **Guard + output-mode errors mirror the written path exactly** (same
  exception types and messages) so streaming and written are interchangeable to
  a caller.

## Scope / non-goals

- No `var_fields` (dosage / custom FORMAT/INFO) — **PR-B3**.
- No REF bytes / `variant-windows` — **PR-B4**.
- No AF filtering for haplotype / annotated / `variant-windows` output (raises,
  same as written).
- `.svar2` remains haplotypes-only; `with_seqs("variants")` on `.svar2` still
  raises `NotImplementedError` at iterate time (unchanged from PR-B1).
- No new on-disk format, no new public top-level symbol (the only public-surface
  change is two new keyword args on `StreamingDataset.with_settings`, matching
  `Dataset.with_settings`).

## Docs to update when PR-B2 lands

Per `CLAUDE.md` (skill + docs-audit gates):

- `skills/genvarloader/SKILL.md` — `StreamingDataset.with_settings` now accepts
  `min_af`/`max_af` (variants output only).
- `docs/source/dataset.md` / `docs/source/faq.md` — streaming AF-filtering note.
- `docs/roadmaps/streaming-dataset.md` — tick PR-B2, update the Wave B status
  line, link this design + #317.
- No `api.md` / `__all__` change (no new exported symbol).

## Testing plan (summary)

| Test | Backends | Asserts |
|---|---|---|
| `test_streaming_variants_parity.py` (extended) | svar1, vcf, pgen | byte-identical `RaggedVariants` vs written oracle under `min_af`, `max_af`, and a combined band; filtered total < unfiltered for ≥1 case |
| AF-missing guard test | a source without cached `AF` | same `RuntimeError` + message as written |
| Rust unit (record engine) | vcf/pgen kernel | region-kept-but-AF-dropped and AF-kept-but-region-dropped compose as AND |

**Rebuild reminder:** Rust changes require `pixi run -e dev maturin develop
--release` before the Python parity tests import the extension (per `CLAUDE.md`)
— else pytest silently tests the stale binary.
