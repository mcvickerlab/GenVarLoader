# Streaming variants `min_af`/`max_af` (Wave B PR-B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Use Sonnet (or weaker) for implementation** per the maintainer's SDD convention; escalate only for a genuinely stuck second pass.

**Goal:** Add `min_af`/`max_af` allele-frequency filtering to streaming `StreamingDataset.with_seqs("variants")`, byte-identical to the written oracle at `jitter=0`, for **SVAR1** (AF from `.svar` `cache_afs()`) and **VCF/BCF** (AF from the VCF `INFO/AF` field, read on *both* the streaming and written paths). PGEN stays **guard-parity** (both raise).

**Scope note — #319 folded in:** an earlier revision scoped this SVAR-only and deferred VCF to #319. The decision (2026-07-21) is to **fix both sides together here**: the written VCF path caches `AF` into its `.gvi` (`info=["AF"]`) and the streaming VCF path reads the same `INFO/AF` live via a genoray `FieldSpec`. Both use genoray's ALT-resolving `resolve_scalar`, so parity holds by construction. **This PR closes #319.**

**Architecture:**
- **SVAR1** — an optional global `afs` table on `Svar1Backend`, folded into `generate_variants`' existing per-variant inline keep.
- **VCF/BCF** — (a) `gvl.write` builds the `.gvi` with `info=["AF"]` when the VCF header has `INFO/AF`; (b) the streaming `VcfWindowFiller` requests an `AF` INFO `FieldSpec` → genoray stages `DenseChunk.info_staged` → a new `DecodedWindow.afs` channel → an AF keep-mask folded into the record-path variants assembly (mirroring the written `_compact_keep`).
- **PGEN** — guard only.

**Tech Stack:** Rust (PyO3/ndarray) for the SVAR1 kernel, the `DecodedWindow.afs` channel, and the record-path keep; Python for the `StreamingDataset` surface, the `gvl.write` INFO wiring, and guards; genoray `FieldSpec`/`_write_gvi_index(info=…)` (**no genoray change — verified against pinned rev `73d25cb`**); pytest + cargo test.

**Design spec:** `docs/superpowers/specs/2026-07-21-streaming-variants-min-max-af-b2-design.md`

## Global Constraints

- **Target branch:** `streaming` (not `main`) — streaming-coordination rules, `CLAUDE.md`.
- **Byte-identical parity** with the written `Dataset` variants path at `jitter=0` is the correctness gate (SVAR1 **and** VCF/BCF).
- **Guard/error surfaces must match the written path exactly** — AF-missing → `RuntimeError("Either this dataset is not backed by an SVAR file, or the SVAR file has not had AFs cached yet." …)` (`_haps.py:333-340`); AF + non-variants output → `NotImplementedError` (`_haps.py:707-710`).
- **AF is a pure per-variant predicate** — fold into the per-variant keep (SVAR1 inline; record path via keep-mask + compaction).
- **No genoray change / no rev bump.** `FieldSpec` INFO-Float staging and `_write_gvi_index(info=…)` both exist at the pinned rev.
- **libdeflate stays on** for the VCF/BCF htslib read (`genoray/Cargo.toml:33`); Task 12 adds a regression guard.
- **No new public exported symbol** — only two new `with_settings` kwargs. No `api.md`/`__all__` change.
- **Rebuild Rust before Python tests:** after any `src/` edit run `pixi run -e dev maturin develop --release`, else pytest imports the stale extension (`CLAUDE.md`).
- **`cargo test` needs libpython on the link path** (memory: cargo-test-libpython-ldpath) — prefer `pixi run -e dev cargo test` (activation sets `LD_LIBRARY_PATH`).
- **Commit style:** conventional commits; end messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Ensure prek/pre-commit hooks are installed before committing (`.pre-commit-config.yaml` present). Docs-only commits may `--no-verify`; code commits should pass hooks.

---

## Phase 1 — Python surface

### Task 1: `StreamingDataset.with_settings(min_af, max_af)` params + storage

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (class fields ~142-150; `with_settings` ~1009-1053; the copy-propagation list ~282-288)
- Test: `tests/dataset/test_streaming_settings.py` (create, or extend the nearest streaming-settings module — check `tests/dataset/` first)

- [ ] **Step 1: Failing test**

```python
# tests/dataset/test_streaming_settings.py
import numpy as np
import pytest
import genvarloader as gvl


def test_with_settings_min_max_af_stored(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert sds._min_af is None and sds._max_af is None
    out = sds.with_settings(min_af=0.1, max_af=0.9)
    assert out._min_af == 0.1 and out._max_af == 0.9
    assert sds._min_af is None and sds._max_af is None  # immutable
    assert out._jitter == sds._jitter                    # copy preserves others
```

- [ ] **Step 2: Verify it fails** — `pixi run -e dev pytest tests/dataset/test_streaming_settings.py::test_with_settings_min_max_af_stored -v` → `AttributeError: _min_af`.

- [ ] **Step 3: Class fields** — after `_deterministic: bool = True` (~150):

```python
    _min_af: "float | None" = None
    _max_af: "float | None" = None
```

- [ ] **Step 4: Copy propagation** — add `"_min_af"`, `"_max_af"` to the settings-copy name tuple (~282-288); keep existing entries exactly.

- [ ] **Step 5: Extend `with_settings`** (~1009) — add the two `*`-kwargs mirroring `jitter`:

```python
    def with_settings(self, *, jitter=None, rng=None, deterministic=None,
                      min_af=None, max_af=None) -> "StreamingDataset":
        ...
        out = copy.copy(self)
        ...
        if min_af is not None:
            object.__setattr__(out, "_min_af", float(min_af))
        if max_af is not None:
            object.__setattr__(out, "_max_af", float(max_af))
        return out
```

Add `min_af`/`max_af` to the docstring `Parameters` (one line each: inclusive lower/upper AF bound for `with_seqs("variants")`; requires an available `AF` — SVAR `cache_afs()`, or a VCF `INFO/AF` field — else raises at iterate time; matches `Dataset.with_settings`).

- [ ] **Step 6: Verify pass.** — [ ] **Step 7: Commit** (`feat(streaming): with_settings(min_af, max_af) params + storage (PR-B2, #317)`).

---

### Task 2: Output-mode guard — `NotImplementedError` for AF + non-variants output

**Files:** Modify `_streaming.py` (`_iter_batches` fail-fast block ~464-500); Test `tests/dataset/test_streaming_settings.py`.

- [ ] **Step 1: Failing test**

```python
def test_af_filter_rejects_non_variants_output(streaming_case):
    regions, reference, variants, _ = streaming_case("svar1")
    sds = (gvl.StreamingDataset(regions, reference=reference, variants=variants)
           .with_seqs("haplotypes").with_settings(min_af=0.1))
    with pytest.raises(NotImplementedError, match="AF"):
        next(iter(sds.to_iter(batch_size=4)))
```

- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Guard** — after `_variants = self._seq_kind is RaggedVariants` (~464), before the SVAR2 fail-fast:

```python
            _af_filter = self._min_af is not None or self._max_af is not None
            if _af_filter and not _variants:
                raise NotImplementedError(
                    'min_af/max_af filtering is only supported for with_seqs("variants") '
                    "output (matching the written Dataset, which raises for "
                    "haplotype/annotated output)."
                )
```

- [ ] **Step 4: Verify pass.** — [ ] **Step 5: Commit** (`feat(streaming): guard min_af/max_af to variants output only (PR-B2, #317)`).

---

## Phase 2 — SVAR1 full support

### Task 3: Rust — `Svar1Backend` AF table + keep-fold + `#[new]` params

**Files:** Modify `src/ffi/stream_engine.rs` — `Svar1Backend` struct (`:109-141`); `generate_variants` (`:257-315`); `Svar1StreamEngine::build` (`:331-368`); `#[new]` + `#[pyo3(signature=…)]` (`:394-432`). Test: same file (`#[cfg(test)]`).

- [ ] **Step 1: Failing Rust unit test** — read the file's existing `#[test]` helpers first (how they stub the store/build `Svar1Backend`); reuse them. Assert region + AF keeps compose as AND:

```rust
    #[test]
    fn svar1_generate_variants_af_and_region_compose_as_and() {
        // globals POS 10/20/30 (0-based 9/19/29), SNPs; AF=[.05,.5,.95].
        // region [15,25) keeps only #2 by position; min_af=.1,max_af=.9 keeps only #2 by AF.
        let backend = make_test_backend(&[9,19,29], &[0,0,0], /*alts*/.., Some(&[0.05f32,0.5,0.95]), Some(0.1), Some(0.9));
        let batch = backend.generate_variants(0, &make_window_all_three(), 0, 1).unwrap();
        assert_eq!(batch.start.as_slice().unwrap(), &[19]);
        assert_eq!(batch.row_offsets.as_slice().unwrap(), &[0, 1]);
    }
```

- [ ] **Step 2: Verify fail** (fields don't exist) — `pixi run -e dev cargo test svar1_generate_variants_af_and_region_compose_as_and 2>&1 | tail -20`.
- [ ] **Step 3: Struct fields** (`:109-141`): `afs: Option<Array1<f32>>`, `min_af: Option<f32>`, `max_af: Option<f32>` (doc: `Some` only when the `.svar` had AF cached and filtering requested; `None` ⇒ no-op).
- [ ] **Step 4: Keep-fold** (`:257-315`): `let af_keep = match &self.afs { Some(a) => { let af=a[gvi as usize]; self.min_af.map_or(true,|m|af>=m) && self.max_af.map_or(true,|m|af<=m) } None => true }; if region_keep && af_keep { kept.push(gvi); }`.
- [ ] **Step 5: `build`** (`:331-368`) — add the three params, set the fields.
- [ ] **Step 6: `#[new]`** (`:394-432`) — trailing pyO3 params `afs: Option<PyReadonlyArray1<f32>>`, `min_af/max_af: Option<f32>` (all `=None` in the signature); convert `afs.map(|a| a.as_array().to_owned())`; pass to `build`.
- [ ] **Step 7: Fix other `build`/`new_rs` call sites** — `grep -n "\.build(\|Svar1StreamEngine::build(" src/ffi/stream_engine.rs`; add `, None, None, None`.
- [ ] **Step 8-9: Unit test + full `cargo test` pass** (`afs=None` no-op keeps existing parity green).
- [ ] **Step 10: Commit** (`feat(streaming): SVAR1 generate_variants folds min_af/max_af into keep (PR-B2, #317)`).

---

### Task 4: SVAR1 `_afs` loading + `has_cached_af` + shared `build_engine` threading

**Files:** Modify `_streaming.py` — `_Svar1Backend.__init__` (`:1148-1293`, after `idx = sv.index.sort("index")` ~`:1200`); `_Svar1Backend.build_engine` (`:1295`); `_iter_batches` `build_engine(...)` call (`:567`). Test: `tests/dataset/test_streaming_settings.py`.

> **Note:** `build_engine` is shared across all three backends. Establish the extended signature `build_engine(self, jobs, batch_size, output_length, annotated=False, variants=False, min_af=None, max_af=None)` here; VCF/PGEN pick it up in Task 8.

- [ ] **Step 1: Failing test (AF actually filters a cached SVAR1)**

```python
import shutil
from genoray import SparseVar

def _af_cached_svar(streaming_case, tmp_path):
    regions, reference, variants, _ = streaming_case("svar1")
    dst = tmp_path / "af.svar"; shutil.copytree(variants, dst)
    sv = SparseVar(str(dst)); sv.cache_afs()
    af = sv.index.sort("index")["AF"].to_numpy()
    return regions, reference, str(dst), af

def test_svar1_min_af_actually_filters(streaming_case, tmp_path):
    regions, reference, svar, af = _af_cached_svar(streaming_case, tmp_path)
    thr = float(np.median(af))
    base = gvl.StreamingDataset(regions, reference=reference, variants=svar).with_seqs("variants")
    filt = base.with_settings(min_af=thr)
    def total(ds):
        n = 0
        for data, r_idx, _s in ds.to_iter(batch_size=4):
            for k in range(len(r_idx)):
                for h in range(ds.ploidy):
                    n += np.atleast_1d(np.asarray(data[k].alt[h])).shape[0]
        return n
    n_base, n_filt = total(base), total(filt)
    assert n_base > 0 and n_filt < n_base
```

- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Load `_afs`** — after `idx = sv.index.sort("index")`: if `"AF" in idx.collect_schema().names()`, `self._afs = idx["AF"].to_numpy().astype(np.float32, copy=False)` else `None` (match the surrounding frozen-dataclass `object.__setattr__` vs plain-assign pattern — read nearby lines).
- [ ] **Step 4: `has_cached_af` property** → `self._afs is not None`.
- [ ] **Step 5: `build_engine`** — add `min_af=None, max_af=None`; pass `afs=(self._afs if (min_af is not None or max_af is not None) else None), min_af=min_af, max_af=max_af` to `Svar1StreamEngine(...)`.
- [ ] **Step 6: Call site** (`:567`) — `backend.build_engine(engine_jobs, batch_size, _out_len, _annotated, _variants, min_af=self._min_af, max_af=self._max_af)`.
- [ ] **Step 7: Rebuild** — `pixi run -e dev maturin develop --release`.
- [ ] **Step 8: Verify pass.** — [ ] **Step 9: Commit** (`feat(streaming): SVAR1 loads cached AF + threads min_af/max_af to engine (PR-B2, #317)`).

---

## Phase 3 — VCF/BCF full support (folds in #319)

### Task 5: Written path — `gvl.write` caches `AF` from VCF `INFO/AF`

**Files:** Modify `python/genvarloader/_dataset/_write.py` (VCF index build ~244-249). Test: `tests/dataset/` (a write-level test, or fold into the Task 11 parity fixture setup).

**Goal:** when writing a VCF-sourced dataset whose header declares an `INFO/AF` field, build the `.gvi` with `info=["AF"]` so `Dataset.open(...).variants.info["AF"]` exists and the written AF filter runs. AF-less VCFs are unchanged.

- [ ] **Step 1: Confirm the genoray header API** — determine how to detect an `INFO/AF` field on a genoray `VCF` object **before** index build (candidates: a header/`available_info_fields`-style call on `VCF`; inspect `genoray/python/genoray/_vcf.py`). Record the exact call. If detection is impractical pre-index, fall back to: always pass `info=["AF"]` **inside a try/except** that retries with no `info=` if genoray errors on an absent field (verify genoray's actual behavior first — do not assume).

- [ ] **Step 2: Failing test** — write a fixture VCF **with** an `INFO/AF` line (small, ≥1 multiallelic site), `gvl.write` it, open, and assert `"AF"` is in `_Variants.available_info_fields(out / "variants.arrow")` (or `Dataset.open(...).variants.info` keys). Also assert an AF-less VCF still writes with no `AF` column.

- [ ] **Step 3: Implement** — at `_write.py:248`, gate the `info=["AF"]`:

```python
            if isinstance(variants, VCF):
                if variants._index is None:
                    if not variants._valid_index():
                        info = ["AF"] if _vcf_has_info_af(variants) else None
                        variants._write_gvi_index(info=info)
                    variants._load_index()
```

Implement `_vcf_has_info_af(variants)` using the Step-1 API. (If a valid index already exists without AF, AF filtering will raise the guard — acceptable; note in the docstring that AF caching happens at index-build time.)

- [ ] **Step 4: Verify pass.** — [ ] **Step 5: Commit** (`feat(write): cache AF from VCF INFO/AF into the .gvi index (PR-B2, #317, #319)`).

---

### Task 6: Streaming VCF — request the `AF` `FieldSpec` + `DecodedWindow.afs` channel

**Files:** Modify `src/record_stream/vcf.rs` (`fields` `:103`/`:127`; constructor; `fill` `:144-`); `src/record_stream/transpose.rs` (`DecodedWindow` `:29-49`; `fill_decoded_window` `:50-`). Test: Rust `#[cfg(test)]` in `transpose.rs`.

- [ ] **Step 1: Failing Rust unit test** — in `transpose.rs` tests, construct a `DenseChunk` literal with a non-empty `info_staged` (one `StagedColumn::Float`) and assert `fill_decoded_window` copies it into `slot.afs`; and that an empty `info_staged` leaves `slot.afs` empty. (The test module already builds `DenseChunk` literals — extend one.)

- [ ] **Step 2: Verify fail** (no `afs` field).
- [ ] **Step 3: `DecodedWindow.afs`** — add `pub afs: Vec<f32>` to the struct (`:29-49`).
- [ ] **Step 4: Copy in `fill_decoded_window`** — `slot.afs.clear(); if let Some(col) = chunk.info_staged.get(af_col_index) { if let StagedColumn::Float(v) = col { slot.afs.extend_from_slice(v); } }` (define `af_col_index` = the position of the AF spec in the requested `fields`; if fields carries only AF it is `0`). Leave `afs` empty when `info_staged` is empty.
- [ ] **Step 5: Request the AF `FieldSpec` in `VcfWindowFiller`** — replace `fields: Vec::new()` (`vcf.rs:127`) with a real spec **gated on a "want AF" flag** threaded into the constructor (so no-filter jobs pay no INFO-decode cost):

```rust
    // vcf.rs — construct when want_af:
    let fields = if want_af {
        vec![FieldSpec { name: "AF".into(), category: FieldCategory::Info,
                         htype: HtslibType::Float, dtype: StorageDtype::F32, default: None }]
    } else { Vec::new() };
```

(`use genoray_core::field::{FieldSpec, FieldCategory, HtslibType, StorageDtype};` — confirm the re-export path; `field.rs:248-255` shows the literal.) It already flows into both `VcfRecordSource::with_sample_indices` (`:166`) and `ChunkAssembler::new` (`:170-179`).

- [ ] **Step 6: Rebuild + Rust unit test pass** (`maturin develop --release`, then `cargo test`).
- [ ] **Step 7: Commit** (`feat(streaming): VCF filler requests INFO/AF FieldSpec into DecodedWindow.afs (PR-B2, #319)`).

---

### Task 7: Streaming VCF — fold the AF keep into the record-path variants assembly

**Files:** Modify the record variants assembly — `src/ffi/mod.rs` (`generate_batch_core` ~1073-1200) and/or `src/variants/windows.rs` (`assemble_variants_mode` ~162-219). Test: Rust `#[cfg(test)]` + the Task 11 parity test is the real gate.

> **Locate first:** PR-B1 landed VCF `with_seqs("variants")`. Confirm **where the record path applies the region clip** for variants output (the SVAR1 clip lives in `stream_engine.rs`; the record path's `assemble_variants_mode` had *no* keep loop as of PR-B1 — it may rely on genoray `skip_out_of_scope`+overlap already restricting the window). Fold the AF keep at that site; if there is no per-variant keep yet, add a keep-mask + compaction pass mirroring the written `keep`/`_compact_keep` (`_flat_variants.py:810-858`).

- [ ] **Step 1: Failing Rust unit** — build a `DecodedWindow` with `afs` set + a small per-hap CSR; assert the record assembly drops variants outside `[min_af, max_af]` and AND-composes with the region clip. (If the keep is easier to assert end-to-end, lean on Task 11 and keep this unit minimal.)
- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** — thread `min_af`/`max_af` into the record backend / `generate_batch_core`; build `af_keep[i] = afs.is_empty() || (min_af.map_or(true,…) && max_af.map_or(true,…))` per window-local variant, AND with the region clip, compact `v_idxs`/offsets (and any parallel field arrays) before allele assembly. When `afs.is_empty()`, the keep is all-true (no-op) — identical to the pre-B2 path.
- [ ] **Step 4: Rebuild + unit pass.**
- [ ] **Step 5: Commit** (`feat(streaming): fold min_af/max_af keep into the VCF record variants assembly (PR-B2, #319)`).

---

### Task 8: VCF/PGEN `has_cached_af` + `build_engine` want-AF wiring + AF-missing guard

**Files:** Modify `_streaming.py` — `_VcfBackend`/`_PgenBackend` `has_cached_af` + `build_engine` (`:1858`/`:1999`); the AF-missing guard in `_iter_batches`. Test: `tests/dataset/test_streaming_settings.py`.

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.parametrize("backend", ["svar1", "pgen"])
def test_af_missing_guard_raises(streaming_case, backend):
    # svar1 here = the UN-cached fixture; pgen has no AF path
    regions, reference, variants, _ = streaming_case(backend)
    sds = (gvl.StreamingDataset(regions, reference=reference, variants=variants)
           .with_seqs("variants").with_settings(min_af=0.1))
    with pytest.raises(RuntimeError, match="AFs cached"):
        next(iter(sds.to_iter(batch_size=4)))

def test_af_missing_guard_raises_vcf_without_info_af(streaming_case):
    # a VCF fixture WITHOUT an INFO/AF header field -> both paths raise
    regions, reference, vcf_no_af, _ = streaming_case("vcf")  # ensure this fixture lacks INFO/AF
    sds = (gvl.StreamingDataset(regions, reference=reference, variants=vcf_no_af)
           .with_seqs("variants").with_settings(min_af=0.1))
    with pytest.raises(RuntimeError, match="AFs cached"):
        next(iter(sds.to_iter(batch_size=4)))
```

- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: `has_cached_af`** — `_PgenBackend` → `False`. `_VcfBackend` → `True` iff the VCF header declares `INFO/AF` (reuse the Task-5 `_vcf_has_info_af`-style detection so streaming ⟺ written agree). Compute once at backend construction.
- [ ] **Step 4: `build_engine` want-AF** — extend `_VcfBackend.build_engine`/`_PgenBackend.build_engine` to the shared signature. `_VcfBackend`: when `min_af`/`max_af` set (and `has_cached_af`), pass a `want_af=True` flag + the bounds into the `VcfWindowFiller`/engine constructor (Task 6). `_PgenBackend`: accept-and-ignore (guarded out upstream).
- [ ] **Step 5: AF-missing guard** — in `_iter_batches`, where `_af_filter` and `backend` are both in scope (before `build_engine`):

```python
                if _af_filter and not backend.has_cached_af:
                    raise RuntimeError(
                        "Either this dataset is not backed by an SVAR file, or the "
                        "SVAR file has not had AFs cached yet."
                    )
```

- [ ] **Step 6: Rebuild + verify pass.** — [ ] **Step 7: Commit** (`feat(streaming): VCF/PGEN has_cached_af + AF-missing guard parity (PR-B2, #317, #319)`).

---

## Phase 4 — Parity

### Task 9: Byte-identical SVAR1 AF parity vs the written oracle

**Files:** Modify `tests/dataset/test_streaming_variants_parity.py` (reuse `_assert_variants_cell_matches`).

- [ ] **Step 1: Parity test** — cache AFs into a `.svar` copy, `gvl.write` a fresh oracle from it, apply `.with_settings(min_af=, max_af=)` on both; assert byte-identical `RaggedVariants` cell-by-cell over the full `(region, sample)` grid; assert `0 < total < unfiltered_total` (partial filter). Parametrize `[(0.3,None),(None,0.7),(0.2,0.8)]`.

```python
@pytest.mark.parametrize("min_af,max_af", [(0.3, None), (None, 0.7), (0.2, 0.8)])
def test_streaming_svar1_af_matches_written(streaming_case, tmp_path, min_af, max_af):
    regions, reference, variants, _ = streaming_case("svar1")
    svar = tmp_path / "af.svar"; shutil.copytree(variants, svar)
    SparseVar(str(svar)).cache_afs()
    out = tmp_path / "ds"; gvl.write(out, regions, variants=str(svar), overwrite=True)
    written = gvl.Dataset.open(out, reference=reference)
    ds = written.with_seqs("variants").with_settings(min_af=min_af, max_af=max_af)
    sds = (gvl.StreamingDataset(regions, reference=reference, variants=str(svar))
           .with_seqs("variants").with_settings(min_af=min_af, max_af=max_af))
    seen, total = set(), 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            total += _assert_variants_cell_matches(data[k], ds[r, s], sds.ploidy)
            seen.add((r, s))
    assert seen == {(r, s) for r in range(ds.shape[0]) for s in range(ds.shape[1])}
    unfiltered = written.with_seqs("variants")
    n_unf = sum(np.atleast_1d(np.asarray(unfiltered[r, s].alt[h])).shape[0]
                for r in range(ds.shape[0]) for s in range(ds.shape[1]) for h in range(sds.ploidy))
    assert 0 < total < n_unf
```

- [ ] **Step 2-4: Run; debug against the oracle if needed** (likely no new code — Tasks 3-4 implement it). If parity fails: check `_afs` dtype/order (global, `float32`) or the oracle not carrying AF.
- [ ] **Step 5: Commit** (`test(streaming): byte-identical SVAR1 min_af/max_af parity vs written (PR-B2, #317)`).

---

### Task 10: Byte-identical VCF/BCF `INFO/AF` parity vs the written oracle

**Files:** Modify `tests/dataset/test_streaming_variants_parity.py`; add/extend a VCF fixture **with `INFO/AF`** (≥1 multiallelic site) under `tests/` fixtures.

- [ ] **Step 1: Fixture** — a small bgzipped+indexed VCF (and, if the fixture harness supports it, a `.bcf` twin) with an `##INFO=<ID=AF,...>` header and per-ALT `AF` values, including a **multiallelic** record so the atomized ALT→AF mapping is exercised. If an existing streaming VCF fixture lacks AF, synthesize AF into a copy.

- [ ] **Step 2: Parity test** — mirror Task 9 but source from the VCF: `gvl.write(out, regions, variants=vcf_with_af)` (Task 5 caches AF into the `.gvi`), open the oracle, and compare to `StreamingDataset(variants=vcf_with_af)`; parametrize `min_af`/`max_af`/band; assert byte-identical `RaggedVariants` over the full grid and `0 < total < unfiltered`. **Explicitly assert the multiallelic record's per-ALT AF** matches on both paths (the primary parity risk).

- [ ] **Step 3: Debug** — if streaming and written diverge, the likely cause is ALT-index resolution: confirm both use genoray `resolve_scalar(source_alt_index)` and that gvl's atomization maps atoms to source ALTs identically on both sides. Add a targeted assertion on the offending `(r,s)` cell's `start`/`alt`/`ilen`.

- [ ] **Step 4: Run the full variants-parity module** — `pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -v` (PR-B1 no-filter + SVAR1 AF + VCF AF all green).
- [ ] **Step 5: Commit** (`test(streaming): byte-identical VCF/BCF INFO-AF parity vs written (PR-B2, #319)`).

---

## Phase 5 — Docs + verification

### Task 11: Docs + roadmap + board

**Files:** `skills/genvarloader/SKILL.md`; `docs/source/dataset.md`; `docs/source/faq.md`; `docs/source/write.md`; `docs/roadmaps/streaming-dataset.md`.

- [ ] **Step 1: SKILL.md** — `StreamingDataset.with_settings` accepts `min_af`/`max_af` for `with_seqs("variants")`; AF source per backend: SVAR (`cache_afs()`), VCF/BCF (`INFO/AF`), PGEN unsupported (raises). Note `gvl.write` now caches `AF` from a VCF `INFO/AF` field.
- [ ] **Step 2: dataset.md / faq.md** — streaming AF-filtering note + per-backend AF-source table; cross-reference the written path's identical guard.
- [ ] **Step 3: write.md** — `gvl.write` writes an `AF` column for VCFs that declare `INFO/AF`.
- [ ] **Step 4: roadmap** — tick PR-B2, update the Wave B status line (B2 done, closes #319; B3/B4 remain), link this design + #317 + #319. Move #319 to this PR on the StreamingDataset board.
- [ ] **Step 5: api.md no-op check** — `pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"` → `none`.
- [ ] **Step 6: Commit** (`docs(streaming): document min_af/max_af variants filtering incl VCF INFO/AF (PR-B2, #317, #319)`).

---

### Task 12: Full-suite verification + libdeflate guard + PR

- [ ] **Step 1: libdeflate regression guard** — confirm the htslib read links libdeflate (`genoray/Cargo.toml:33` `rust-htslib { features = ["libdeflate"] }`; `libdeflate-sys` in gvl `Cargo.lock`). Add a durable check: a `cargo test`/build assertion, or at minimum document the invariant + a `cargo tree -i libdeflate-sys` verification step here. Record the command + expected output.
- [ ] **Step 2: Rebuild + full Python tree** — `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests -q 2>&1 | tail -25` (whole tree — scoped runs skip `tests/unit/`).
- [ ] **Step 3: Full Rust suite** — `pixi run -e dev cargo test 2>&1 | tail -15`.
- [ ] **Step 4: Lint/typecheck/docstring** — `ruff check python/ tests/`; `ruff format --check python/ tests/`; `typecheck`; `python scripts/docstring_style.py --check python/genvarloader`.
- [ ] **Step 5: Push + draft PR into `streaming`**

```bash
git push -u origin spec/streaming-b2-minmaxaf
gh pr create --draft --base streaming \
  --title "feat(streaming): variants min_af/max_af — Wave B PR-B2 (#317, closes #319)" \
  --body "$(cat <<'EOF'
Streaming `with_seqs("variants")` `min_af`/`max_af` filtering (**Wave B PR-B2**, #317; **closes #319**).

- **SVAR1** — full support: optional global `afs` table folded into `generate_variants`' per-variant inline keep; byte-identical to the written oracle at `jitter=0` (`SparseVar.cache_afs()`).
- **VCF/BCF** — full support: AF read from the VCF `INFO/AF` field on **both** paths — the written `.gvi` is built with `info=["AF"]` (`gvl.write`) and the streaming filler requests an `AF` `FieldSpec` (genoray `info_staged` → `DecodedWindow.afs` → record keep). Same genoray `resolve_scalar` ALT resolution ⇒ byte-identical parity, incl. multiallelic.
- **PGEN** — guard-parity: no INFO ⇒ both paths raise the AF-missing `RuntimeError`.
- **Guards mirror the written path exactly:** AF + non-variants output → `NotImplementedError`; AF unavailable → `RuntimeError`.
- **No genoray change / no rev bump** (verified against pinned rev). libdeflate stays on for the htslib read.

Closes #317, #319. Relates to #304. Design: `docs/superpowers/specs/2026-07-21-streaming-variants-min-max-af-b2-design.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: Board** — `gh project item-add 1 --owner mcvickerlab --url <PR_URL>`; move #317 (+ #319) to in-review per team convention.

---

## Self-Review

**Spec coverage:**
- SVAR1 full support → Tasks 3, 4, 9. ✓
- VCF/BCF full support (written `.gvi info=["AF"]` + streaming `FieldSpec`→`afs`→record keep) → Tasks 5, 6, 7, 8, 10. ✓
- PGEN guard-parity → Task 8. ✓
- `with_settings(min_af, max_af)` surface → Task 1. ✓
- Output-mode guard (`NotImplementedError`) → Task 2. ✓
- AF-missing guard (`RuntimeError`, reconciled per backend) → Task 8. ✓
- Byte parity (SVAR1 + VCF/BCF incl. multiallelic) → Tasks 9, 10. ✓
- Rust units (SVAR1 AND-compose; record `afs` copy + keep) → Tasks 3, 6, 7. ✓
- libdeflate guard → Task 12. ✓
- Docs (SKILL/dataset/faq/write/roadmap) + api.md no-op → Task 11. ✓
- #319 closed → title/body/roadmap/board. ✓

**Open verification points (flagged, not hidden work):**
1. **genoray VCF-header `INFO/AF` detection API** (Task 5 Step 1) — exact call TBD by reading `genoray/python/genoray/_vcf.py`; fallback specified.
2. **Record-path region-clip location** (Task 7) — PR-B1's `assemble_variants_mode` had no keep loop; executor must locate the clip and fold AF there, or add a keep+compaction pass. Parity test (Task 10) is the gate.
3. **`FieldSpec` enum re-export path** (Task 6) — confirm `genoray_core::field::{FieldCategory, HtslibType, StorageDtype}` import path.
4. **Multiallelic ALT→AF parity** (Task 10) — the primary parity risk; explicit multiallelic assertion required.

**Type consistency:** `_afs: NDArray[np.float32] | None` (Py) ↔ `afs: Option<Array1<f32>>` (Rust SVAR1) ↔ `PyReadonlyArray1<f32>` (`#[new]`); `DecodedWindow.afs: Vec<f32>` (record path); `has_cached_af: bool` on all three backends; `build_engine(..., min_af=None, max_af=None)` shared signature + single call site; `min_af`/`max_af` are `float | None` end to end.

**Ordering:** Phase 1 (surface) → Phase 2 (SVAR1) → Phase 3 (VCF/BCF: write → filler/channel → record keep → guards) → Phase 4 (parity) → Phase 5 (docs/verify). Tasks 5 and 6 are independent (write-path vs Rust filler) and may run in parallel; Task 7 depends on Task 6; Task 8 depends on Tasks 5-7; Task 10 depends on Tasks 5-8.
