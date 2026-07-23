# Streaming variants Wave B — PR-B3a / PR-B3b / PR-B4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the streaming variants-output surface — `var_fields` (per-variant INFO
columns + `ref`, then SVAR1-only per-call FORMAT fields) and
`with_seqs("variant-windows")` — byte-identical to the written `Dataset`.

**Architecture:** Three stacked PRs on branch `spec/streaming-waveb-b3b4` (off `streaming`).
PR-B3a generalizes PR-B2's single hardcoded `AF` INFO column into N staged columns on
`DecodedWindow`, adds REF bytes, and exposes `var_fields` on
`StreamingDataset.with_settings`. PR-B3b adds per-call FORMAT fields for SVAR1 only
(VCF/PGEN raise, matching what a written dataset from those sources exposes). PR-B4 wires
`with_seqs("variant-windows")` to the **already-existing** `assemble_windows_mode` kernel.

**Tech Stack:** Rust (PyO3/ndarray, `genoray_core`), Python 3.10+ (numpy, polars,
`seqpro.rag.Ragged`), pixi, pytest, maturin.

**Spec:** `docs/superpowers/specs/2026-07-22-streaming-variants-wave-b-b3-b4-design.md`

## Global Constraints

- **Parity oracle is byte-identical**: `gvl.write()` + `Dataset[r, s]` at `jitter=0`. Any
  divergence is a bug in the new code until proven otherwise.
- **Rebuild before testing Rust**: `pixi run -e dev maturin develop --release` after ANY
  edit under `src/`. `pytest` does **not** rebuild — it silently imports the stale
  extension.
- **Run the full tree before pushing**: `pixi run -e dev pytest tests -q`. Scoped runs skip
  `tests/unit/`.
- **Lint covers both trees**: `pixi run -e dev ruff check python/ tests/` and
  `pixi run -e dev ruff format python/ tests/`, plus `pixi run -e dev typecheck`.
- **Google-style docstrings** in `python/genvarloader/` (`Args:`/`Returns:`/`Raises:`, no
  NumPy underlines). Verify with `python scripts/docstring_style.py --check python/genvarloader`.
- **Default `var_fields` is `["alt", "ilen", "start"]`** — every existing caller must stay
  byte-unchanged.
- **Conventional commits** (commitizen enforces this in a `commit-msg` hook).
- **No new `__all__` symbols expected.** If one appears, `docs/source/api.md` must gain an
  autodoc entry.
- **The `pyrefly-check` pre-commit hook** shells into `pixi run -e dev`; it needs a working
  dev env in this worktree. Set one up before the first code commit (see Task 0).

## Parallelism

Phase 1 (PR-B3a) is a serial dependency chain — Tasks 1→6 each build on the previous one's
Rust types. **Phases 2 and 3 (PR-B3b and PR-B4) are independent of each other** and can run
in parallel once Phase 1 lands: B3b touches `_Svar1Backend` + `src/svar1/`, B4 touches
`generate_variant_windows` + `with_seqs`. They both edit `python/genvarloader/_dataset/_streaming.py`,
so if run concurrently, use `superpowers:dispatching-parallel-agents` with separate worktrees
and reconcile `_streaming.py` at merge, or simply run them sequentially — the file is 2191
lines and the two edits land in different methods (`with_settings` vs `with_seqs`/`_iter_batches`).

**Recommended:** run Phase 1 serially, then Phases 2 and 3 in parallel via
`superpowers:dispatching-parallel-agents` + `superpowers:subagent-driven-development`, using
Sonnet or weaker for implementation.

---

## Task 0: Dev environment for this worktree

**Files:** none (environment only)

**Interfaces:**
- Consumes: nothing
- Produces: a working `pixi run -e dev` in this worktree, so every later task's test and
  lint commands work and the `pyrefly-check` pre-commit hook passes.

- [ ] **Step 1: Share the main checkout's cargo target dir**

The Rust build from scratch is slow. Point this worktree at the main checkout's target dir
so linking reuses existing artifacts:

```bash
export CARGO_TARGET_DIR=/carter/users/dlaub/projects/GenVarLoader/target
echo 'export CARGO_TARGET_DIR=/carter/users/dlaub/projects/GenVarLoader/target' >> ~/.bashrc
```

- [ ] **Step 2: Install the dev environment**

```bash
pixi install -e dev
```

Expected: completes without error (may take several minutes on first run).

- [ ] **Step 3: Build the extension**

```bash
pixi run -e dev maturin develop --release
```

Expected: `Installed genvarloader-<version>`.

- [ ] **Step 4: Verify the baseline is green**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -q
```

Expected: all tests pass. If they do not, STOP — the branch base is broken and nothing
below can be trusted.

---

# Phase 1 — PR-B3a: per-variant `var_fields`

## Task 1: Generalize `DecodedWindow.afs` to N staged INFO columns

**Files:**
- Modify: `src/record_stream/transpose.rs:30-90` (`DecodedWindow`, `fill_decoded_window`)
- Modify: `src/record_stream/vcf.rs:118-150` (`VcfWindowFiller::new` field list)
- Modify: `src/record_stream/pgen.rs:650-660` (empty `info_cols` in the chunk it builds)
- Test: `src/record_stream/transpose.rs` (in-file `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `pub enum InfoVals { I32(Vec<i32>), F32(Vec<f32>) }` in `src/record_stream/transpose.rs`
  - `pub struct InfoCol { pub name: String, pub values: InfoVals }` in the same module
  - `DecodedWindow.info_cols: Vec<InfoCol>` — one entry per requested non-AF INFO field,
    each of length `v` (the window's variant count), in the order the fields were requested
  - `VcfWindowFiller::new(vcf_path, samples, ploidy, fasta_path, want_af, info_fields)` where
    `info_fields: &[(String, bool)]` is `(name, is_float)` pairs

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block at the bottom of `src/record_stream/transpose.rs`:

```rust
    /// Requested non-AF INFO fields land in `info_cols` in request order, with the
    /// dtype genoray staged them as. `info_staged[0]` stays reserved for AF (PR-B2),
    /// so a chunk with AF + 2 extra fields yields `afs` plus 2 `info_cols`.
    #[test]
    fn info_cols_copied_from_staged_columns_in_request_order() {
        let mut chunk = dense_fixture();
        chunk.info_staged = vec![
            StagedColumn::Float(vec![0.1f32, 0.9]),
            StagedColumn::Int(vec![7i32, 8]),
            StagedColumn::Float(vec![1.5f32, 2.5]),
        ];
        let names = vec!["DP".to_string(), "QUAL2".to_string()];
        let mut w = DecodedWindow::default();
        fill_decoded_window(&mut w, &chunk, true, &names);
        assert_eq!(w.afs, vec![0.1f32, 0.9]);
        assert_eq!(w.info_cols.len(), 2);
        assert_eq!(w.info_cols[0].name, "DP");
        assert_eq!(w.info_cols[0].values, InfoVals::I32(vec![7, 8]));
        assert_eq!(w.info_cols[1].name, "QUAL2");
        assert_eq!(w.info_cols[1].values, InfoVals::F32(vec![1.5, 2.5]));
    }

    /// No AF requested: `info_staged[0]` is the first *requested* field, not AF.
    #[test]
    fn info_cols_start_at_zero_when_af_not_requested() {
        let mut chunk = dense_fixture();
        chunk.info_staged = vec![StagedColumn::Int(vec![3i32, 4])];
        let names = vec!["DP".to_string()];
        let mut w = DecodedWindow::default();
        fill_decoded_window(&mut w, &chunk, false, &names);
        assert!(w.afs.is_empty());
        assert_eq!(w.info_cols.len(), 1);
        assert_eq!(w.info_cols[0].values, InfoVals::I32(vec![3, 4]));
    }

    /// Slot recycling: a window with info columns then one without must not leak the
    /// previous window's columns (same defect class as the existing `afs` reuse test).
    #[test]
    fn info_cols_cleared_on_slot_reuse() {
        let mut with_cols = dense_fixture();
        with_cols.info_staged = vec![StagedColumn::Int(vec![3i32, 4])];
        let mut w = DecodedWindow::default();
        fill_decoded_window(&mut w, &with_cols, false, &["DP".to_string()]);
        assert_eq!(w.info_cols.len(), 1);
        let without = dense_fixture();
        fill_decoded_window(&mut w, &without, false, &[]);
        assert!(w.info_cols.is_empty());
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
export LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib:$LD_LIBRARY_PATH
pixi run -e dev cargo test --release info_cols
```

Expected: FAIL — `cannot find type InfoVals`, `no field info_cols`, and
`fill_decoded_window` takes 3 arguments not 4.

- [ ] **Step 3: Add the types and the field**

In `src/record_stream/transpose.rs`, above `pub struct DecodedWindow`:

```rust
/// Per-variant staged INFO values for one requested field. genoray's `StagedColumn`
/// has exactly two variants, so "arbitrary dtype" collapses to these two.
#[derive(Debug, Clone, PartialEq)]
pub enum InfoVals {
    I32(Vec<i32>),
    F32(Vec<f32>),
}

impl InfoVals {
    pub fn len(&self) -> usize {
        match self {
            InfoVals::I32(v) => v.len(),
            InfoVals::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named per-variant INFO column carried on the window (Wave B PR-B3a).
#[derive(Debug, Clone, PartialEq)]
pub struct InfoCol {
    pub name: String,
    pub values: InfoVals,
}
```

Add the field to `DecodedWindow` immediately after `pub afs: Vec<f32>`:

```rust
    /// Requested non-AF INFO columns, one per name passed to `fill_decoded_window`,
    /// in request order. Each has one value per window variant. Empty when no
    /// `var_fields` INFO column was requested (Wave B PR-B3a).
    pub info_cols: Vec<InfoCol>,
```

- [ ] **Step 4: Populate them in `fill_decoded_window`**

Change the signature to take the requested names, and replace the `afs` block. The existing
AF block currently reads `chunk.info_staged.first()`; AF now occupies index 0 only when
requested, so the extra columns start at `af_offset`:

```rust
pub fn fill_decoded_window(
    w: &mut DecodedWindow,
    chunk: &DenseChunk,
    want_af: bool,
    info_names: &[String],
) {
    // ... existing v_starts / ilens / alt_alleles / alt_offsets / geno_* / global_v_idxs
    // population is unchanged ...

    // AF occupies info_staged[0] iff it was requested (PR-B2); the var_fields INFO
    // columns follow it in request order (PR-B3a).
    w.afs.clear();
    let af_offset = if want_af {
        if let Some(genoray_core::types::StagedColumn::Float(v)) = chunk.info_staged.first() {
            w.afs.extend_from_slice(v);
        }
        1
    } else {
        0
    };

    w.info_cols.clear();
    for (i, name) in info_names.iter().enumerate() {
        let values = match chunk.info_staged.get(af_offset + i) {
            Some(genoray_core::types::StagedColumn::Int(v)) => InfoVals::I32(v.clone()),
            Some(genoray_core::types::StagedColumn::Float(v)) => InfoVals::F32(v.clone()),
            None => continue,
        };
        w.info_cols.push(InfoCol { name: name.clone(), values });
    }
}
```

- [ ] **Step 5: Update the two callers**

In `src/record_stream/vcf.rs`, store the requested fields on the filler and build the
`FieldSpec` list from them. Replace the `want_af`-only field construction in
`VcfWindowFiller::new`:

```rust
    pub fn new(
        vcf_path: &str,
        samples: &[&str],
        ploidy: usize,
        fasta_path: Option<&str>,
        want_af: bool,
        info_fields: &[(String, bool)],
    ) -> anyhow::Result<Self> {
        let sample_indices = VcfRecordSource::resolve_sample_indices(vcf_path, samples)?;
        VCF_SAMPLE_RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
        let mut fields: Vec<FieldSpec> = Vec::new();
        if want_af {
            fields.push(FieldSpec {
                name: "AF".into(),
                category: FieldCategory::Info,
                htype: HtslibType::Float,
                dtype: StorageDtype::F32,
                default: None,
            });
        }
        for (name, is_float) in info_fields {
            fields.push(FieldSpec {
                name: name.clone().into(),
                category: FieldCategory::Info,
                htype: if *is_float { HtslibType::Float } else { HtslibType::Integer },
                dtype: if *is_float { StorageDtype::F32 } else { StorageDtype::I32 },
                default: None,
            });
        }
        let info_names: Vec<String> = info_fields.iter().map(|(n, _)| n.clone()).collect();
        Ok(Self {
            vcf_path: vcf_path.to_string(),
            sample_indices,
            ploidy,
            fasta_path: fasta_path.map(str::to_string),
            check_ref: CheckRef::Exclude,
            overlap: OverlapMode::Variant,
            fields,
            want_af,
            info_names,
            htslib_threads: 1,
            chunk_size: DEFAULT_CHUNK_SIZE,
        })
    }
```

Add `want_af: bool` and `info_names: Vec<String>` to the `VcfWindowFiller` struct, and in
its `fill` method pass them through: `fill_decoded_window(w, &chunk, self.want_af, &self.info_names)`.

In `src/record_stream/pgen.rs`, PGEN never stages INFO, so its `fill` passes
`fill_decoded_window(w, &chunk, false, &[])`.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pixi run -e dev cargo test --release info_cols
```

Expected: 3 passed.

- [ ] **Step 7: Run the whole Rust suite**

```bash
pixi run -e dev cargo test --release
```

Expected: 0 failed.

- [ ] **Step 8: Commit**

```bash
git add src/record_stream/transpose.rs src/record_stream/vcf.rs src/record_stream/pgen.rs
git commit -m "feat(streaming): N staged INFO columns on DecodedWindow (PR-B3a, #304)"
```

---

## Task 2: Carry INFO columns through `VariantsBatch` and the variants consumer

**Files:**
- Modify: `src/variants/mod.rs:13-20` (`VariantsBatch`)
- Modify: `src/record_stream/engine.rs:247-318` (`RecordBackend::generate_variants`)
- Modify: `src/ffi/stream_core.rs:81-89` (`EngineBackend::generate_variants` doc only)
- Test: `src/record_stream/engine.rs` (in-file `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: `InfoVals`, `InfoCol`, `DecodedWindow.info_cols` (Task 1).
- Produces: `VariantsBatch.info_out: Vec<(String, InfoVals)>` — one entry per window INFO
  column, gathered and keep-compacted in lockstep with `start`/`ilen`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `src/record_stream/engine.rs`, next to the
existing `generate_variants_*` tests:

```rust
    /// PR-B3a: a window INFO column rides along, gathered by the SAME kept `v_idxs`
    /// and therefore aligned with `start`/`ilen` element-for-element.
    #[test]
    fn generate_variants_gathers_info_columns_aligned_with_start() {
        let (backend, slot) = variants_fixture_with_info(vec![InfoCol {
            name: "DP".into(),
            values: InfoVals::I32(vec![10, 20, 30]),
        }]);
        let batch = backend.generate_variants(0, &slot, 0, 1).unwrap();
        assert_eq!(batch.info_out.len(), 1);
        assert_eq!(batch.info_out[0].0, "DP");
        let InfoVals::I32(dp) = &batch.info_out[0].1 else {
            panic!("expected I32 column");
        };
        // One DP value per kept variant, same count as `start`.
        assert_eq!(dp.len(), batch.start.len());
        // And the values must be the ones indexed by the kept v_idxs, not a prefix.
        for (i, &s) in batch.start.iter().enumerate() {
            let vi = slot.v_starts.iter().position(|&x| x == s).unwrap();
            assert_eq!(dp[i], 10 * (vi as i32 + 1));
        }
    }
```

Add the fixture helper alongside the existing test helpers in the same module:

```rust
    /// A 1-region / 1-sample / ploidy-1 window whose 3 variants all fall inside the
    /// region, carrying the supplied INFO columns. Mirrors `variants_fixture()` but
    /// lets the caller attach `info_cols`.
    fn variants_fixture_with_info(
        info_cols: Vec<InfoCol>,
    ) -> (RecordBackend, DecodedWindow) {
        let (backend, mut slot) = variants_fixture();
        slot.info_cols = info_cols;
        (backend, slot)
    }
```

If `variants_fixture()` does not already exist under that name, factor the setup out of
`generate_variants_clips_to_window_and_gathers_fields` into it first — that test builds
exactly this shape.

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run -e dev cargo test --release generate_variants_gathers_info_columns
```

Expected: FAIL — `no field info_out on VariantsBatch`.

- [ ] **Step 3: Add the field to `VariantsBatch`**

In `src/variants/mod.rs`, add to the struct:

```rust
    /// Ride-along per-variant INFO columns (Wave B PR-B3a), one entry per requested
    /// `var_fields` INFO column, each with exactly one value per kept variant — i.e.
    /// the same length as `start`/`ilen`, gathered by the same kept `v_idxs`.
    pub info_out: Vec<(String, crate::record_stream::transpose::InfoVals)>,
```

- [ ] **Step 4: Gather them in `generate_variants`**

In `src/record_stream/engine.rs`, after the existing `assemble_variants_window` call and
before the `Ok(VariantsBatch { .. })`:

```rust
        // PR-B3a: ride-along INFO columns, gathered by the SAME kept v_idxs the
        // assembly above used, so every column is aligned with `start`/`ilen`.
        // The region/AF keep already happened when v_idxs was built — no second mask.
        let info_out: Vec<(String, InfoVals)> = slot
            .info_cols
            .iter()
            .map(|col| {
                let vals = match &col.values {
                    InfoVals::I32(src) => {
                        InfoVals::I32(v_idxs.iter().map(|&vi| src[vi as usize]).collect())
                    }
                    InfoVals::F32(src) => {
                        InfoVals::F32(v_idxs.iter().map(|&vi| src[vi as usize]).collect())
                    }
                };
                (col.name.clone(), vals)
            })
            .collect();
```

and add `info_out,` to the returned struct literal.

- [ ] **Step 5: Fix the other `VariantsBatch` construction sites**

`Svar1Backend::generate_variants` in `src/ffi/stream_engine.rs` also builds a
`VariantsBatch`. Give it `info_out: Vec::new()` for now — Task 3 fills it in for SVAR1.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
pixi run -e dev cargo test --release
```

Expected: 0 failed.

- [ ] **Step 7: Commit**

```bash
git add src/variants/mod.rs src/record_stream/engine.rs src/ffi/stream_engine.rs
git commit -m "feat(streaming): ride-along INFO columns on VariantsBatch (PR-B3a, #304)"
```

---

## Task 3: REF bytes on all three backends

**Files:**
- Modify: `src/variants/mod.rs:104-118` (`assemble_variants_window`)
- Modify: `src/record_stream/engine.rs` (`RecordBackend::generate_variants` REF slice)
- Modify: `src/ffi/stream_engine.rs` (`Svar1Backend` REF table)
- Modify: `python/genvarloader/_dataset/_streaming.py:1281` (keep SVAR1's REF)
- Test: `src/variants/mod.rs` (in-file tests)

**Interfaces:**
- Consumes: `VariantsBatch` (Task 2).
- Produces:
  - `VariantsBatch.ref_data: Option<Array1<u8>>` and `ref_seq_offsets: Option<Array1<i64>>`
  - `assemble_variants_window(v_idxs, v_starts, ilens, alt_alleles, alt_offsets, ref_src)`
    where `ref_src: Option<(ArrayView1<u8>, ArrayView1<i64>)>` is a per-variant REF byte
    table, returning `(alt_data, alt_seq_offsets, start, ilen, ref_data, ref_seq_offsets)`

- [ ] **Step 1: Write the failing test**

Add to `src/variants/mod.rs`'s test module:

```rust
    /// PR-B3a: with a per-variant REF table, `assemble_variants_window` gathers REF
    /// bytes exactly the way it gathers ALT bytes.
    #[test]
    fn assemble_variants_window_gathers_ref_when_table_supplied() {
        let v_idxs = Array1::from_vec(vec![2i32, 0]);
        let v_starts = Array1::from_vec(vec![10i32, 20, 30]);
        let ilens = Array1::from_vec(vec![0i32, 0, 0]);
        let alt = Array1::from_vec(b"AACCGG".to_vec());
        let alt_off = Array1::from_vec(vec![0i64, 2, 4, 6]);
        let refe = Array1::from_vec(b"TTTGGGCCC".to_vec());
        let ref_off = Array1::from_vec(vec![0i64, 3, 6, 9]);
        let (_a, _ao, _s, _i, rd, ro) = assemble_variants_window(
            v_idxs.view(),
            v_starts.view(),
            ilens.view(),
            alt.view(),
            alt_off.view(),
            Some((refe.view(), ref_off.view())),
        );
        let rd = rd.expect("ref requested");
        let ro = ro.expect("ref requested");
        // v_idx 2 -> "CCC", v_idx 0 -> "TTT"
        assert_eq!(rd.to_vec(), b"CCCTTT".to_vec());
        assert_eq!(ro.to_vec(), vec![0i64, 3, 6]);
    }

    /// No REF table supplied ⇒ both REF outputs are None (default `var_fields`).
    #[test]
    fn assemble_variants_window_ref_is_none_without_table() {
        let v_idxs = Array1::from_vec(vec![0i32]);
        let v_starts = Array1::from_vec(vec![10i32]);
        let ilens = Array1::from_vec(vec![0i32]);
        let alt = Array1::from_vec(b"A".to_vec());
        let alt_off = Array1::from_vec(vec![0i64, 1]);
        let (_a, _ao, _s, _i, rd, ro) = assemble_variants_window(
            v_idxs.view(),
            v_starts.view(),
            ilens.view(),
            alt.view(),
            alt_off.view(),
            None,
        );
        assert!(rd.is_none());
        assert!(ro.is_none());
    }
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev cargo test --release assemble_variants_window
```

Expected: FAIL — `this function takes 5 arguments but 6 were supplied`.

- [ ] **Step 3: Extend `assemble_variants_window`**

```rust
pub fn assemble_variants_window(
    v_idxs: ArrayView1<i32>,
    v_starts: ArrayView1<i32>,
    ilens: ArrayView1<i32>,
    alt_alleles: ArrayView1<u8>,
    alt_offsets: ArrayView1<i64>,
    ref_src: Option<(ArrayView1<u8>, ArrayView1<i64>)>,
) -> (
    Array1<u8>,
    Array1<i64>,
    Array1<i32>,
    Array1<i32>,
    Option<Array1<u8>>,
    Option<Array1<i64>>,
) {
    let (alt_data, alt_seq_offsets) = gather_alleles(v_idxs, alt_alleles, alt_offsets);
    let start: Array1<i32> = v_idxs.iter().map(|&vi| v_starts[vi as usize]).collect();
    let ilen: Array1<i32> = v_idxs.iter().map(|&vi| ilens[vi as usize]).collect();
    let (ref_data, ref_seq_offsets) = match ref_src {
        Some((bytes, offs)) => {
            let (d, o) = gather_alleles(v_idxs, bytes, offs);
            (Some(d), Some(o))
        }
        None => (None, None),
    };
    (alt_data, alt_seq_offsets, start, ilen, ref_data, ref_seq_offsets)
}
```

Add `ref_data: Option<Array1<u8>>` and `ref_seq_offsets: Option<Array1<i64>>` to
`VariantsBatch`, and thread them through both `generate_variants` impls.

- [ ] **Step 4: Build the VCF/PGEN REF table by reference-slice**

In `src/record_stream/engine.rs`'s `generate_variants`, before the assembly call, build a
per-window-variant REF byte table from the contig reference. `ref_len = alt_len - ilen`
where `alt_len` comes from the window's `alt_offsets`:

```rust
        // PR-B3a: VCF/PGEN REF bytes. genoray's DenseChunk does not expose REF, so
        // slice the contig reference at [start, start + ref_len) with
        // ref_len = alt_len - ilen — exact for the normalized, left-aligned biallelic
        // variants gvl requires. `c.ref_bytes` is the WHOLE contig (build_engine
        // materializes each touched contig entire), so `v_start` is a direct index.
        let ref_table = if self.want_ref {
            let c = &self.contigs[job.contig_idx];
            let n_v = slot.v_starts.len();
            let mut bytes: Vec<u8> = Vec::new();
            let mut offs: Vec<i64> = Vec::with_capacity(n_v + 1);
            offs.push(0);
            for vi in 0..n_v {
                let s = slot.v_starts[vi] as i64;
                let alt_len = slot.alt_offsets[vi + 1] - slot.alt_offsets[vi];
                let ref_len = alt_len - slot.ilens[vi] as i64;
                for k in 0..ref_len {
                    let pos = s + k;
                    let b = if pos < 0 || pos as usize >= c.ref_bytes.len() {
                        self.pad_char
                    } else {
                        c.ref_bytes[pos as usize]
                    };
                    bytes.push(b);
                }
                offs.push(bytes.len() as i64);
            }
            Some((bytes, offs))
        } else {
            None
        };
```

then pass
`ref_table.as_ref().map(|(b, o)| (ArrayView1::from(b.as_slice()), ArrayView1::from(o.as_slice())))`
as the new argument. Add a `want_ref: bool` field to `RecordBackend`, set from the
constructor.

- [ ] **Step 5: Keep SVAR1's REF instead of discarding it**

In `python/genvarloader/_dataset/_streaming.py`, `_Svar1Backend.__init__` currently does:

```python
        v_starts, ilens, ref, alt = _variant_arrays_from_table(idx, one_based=True)
        if ref is None:
            raise ValueError(f"SVAR1 store at {svar_path} has no REF allele column.")
```

Keep the check, and store the table:

```python
        v_starts, ilens, ref, alt = _variant_arrays_from_table(idx, one_based=True)
        if ref is None:
            raise ValueError(f"SVAR1 store at {svar_path} has no REF allele column.")
        # Wave B PR-B3a (#304): REF was read and discarded here. It is a `var_field`
        # (and the `ref="allele"` input for variant-windows), so keep the global
        # per-variant byte table alongside the ALT one -- same layout, same gather.
        self._ref_alleles = np.ascontiguousarray(ref.data.view(np.uint8), np.uint8)
        self._ref_offsets = np.ascontiguousarray(ref.offsets, np.int64)
```

Thread both into `Svar1StreamEngine`'s constructor and store them on `Svar1Backend`, so its
`generate_variants` can pass `Some((ref_alleles, ref_offsets))` when `want_ref`.

- [ ] **Step 6: Rebuild and run**

```bash
pixi run -e dev maturin develop --release && pixi run -e dev cargo test --release
```

Expected: 0 failed.

- [ ] **Step 7: Commit**

```bash
git add src/variants/mod.rs src/record_stream/engine.rs src/ffi/stream_engine.rs python/genvarloader/_dataset/_streaming.py
git commit -m "feat(streaming): REF bytes for variants output on all three backends (PR-B3a, #304)"
```

---

## Task 4: Marshal the new columns across the FFI

**Files:**
- Modify: `src/record_stream/engine.rs:621-640` (`next_batch_variants`)
- Modify: `src/ffi/stream_engine.rs` (SVAR1's `next_batch_variants`)
- Test: `tests/dataset/test_streaming_variants_parity.py`

**Interfaces:**
- Consumes: `VariantsBatch.info_out`, `.ref_data`, `.ref_seq_offsets` (Tasks 2-3).
- Produces: the `next_batch_variants` dict gains optional keys `ref` (u8) and
  `ref_offsets` (i64), plus one key per INFO column name carrying its raw numpy array.

- [ ] **Step 1: Extend both marshalers**

In `src/record_stream/engine.rs`'s `next_batch_variants`, after the existing `set_item`
calls:

```rust
                if let (Some(rd), Some(ro)) = (batch.ref_data, batch.ref_seq_offsets) {
                    dict.set_item("ref", rd.into_pyarray(py))?;
                    dict.set_item("ref_offsets", ro.into_pyarray(py))?;
                }
                for (name, vals) in batch.info_out {
                    match vals {
                        InfoVals::I32(v) => {
                            dict.set_item(name, Array1::from_vec(v).into_pyarray(py))?
                        }
                        InfoVals::F32(v) => {
                            dict.set_item(name, Array1::from_vec(v).into_pyarray(py))?
                        }
                    }
                }
```

Apply the identical block to `Svar1StreamEngine::next_batch_variants` in
`src/ffi/stream_engine.rs`.

- [ ] **Step 2: Rebuild**

```bash
pixi run -e dev maturin develop --release
```

Expected: `Installed genvarloader-<version>`.

- [ ] **Step 3: Confirm the default path is byte-unchanged**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -q
```

Expected: all pass — no `var_fields` is requested yet, so `ref`/INFO keys are absent and
the existing packing is untouched.

- [ ] **Step 4: Commit**

```bash
git add src/record_stream/engine.rs src/ffi/stream_engine.rs
git commit -m "feat(streaming): marshal ref + INFO columns from next_batch_variants (PR-B3a, #304)"
```

---

## Task 5: Python surface — `var_fields`, `available_var_fields`, `active_var_fields`

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`with_settings` ~1042, `_iter_batches`
  ~615-655, `_Svar1Backend.__init__` ~1196, `_VcfBackend` ~1965, `_PgenBackend` ~2122,
  `build_engine` ~1377)
- Test: `tests/dataset/test_streaming_config.py`

**Interfaces:**
- Consumes: the extended `next_batch_variants` dict (Task 4).
- Produces:
  - `StreamingDataset.with_settings(..., var_fields: list[str] | None = None)`
  - `StreamingDataset.available_var_fields -> list[str]`
  - `StreamingDataset.active_var_fields -> list[str]`
  - `backend.available_var_fields -> list[str]` on each of `_Svar1Backend`, `_VcfBackend`,
    `_PgenBackend`

- [ ] **Step 1: Write the failing tests**

Add to `tests/dataset/test_streaming_config.py`:

```python
def test_available_var_fields_svar1_includes_ref(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert "ref" in sds.available_var_fields
    assert {"alt", "ilen", "start"} <= set(sds.available_var_fields)


def test_active_var_fields_defaults_to_alt_ilen_start(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert sds.active_var_fields == ["alt", "ilen", "start"]


def test_unknown_var_field_raises(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    with pytest.raises(ValueError, match="not available"):
        sds.with_settings(var_fields=["alt", "start", "NOPE"])


def test_var_fields_with_haplotypes_output_raises(streaming_case):
    regions, reference, variants, _written = streaming_case("svar1")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=["alt", "start", "ref"])
        .with_seqs("haplotypes")
    )
    with pytest.raises(NotImplementedError, match="var_fields"):
        next(iter(sds.to_iter(batch_size=2)))


def test_pgen_var_fields_limited_to_ref(streaming_case):
    regions, reference, variants, _written = streaming_case("pgen")
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert "ref" in sds.available_var_fields
    assert sds.available_var_fields == ["alt", "ilen", "start", "ref"]
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_config.py -q -k var_field
```

Expected: FAIL — `StreamingDataset has no attribute available_var_fields`.

- [ ] **Step 3: Add `available_var_fields` to each backend**

`_Svar1Backend.__init__`, after reading `idx` and before canonicalization:

```python
        # Wave B PR-B3a (#304): numeric index columns are the requestable INFO fields,
        # filtered exactly as `_Variants.available_info_fields` does on the written
        # path (`_haps.py`): numeric, minus the positional POS/ILEN columns. `ref` is
        # always available -- the store is rejected above if it has no REF column.
        _schema = pl.scan_ipc(Path(svar_path) / "index.arrow").collect_schema()
        _info = [
            k
            for k, v in _schema.items()
            if v.is_numeric() and k not in {"POS", "ILEN"}
        ]
        self.available_var_fields = ["alt", "ilen", "start", *_info, "ref"]
```

`_VcfBackend.__init__` (reusing the PR-B2 helper), and record the dtype so
`build_engine` can pass `(name, is_float)`:

```python
        # Wave B PR-B3a (#304): declared numeric INFO fields are requestable.
        # Non-numeric types (Flag/String/Character) are excluded, matching the
        # written path's numeric-only `available_info_fields` filter.
        self._info_dtypes = _declared_info_numeric_dtypes(self._vcf_path)
        self.available_var_fields = [
            "alt",
            "ilen",
            "start",
            *self._info_dtypes,
            "ref",
        ]
```

`_PgenBackend.__init__`:

```python
        # PGEN has no INFO path (see the PR-B2 AF guard); `ref` is the only
        # non-builtin var_field it can serve.
        self.available_var_fields = ["alt", "ilen", "start", "ref"]
```

Write `_declared_info_numeric_dtypes(path) -> dict[str, bool]` next to the existing
`_declared_info_fields` helper, returning `{name: is_float}` for every `INFO` header line
whose `Type=` is `Integer` or `Float`.

- [ ] **Step 4: Add the public surface**

On `StreamingDataset`, add the field `_var_fields: list[str] | None = None` and:

```python
    @property
    def available_var_fields(self) -> list[str]:
        """Variant fields this source can serve, in a stable order.

        Derived from the live source (index schema for SVAR, VCF header for
        VCF/BCF), so it can differ per backend — unlike the written
        :attr:`Dataset.available_var_fields`, which reads the written artifact.

        Returns:
            Field names requestable via :meth:`with_settings`.
        """
        if self._backend is None:
            return ["alt", "ilen", "start"]
        return list(self._backend.available_var_fields)

    @property
    def active_var_fields(self) -> list[str]:
        """The variant fields currently selected.

        Returns:
            The configured ``var_fields``, or the default ``["alt", "ilen", "start"]``.
        """
        return list(self._var_fields) if self._var_fields is not None else [
            "alt",
            "ilen",
            "start",
        ]
```

In `with_settings`, add the `var_fields` parameter and validate:

```python
        if var_fields is not None:
            missing = [f for f in var_fields if f not in self.available_var_fields]
            if missing:
                raise ValueError(
                    f"var_fields {missing} are not available for this source. "
                    f"Available: {self.available_var_fields}."
                )
            to_evolve["_var_fields"] = list(var_fields)
```

- [ ] **Step 5: Guard non-variants output**

In `_iter_batches`, beside the existing `_af_filter and not _variants` guard:

```python
            # Wave B PR-B3a (#304): `var_fields` only shapes `with_seqs("variants")`
            # output. The written path silently ignores it elsewhere; streaming fails
            # fast instead, matching how it treats every other ignorable setting
            # (see the jitter/out_len/annotated guard below and PR-B2's AF guard).
            if self._var_fields is not None and not _variants:
                raise NotImplementedError(
                    'var_fields only applies to with_seqs("variants") output; '
                    f"got with_seqs({self._seq_kind_name!r})."
                )
```

- [ ] **Step 6: Pass the request into `build_engine` and pack the output**

`build_engine` gains `var_fields: list[str] | None`; each backend derives
`want_ref = "ref" in var_fields` and the INFO name/dtype list, and forwards them to its
engine constructor.

In `_iter_batches`'s `_variants` packing block, after building `alt`/`start`/`ilen`:

```python
                            extra: dict[str, Ragged] = {}
                            ref_rag = None
                            if "ref" in nxt:
                                ref_char = np.asarray(nxt["ref"], np.uint8).view("S1")
                                ref_rag = (
                                    Ragged.from_offsets(
                                        ref_char,
                                        (b_times_p, None, None),
                                        [row_off, np.asarray(nxt["ref_offsets"], np.int64)],
                                    )
                                    .to_strings()
                                    .reshape(hi - lo, backend.ploidy, None)
                                )
                            for name in self.active_var_fields:
                                if name in ("alt", "start", "ilen", "ref"):
                                    continue
                                extra[name] = Ragged.from_offsets(
                                    np.asarray(nxt[name]),
                                    (b_times_p, None),
                                    row_off,
                                ).reshape(hi - lo, backend.ploidy, None)
                            out = RaggedVariants(
                                alt=alt,
                                start=start,
                                ilen=ilen if "ilen" in self.active_var_fields else None,
                                ref=ref_rag,
                                **extra,
                            )
```

`RaggedVariants.__init__` already accepts `ref`, `ilen`, and `**fields`
(`_rag_variants.py:210-232`), so no change is needed there.

- [ ] **Step 7: Run the tests to verify they pass**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_config.py -q -k var_field
```

Expected: 5 passed.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_config.py
git commit -m "feat(streaming): var_fields on with_settings + available/active_var_fields (PR-B3a, #304)"
```

---

## Task 6: PR-B3a parity suite + docs

**Files:**
- Modify: `tests/dataset/test_streaming_variants_parity.py`
- Modify: `docs/source/dataset.md`, `docs/source/faq.md`, `skills/genvarloader/SKILL.md`
- Modify: `docs/roadmaps/streaming-dataset.md`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: the byte-identical gate for PR-B3a.

- [ ] **Step 1: Write the parity tests**

Add to `tests/dataset/test_streaming_variants_parity.py`:

```python
@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_ref_var_field_matches_written(streaming_case, backend):
    """`ref` is byte-identical to the written path on every backend.

    SVAR1 reads REF from its index; VCF/PGEN slice it out of the contig reference at
    [start, start + alt_len - ilen). The written oracle carries REF directly, so this
    is the gate on the reference-slice assumption.
    """
    regions, reference, variants, written = streaming_case(backend)
    fields = ["alt", "ilen", "start", "ref"]
    ds = written.with_settings(var_fields=fields).with_seqs("variants")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=fields)
        .with_seqs("variants")
    )
    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data.ref[k, h]), np.asarray(expected.ref[h])
                )
                total += np.atleast_1d(np.asarray(data.ref[k, h])).shape[0]
    assert total > 0, "vacuous pass: no variants compared"


@pytest.mark.parametrize("field_name", ["AF", "DP"])
def test_streaming_vcf_info_var_field_matches_written(af_vcf_case, field_name):
    """A Float INFO field (AF) and an Integer INFO field (DP) both ride along
    byte-identically. DP is the dtype canary: the written column's dtype comes from
    genoray's index writer, streaming's from the staged FieldSpec, and they must agree.
    """
    regions, reference, variants, written = af_vcf_case
    fields = ["alt", "ilen", "start", field_name]
    ds = written.with_settings(var_fields=fields).with_seqs("variants")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=fields)
        .with_seqs("variants")
    )
    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            for h in range(sds.ploidy):
                got = np.asarray(data[field_name][k, h])
                exp = np.asarray(expected[field_name][h])
                assert got.dtype == exp.dtype, (
                    f"{field_name} dtype divergence: streaming {got.dtype} "
                    f"vs written {exp.dtype}"
                )
                np.testing.assert_array_equal(got, exp)
                total += np.atleast_1d(got).shape[0]
    assert total > 0, "vacuous pass: no variants compared"


@pytest.mark.parametrize("backend", BACKENDS)
def test_available_var_fields_matches_written(streaming_case, backend):
    """Streaming derives the field set from the live source, the written path from the
    written artifact. They must agree, or one of the two definitions has drifted."""
    regions, reference, variants, written = streaming_case(backend)
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert set(sds.available_var_fields) == set(written.available_var_fields)
```

Extend `_build_af_vcf` (the existing helper at `test_streaming_variants_parity.py:189`) to
also emit an `##INFO=<ID=DP,Number=1,Type=Integer,...>` header line and a per-record `DP`
value, and expose it through a module-scoped `af_vcf_case` fixture that returns
`(regions, reference, variants, written)` the way `streaming_case` does.

- [ ] **Step 2: Run to verify failure, then iterate to green**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -q
```

Expected initially: the `DP` case may FAIL on dtype. If it does, that is the risk the spec
named — normalize the streaming dtype to the written column's dtype in the Python packing
step of Task 5 and document it, rather than loosening the assertion.

- [ ] **Step 3: Run the interior-exclusion and narrowed-window fixtures**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py tests/dataset/test_variants_region_clip.py -q
```

Expected: all pass. A ride-along column compacted by the wrong mask shows up here as
misaligned values, not a crash.

- [ ] **Step 4: Update the docs**

- `docs/source/dataset.md` — document `StreamingDataset.with_settings(var_fields=...)`,
  `available_var_fields`, `active_var_fields`, and that the available set is
  backend-derived (SVAR1: index columns + `ref`; VCF: numeric INFO + `ref`; PGEN: `ref`).
- `docs/source/faq.md` — note that `var_fields` applies only to `with_seqs("variants")`
  and raises `NotImplementedError` otherwise (stricter than the written path).
- `skills/genvarloader/SKILL.md` — add `var_fields` to the `StreamingDataset` section.
- `docs/roadmaps/streaming-dataset.md` — mark PR-B3a done under the Wave B entry.

- [ ] **Step 5: Verify `api.md` is still in sync**

```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```

Expected: `MISSING: none`.

- [ ] **Step 6: Full tree + lint**

```bash
pixi run -e dev pytest tests -q
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
```

Expected: 0 failed, no lint errors.

- [ ] **Step 7: Commit**

```bash
git add tests/ docs/ skills/
git commit -m "test(streaming): var_fields + ref byte-identical parity, all backends (PR-B3a, #304)"
```

---

# Phase 2 — PR-B3b: per-call FORMAT fields (SVAR1 only)

## Task 7: SVAR1 FORMAT/dosage fields

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_Svar1Backend`)
- Modify: `src/ffi/stream_engine.rs` (`Svar1Backend::generate_variants`)
- Test: `tests/dataset/test_streaming_variants_parity.py`

**Interfaces:**
- Consumes: `available_var_fields` (Task 5), `VariantsBatch.info_out` (Task 2).
- Produces: `_Svar1Backend.available_var_fields` additionally lists `dosage` (iff
  `dosages.npy` exists) and every key of `_svar_format_fields(svar_path)`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("min_af,max_af", [(None, None), (0.2, 0.8)])
def test_streaming_svar1_dosage_matches_written(streaming_case, min_af, max_af):
    """SVAR1 `dosage` rides along byte-identically, and is compacted by the SAME
    AF/region keep mask as `start`/`ilen` (hence the AF-filtered parametrization)."""
    regions, reference, variants, written = streaming_case("svar1")
    fields = ["alt", "ilen", "start", "dosage"]
    ds = written.with_settings(
        var_fields=fields, min_af=min_af, max_af=max_af
    ).with_seqs("variants")
    sds = (
        gvl.StreamingDataset(regions, reference=reference, variants=variants)
        .with_settings(var_fields=fields, min_af=min_af, max_af=max_af)
        .with_seqs("variants")
    )
    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            for h in range(sds.ploidy):
                got = np.asarray(data.dosage[k, h])
                np.testing.assert_array_equal(got, np.asarray(expected.dosage[h]))
                total += np.atleast_1d(got).shape[0]
    assert total > 0, "vacuous pass: no variants compared"


@pytest.mark.parametrize("backend", ["vcf", "pgen"])
def test_dosage_var_field_rejected_on_record_backends(streaming_case, backend):
    """`gvl.write` never persists dosage for VCF/PGEN sources, so a written dataset
    from the same source cannot serve it either — streaming declines symmetrically."""
    regions, reference, variants, _written = streaming_case(backend)
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    assert "dosage" not in sds.available_var_fields
    with pytest.raises(ValueError, match="not available"):
        sds.with_settings(var_fields=["alt", "start", "dosage"])
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -q -k dosage
```

Expected: FAIL — `dosage` is not in `available_var_fields`, so `with_settings` raises in the
first test too.

- [ ] **Step 3: Discover and memmap the per-call fields**

In `_Svar1Backend.__init__`, after the `available_var_fields` block from Task 5:

```python
        # Wave B PR-B3b (#304): per-call FORMAT fields live on the SAME hap-major CSR
        # offsets as `variant_idxs.npy`, so each is a plain memmap the window fill can
        # slice with the ranges `find_ranges` already produced -- no new search, no new
        # offsets. Field discovery reuses the written path's helper verbatim.
        from ._haps import _svar_format_fields

        self._call_fields: dict[str, np.memmap] = {}
        _dosage_path = Path(svar_path) / "dosages.npy"
        if _dosage_path.exists():
            self._call_fields["dosage"] = np.memmap(
                _dosage_path, dtype=DOSAGE_TYPE, mode="r"
            )
        for _name, _dt in _svar_format_fields(Path(svar_path)).items():
            self._call_fields[_name] = np.memmap(
                Path(svar_path) / f"{_name}.npy", dtype=_dt, mode="r"
            )
        self.available_var_fields = [
            *self.available_var_fields,
            *[f for f in self._call_fields if f not in self.available_var_fields],
        ]
```

- [ ] **Step 4: Gather them in the SVAR1 walk**

Per-call fields are stored **parallel to `variant_idxs.npy` on the same CSR offsets** (the
written path builds `dosages` with `Ragged.from_offsets(dosages_mm, rag_shape, offsets)` —
the *same* `offsets` as `genotypes`). So the field value for a kept entry is indexed by the
CSR **position** `o`, not by the variant id `gvi`. That makes it a one-line push inside the
existing keep branch of `Svar1Backend::generate_variants` (`src/ffi/stream_engine.rs:295-321`).

Add a field to `Svar1Backend`:

```rust
    /// Per-call FORMAT/dosage columns (Wave B PR-B3b), parallel to the store's
    /// `variant_idxs` CSR — indexed by CSR position, NOT by variant id.
    call_fields: Vec<(String, CallVals)>,
```

with

```rust
pub enum CallVals {
    F32(ndarray::Array1<f32>),
    I32(ndarray::Array1<i32>),
}
```

Declare the accumulators next to `kept`:

```rust
        let mut call_out: Vec<(String, CallVals)> = self
            .call_fields
            .iter()
            .map(|(n, v)| {
                let empty = match v {
                    CallVals::F32(_) => CallVals::F32(ndarray::Array1::zeros(0)),
                    CallVals::I32(_) => CallVals::I32(ndarray::Array1::zeros(0)),
                };
                (n.clone(), empty)
            })
            .collect();
        let mut call_bufs: Vec<Vec<f64>> = vec![Vec::new(); self.call_fields.len()];
```

and inside the existing `if v_start < r_e as i64 && v_end > r_s as i64 && af_keep {` block,
right after `kept.push(gvi);`:

```rust
                    // PR-B3b: same keep, same order — one value per KEPT entry, so the
                    // column stays aligned with `start`/`ilen` element-for-element.
                    for (fi, (_, vals)) in self.call_fields.iter().enumerate() {
                        call_bufs[fi].push(match vals {
                            CallVals::F32(a) => a[o] as f64,
                            CallVals::I32(a) => a[o] as f64,
                        });
                    }
```

After the walk, narrow each buffer back to its declared dtype and emit through
`VariantsBatch.info_out` (the channel Task 2 built, so Task 4's marshaling already handles
it — no new FFI):

```rust
        let mut info_out: Vec<(String, InfoVals)> = Vec::new();
        for (fi, (name, vals)) in self.call_fields.iter().enumerate() {
            let buf = &call_bufs[fi];
            info_out.push((
                name.clone(),
                match vals {
                    CallVals::F32(_) => {
                        InfoVals::F32(buf.iter().map(|&x| x as f32).collect())
                    }
                    CallVals::I32(_) => {
                        InfoVals::I32(buf.iter().map(|&x| x as i32).collect())
                    }
                },
            ));
        }
```

and set `info_out,` on the returned `VariantsBatch`. Drop the now-unused `call_out` binding
if the compiler flags it — it is only there to make the dtype pairing explicit while
writing; `self.call_fields` already carries it.

Pass the memmapped arrays from Task 7 Step 3 into `Svar1StreamEngine`'s constructor as
`PyReadonlyArray1`s and store them as `call_fields`.

- [ ] **Step 5: Rebuild, run**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -q -k "dosage or var_field"
```

Expected: all pass.

- [ ] **Step 6: Docs**

- `docs/source/dataset.md` — note that `dosage` and custom FORMAT fields are SVAR-only,
  on both the streaming and written paths.
- `skills/genvarloader/SKILL.md` — same note.
- `docs/roadmaps/streaming-dataset.md` — mark PR-B3b done.

- [ ] **Step 7: Full tree, lint, commit**

```bash
pixi run -e dev pytest tests -q
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add -A
git commit -m "feat(streaming): SVAR1 per-call FORMAT/dosage var_fields (PR-B3b, #304)"
```

---

# Phase 3 — PR-B4: `with_seqs("variant-windows")`

## Task 8: Rust `generate_variant_windows`

**Files:**
- Modify: `src/ffi/stream_core.rs:81-89` (add the trait method)
- Modify: `src/record_stream/engine.rs` (`RecordBackend` impl)
- Modify: `src/ffi/stream_engine.rs` (`Svar1Backend` impl)
- Test: `src/record_stream/engine.rs` (in-file tests)

**Interfaces:**
- Consumes: `assemble_windows_mode` (already exists, `src/variants/windows.rs:227`),
  `VariantsBatch` (Task 2), the REF table (Task 3).
- Produces:
  - `pub struct VariantWindowsBatch { pub scalars: VariantsBatch, pub tok_bufs: Vec<(String, TokBuf)> }`
  - `enum TokBuf { U8(Array1<u8>, Array1<i64>), I32(Array1<i32>, Array1<i64>) }`
  - `EngineBackend::generate_variant_windows(job_idx, slot, row_lo, row_hi) -> anyhow::Result<VariantWindowsBatch>`,
    defaulting to `anyhow::bail!("variant-windows output is not supported by this backend")`

- [ ] **Step 1: Write the failing test**

```rust
    /// PR-B4: variant-windows reuses `assemble_windows_mode` and emits one token
    /// buffer per configured side. ref="window"/alt="window" ⇒ ref_window + alt_window.
    #[test]
    fn generate_variant_windows_emits_both_window_buffers() {
        let (backend, slot) = variants_windows_fixture(1, 1, 2); // ref_mode, alt_mode, flank
        let batch = backend.generate_variant_windows(0, &slot, 0, 1).unwrap();
        let names: Vec<&str> = batch.tok_bufs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["ref_window", "alt_window"]);
        // Scalar fields ride along with the same row_offsets as plain variants output.
        assert_eq!(
            batch.scalars.row_offsets.len(),
            slot.geno_offsets.len(),
        );
    }

    /// ref="allele" emits the bare tokenized REF allele, which needs the REF table.
    #[test]
    fn generate_variant_windows_ref_allele_uses_ref_table() {
        let (backend, slot) = variants_windows_fixture(2, 1, 2);
        let batch = backend.generate_variant_windows(0, &slot, 0, 1).unwrap();
        let names: Vec<&str> = batch.tok_bufs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["ref", "alt_window"]);
    }
```

Write `variants_windows_fixture(ref_mode, alt_mode, flank)` next to `variants_fixture`,
building a `RecordBackend` configured with a u8 token LUT (`build_token_lut`'s output shape:
a 256-entry array) and a contig reference long enough to flank every variant.

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev cargo test --release generate_variant_windows
```

Expected: FAIL — `no method named generate_variant_windows`.

- [ ] **Step 3: Implement**

Add the default trait method to `EngineBackend`, then implement it on `RecordBackend` and
`Svar1Backend`. The body reuses the **existing** per-row CSR walk + region/AF keep from
`generate_variants` (factor that walk into a shared
`fn kept_v_idxs(&self, job, slot, row_lo, row_hi) -> (Array1<i32>, Array1<i64>)` so both
methods call it — DRY, and it guarantees the two output modes select identically), then:

```rust
        let n_v = v_idxs.len();
        let v_contigs = Array1::<i32>::zeros(n_v); // single-contig window by construction
        let c = &self.contigs[job.contig_idx];
        let reference = ArrayView1::from(c.ref_bytes.as_slice());
        let ref_offsets = Array1::from(vec![0i64, c.ref_bytes.len() as i64]);
        let bufs = crate::variants::windows::assemble_windows_mode(
            v_idxs.view(),
            row_offsets.view(),
            self.win_ref_mode,
            self.win_alt_mode,
            ArrayView1::from(slot.alt_alleles.as_slice()),
            ArrayView1::from(slot.alt_offsets.as_slice()),
            ref_table.as_ref().map(|(b, _)| ArrayView1::from(b.as_slice())),
            ref_table.as_ref().map(|(_, o)| ArrayView1::from(o.as_slice())),
            self.win_flank_len,
            self.token_lut.view(),
            v_contigs.view(),
            ArrayView1::from(slot.v_starts.as_slice()),
            ArrayView1::from(slot.ilens.as_slice()),
            reference,
            ref_offsets.view(),
            self.pad_char,
        );
```

`assemble_windows_mode` is generic over `Tok`; monomorphize on the LUT dtype the backend was
constructed with, mirroring how `assemble_variant_buffers_u8`/`_i32` split in
`src/ffi/mod.rs:533-607`.

- [ ] **Step 4: Run to verify pass**

```bash
pixi run -e dev cargo test --release
```

Expected: 0 failed.

- [ ] **Step 5: Commit**

```bash
git add src/
git commit -m "feat(streaming): generate_variant_windows via assemble_windows_mode (PR-B4, #304)"
```

---

## Task 9: FFI + Python `with_seqs("variant-windows")`

**Files:**
- Modify: `src/record_stream/engine.rs`, `src/ffi/stream_engine.rs` (`next_batch_variant_windows`)
- Modify: `python/genvarloader/_dataset/_streaming.py` (`with_seqs` ~989, `_iter_batches`,
  `build_engine`)
- Test: `tests/dataset/test_streaming_config.py`

**Interfaces:**
- Consumes: `VariantWindowsBatch` (Task 8).
- Produces:
  - `next_batch_variant_windows() -> dict[str, ndarray] | None` with keys `start`, `ilen`,
    `offsets`, plus `<name>`/`<name>_offsets` per token buffer
  - `StreamingDataset.with_seqs("variant-windows", opt: VarWindowOpt)`
  - `to_iter` yields `dict[str, Ragged]`, matching `_FlatVariantWindows.to_ragged()`

- [ ] **Step 1: Write the failing test**

```python
def test_with_seqs_variant_windows_returns_dict_of_ragged(streaming_case):
    from genvarloader._dataset._flat_variants import VarWindowOpt

    regions, reference, variants, _written = streaming_case("svar1")
    opt = VarWindowOpt(
        flank_length=4, token_alphabet=b"ACGT", unknown_token=4, ref="window", alt="window"
    )
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variant-windows", opt)
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=2)))
    assert isinstance(data, dict)
    assert {"ref_window", "alt_window", "start"} <= set(data)


def test_variant_windows_rejected_on_svar2(streaming_svar2_case):
    from genvarloader._dataset._flat_variants import VarWindowOpt

    regions, reference, variants = streaming_svar2_case
    opt = VarWindowOpt(
        flank_length=4, token_alphabet=b"ACGT", unknown_token=4
    )
    sds = gvl.StreamingDataset(regions, reference=reference, variants=variants)
    with pytest.raises(NotImplementedError):
        sds.with_seqs("variant-windows", opt)
```

- [ ] **Step 2: Run to verify failure**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_config.py -q -k variant_windows
```

Expected: FAIL — `with_seqs("variant-windows") is not implemented`.

- [ ] **Step 3: Add the FFI marshaler**

Mirroring `next_batch_variants`:

```rust
                dict.set_item("start", batch.scalars.start.into_pyarray(py))?;
                dict.set_item("ilen", batch.scalars.ilen.into_pyarray(py))?;
                dict.set_item("offsets", batch.scalars.row_offsets.into_pyarray(py))?;
                for (name, buf) in batch.tok_bufs {
                    match buf {
                        TokBuf::U8(d, o) => {
                            dict.set_item(format!("{name}_offsets"), o.into_pyarray(py))?;
                            dict.set_item(name, d.into_pyarray(py))?;
                        }
                        TokBuf::I32(d, o) => {
                            dict.set_item(format!("{name}_offsets"), o.into_pyarray(py))?;
                            dict.set_item(name, d.into_pyarray(py))?;
                        }
                    }
                }
```

- [ ] **Step 4: Extend `with_seqs`**

```python
    def with_seqs(
        self,
        kind: Literal["haplotypes", "annotated", "variants", "variant-windows"],
        opt: "VarWindowOpt | None" = None,
    ) -> "StreamingDataset":
```

For `"variant-windows"`: require `opt`, reject it for the SVAR2 backend, build the LUT with
`build_token_lut(opt.token_alphabet, opt.unknown_token)` exactly as `_impl.py` does, and
store `(lut, lut_dtype, opt)` for `build_engine`. Reject `opt` passed with any other kind.

- [ ] **Step 5: Pack the output**

In `_iter_batches`, add a `_variant_windows` branch that builds the dict:

```python
                        if _variant_windows:
                            b_times_p = (hi - lo) * backend.ploidy
                            row_off = np.asarray(nxt["offsets"], np.int64)
                            out = {}
                            for name in ("ref_window", "alt_window", "ref", "alt"):
                                if name not in nxt:
                                    continue
                                out[name] = Ragged.from_offsets(
                                    np.asarray(nxt[name]),
                                    (hi - lo, backend.ploidy, None, None),
                                    [row_off, np.asarray(nxt[f"{name}_offsets"], np.int64)],
                                )
                            out["start"] = Ragged.from_offsets(
                                np.asarray(nxt["start"], np.int32),
                                (b_times_p, None),
                                row_off,
                            ).reshape(hi - lo, backend.ploidy, None)
                            out["ilen"] = Ragged.from_offsets(
                                np.asarray(nxt["ilen"], np.int32),
                                (b_times_p, None),
                                row_off,
                            ).reshape(hi - lo, backend.ploidy, None)
```

This matches `_FlatWindow.to_ragged()`'s `(b, p, ~v, ~w)` two-ragged-axis shape
(`_flat_variants.py:200-222`) and `_Flat.to_ragged()`'s `(b, p, ~v)` for the scalars.

- [ ] **Step 6: Run to verify pass**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset/test_streaming_config.py -q -k variant_windows
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add src/ python/ tests/
git commit -m "feat(streaming): with_seqs(\"variant-windows\") surface + FFI (PR-B4, #304)"
```

---

## Task 10: PR-B4 parity suite + docs

**Files:**
- Modify: `tests/dataset/test_streaming_variants_parity.py`
- Modify: `docs/source/dataset.md`, `docs/source/faq.md`, `skills/genvarloader/SKILL.md`,
  `docs/roadmaps/streaming-dataset.md`

**Interfaces:**
- Consumes: everything from Tasks 8-9.
- Produces: the byte-identical gate for PR-B4.

- [ ] **Step 1: Write the parity test**

```python
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("ref_mode", ["window", "allele"])
@pytest.mark.parametrize("alt_mode", ["window", "allele"])
@pytest.mark.parametrize("alphabet,unk", [(b"ACGT", 4), (b"ACGTN", 5)])
def test_streaming_variant_windows_matches_written(
    streaming_case, backend, ref_mode, alt_mode, alphabet, unk
):
    """All four (ref, alt) mode combinations, both token dtypes, all three backends,
    byte-identical against the written `_FlatVariantWindows.to_ragged()` output."""
    from genvarloader._dataset._flat_variants import VarWindowOpt

    regions, reference, variants, written = streaming_case(backend)
    opt = VarWindowOpt(
        flank_length=4,
        token_alphabet=alphabet,
        unknown_token=unk,
        ref=ref_mode,
        alt=alt_mode,
    )
    ds = written.with_seqs("variant-windows", opt)
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variant-windows", opt)

    total = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            assert set(data) == set(expected), (
                f"field-set mismatch: streaming {sorted(data)} vs written {sorted(expected)}"
            )
            for name in data:
                for h in range(sds.ploidy):
                    got = np.asarray(data[name][k, h])
                    exp = np.asarray(expected[name][h])
                    assert got.dtype == exp.dtype, f"{name} dtype divergence"
                    np.testing.assert_array_equal(got, exp)
            total += np.atleast_1d(np.asarray(data["start"][k, 0])).shape[0]
    assert total > 0, "vacuous pass: no variants compared"


@pytest.mark.parametrize("backend", BACKENDS)
def test_variant_windows_empty_group_matches_written(empty_region_case, backend):
    """A (region, sample, ploid) cell with no in-window variant stays empty on both
    paths at the default `dummy_variant=None` — no sentinel fill on either side."""
    from genvarloader._dataset._flat_variants import VarWindowOpt

    regions, reference, variants, written = empty_region_case(backend)
    opt = VarWindowOpt(
        flank_length=4, token_alphabet=b"ACGT", unknown_token=4
    )
    ds = written.with_seqs("variant-windows", opt)
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variant-windows", opt)
    saw_empty = False
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            expected = ds[int(r_idx[k]), int(s_idx[k])]
            for h in range(sds.ploidy):
                got = np.atleast_1d(np.asarray(data["start"][k, h]))
                exp = np.atleast_1d(np.asarray(expected["start"][h]))
                np.testing.assert_array_equal(got, exp)
                if got.shape[0] == 0:
                    saw_empty = True
    assert saw_empty, "vacuous pass: fixture had no empty groups"
```

Add an `empty_region_case` fixture that includes at least one BED region positioned away
from every variant, so `saw_empty` is reachable.

- [ ] **Step 2: Run and iterate to green**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -q -k variant_windows
```

Expected: all pass (48 parametrized cases for the main test).

- [ ] **Step 3: Docs**

- `docs/source/dataset.md` — document `StreamingDataset.with_seqs("variant-windows", opt)`,
  its `dict[str, Ragged]` return, and that `unphased_union`/`dummy_variant` are not
  supported on the streaming path.
- `docs/source/faq.md` — same limitations.
- `skills/genvarloader/SKILL.md` — add the new `with_seqs` kind (CLAUDE.md requires this for
  any new accepted literal value).
- `docs/roadmaps/streaming-dataset.md` — mark PR-B4 done; Wave B and #304 complete.

- [ ] **Step 4: Full verification**

```bash
pixi run -e dev pytest tests -q
pixi run -e dev cargo test --release
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
pixi run -e dev python scripts/docstring_style.py --check python/genvarloader
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```

Expected: 0 failed, no lint errors, `MISSING: none`.

- [ ] **Step 5: Commit and open the PR**

```bash
git add -A
git commit -m "test(streaming): variant-windows byte-identical parity, all modes/backends (PR-B4, #304)"
git push -u origin spec/streaming-waveb-b3b4
gh pr create --draft --base streaming \
  --title 'feat(streaming): var_fields + variant-windows — Wave B PR-B3/PR-B4 (#304)' \
  --body 'Completes the streaming variants-output surface. Closes #304.'
```

---

## Self-Review Notes

**Spec coverage:** every spec section maps to a task — B3a public surface → Task 5; N staged
INFO columns → Tasks 1-2; REF bytes → Task 3; the INFO dtype risk → Task 6 Step 2 (the `DP`
canary, with the normalization fallback named); B3b SVAR1 FORMAT → Task 7; B3b VCF/PGEN
guard-parity → Task 7 Step 1's second test; B4 output type → Task 9 Step 5; B4 token LUT →
Task 9 Step 4; B4 Rust → Task 8; the guards section → Task 5 Steps 4-5 and Task 9 Step 4;
the testing plan → Tasks 6, 7, 10.

**Phase independence confirmed:** Task 7 (Phase 2) gathers per-call fields inline in
`Svar1Backend::generate_variants`'s existing walk, indexing by CSR position `o`. It does
**not** depend on Task 8's `kept_v_idxs` refactor, so Phases 2 and 3 are genuinely parallel.
If both run concurrently, the only overlap is `src/ffi/stream_engine.rs` — Task 7 edits
`generate_variants`, Task 8 adds `generate_variant_windows`, so the merge is additive.

**Type consistency check:** `InfoVals` / `InfoCol` (Task 1) are used by Tasks 2, 4, 7;
`VariantsBatch.info_out` (Task 2) is consumed by Tasks 4 and 7; `VariantsBatch.ref_data` /
`ref_seq_offsets` (Task 3) by Task 4; `assemble_variants_window`'s 6-argument form (Task 3)
is used by both `generate_variants` impls; `available_var_fields` (Task 5) is extended by
Task 7 and asserted in Task 6. `VariantWindowsBatch` / `TokBuf` (Task 8) are consumed only by
Task 9. No name appears with two spellings.
