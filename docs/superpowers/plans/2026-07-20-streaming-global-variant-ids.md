# Per-Variant Global Variant IDs (record-stream backends) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `StreamingDataset`'s VCF/PGEN record-stream backends emit **dataset-global** variant
ids per variant (not a scalar `var_base` + window-local index), so `with_seqs("annotated")` and (later)
`with_seqs("variants")` are byte-identical to the written `Dataset` even when the region-overlap
filter excludes interior variants behind a spanning deletion.

**Architecture:** genoray's `DenseChunk` gains a per-variant `global_idx` column (mirroring the SVAR2
`pack_vk_src` call-index). PGEN populates it from the `.pvar` absolute row index it already tracks;
gvl consumes it into a new `DecodedWindow.global_v_idxs` array and remaps annotation ids by
per-variant gather instead of adding a scalar `var_base`. VCF — which cannot self-compute the id —
is a separate, spike-gated phase because its parity hinges on an overlap-semantics question that
must be empirically resolved first.

**Tech Stack:** Rust (genoray crate `genoray_core`; gvl `src/`), PyO3, Python (`genvarloader`),
pytest, cargo test, maturin, pixi.

## Global Constraints

- **Byte-identical parity** vs `gvl.write()` + `Dataset[r, s]` under `with_seqs("annotated")` at
  `jitter=0` — the migration contract. New behavior must not regress existing parity tests.
- **genoray is a git dependency pinned by `rev`** (Cargo.toml: `git = "https://github.com/d-laub/genoray.git", rev = "…"`). A genoray change ships by merging to genoray `main`, then bumping both `svar2-codec` and `genoray_core` `rev` to the same commit (CLAUDE.md → Development Notes).
- **Rebuild Rust before pytest**: after editing anything under `src/` (or bumping the genoray rev),
  run `pixi run -e dev maturin develop --release` or pytest imports the **stale** compiled
  extension. `cargo test` compiles from source and is unaffected.
- **Rust tests need `LD_LIBRARY_PATH`** to `.pixi/envs/dev/lib` to load libpython (see the
  `cargo-test` pixi task; do not invoke `cargo test` bare).
- **Full tree before pushing** shared-code/symbol changes: `pixi run -e dev pytest tests -q` (scoped
  runs skip `tests/unit/`).
- **1:1 record↔atom contract**: gvl forces bcftools atomization (`bcftools norm -f ref -a -m -any`)
  before `gvl.write`, and `_reject_unsupported_variants` rejects multi-allelic/symbolic at write
  (`_write.py:655`). The canonical global id is therefore the file-order record rank per contig
  (`with_row_index` over the `.gvi`/`.pvar`, `_var_ranges.py:132`). Do NOT design for multi-atom
  records; assert the invariant instead.
- **Target branch:** `streaming`. genoray change is a cross-repo PR to `d-laub/genoray` `main`.

## Phasing & parallelism

- **Phase 1 (genoray PR)** and **Phase 2 (gvl consume, PGEN+SVAR1)** are sequential: Phase 2 needs
  the genoray rev merged + bumped. Within Phase 1, Tasks 1.1/1.2 are sequential (1.2 uses 1.1's
  field). Within Phase 2, Task 2.4's two fixtures are parallelizable.
- **Phase 3 (VCF)** is gated on Task 3.1 (a verification spike) and may land as a **separate gvl PR**
  after Phase 2 — do not block PGEN/SVAR1 correctness on it.
- Per repo convention: dispatch independent implementation tasks via
  `superpowers:dispatching-parallel-agents` + `superpowers:subagent-driven-development`, using
  **Sonnet or weaker** for implementation.

---

## PHASE 1 — genoray: `DenseChunk.global_idx` carrier + PGEN population

> Cross-repo PR against `d-laub/genoray` `main`. All paths in this phase are in the genoray repo.
> Use the pinned-rev checkout as the working reference:
> `~/.cargo/git/checkouts/genoray-26e7da4241f8ed6f/1d756ad/` — but implement against a fresh clone of
> genoray `main` (bump target). Confirm every file:line against the actual `main` source; the line
> numbers below are from rev `1d756ad` and may drift.

### Task 1.1: Add the `global_idx` per-variant column to `DenseChunk` and thread it through assembly

**Files:**
- Modify: `src/types.rs` (`DenseChunk` struct, ~`:137-179`)
- Modify: `src/record_source.rs` (`RawRecord` struct, ~`:13-33`)
- Modify: `src/chunk_assembler.rs` (`PendingAtom` ~`:15-37`; `AtomMeta` ~`:183-195`;
  `decompose_raw_record` ~`:349-358`; `flush_window` ~`:256-264`; metadata loop ~`:644-715`)
- Test: `src/chunk_assembler.rs` `#[cfg(test)] mod tests` (extend
  `decompose_raw_record_threads_each_atoms_own_alt_index_to_dense_format` ~`:841-886`)

**Interfaces:**
- Produces: `DenseChunk.global_idx: Vec<i32>` — one entry per surviving variant, parallel to
  `pos`/`ilens`/`alt_offsets`, holding that variant's dataset-global id. `RawRecord.global_idx: i32`
  — the source-supplied global id for a record (default/sentinel `-1` when the source does not set
  it, so existing non-record-stream callers compile).

- [ ] **Step 1: Write the failing test** — extend the existing atomization test to assert each atom
  carries the record's `global_idx`. In `src/chunk_assembler.rs` tests, add:

```rust
#[test]
fn decompose_threads_record_global_idx_onto_each_atom() {
    // A single biallelic SNP record tagged with global id 7 must yield one
    // PendingAtom whose global_idx == 7 (the 1:1 record<->atom contract).
    let mut pending: Vec<PendingAtom> = Vec::new();
    let rec = RawRecord {
        // ...mirror the existing decompose test's RawRecord construction...
        global_idx: 7,
        ..raw_record_snp_fixture()   // reuse the existing fixture helper
    };
    let dropped = decompose_raw_record(
        rec.pos, &rec.ref_allele, &rec.alts, &mut pending, /*skip_out_of_scope*/ false,
    ).unwrap();
    assert_eq!(dropped, 0);
    assert_eq!(pending.len(), 1, "biallelic SNP must atomize 1:1");
    assert_eq!(pending[0].global_idx, 7);
}
```

Note: match the existing test's exact `decompose_raw_record` signature and fixture helper (read
`:841-886` first; if the helper doesn't exist, inline a minimal `RawRecord` as that test does).

- [ ] **Step 2: Run it to verify it fails**

Run: `cd <genoray> && LD_LIBRARY_PATH=… cargo test -p genoray decompose_threads_record_global_idx -- --nocapture`
Expected: FAIL — `global_idx` is not a field of `RawRecord`/`PendingAtom` (compile error).

- [ ] **Step 3: Add the fields and plumbing (minimal)**

- `src/types.rs`: add `pub global_idx: Vec<i32>,` to `DenseChunk` (place it next to `pos`).
- `src/record_source.rs`: add `pub global_idx: i32,` to `RawRecord`; set it to `-1` at every
  existing construction site (grep `RawRecord {` across the crate) except the record-stream sources
  (Task 1.2 sets those).
- `src/chunk_assembler.rs`:
  - `PendingAtom`: add `pub global_idx: i32,`.
  - `decompose_raw_record`: it currently takes `(pos, ref_allele, alts, pending, skip_out_of_scope)`.
    Thread the record's `global_idx` in (add a param, OR pass the whole `RawRecord`) and set
    `global_idx: rec.global_idx` in the `pending.push(PendingAtom { … })` at `:349-358`. Add
    `debug_assert!(atoms.len() == 1 || skip_out_of_scope, "1:1 record↔atom contract");` after
    atomization to make a violated invariant loud.
  - `AtomMeta`: add `global_idx: i32`; copy it from the pending atom in `flush_window` (`:256-264`).
  - Metadata loop (`:644-675`): declare `let mut global_idx: Vec<i32> = Vec::with_capacity(v);` and
    add `global_idx.push(a.global_idx);` next to `pos.push(a.pos);`.
  - `DenseChunk { … }` construction (`:704-715`): add `global_idx,`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p genoray decompose_threads_record_global_idx`
Expected: PASS.

- [ ] **Step 5: Run the full genoray suite (nothing regressed)**

Run: `LD_LIBRARY_PATH=… cargo test -p genoray`
Expected: PASS (the new field defaults to `-1`; existing assertions on `DenseChunk` fields are
unaffected — they don't check `global_idx`).

- [ ] **Step 6: Commit**

```bash
git add src/types.rs src/record_source.rs src/chunk_assembler.rs
git commit -m "feat(chunk): carry per-variant global_idx through DenseChunk assembly"
```

### Task 1.2: PGEN record source populates `global_idx` from the `.pvar` absolute row index

**Files:**
- Modify: `src/pvar.rs` (`PvarReader`, expose `vidx` ~`:33`, `:191`)
- Modify: `src/pgen_reader.rs` (`PgenRecordSource::next_record` ~`:94-207`, esp. after
  `next_variant()` at ~`:154`)
- Test: `src/pgen_reader.rs` tests (mirror `src/pvar.rs:282` `var_start_skips_leading_variants`)

**Interfaces:**
- Consumes: `RawRecord.global_idx` (Task 1.1).
- Produces: PGEN survivors carry `global_idx == .pvar row index == var_start + offset`, correct
  across region-overlap gaps (the row counter advances for skipped candidates too).

- [ ] **Step 1: Write the failing test** — a narrowed PGEN scan where the extent filter skips a
  leading candidate must still tag survivors with their true (gapped) `.pvar` row indices.

```rust
#[test]
fn pgen_source_tags_survivors_with_true_pvar_row_index() {
    // Fixture .pvar/.pgen with variants at rows 0..=4 on one contig. Query a
    // region that (via extent-overlap) keeps rows {2,3,4} and drops {0,1}.
    // Assert next_record() yields RawRecords with global_idx 2,3,4 in order.
    let mut src = pgen_source_fixture(/*var_start*/ 0, /*var_end*/ 5, /*region*/ (region_lo, region_hi));
    let mut ids = Vec::new();
    while let Some(rec) = src.next_record().unwrap() { ids.push(rec.global_idx); }
    assert_eq!(ids, vec![2, 3, 4]);
}
```

Use the existing PGEN test fixtures/helpers in `src/pgen_reader.rs` tests; if none expose a narrowed
region, construct one mirroring `pvar.rs`'s fixture builder.

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p genoray pgen_source_tags_survivors_with_true_pvar_row_index`
Expected: FAIL — `global_idx` is `-1` (default from Task 1.1), not the row index.

- [ ] **Step 3: Implement** — expose and read the `.pvar` absolute row index.

- `src/pvar.rs`: add `pub fn current_vidx(&self) -> usize { self.vidx }` (`vidx` is the pre-increment
  absolute row index initialized by `var_start` skipping at `:118-126` and incremented per
  `next_variant` at `:191`).
- `src/pgen_reader.rs` `next_record`: capture `let gidx = self.pvar.current_vidx();` immediately
  after the `next_variant()` call (`:154`), **before** the region-overlap exclusion block
  (`:164-207`), and set `rec.global_idx = gidx as i32` on the emitted `RawRecord`. Because the
  exclusion happens after `next_variant`, `vidx` has already advanced for dropped candidates — each
  survivor keeps its own true row index.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p genoray pgen_source_tags_survivors_with_true_pvar_row_index`
Expected: PASS.

- [ ] **Step 5: Full genoray suite + commit**

Run: `LD_LIBRARY_PATH=… cargo test -p genoray` → PASS.

```bash
git add src/pvar.rs src/pgen_reader.rs
git commit -m "feat(pgen): tag record-stream survivors with true .pvar global row index"
```

- [ ] **Step 6: Open the genoray PR, merge to `main`, note the merge commit SHA** (needed for the
  Phase 2 rev bump). PR title: `feat: per-variant global_idx on DenseChunk (PGEN populated)`.

---

## PHASE 2 — gvl: consume `global_idx`, retire scalar `var_base` (PGEN + SVAR1)

> All paths in this phase are in the gvl repo/worktree.

### Task 2.1: Bump the genoray rev

**Files:** Modify `Cargo.toml` (both `svar2-codec` and `genoray_core` `rev`).

- [ ] **Step 1:** Set both `rev` values to the Phase 1 merge SHA (they must share one rev).
- [ ] **Step 2:** Rebuild: `pixi run -e dev maturin develop --release`. Expected: builds clean;
  `DenseChunk` now exposes `global_idx`.
- [ ] **Step 3:** Commit: `git add Cargo.toml Cargo.lock && git commit -m "build(deps): bump genoray rev for DenseChunk.global_idx"`

### Task 2.2: `DecodedWindow.global_v_idxs` + `fill_decoded_window` copies it

**Files:**
- Modify: `src/record_stream/transpose.rs` (`DecodedWindow` `:29-46`; `fill_decoded_window` `:51-116`)
- Test: `src/record_stream/transpose.rs` tests (extend `transpose_emits_variant_indices_hap_major`)

**Interfaces:**
- Produces: `DecodedWindow.global_v_idxs: Vec<i32>` — length `n_var` (one per window-local variant
  column), `global_v_idxs[local] == dataset-global id`. Filled from `chunk.global_idx`.

- [ ] **Step 1: Write the failing test** — extend the transpose fixture to carry `global_idx` and
  assert it copies through. Add `global_idx: vec![5, 8]` to the `dense_fixture()` `DenseChunk` (2
  variants) and assert:

```rust
assert_eq!(slot.global_v_idxs, vec![5, 8]);
```

(Also add `global_idx: Vec::new()` / `global_idx: vec![…]` to every `DenseChunk { … }` literal in the
transpose tests and in `vcf.rs`/`pgen.rs` empty-window construction so the crate compiles.)

- [ ] **Step 2: Run it to verify it fails**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib cargo test -p genvarloader transpose_emits_variant_indices_hap_major`
Expected: FAIL — `global_v_idxs` is not a field.

- [ ] **Step 3: Implement**

- `DecodedWindow`: add `pub global_v_idxs: Vec<i32>,`. **Remove `pub var_base: i64,`** (retired) —
  or, to stage safely, keep it unused for this task and delete in Task 2.5. Recommended: remove now
  and fix the fallout in 2.3/2.4 within this phase.
- `fill_decoded_window`: add `slot.global_v_idxs.clear(); slot.global_v_idxs.extend_from_slice(&chunk.global_idx);`
  next to the `v_starts` copy. Update the empty-`DenseChunk` literals (vcf.rs/pgen.rs) to include
  `global_idx: Vec::new()`.

- [ ] **Step 4: Run the test to verify it passes** → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/record_stream/transpose.rs src/record_stream/vcf.rs src/record_stream/pgen.rs
git commit -m "feat(streaming): carry per-variant global_v_idxs on DecodedWindow"
```

### Task 2.3: `generate_batch_core` remaps annotation ids by per-variant gather (not `var_base` add)

**Files:**
- Modify: `src/ffi/mod.rs` (`generate_batch_core` signature + the `var_base` add-back block
  `:1113-1121`; the two fused wrappers that pass `var_base`)
- Modify: `src/record_stream/engine.rs` (`RecordBackend::generate` `:199-217` — pass
  `slot.global_v_idxs` instead of `slot.var_base`)
- Modify: `src/ffi/stream_engine.rs` (SVAR1 `generate` — passes `var_base=0` at `:220`; SVAR1's
  `geno_v_idxs` are already global, so pass `None`)
- Test: an `src/ffi` or `src/record_stream/engine.rs` Rust test asserting the gather remap

**Interfaces:**
- Consumes: `DecodedWindow.global_v_idxs` (2.2).
- `generate_batch_core(…, global_v_idxs: Option<ndarray::ArrayView1<i32>>, parallel)` — replaces the
  `var_base: i64` parameter. When `Some`, after reconstruction each non-negative `annot_v_idxs`
  entry (a window-LOCAL column index) is remapped: `annot_v[i] = global_v_idxs[annot_v[i]]`. When
  `None` (SVAR1, whose `geno_v_idxs` are already global), `annot_v` is left as-is.

- [ ] **Step 1: Write the failing test** — a record-path `generate` over a window whose
  `global_v_idxs = [0, 2]` (an interior variant excluded upstream) must emit annotation ids `{0, 2}`,
  not `{0, 1}`. Build a `RecordBackend` (or call `generate_batch_core` directly) with a 2-variant
  window where both haps carry both variants, `global_v_idxs = [0, 2]`, and assert the returned
  `annot_v_idxs` contains `2` (never `1`).

```rust
#[test]
fn generate_remaps_local_annot_ids_to_global_via_gather() {
    // window: 2 local variants, global_v_idxs = [0, 2]; a hap carrying both
    // must annotate with global ids 0 and 2 (the gapped set), never 1.
    let (data, annot_v, _pos, _off) = generate_batch_core(
        /* … 2-variant fixture … */,
        Some(ndarray::ArrayView1::from(&[0i32, 2])), // global_v_idxs
        /*parallel*/ false,
    );
    let ids: Vec<i32> = annot_v.unwrap().iter().copied().filter(|&x| x >= 0).collect();
    assert!(ids.contains(&2) && !ids.contains(&1), "got {ids:?}");
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib cargo test -p genvarloader generate_remaps_local_annot_ids_to_global`
Expected: FAIL — signature still takes `var_base: i64` (compile error), or ids are `{0,1}`.

- [ ] **Step 3: Implement**

- `generate_batch_core`: change the `var_base: i64` param to `global_v_idxs: Option<ArrayView1<i32>>`.
  Replace the `:1113-1121` block with:

```rust
if let Some(gv) = global_v_idxs {
    if let Some(av) = annot_v.as_mut() {
        for x in av.iter_mut() {
            if *x >= 0 {
                *x = gv[*x as usize]; // window-local col -> dataset-global id
            }
        }
    }
}
```

- `RecordBackend::generate` (`engine.rs:215`): pass
  `Some(ndarray::ArrayView1::from(slot.global_v_idxs.as_slice()))`.
- SVAR1 `generate` (`stream_engine.rs:220`) and the two fused plain wrappers (`ffi/mod.rs:668,843`,
  `:1194-1195`): pass `None` (SVAR1 `geno_v_idxs` are already global; the fused plain path never
  emits annotation).

- [ ] **Step 4: Run the test to verify it passes** → PASS.

- [ ] **Step 5: Rebuild + run the existing annotated parity test (must still pass for SVAR1/whole-contig)**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_annotated_parity.py::test_annotated_matches_written -q`
Expected: PASS for all three backends (whole-contig fixtures were already correct; the remap is a
no-op there since `global_v_idxs == 0..n`).

- [ ] **Step 6: Commit**

```bash
git add src/ffi/mod.rs src/record_stream/engine.rs src/ffi/stream_engine.rs
git commit -m "feat(streaming): remap annot var_idxs via per-variant global gather; retire var_base"
```

### Task 2.4: PGEN parity — flip the xfail + add an interior-exclusion fixture

**Files:**
- Modify: `tests/dataset/test_streaming_annotated_parity.py`
  (`test_annotated_pgen_narrowed_window_var_idxs_gap` `:125-175` — remove the `xfail` marker)
- Create: an interior-exclusion PGEN fixture + test in the same file
- Test data: reuse `pgen_snp_ins_del_multi` (conftest) and/or add a spanning-deletion fixture in
  `tests/dataset/conftest.py`

**Interfaces:**
- Consumes: the corrected global-id path (2.1–2.3).

- [ ] **Step 1: Un-xfail the existing narrowed-window gap test.** Delete the `@pytest.mark.xfail(…)`
  decorator on `test_annotated_pgen_narrowed_window_var_idxs_gap` (`:125-131`) and update its
  docstring to state it is now a passing regression lock (the narrowed region `[71,200)` first-kept
  variant is global id 2; the streamed `var_idxs` must be `[2,3,4]`).

- [ ] **Step 2: Run it to verify it now PASSES**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_annotated_parity.py::test_annotated_pgen_narrowed_window_var_idxs_gap -q`
Expected: PASS (previously `xfail`). If it still fails, the global-id plumbing is wrong — debug 2.2/2.3.

- [ ] **Step 3: Write a failing interior-exclusion test.** Add a fixture (in `conftest.py`) whose
  variants include a spanning deletion that, for a chosen region, is kept while an interior variant
  behind it is dropped — so the written oracle's `var_idxs` are a **gapped** set (e.g. `{0, 2}`).
  Add `test_annotated_pgen_interior_exclusion_var_idxs_gapped` asserting streamed `var_idxs` equal
  the written `var_idxs` (which will contain the gap):

```python
def test_annotated_pgen_interior_exclusion_var_idxs_gapped(pgen_spanning_del, tmp_path):
    ds, sds = _interior_exclusion_case(pgen_spanning_del, tmp_path)  # region keeps DEL, drops interior SNP
    saw_gap = False
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for row in range(len(r_idx)):
            exp = ds[int(r_idx[row]), int(s_idx[row])]
            for h in range(sds.ploidy):
                got = np.asarray(data.var_idxs[row][h]); want = np.asarray(exp.var_idxs[h])
                np.testing.assert_array_equal(got, want)
                v = want[want >= 0]
                if v.size >= 2 and np.any(np.diff(v) > 1):
                    saw_gap = True
    assert saw_gap, "fixture must exercise a non-contiguous (gapped) global var_idxs set"
```

Fixture note: a spanning deletion `REF=ACGTACGT` at pos P (extent P..P+8) plus an interior SNP at
P+3, with a query region `[P+5, P+9)` — the DEL's extent reaches the region (kept), the interior SNP
does not (dropped). Confirm the written `Dataset` produces a gapped `var_idxs` for this cell first
(that's the oracle); the `saw_gap` assert guards against a fixture that doesn't actually gap.

- [ ] **Step 4: Run it to verify it passes** → PASS. If the streamed set is contiguous while the
  oracle is gapped, 2.2/2.3 are wrong.

- [ ] **Step 5: Full annotated parity suite + commit**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_annotated_parity.py -q` → PASS.

```bash
git add tests/dataset/test_streaming_annotated_parity.py tests/dataset/conftest.py
git commit -m "test(streaming): PGEN annotated var_idxs global + interior-exclusion gap (#305)"
```

### Task 2.5: Retire `var_base` residue + close-out

**Files:** `src/record_stream/vcf.rs` (`slot.var_base = 0` + its doc comment `:153-170`),
`src/record_stream/pgen.rs` (`slot.var_base = 0` + doc `:595-614`), `src/record_stream/transpose.rs`
(if `var_base` not already removed in 2.2), any remaining `var_base` references.

- [ ] **Step 1:** grep `var_base` across `src/` and `python/`; remove the now-dead scalar and its doc
  comments. VCF's `fill` sets `global_v_idxs` per Task 3 (until then it decodes with genoray's
  default `-1` — see Phase 3; PGEN's `fill` needs no per-variant work since genoray populates it).
- [ ] **Step 2:** Rebuild + full tree: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests -q`. Expected: PASS.
- [ ] **Step 3:** Commit: `git commit -am "refactor(streaming): remove retired scalar var_base"`
- [ ] **Step 4:** Update `#311` (mark the annotated interior-exclusion bug fixed by this PR) and
  `docs/roadmaps/streaming-dataset.md` (Plan 5 status). PGEN + SVAR1 annotated global ids are now
  correct; VCF remains behind Phase 3.

---

## PHASE 3 — VCF global ids (spike-gated; may be a separate gvl PR)

> **Do not start Phase 3 concretely until Task 3.1 resolves the overlap-semantics question.** The
> VCF global-id VALUE must come from gvl's `.gvi` oracle (`vcf._var_idxs`), threaded so it aligns
> positionally with the Rust survivors. That alignment is only valid if the Rust `extent_overlaps`
> kept-set equals the Python `var_indices` kept-set — which is UNVERIFIED and shows a credible
> mismatch at deletion anchor bases (Rust anchor-trimmed `[pos+1, …)` vs Python `[POS-1, …)`).

### Task 3.1 (SPIKE): empirically compare Rust `extent_overlaps` vs Python `var_indices` kept-sets

**Deliverable:** a differential test over deletion-boundary fixtures that either (a) proves the two
kept-sets are identical across SNP/INS/DEL/spanning-DEL at every region-boundary offset, or (b)
characterizes the exact divergence.

- [ ] **Step 1:** Build a small VCF fixture with a deletion `ACGT>A` and neighbors; for a sweep of
  query regions around the deletion's anchor/extent boundaries, compute:
  - Python: `vcf._var_idxs(contig, [start], [end])` (the write-path oracle, `_var_ranges.py`).
  - Rust: the survivors the `VcfWindowFiller` decodes for the same region (`debug_decode_window`).
- [ ] **Step 2:** Assert set-equality per region. Record every divergence (expected at the anchor
  base per the `_var_end_expr` vs `extent_overlaps` analysis).
- [ ] **Step 3: Decision fork based on the result:**
  - **If identical:** proceed to Task 3.2 (thread `vcf._var_idxs` ids through, aligned positionally).
  - **If they diverge:** the fix is to make the two overlap rules agree first — either align the
    streaming `OverlapMode` to `_var_end_expr` (a genoray/gvl overlap-mode choice) or fix
    `var_indices`/`_var_end_expr` to anchor-trim (a written-path change, coordinate with #202 since
    it also concerns variant-region overlap). This is a design decision to surface, not code to
    guess. **Stop and report the divergence + recommended reconciliation before writing 3.2.**

### Task 3.2 (conditional): supply VCF global ids from `vcf._var_idxs`, carried on `global_idx`

> Concrete steps depend on 3.1's outcome; sketch only. Precompute, per window/job at
> `VcfWindowFiller` construction (the traversal plan is known), the per-region global-id array via
> `vcf._var_idxs(contig, regions)`; thread it to the Rust `VcfRecordSource` as a caller-supplied
> id stream consumed in emission order (so alignment is by construction), populating `RawRecord.global_idx`.
> Then `fill_decoded_window` carries it to `global_v_idxs` exactly as PGEN does. Add VCF interior-
> exclusion + multi-contig annotated parity fixtures mirroring Task 2.4. Retire VCF's `-1` default.

---

## Self-Review notes (author)

- **Spec coverage:** #305 (global ids) — Phases 1–3; latent Wave A annotated bug (#311) — fixed by
  Phase 2 for PGEN/SVAR1, Phase 3 for VCF. `var_base` retirement — Task 2.5.
- **Confidence:** Phases 1–2 (genoray carrier + PGEN + gvl consume) are high-confidence and concrete.
  Phase 3 (VCF) is deliberately spike-gated — the overlap-semantics mismatch is real and unverified,
  and writing its code before Task 3.1 would be guessing.
- **Follow-on plans (separate docs):** PR-B0 (#202 clip), PR-B1 (variants output), PR-B2/B3/B4 per
  the spec's PR stack — each gets its own plan once this foundation lands.
