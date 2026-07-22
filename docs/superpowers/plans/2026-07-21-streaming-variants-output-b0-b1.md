# Streaming variants output (Wave B, PR-B0 + PR-B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the write-free `StreamingDataset` a byte-identical `with_seqs("variants")` output (`RaggedVariants`, default fields `alt`/`start`/`ilen`), and first correct the written-path oracle so its variants are clipped to the region window (#202).

**Architecture:** Two PRs on one branch. **PR-B0** (#202) threads a region-extent-overlap filter into the written `get_variants_flat` so `with_seqs("variants")` no longer leaks variants outside the queried window — the parity oracle must be correct before streaming asserts against it. **PR-B1** adds a variants consumer to the streaming `RecordBackend` (VCF/PGEN) and the SVAR1 backend: for each `(region, sample, ploid)` it gathers window-local variant indices from the per-hap CSR, applies the **same** region-overlap clip, and feeds the **existing** `assemble_variant_buffers_*` Rust kernel plus scalar gathers — the same kernel the written path uses, so the two paths converge on one assembly. Per #313, variants output is self-contained from the window table (`alt`/`start`/`ilen` values only) and needs **no** dataset-global variant id.

**Tech Stack:** Rust (PyO3, ndarray) for the engine/kernel; Python (numpy, `seqpro.rag.Ragged`) for the surface; pytest byte-identical parity vs `gvl.write()` + `Dataset[r, s]`; `maturin develop --release` build; `pixi run -e dev` for all tasks.

## Global Constraints

- **Branch/target:** all work targets the long-lived `streaming` integration branch (Wave A already merged via #308). Do **not** target `main`. Work in the worktree `.claude/worktrees/streaming-waveb-b0b1` (branch `streaming-waveb-b0b1`, based on `origin/streaming`).
- **Parity oracle is byte-identical** vs `gvl.write()` + `Dataset.open()[r, s]` (modulo jitter/rng — not exercised here). PR-B1's oracle is the **corrected** (post-PR-B0) written `with_seqs("variants")`.
- **Rebuild Rust before pytest:** after any edit under `src/`, run `pixi run -e dev maturin develop --release` or pytest silently imports the stale extension. `cargo test` compiles from source and is unaffected.
- **Variants output carries NO global variant id** (#313 correction): `RaggedVariants`/`_FlatVariants` expose field *values* (`alt`/`start`/`ilen`/`dosage`), never a `v_idxs` field. Do not thread `global_v_idxs` into the variants path.
- **Region-overlap predicate (exact, single source of truth for both PRs):** a variant with genomic start `v_start` (`ffi_static.v_starts` / `DecodedWindow.v_starts`) and indel length `v_ilen` (`ffi_static.ilens` / `DecodedWindow.ilens`) has reference extent `[v_start, v_end)` where `v_end = v_start - min(v_ilen, 0) + 1` (matches `src/reconstruct/mod.rs:705`). It **overlaps** a region window `[r_start, r_end)` iff `v_start < r_end && v_end > r_start`. This includes upstream deletions that span into the window (consistent with how the linear reconstruction applies them, `src/reconstruct/mod.rs:96-193`), so it is a superset-safe match for "variants that affect this region's haplotype".
- **Conventional commits** (commitizen). Prefix streaming commits with the affected scope, e.g. `fix(streaming):`, `feat(streaming):`, `test(streaming):`.
- **Public-API + docs gates (CLAUDE.md):** PR-B1 adds a public `with_seqs` value, so `skills/genvarloader/SKILL.md`, `docs/source/api.md`, and the relevant `docs/source/*.md` must be updated in the same PR (Task 5). `api.md` must stay in sync with `__all__`. Update `docs/roadmaps/streaming-dataset.md` (Plan 5) status markers and the StreamingDataset project board as each PR lands.
- **Lint/type gates before each PR:** `pixi run -e dev ruff check python/ tests/`, `pixi run -e dev ruff format python/ tests/`, `pixi run -e dev typecheck`, and (shared-code change) the full `pixi run -e dev pytest tests -q` sweep.

---

## File Structure

**PR-B0 (written-path #202 fix):**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` — `get_variants_flat` gains the region-overlap keep mask (folded into the existing AF `keep`/`_compact_keep` block).
- Test: `tests/dataset/test_variants_region_clip.py` (new) — written-path clip regression + post-condition invariant.

**PR-B1 (streaming variants output):**
- Modify: `src/record_stream/engine.rs` — `RecordBackend` gains a variants generate path + `RecordStreamEngine::next_batch_variants`; `new`/`new_rs` gains a `variants: bool` flag.
- Modify: `src/record_stream/mod.rs` and/or `src/variants/mod.rs` — a shared `assemble_variants_window` helper (window CSR → flat variant buffers), reused by both backends.
- Modify: `src/ffi/stream_engine.rs` — `Svar1Backend` variants generate path + `Svar1StreamEngine::next_batch_variants` + `variants: bool` flag.
- Modify: `src/lib.rs` — no new registration expected (methods hang off already-registered engine classes); confirm.
- Modify: `python/genvarloader/_dataset/_streaming.py` — `with_seqs` accepts `"variants"`; `_iter_batches` variants branch; `build_engine` threads `variants`.
- Test: `tests/dataset/test_streaming_variants_parity.py` (new) — multi-backend byte-identical parity.
- Docs: `skills/genvarloader/SKILL.md`, `docs/source/api.md`, `docs/source/dataset.md` (or `write.md` as apt), `docs/roadmaps/streaming-dataset.md`.

**Parallelism:** Task 1 (PR-B0) is independent and can run concurrently with Task 2's start. Task 3 depends on Task 2; Task 4 depends on Task 3 (reuses the Python `_iter_batches` variants wiring); Task 5 is last. Suggested execution: **Task 1 ∥ Task 2 → Task 3 → Task 4 → Task 5.**

---

## Task 1: PR-B0 — region-overlap clip in the written variants path (#202)

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py:808-833` (inside `get_variants_flat`)
- Test: `tests/dataset/test_variants_region_clip.py` (create)

**Interfaces:**
- Consumes: `get_variants_flat(haps, idx, regions=None)` — `regions` is `(b, 3)` int `(contig_idx, start, end)` aligned with `idx`, or `None` (no clip). `haps.ffi_static.v_starts` (i32, indexed by variant idx), `haps.ffi_static.ilens` (i32). `v_idxs` (per-`(b*ploidy)`-row variant indices), `unfiltered_row_offsets` (length `b*ploidy + 1`).
- Produces: a `get_variants_flat` whose returned variants all overlap their cell's region window per the Global-Constraints predicate. No signature change.

- [ ] **Step 1: Write the failing test — every returned variant overlaps its window**

Create `tests/dataset/test_variants_region_clip.py`. Request `ilen` so the extent predicate can be asserted directly. This is a post-condition invariant that fails today because PGEN's genotype query is contig-scoped and leaks out-of-window variants (#202).

```python
"""#202: written `with_seqs("variants")` must clip variants to the region window."""

import numpy as np
import pytest

import genvarloader as gvl


def test_written_variants_are_clipped_to_window(pgen_snp_ins_del_multi, tmp_path):
    """Every returned variant's extent overlaps its cell's region window (#202)."""
    f = pgen_snp_ins_del_multi
    gvl.write(tmp_path / "ds", f.regions, variants=str(f.pgen))
    ds = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta).with_seqs(
        "variants", var_fields=["alt", "ilen", "start"]
    )

    regions = np.asarray(f.regions.select(["chromStart", "chromEnd"]).to_numpy())
    n_regions, n_samples = ds.shape

    for r in range(n_regions):
        r_start, r_end = int(regions[r, 0]), int(regions[r, 1])
        for s in range(n_samples):
            rv = ds[r, s]
            for h in range(ds.ploidy):
                starts = np.asarray(rv.start[h]).astype(np.int64)
                ilens = np.asarray(rv.ilen[h]).astype(np.int64)
                v_end = starts - np.minimum(ilens, 0) + 1  # matches reconstruct/mod.rs:705
                overlaps = (starts < r_end) & (v_end > r_start)
                assert overlaps.all(), (
                    f"cell ({r},{s}) hap {h}: variant outside window "
                    f"[{r_start},{r_end}); starts={starts.tolist()}"
                )
```

(Confirm the `f.regions` column names against the `pgen_snp_ins_del_multi` fixture in `tests/dataset/conftest.py`; if the BED columns are named differently, select the start/end columns accordingly.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_variants_region_clip.py -v`
Expected: FAIL — at least one cell returns a variant whose extent does not overlap its window (the #202 leak; PGEN's genotype query is contig-scoped, so out-of-window variants appear).

- [ ] **Step 3: Implement the region-overlap keep mask in `get_variants_flat`**

In `python/genvarloader/_dataset/_flat_variants.py`, insert the region-overlap mask **after** the AF `keep` block (line 817) and **before** the dosage gather + compaction (line 819+), so the existing `_compact_keep` at lines 830-833 applies the combined mask to `v_idxs`/`row_offsets`/`dosage` in one pass. Do NOT add a second compaction.

```python
    # AF filtering (mirrors _get_variants) ... [existing lines 810-817] ...

    # #202 region-extent-overlap clip. The written variants path previously gathered
    # v_idxs straight from the sparse genotype set and never filtered by region extent,
    # so RaggedVariants leaked variants outside the queried window (boundary-overlapping
    # indels; for PGEN, other variants on the same contig). Clip to the same set the
    # haplotype reconstruction uses: a variant [v_start, v_end) is kept iff it overlaps
    # the region window [r_start, r_end), with v_end = v_start - min(ilen, 0) + 1
    # (matches src/reconstruct/mod.rs). regions is None on the no-region call path
    # (_haps.py get_variants_flat(self, idx)); skip the clip there.
    if regions is not None:
        regions_arr = np.asarray(regions)
        # Per (b*ploidy) row region extents (C-order: b cells, each ploidy rows).
        row_r_start = np.repeat(regions_arr[:, 1].astype(np.int64), ploidy)
        row_r_end = np.repeat(regions_arr[:, 2].astype(np.int64), ploidy)
        # Broadcast per variant via the per-row variant counts.
        counts = np.diff(unfiltered_row_offsets)
        v_r_start = np.repeat(row_r_start, counts)
        v_r_end = np.repeat(row_r_end, counts)
        v_start = np.asarray(haps.ffi_static.v_starts, np.int64)[v_idxs]
        v_ilen = np.asarray(haps.ffi_static.ilens, np.int64)[v_idxs]
        v_end = v_start - np.minimum(v_ilen, 0) + 1
        region_keep = (v_start < v_r_end) & (v_end > v_r_start)
        keep = region_keep if keep is None else (keep & region_keep)
```

Note: `regions_arr[:, 0]` is the contig index; `[:, 1]`/`[:, 2]` are start/end. `counts` uses `unfiltered_row_offsets` (the pre-AF-compaction offsets), matching what `v_idxs` indexes at this point. The existing compaction block at lines 830-833 already handles `keep is not None`.

- [ ] **Step 4: Run the new test + the existing variants tests to verify pass + no regression**

Run: `pixi run -e dev pytest tests/dataset/test_variants_region_clip.py -v`
Expected: PASS.

Run the existing variants-output tests (byte-identical `to_ragged`, windows, var_fields) to confirm in-window variants are unaffected:
Run: `pixi run -e dev pytest tests/dataset tests/unit -k "variant or flat" -q`
Expected: PASS (no regression). If any pre-existing test asserted a golden set that *included* a now-clipped out-of-window variant, that test encoded the #202 bug — update its expectation to the clipped set and note it in the commit body.

- [ ] **Step 5: Add an annotated cross-check (independent correctness anchor)**

Add to `tests/dataset/test_variants_region_clip.py`: the distinct variants that actually appear in `with_seqs("annotated")` (its non-`-1` per-position `var_idxs`, which is already parity-tested and correct) must be a **subset** of the clipped `with_seqs("variants")` set for the same cell (a variant can be present-but-fully-overwritten, so equality is not guaranteed; subset is the invariant). Compare by genomic start position (variants output has no id field):

```python
def test_annotated_variants_are_a_subset_of_clipped_variants(
    pgen_snp_ins_del_multi, tmp_path
):
    f = pgen_snp_ins_del_multi
    gvl.write(tmp_path / "ds", f.regions, variants=str(f.pgen))
    base = gvl.Dataset.open(tmp_path / "ds", reference=f.fasta)
    ann = base.with_seqs("annotated")
    var = base.with_seqs("variants", var_fields=["alt", "ilen", "start"])

    n_regions, n_samples = base.shape
    v_starts = np.asarray(base._haps.ffi_static.v_starts).astype(np.int64)  # type: ignore[attr-defined]

    for r in range(n_regions):
        for s in range(n_samples):
            a = ann[r, s]
            v = var[r, s]
            for h in range(base.ploidy):
                a_ids = np.asarray(a.var_idxs[h])
                a_ids = a_ids[a_ids >= 0]
                appeared_starts = set(v_starts[a_ids].tolist())
                clipped_starts = set(np.asarray(v.start[h]).astype(np.int64).tolist())
                assert appeared_starts <= clipped_starts, (
                    f"cell ({r},{s}) hap {h}: annotated used a variant the clipped "
                    f"variants output dropped"
                )
```

Run: `pixi run -e dev pytest tests/dataset/test_variants_region_clip.py -v`
Expected: PASS. (If this fails, the overlap predicate is wrong — STOP and escalate; do not weaken the assertion.)

- [ ] **Step 6: Lint, format, commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_flat_variants.py tests/dataset/test_variants_region_clip.py
git commit -m "fix(streaming): clip written with_seqs(\"variants\") to the region window (#202)

get_variants_flat gathered v_idxs straight from the sparse genotype set and
never filtered by region extent, so RaggedVariants leaked variants outside the
queried window (boundary indels; PGEN contig-wide). Fold a region-overlap keep
mask into the existing AF keep/_compact_keep block, matching the haplotype
reconstruction's variant-inclusion extent. Corrects the parity oracle ahead of
streaming variants output (Wave B PR-B1).

Closes #202.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: PR-B1 — RecordBackend (VCF/PGEN) variants generate core (Rust)

**Files:**
- Modify: `src/record_stream/engine.rs` — add a variants generate path on `RecordBackend`.
- Create/Modify: a shared helper `assemble_variants_window` in `src/record_stream/mod.rs` (or `src/variants/mod.rs` if it fits the existing module) — window CSR → flat variant buffers.
- Test: Rust `#[cfg(test)]` unit test in `src/record_stream/engine.rs` (or `mod.rs`), runnable via `cargo test`.

**Interfaces:**
- Consumes: `DecodedWindow { v_starts: Vec<i32>, ilens: Vec<i32>, alt_alleles: Vec<u8>, alt_offsets: Vec<i64>, geno_v_idxs: Vec<i32>, geno_offsets: Vec<i64>, global_v_idxs: Vec<i32> }` (`src/record_stream/transpose.rs:29-45`); `RecordBackend` fields (`ploidy`, `regions`, per-job region extents). The per-hap CSR is window-local with **no region dimension** — replicate a sample's run across regions via `bi % n_samples` (see `generate`, `engine.rs:186-195`).
- Produces: a function that, for a `[row_lo, row_hi)` batch slice, returns flat `RaggedVariants`-shaped buffers:
  - `alt_data: Array1<u8>`, `alt_seq_offsets: Array1<i64>` (per-variant ALT bytes, ragged),
  - `start_data: Array1<i32>` (one per variant),
  - `ilen_data: Array1<i32>` (one per variant),
  - `row_offsets: Array1<i64>` (per-`(b*ploidy)`-row, or per-`b`-row after unphased-union fold — match the haplotype path's `eff_ploidy` handling; for the first cut assume phased ploidy and length `n_rows*ploidy + 1`).
  Signature (suggested):
  ```rust
  fn generate_variants(
      &self,
      job_idx: usize,
      slot: &DecodedWindow,
      row_lo: usize,
      row_hi: usize,
  ) -> anyhow::Result<VariantsBatch>;
  // struct VariantsBatch { alt_data, alt_seq_offsets, start, ilen, row_offsets }
  ```

- [ ] **Step 1: Write the failing Rust unit test**

In `src/record_stream/engine.rs` under `#[cfg(test)]`, build a tiny `DecodedWindow` with two variants (one SNP inside the window, one variant whose extent falls outside) and a one-hap CSR referencing both, plus a `RecordBackend` with one region `[10, 20)`. Assert `generate_variants` returns only the in-window variant, with correct `alt`/`start`/`ilen`.

```rust
#[test]
fn generate_variants_clips_to_window_and_gathers_fields() {
    // window table: v0 SNP at pos=12 (in [10,20)); v1 SNP at pos=25 (outside)
    let slot = DecodedWindow {
        v_starts: vec![12, 25],
        ilens: vec![0, 0],
        alt_alleles: vec![b'A', b'C'],
        alt_offsets: vec![0, 1, 2],
        geno_v_idxs: vec![0, 1],       // hap 0 has both local variants
        geno_offsets: vec![0, 2],      // one hap, CSR [0,2)
        global_v_idxs: vec![100, 101], // ignored for variants output
    };
    let backend = /* RecordBackend with ploidy=1, one region (10,20), n_samples=1 */;
    let out = backend.generate_variants(0, &slot, 0, 1).unwrap();
    // only v0 survives the overlap clip
    assert_eq!(out.start.as_slice().unwrap(), &[12]);
    assert_eq!(out.ilen.as_slice().unwrap(), &[0]);
    assert_eq!(out.alt_data.as_slice().unwrap(), &[b'A']);
    assert_eq!(out.row_offsets.as_slice().unwrap(), &[0, 1]);
}
```

(Mirror the existing test-fixture construction in `engine.rs`'s `#[cfg(test)]` module for how a minimal `RecordBackend`/`WindowJob` is built; reuse those helpers.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd .claude/worktrees/streaming-waveb-b0b1 && LD_LIBRARY_PATH=$PWD/../../../.pixi/envs/dev/lib pixi run -e dev cargo test -p genvarloader generate_variants_clips_to_window -- --nocapture`
Expected: FAIL to compile (`generate_variants` not defined). (`cargo test` needs `LD_LIBRARY_PATH` to the dev env's `lib` for libpython — see the memory note; if `cargo test` is wired as a pixi task use that.)

- [ ] **Step 3: Implement `generate_variants` + the shared `assemble_variants_window` helper**

Implement in two pieces:

(a) Per-row CSR + region-clip gather, mirroring `generate`'s `rb` and `bi % n_samples` replication (`engine.rs:162-195`), but instead of reconstructing haplotypes, produce per-row **kept window-local variant indices** and per-row region extents:

```rust
// for each output row bi in [row_lo, row_hi): region (r_s, r_e), sample si = bi % n_samples
// for each ploid p: hap h = si*ploidy + p; CSR range slot.geno_offsets[h..h+1]
// for each local vidx in slot.geno_v_idxs[csr_lo..csr_hi]:
//     v_start = slot.v_starts[vidx]; v_end = v_start - min(slot.ilens[vidx], 0) + 1
//     keep iff v_start < r_e && v_end > r_s   // Global-Constraints predicate
```

(b) A shared `assemble_variants_window` that takes the kept per-row `v_idxs` + `row_offsets` + the window static table (`slot.v_starts`, `slot.ilens`, `slot.alt_alleles`, `slot.alt_offsets`) and builds the flat buffers. **Reuse `gather_alleles`** (`src/variants/mod.rs:52-76`) for the ragged `alt` bytes, and simple index-gathers for `start`/`ilen`:

```rust
let (alt_data, alt_seq_offsets) = crate::variants::gather_alleles(
    v_idxs.view(), slot.alt_alleles.view(), slot.alt_offsets.view(),
);
let start: Array1<i32> = v_idxs.iter().map(|&vi| slot.v_starts[vi as usize]).collect();
let ilen:  Array1<i32> = v_idxs.iter().map(|&vi| slot.ilens[vi as usize]).collect();
```

This matches the written path's `_assemble_variant_buffers_rust(0, v_idxs, row_offsets, stat.alt_alleles, stat.alt_offsets, ...)` for `alt` (`_flat_variants.py:959-979`) and its scalar `start`/`ilen` gather — byte-identical inputs, so byte-identical outputs. **Do not** map local→global (`global_v_idxs` is unused here).

- [ ] **Step 4: Run the Rust test to verify it passes**

Run: `cd .claude/worktrees/streaming-waveb-b0b1 && LD_LIBRARY_PATH=$PWD/../../../.pixi/envs/dev/lib pixi run -e dev cargo test -p genvarloader generate_variants_clips_to_window`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/record_stream/engine.rs src/record_stream/mod.rs
git commit -m "feat(streaming): RecordBackend variants generate core — window CSR -> flat variant buffers

For a batch row slice, gather each (region,sample,ploid)'s window-local variant
indices from the per-hap CSR, apply the region-overlap clip (matching #202), and
assemble ragged alt bytes + scalar start/ilen via the shared gather helpers.
Reuses gather_alleles; no dataset-global id (variants output is self-contained,
per #313). VCF/PGEN.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: PR-B1 — `next_batch_variants` FFI + Python wiring + VCF/PGEN parity

**Files:**
- Modify: `src/record_stream/engine.rs` — `RecordStreamEngine::next_batch_variants` pymethod; `variants: bool` constructor flag on `new`/`new_rs`; stored on `RecordBackend`.
- Modify: `python/genvarloader/_dataset/_streaming.py` — `with_seqs` accepts `"variants"`; `_iter_batches` variants branch; `_VcfBackend.build_engine`/`_PgenBackend.build_engine` thread `variants`.
- Test: `tests/dataset/test_streaming_variants_parity.py` (create) — VCF + PGEN.

**Interfaces:**
- Consumes: `RecordBackend::generate_variants` (Task 2). The Wave A annotated template: `next_batch_annotated` (`engine.rs:456-484`), `annotated: bool` flag (`new` default `engine.rs:299`), `next_batch_core` (`engine.rs:267-272`).
- Produces:
  - `RecordStreamEngine.next_batch_variants(py) -> Option[dict[str, np.ndarray]]` returning a dict with keys `alt`, `alt_offsets`, `start`, `ilen`, `offsets` (row offsets), or `None` at end-of-stream. Raises `RuntimeError` if the engine was not built with `variants=True`.
  - `StreamingDataset.with_seqs("variants")` → `_seq_kind = RaggedVariants`.
  - `build_engine(engine_jobs, batch_size, out_len, annotated, variants)` on the VCF/PGEN backends.

- [ ] **Step 1: Write the failing VCF/PGEN parity test**

Create `tests/dataset/test_streaming_variants_parity.py`, following the `test_streaming_pgen_parity.py:110-141` harness shape. Compare each streamed `RaggedVariants` cell field-by-field against the corrected written oracle (`.with_seqs("variants")`).

```python
"""Wave B PR-B1: streaming with_seqs("variants") is byte-identical to the written path."""

import numpy as np
import pytest

import genvarloader as gvl

BACKENDS = ["vcf", "pgen"]  # svar1 added in Task 4


def _assert_variants_cell_matches(streamed, expected, ploidy):
    for h in range(ploidy):
        np.testing.assert_array_equal(
            np.asarray(streamed.alt[h]), np.asarray(expected.alt[h])
        )
        np.testing.assert_array_equal(
            np.asarray(streamed.start[h]), np.asarray(expected.start[h])
        )
        np.testing.assert_array_equal(
            np.asarray(streamed.ilen[h]), np.asarray(expected.ilen[h])
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_streaming_variants_matches_written(streaming_case, backend):
    regions, reference, variants, written = streaming_case(backend)
    ds = written.with_seqs("variants", var_fields=["alt", "ilen", "start"])
    sds = gvl.StreamingDataset(
        regions, reference=reference, variants=variants
    ).with_seqs("variants")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4):
        for k in range(len(r_idx)):
            r, s = int(r_idx[k]), int(s_idx[k])
            _assert_variants_cell_matches(data[k], ds[r, s], sds.ploidy)
            seen.add((r, s))
    assert seen == {(r, s) for r in range(ds.shape[0]) for s in range(ds.shape[1])}
```

(If `streaming_case` does not yet include `"vcf"`, extend `BACKENDS` in `tests/dataset/conftest.py:1249-1266` to map `"vcf"` — variants output has no fail-fast, unlike annotated.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -v`
Expected: FAIL — `StreamingDataset.with_seqs("variants")` raises `NotImplementedError` today (`_streaming.py:934-938`).

- [ ] **Step 3: Add the Rust `next_batch_variants` pymethod + `variants` flag**

In `src/record_stream/engine.rs`:
- Add `variants: bool` to `RecordBackend` (mirror `annotated`, `engine.rs:97`) and to `new`/`new_rs` (`engine.rs:244`, `299`; default `false`).
- Add a `next_batch_variants` pymethod parallel to `next_batch_annotated` (`engine.rs:456-484`). It drives the same per-job iteration but calls `generate_variants` (Task 2) instead of `generate`, and marshals `VariantsBatch` into a `PyDict` with `alt`/`alt_offsets`/`start`/`ilen`/`offsets`. Raise `PyRuntimeError` if `!self.backend.variants`.

Follow how `next_batch_annotated` obtains the next slice from the shared iteration state; if the variants payload does not fit `next_batch_core`'s 4-tuple, add a sibling core (e.g. `next_batch_variants_core`) that calls `generate_variants` — do not overload the haplotype core's return shape.

- [ ] **Step 4: Wire the Python surface**

In `python/genvarloader/_dataset/_streaming.py`:

(a) `with_seqs` (`_streaming.py:920-941`): extend the `Literal` and `kind_map`:
```python
def with_seqs(
    self, kind: Literal["haplotypes", "annotated", "variants"]
) -> "StreamingDataset":
    ...
    kind_map = {
        "haplotypes": RaggedSeqs,
        "annotated": RaggedAnnotatedHaps,
        "variants": RaggedVariants,
    }
    if kind not in kind_map:
        raise NotImplementedError(
            f"StreamingDataset.with_seqs({kind!r}) is not implemented; "
            'supported: "haplotypes", "annotated", "variants". '
            '"variant-windows" and "reference" are later Wave B / follow-ups.'
        )
    ...
```
(Import `RaggedVariants` at the top of `_streaming.py` if not already.)

(b) `_iter_batches` (`_streaming.py:435-608`): add a `_variants = self._seq_kind is RaggedVariants` gate alongside `_annotated`. Thread `variants=_variants` into `build_engine`. Select the batch puller:
```python
if _variants:
    next_batch = engine.next_batch_variants
elif _annotated:
    next_batch = engine.next_batch_annotated
else:
    next_batch = engine.next_batch
```
And add a packing block that builds `RaggedVariants` from the returned dict (mirror the annotated `Ragged.from_offsets` packing at `_streaming.py:591-601`):
```python
if _variants:
    d = batch  # dict from next_batch_variants
    alt = Ragged.from_offsets(
        np.asarray(d["alt"], np.uint8).view("S1"), out_shape, np.asarray(d["alt_offsets"], np.int64)
    )
    start = Ragged.from_offsets(np.asarray(d["start"], np.int32), out_shape, np.asarray(d["offsets"], np.int64))
    ilen = Ragged.from_offsets(np.asarray(d["ilen"], np.int32), out_shape, np.asarray(d["offsets"], np.int64))
    data = RaggedVariants(alt=alt, start=start, ilen=ilen)
```
(Match the exact `out_shape`/offset conventions used by the annotated branch; `RaggedVariants.__init__` re-points `start`/`ilen` onto `alt.offsets` internally, so passing the shared row `offsets` is consistent — see `_rag_variants.py:210-232`.)

(c) `_VcfBackend.build_engine` (`_streaming.py:1805-1810`) and `_PgenBackend.build_engine` (`_streaming.py:1944-1949`): add `variants: bool = False` and forward it to the engine constructor (`_streaming.py:1870`, `2007`).

- [ ] **Step 5: Rebuild Rust and run the parity test**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -v
```
Expected: PASS (VCF + PGEN byte-identical vs corrected written oracle).

- [ ] **Step 6: Lint, format, commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add src/record_stream/engine.rs python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_variants_parity.py tests/dataset/conftest.py
git commit -m "feat(streaming): with_seqs(\"variants\") for VCF/PGEN — RaggedVariants byte-identical to written

next_batch_variants marshals the RecordBackend variants core to numpy; _streaming
gains a variants branch (build_engine variants flag, RaggedVariants packing).
Parity vs the corrected (#202) written with_seqs(\"variants\") on VCF + PGEN.

Relates to #304.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: PR-B1 — SVAR1 backend variants output + parity

**Files:**
- Modify: `src/ffi/stream_engine.rs` — `Svar1Backend` variants generate path + `Svar1StreamEngine::next_batch_variants` + `variants: bool` flag.
- Modify: `python/genvarloader/_dataset/_streaming.py` — `_Svar1Backend.build_engine` threads `variants` (`_streaming.py:1250-1255`, forward at `1343`).
- Test: extend `tests/dataset/test_streaming_variants_parity.py` — add `"svar1"` to `BACKENDS`.

**Interfaces:**
- Consumes: SVAR1's window read (the store's static variant table + per-hap window CSR — the same structure the SVAR1 haplotype path already uses in `Svar1StreamEngine`). Reuse the shared `assemble_variants_window` helper from Task 2 so SVAR1 and RecordBackend share one assembly.
- Produces: `Svar1StreamEngine.next_batch_variants(py) -> Option[dict]` identical in shape to Task 3's dict; `_Svar1Backend.build_engine(..., variants=False)`.

- [ ] **Step 1: Add `"svar1"` to the parity test (failing)**

Edit `tests/dataset/test_streaming_variants_parity.py`: `BACKENDS = ["svar1", "vcf", "pgen"]`.

Run: `pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -v -k svar1`
Expected: FAIL — SVAR1 engine has no `next_batch_variants` / `variants` flag yet.

- [ ] **Step 2: Implement the SVAR1 variants path**

In `src/ffi/stream_engine.rs`, mirror Task 3's additions on the SVAR1 side: `variants: bool` flag on `Svar1Backend` (`stream_engine.rs:132`), a `next_batch_variants` pymethod parallel to `next_batch_annotated` (`stream_engine.rs:453-481`), driving the SVAR1 window read and calling the **shared** `assemble_variants_window` helper (Task 2 §3b) with SVAR1's window static table + per-hap CSR + region extents (apply the same overlap clip). No global-id remap.

Thread `variants` through `_Svar1Backend.build_engine` in `_streaming.py` (add param, forward at the constructor call `_streaming.py:1343`). The Python `_iter_batches` variants branch (Task 3 §4b) is backend-agnostic and already handles SVAR1 once `next_batch_variants` exists.

- [ ] **Step 3: Rebuild and run the full parity test (all three backends)**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset/test_streaming_variants_parity.py -v
```
Expected: PASS for `svar1`, `vcf`, `pgen`.

- [ ] **Step 4: Commit**

```bash
git add src/ffi/stream_engine.rs python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_variants_parity.py
git commit -m "feat(streaming): with_seqs(\"variants\") for SVAR1 — shared variants assembly

Svar1StreamEngine gains next_batch_variants + variants flag, reusing the shared
assemble_variants_window helper (region-clipped window CSR -> flat variant
buffers). Byte-identical parity on svar1/vcf/pgen.

Relates to #304.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Docs, skill, api.md, roadmap, board

**Files:**
- Modify: `skills/genvarloader/SKILL.md` — streaming `with_seqs` now supports `"variants"` (which backends, default fields, region-clip semantics, no id field).
- Modify: `docs/source/dataset.md` (and `api.md` if `StreamingDataset.with_seqs` signature is documented there) — the new `"variants"` value.
- Modify: `docs/source/faq.md` or `write.md` if they claim streaming is haplotypes-only.
- Modify: `docs/roadmaps/streaming-dataset.md` — tick PR-B0/PR-B1 under Plan 5, set status markers, add PR links.

- [ ] **Step 1: Update the skill + prose docs**

Edit `skills/genvarloader/SKILL.md`'s streaming section: `StreamingDataset(...).with_seqs("variants")` returns `RaggedVariants` (fields `alt`/`start`/`ilen`), supported on SVAR1/VCF/PGEN, variants clipped to the region window, byte-identical to the written path. Note `min_af`/`max_af`, `var_fields`, and `variant-windows` are later Wave B PRs (B2–B4). Grep for any "haplotypes/annotated only" streaming claims and correct them.

- [ ] **Step 2: Verify api.md ↔ `__all__` sync**

Run: `python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none` (no new public symbol is added — `with_seqs` is an existing method — but run the gate per CLAUDE.md).

- [ ] **Step 3: Update the roadmap + board**

Edit `docs/roadmaps/streaming-dataset.md`: under Plan 5 / Wave B, mark PR-B0 (#202) and PR-B1 (#304, variants output) done with PR links and a one-line result. Update the StreamingDataset project board (#304 → In progress/Done as apt; note B2–B4 remain).

- [ ] **Step 4: Full sweep, lint, commit**

```bash
pixi run -e dev pytest tests -q
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
git add skills/ docs/
git commit -m "docs(streaming): document with_seqs(\"variants\") + tick Wave B PR-B0/PR-B1 roadmap

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Open the draft PR(s) into `streaming`**

Per the streaming coordination rules, PR-B0 and PR-B1 may land as separate PRs into `streaming` (PR-B0 stands alone / cherry-pickable; PR-B1 stacks on it). Open draft PRs with `gh pr create --draft --base streaming`, cross-linking #202 (B0) and #304 (B1), and add them to the StreamingDataset project.

---

## Self-Review

**Spec coverage (against `2026-07-20-streaming-variants-output-wave-b-design.md` §PR-B0, §PR-B1):**
- §PR-B0 region clip both paths → Task 1 (written) + Task 2/§3a + Task 4 (streaming clip). ✓
- §PR-B1 `with_seqs("variants")` → RaggedVariants, default fields, region-clipped, reuse assemble kernel → Tasks 2–4. ✓
- §PR-B1 multi-backend parity (SVAR1/VCF/PGEN) → Tasks 3 (VCF/PGEN) + 4 (SVAR1). ✓
- #313 correction (variants output is self-contained, no global id) → encoded in Global Constraints + Task 2/§3 (no `global_v_idxs` remap). ✓
- Out of scope (this plan): `min_af`/`max_af` (PR-B2), `var_fields` dosage/FORMAT (PR-B3), REF bytes + `variant-windows` (PR-B4). Called out in Task 5 §3 and the docs. ✓

**Placeholder scan:** No TBD/TODO/placeholder code. Every test and code step shows the actual content. Two spots ask the implementer to *confirm a name against the fixture/existing branch* (Task 1 Step 1 BED columns; Task 3 Step 1 `streaming_case` "vcf" key) — these are verification instructions with a concrete fallback, not missing content.

**Type consistency:** `next_batch_variants` returns a `dict` with keys `alt`/`alt_offsets`/`start`/`ilen`/`offsets` in Tasks 3 and 4 (same shape). `generate_variants`/`VariantsBatch` fields (`alt_data`/`alt_seq_offsets`/`start`/`ilen`/`row_offsets`) are consistent between Task 2 (definition) and Tasks 3–4 (consumption). The overlap predicate `v_end = v_start - min(ilen,0) + 1` is stated once in Global Constraints and referenced (not re-derived divergently) in Tasks 1, 2, 4. `variants: bool` flag naming is consistent across `RecordBackend`, `Svar1Backend`, and `build_engine`.

**Open risk flagged for the implementer:** if Task 1 Step 5's annotated-subset cross-check fails, the overlap predicate (vs. containment) is wrong — stop and escalate rather than weakening the assertion; the reviewed spec chose overlap and the linear reconstruction applies upstream spanning DELs, so overlap is expected to hold.
