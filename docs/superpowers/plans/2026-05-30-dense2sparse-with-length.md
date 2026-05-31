# `_dense2sparse_with_length` for VCF/PGEN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route VCF/PGEN dataset writes through genoray 2.7.0's `_dense2sparse_with_length` so they produce per-haplotype-minimal sparse output identical to SVAR when `extend_to_length=True`, honor `extend_to_length=False` (no extension), and compute per-region `chromEnd` (`max_ends`) the same way the SVAR path does.

**Architecture:** `python/genvarloader/_dataset/_write.py` holds the write pipeline. The two near-duplicate per-region generators `_vcf_region_chunks` and `_pgen_region_chunks` are refactored to (a) accept `extend_to_length`, (b) assemble each region's full dense window across genoray memory-chunks, and (c) delegate the dense→sparse conversion to one new shared helper `_window_to_sparse`. The aggregation helper `_write_phased_chunked` keeps its `(list[Ragged], region_end, desc)` contract. Per-region `region_end` is computed from the max retained variant index via `v_ends`, matching `_write_from_svar`.

**Tech Stack:** Python, genoray 2.7.0 (`VCF`, `PGEN`, `genoray._svar.dense2sparse` / `_dense2sparse_with_length`), polars, numpy, awkward, seqpro `Ragged`, pytest.

**Spec:** `docs/superpowers/specs/2026-05-30-dense2sparse-with-length-design.md`

**Background facts (verified against the worktree):**
- genoray `2.7.0` is installed; dep already bumped in `pixi.toml` (`genoray = "==2.7.0"`) and `pyproject.toml` (`genoray>=2.7.0,<3`).
- `genoray._svar._dense2sparse_with_length(genos, var_idxs, q_start, q_end, v_starts, ilens, dosages=None)` returns a `Ragged[V_IDX_TYPE]` of shape `(samples, ploidy, ~variants)`. It needs the **entire region window** at once. `genos` shape is `(samples, ploidy, variants)`; `var_idxs` is window-aligned global indices; `q_start`/`q_end` are the 0-based half-open *original* (unextended) query span; `v_starts = POS-1` and `ilens = ILEN` are window-aligned (positionally aligned with `var_idxs`).
- The VCF/PGEN variant index (`vcf._index` / `pgen._index`) has columns including `POS` (i32) and `ILEN` (list[i32]; first element for bi-allelic).
- `VCF._chunk_ranges_with_length(contig, starts, ends, max_mem, mode)` yields one generator per region; each region generator yields `(chunk_genos, chunk_end, n_ext)` per memory-chunk, with `n_ext > 0` only on the last chunk (extension variants appended).
- `PGEN._chunk_ranges_with_length(contig, starts, ends, max_mem)` yields one generator per region; each yields `(genos, chunk_end, chunk_idxs)` — `chunk_idxs` already includes the extension tail.
- Non-length APIs (for `extend_to_length=False`): `VCF.chunk(contig, start, end, max_mem, mode)` (single range) and `PGEN.chunk_ranges(contig, starts, ends, max_mem, mode)` (multi-range). Both return exactly the in-range variants. `VCF._var_idxs(contig, starts, ends)` returns `(v_idx, offsets)` for the unextended in-range variants.
- `_write_from_svar` computes `v_ends = POS - ILEN_first.clip(upper_bound=0)` and sets each region's `chromEnd` to `v_ends[max retained v_idx]`, falling back to the input `chromEnd` when the region has no variants. This same formula must apply to VCF/PGEN, for both `True` and `False`.
- Test ground-truth datasets (`tests/data/phased_dataset.{vcf,pgen,svar}.gvl`) are produced by `pixi run -e dev gen` (`tests/data/generate_ground_truth.py`), all from the same BED + the same underlying variants. Conftest fixtures: `phased_vcf_gvl`, `phased_pgen_gvl`, `phased_svar_gvl`, `reference`.

**Current code anchors (pre-change line numbers in `python/genvarloader/_dataset/_write.py`):**
- imports: line 18 `from genoray._svar import dense2sparse`
- `_write_from_vcf` 379-396; `_vcf_region_chunks` 399-449
- `_write_from_pgen` 452-466; `_pgen_region_chunks` 469-492
- `_write_phased_chunked` 495-558
- `write()` calls these at 241-248

---

## Task 1: Add the shared `_window_to_sparse` helper (TDD)

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (import on line 18; add helper near the other `_write_*`/`_*_region_chunks` helpers, e.g. just above `_vcf_region_chunks` at line 399)
- Test: `tests/unit/dataset/genotypes/test_window_to_sparse.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/genotypes/test_window_to_sparse.py`:

```python
"""Unit tests for `_window_to_sparse`: the shared dense->sparse conversion that
dispatches between plain `dense2sparse` (no extension) and genoray's
`_dense2sparse_with_length` (per-haplotype-minimal extension)."""

import awkward as ak
import numpy as np
from genoray._types import V_IDX_TYPE

from genvarloader._dataset._write import _window_to_sparse


def _window():
    """A 1-sample, 2-haplotype window over query [0, 4).

    Two variants both starting inside the query:
      - v0 @ start=1, ILEN=-3 (3bp deletion)
      - v1 @ start=5, ILEN=0  (SNP) -- starts AFTER q_end=4; it is an
        extension variant only needed by a haplotype shortened by the deletion.
    hap0 carries both v0 and v1; hap1 carries only v1.
    """
    # (samples, ploidy, variants)
    genos = np.array([[[1, 1], [0, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=V_IDX_TYPE)
    v_starts = np.array([1, 5], dtype=np.int32)
    ilens = np.array([-3, 0], dtype=np.int32)
    q_start, q_end = 0, 4
    return genos, var_idxs, q_start, q_end, v_starts, ilens


def test_no_extend_keeps_all_carried_variants():
    genos, var_idxs, q_start, q_end, v_starts, ilens = _window()
    rag = _window_to_sparse(
        genos, var_idxs, q_start, q_end, v_starts, ilens, extend_to_length=False
    )
    # plain dense2sparse: every haplotype keeps exactly the variants it carries.
    # hap0 carries v0,v1 -> [0, 1]; hap1 carries v1 -> [1]
    assert ak.to_list(rag) == [[[0, 1], [1]]]


def test_extend_trims_per_haplotype_to_length():
    genos, var_idxs, q_start, q_end, v_starts, ilens = _window()
    rag = _window_to_sparse(
        genos, var_idxs, q_start, q_end, v_starts, ilens, extend_to_length=True
    )
    # hap0 was shortened by the 3bp deletion at v0, so it needs the extension
    # variant v1 to reach length 4 -> [0, 1].
    # hap1 has no deletion; it already spans [0,4) with just the SNP v1, so the
    # length walk does NOT pull anything past q_end -> [1].
    assert ak.to_list(rag) == [[[0, 1], [1]]]


def test_extend_drops_unneeded_extension_for_full_length_haplotype():
    """A haplotype with no deletion must not absorb extension variants it
    doesn't need (this is the over-extension bug the change fixes)."""
    # hap0: SNP only (no deletion); hap1: SNP only. Query [0,4).
    # v0 @ start=1 ILEN=0 (in-query SNP), v1 @ start=5 ILEN=0 (past q_end).
    genos = np.array([[[1, 1], [1, 1]]], dtype=np.int8)
    var_idxs = np.array([0, 1], dtype=V_IDX_TYPE)
    v_starts = np.array([1, 5], dtype=np.int32)
    ilens = np.array([0, 0], dtype=np.int32)
    rag = _window_to_sparse(
        genos, var_idxs, 0, 4, v_starts, ilens, extend_to_length=True
    )
    # Neither haplotype is shortened, so neither needs v1 (which starts at 5,
    # outside [0,4)). Both keep only v0.
    assert ak.to_list(rag) == [[[0], [0]]]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/genotypes/test_window_to_sparse.py -v`
Expected: FAIL with `ImportError: cannot import name '_window_to_sparse'`.

- [ ] **Step 3: Add the import and the helper**

In `python/genvarloader/_dataset/_write.py`, change the import on line 18 from:

```python
from genoray._svar import dense2sparse
```

to:

```python
from genoray._svar import _dense2sparse_with_length, dense2sparse
```

Then add this helper immediately above `def _vcf_region_chunks(` (currently line 399):

```python
def _window_to_sparse(
    genos: NDArray[np.integer],
    var_idxs: NDArray[V_IDX_TYPE],
    q_start: int,
    q_end: int,
    v_starts: NDArray[np.int32],
    ilens: NDArray[np.int32],
    extend_to_length: bool,
) -> Ragged:
    """Convert a full dense region window into per-haplotype sparse genotypes.

    ``genos`` has shape ``(samples, ploidy, variants)`` and must cover the
    entire region window (all genoray memory-chunks concatenated along the
    variant axis). ``var_idxs`` are the window's global variant indices.
    ``v_starts`` (``POS - 1``) and ``ilens`` (``ILEN``) are window-aligned,
    positionally aligned with ``var_idxs``.

    When ``extend_to_length`` is ``True`` this defers to genoray's
    ``_dense2sparse_with_length``, which walks each haplotype's length and keeps
    only the variants it needs to reach ``q_end`` (per-haplotype-minimal,
    identical to ``SparseVar.read_ranges_with_length``). When ``False`` it falls
    back to plain ``dense2sparse`` (every haplotype keeps exactly the variants it
    carries within the window, with no length extension).
    """
    if extend_to_length:
        return _dense2sparse_with_length(
            genos, var_idxs, q_start, q_end, v_starts, ilens
        )
    return dense2sparse(genos, var_idxs)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/genotypes/test_window_to_sparse.py -v`
Expected: PASS (3 passed).

Note: if the `extend` assertions in steps 1's `test_extend_*` reveal that genoray's
length-walk semantics differ from the inline comments (e.g. an inclusive vs.
exclusive boundary at `q_end`), update the *expected* values to whatever
`_dense2sparse_with_length` actually returns for the same input — genoray is the
source of truth for the semantics. Do not change the helper to "fix" the result;
the helper is a thin pass-through. Re-run until green.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/unit/dataset/genotypes/test_window_to_sparse.py
rtk git commit -m "feat(write): add _window_to_sparse dense->sparse dispatch helper"
```

---

## Task 2: Refactor `_vcf_region_chunks` to assemble full windows + max_ends

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` — `_write_from_vcf` (379-396) and `_vcf_region_chunks` (399-449)

**Context:** The new generator must (a) take `extend_to_length`, (b) build the per-contig `POS`/`ILEN`/`v_ends` numpy arrays once from `vcf._index`, (c) per region assemble the full dense window + window-aligned `var_idxs`/`v_starts`/`ilens`, (d) call `_window_to_sparse` once (extend) or `dense2sparse` per chunk (no-extend), and (e) compute `region_end` from the max retained global variant index via `v_ends`, falling back to the region's input end.

- [ ] **Step 1: Update `_write_from_vcf` to pass `extend_to_length` through**

Replace the body's final line. Current `_write_from_vcf` (379-396) ends with:

```python
    return _write_phased_chunked(out_dir, bed, _vcf_region_chunks(bed, vcf, max_mem))
```

Change it to:

```python
    return _write_phased_chunked(
        out_dir, bed, _vcf_region_chunks(bed, vcf, max_mem, extend_to_length)
    )
```

(`_write_from_vcf` already receives `extend_to_length` as its 5th parameter — see its signature at line 380.)

- [ ] **Step 2: Replace `_vcf_region_chunks` with the full-window version**

Replace the entire current `_vcf_region_chunks` (399-449) with:

```python
def _vcf_region_chunks(
    bed: pl.DataFrame, vcf: VCF, max_mem: int, extend_to_length: bool
) -> Iterator[tuple[list[Ragged], Any, str | None]]:
    assert vcf._index is not None
    pos = vcf._index["POS"].to_numpy()
    ilen_all = vcf._index["ILEN"].list.first().to_numpy()
    # end position of each variant = POS + deletion length (matches _write_from_svar)
    v_ends = pos - np.clip(ilen_all, a_min=None, a_max=0)

    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        # unextended in-range variant indices, split per region
        v_idx, v_offsets = vcf._var_idxs(contig, starts, ends)
        unextended_idxs = np.array_split(
            v_idx.astype(V_IDX_TYPE), v_offsets[1:-1]
        )

        contig_desc = f"Processing genotypes for {df.height} regions on contig {contig}"
        first_in_contig = True

        if extend_to_length:
            region_iter = vcf._chunk_ranges_with_length(
                contig, starts, ends, max_mem, VCF.Genos8
            )
        else:
            # one generator per region; VCF.chunk takes a single range
            region_iter = (
                vcf.chunk(contig, s, e, max_mem, VCF.Genos8)
                for s, e in zip(starts, ends)
            )

        for ri, range_ in enumerate(region_iter):
            q_start = int(starts[ri])
            q_end = int(ends[ri])
            reg_unext = unextended_idxs[ri]
            desc = contig_desc if first_in_contig else None
            first_in_contig = False

            if extend_to_length:
                # assemble the full window across memory-chunks
                chunk_genos_list: list[NDArray] = []
                n_ext_total = 0
                for _, is_last, (chunk_genos, _chunk_end, n_ext) in mark_ends(range_):
                    chunk_genos_list.append(chunk_genos)
                    if is_last:
                        n_ext_total = n_ext
                genos = np.concatenate(chunk_genos_list, axis=-1)

                if reg_unext.size == 0 and n_ext_total == 0:
                    # empty region: no variants for any sample
                    yield [dense2sparse(genos, reg_unext)], q_end, desc
                    continue

                if n_ext_total > 0:
                    ext_start = int(reg_unext[-1]) + 1
                    ext_idxs = np.arange(
                        ext_start, ext_start + n_ext_total, dtype=V_IDX_TYPE
                    )
                    var_idxs = np.concatenate([reg_unext, ext_idxs])
                else:
                    var_idxs = reg_unext

                v_starts = (pos[var_idxs] - 1).astype(np.int32)
                ilens = ilen_all[var_idxs].astype(np.int32)
                rag = _window_to_sparse(
                    genos, var_idxs, q_start, q_end, v_starts, ilens, True
                )
                region_end = _region_end(rag, v_ends, q_end)
                yield [rag], region_end, desc
            else:
                # no extension: convert each chunk independently with plain
                # dense2sparse; var_idxs are exactly the unextended in-range ones
                ls_sparse: list[Ragged] = []
                offset = 0
                for genos in range_:
                    n_vars = genos.shape[-1]
                    chunk_idxs = reg_unext[offset : offset + n_vars]
                    offset += n_vars
                    ls_sparse.append(dense2sparse(genos, chunk_idxs))
                region_end = _region_ends_from_list(ls_sparse, v_ends, q_end)
                yield ls_sparse, region_end, desc
```

- [ ] **Step 3: Add the `region_end` helpers**

Immediately above `_window_to_sparse` (or directly above `_vcf_region_chunks`), add:

```python
def _region_end(rag: Ragged, v_ends: NDArray, fallback_end: int) -> int:
    """Per-region chromEnd = end position of the furthest retained variant.

    ``rag`` is a sparse ``(samples, ploidy, ~variants)`` Ragged of global
    variant indices. Returns ``v_ends[max idx]`` across all haplotypes, or
    ``fallback_end`` when no variant is retained (mirrors _write_from_svar).
    """
    if rag.data.size == 0:
        return int(fallback_end)
    return int(v_ends[int(rag.data.max())])


def _region_ends_from_list(
    ls_sparse: list[Ragged], v_ends: NDArray, fallback_end: int
) -> int:
    """Same as `_region_end` but over a list of per-chunk Ragged arrays."""
    max_idx = -1
    for rag in ls_sparse:
        if rag.data.size:
            max_idx = max(max_idx, int(rag.data.max()))
    if max_idx < 0:
        return int(fallback_end)
    return int(v_ends[max_idx])
```

- [ ] **Step 4: Sanity-check imports/symbols compile**

Run: `pixi run -e dev python -c "import genvarloader._dataset._write"`
Expected: no output, exit 0. (If `VCF.Genos8` is not an attribute, use the module-level `Genos8` import path genoray exposes — verify with `pixi run -e dev python -c "from genoray import VCF; print(VCF.Genos8)"`; the pre-change code referenced `VCF.Genos8` at the old line 423, so this should hold.)

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "refactor(write): assemble full VCF windows, dispatch via _window_to_sparse, fix max_ends"
```

---

## Task 3: Refactor `_pgen_region_chunks` to match

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` — `_write_from_pgen` (452-466) and `_pgen_region_chunks` (469-492)

**Context:** PGEN's length generator yields `(genos, chunk_end, chunk_idxs)` with `chunk_idxs` already including the extension tail, so window `var_idxs` is just the concatenation of per-chunk `chunk_idxs` — no `n_ext`/`arange` reconstruction. For the no-extend path, `PGEN.chunk_ranges(contig, starts, ends, max_mem)` yields one generator per region; pair it with unextended indices from `pgen.var_idxs(contig, starts, ends)` (verified: a **public** method, no leading underscore, returning `(v_idx, offsets)` exactly like `VCF._var_idxs`). `pgen._index` is the loaded variant index with `POS` (i32) and `ILEN` (list[i32]) columns — populated by `pgen._init_index()`, which `write()` already calls before `_write_from_pgen`.

- [ ] **Step 1: Update `_write_from_pgen` to pass `extend_to_length`**

Current `_write_from_pgen` (452-466) ends with:

```python
    return _write_phased_chunked(out_dir, bed, _pgen_region_chunks(bed, pgen, max_mem))
```

Change to:

```python
    return _write_phased_chunked(
        out_dir, bed, _pgen_region_chunks(bed, pgen, max_mem, extend_to_length)
    )
```

- [ ] **Step 2: Replace `_pgen_region_chunks` with the full-window version**

Replace the entire current `_pgen_region_chunks` (469-492) with:

```python
def _pgen_region_chunks(
    bed: pl.DataFrame, pgen: PGEN, max_mem: int, extend_to_length: bool
) -> Iterator[tuple[list[Ragged], Any, str | None]]:
    assert pgen._index is not None
    pos = pgen._index["POS"].to_numpy()
    ilen_all = pgen._index["ILEN"].list.first().to_numpy()
    v_ends = pos - np.clip(ilen_all, a_min=None, a_max=0)

    for (contig,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = df["chromStart"].to_numpy()
        ends = df["chromEnd"].to_numpy()
        contig_desc = f"Processing genotypes for {df.height} regions on contig {contig}"
        first_in_contig = True

        if extend_to_length:
            region_iter = pgen._chunk_ranges_with_length(contig, starts, ends, max_mem)
        else:
            v_idx, v_offsets = pgen.var_idxs(contig, starts, ends)
            unextended_idxs = np.array_split(
                v_idx.astype(V_IDX_TYPE), v_offsets[1:-1]
            )
            region_iter = pgen.chunk_ranges(contig, starts, ends, max_mem)

        for ri, range_ in enumerate(region_iter):
            q_start = int(starts[ri])
            q_end = int(ends[ri])
            desc = contig_desc if first_in_contig else None
            first_in_contig = False

            if extend_to_length:
                genos_list: list[NDArray] = []
                idx_list: list[NDArray] = []
                for genos, _chunk_end, chunk_idxs in range_:
                    genos_list.append(genos.astype(np.int8))
                    idx_list.append(chunk_idxs.astype(V_IDX_TYPE))
                genos = np.concatenate(genos_list, axis=-1)
                var_idxs = (
                    np.concatenate(idx_list)
                    if idx_list
                    else np.empty(0, dtype=V_IDX_TYPE)
                )

                if var_idxs.size == 0:
                    yield [dense2sparse(genos, var_idxs)], q_end, desc
                    continue

                v_starts = (pos[var_idxs] - 1).astype(np.int32)
                ilens = ilen_all[var_idxs].astype(np.int32)
                rag = _window_to_sparse(
                    genos, var_idxs, q_start, q_end, v_starts, ilens, True
                )
                region_end = _region_end(rag, v_ends, q_end)
                yield [rag], region_end, desc
            else:
                reg_unext = unextended_idxs[ri]
                ls_sparse: list[Ragged] = []
                offset = 0
                for genos in range_:
                    n_vars = genos.shape[-1]
                    chunk_idxs = reg_unext[offset : offset + n_vars]
                    offset += n_vars
                    ls_sparse.append(dense2sparse(genos.astype(np.int8), chunk_idxs))
                region_end = _region_ends_from_list(ls_sparse, v_ends, q_end)
                yield ls_sparse, region_end, desc
```

- [ ] **Step 3: Sanity-check it imports**

Run: `pixi run -e dev python -c "import genvarloader._dataset._write"`
Expected: exit 0, no error.

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "refactor(write): assemble full PGEN windows, dispatch via _window_to_sparse, fix max_ends"
```

---

## Task 4: Regenerate test ground-truth

**Files:**
- Modify (regenerated artifacts): `tests/data/phased_dataset.vcf.gvl/`, `tests/data/phased_dataset.pgen.gvl/`, `tests/data/phased_dataset.svar.gvl/` and any other artifacts the `gen` task rewrites.

**Context:** VCF/PGEN stored genotypes change (now per-haplotype-minimal) and `chromEnd` values may change, so the committed datasets must be regenerated from the updated writer. The SVAR dataset is produced by the unchanged SVAR path; it should be byte-stable but may be rewritten by the script regardless.

- [ ] **Step 1: Run the generator**

Run: `pixi run -e dev gen`
Expected: completes without error; logs "Finished writing." for each dataset. The VCF/PGEN datasets under `tests/data/` are rewritten.

- [ ] **Step 2: Inspect what changed**

Run: `rtk git status` and `rtk git diff --stat tests/data/`
Expected: changes confined to `tests/data/phased_dataset.*` (genotype/offset binaries, regenerated `input_regions.arrow`/metadata). If unrelated files changed, investigate before committing.

- [ ] **Step 3: Commit the regenerated data**

```bash
rtk git add tests/data/
rtk git commit -m "test(data): regenerate ground-truth for per-haplotype-minimal VCF/PGEN writes"
```

---

## Task 5: VCF/PGEN-vs-SVAR parity test (acceptance criterion)

**Files:**
- Test: `tests/integration/dataset/test_vcf_pgen_svar_parity.py` (new)

**Context:** The acceptance criterion is that VCF and PGEN datasets produce sparse output identical to SVAR for the same regions/samples. `with_seqs("variants")` exposes the per-haplotype variant set as a `RaggedVariants` awkward array with fields `start`, `alt`, and `ilen`/`ref` of shape `(batch, ploidy, ~variants)`. Comparing `start` and `ilen` across the three sources is the precise sparse-output check (stronger than reconstructed-haplotype bytes, which clipping to region length could mask). Indexing over all regions × all samples uses `ds.n_regions` / `ds.n_samples`.

- [ ] **Step 1: Write the parity test**

Create `tests/integration/dataset/test_vcf_pgen_svar_parity.py`:

```python
"""Acceptance test: VCF and PGEN datasets must produce sparse variant output
identical to the SVAR dataset built from the same variants + BED.

This is the guarantee behind routing VCF/PGEN writes through genoray's
`_dense2sparse_with_length` (per-haplotype-minimal, matching SVAR)."""

import awkward as ak
import numpy as np
import pytest

import genvarloader as gvl


def _all_variants(path, reference):
    ds = gvl.Dataset.open(path, reference=reference).with_seqs("variants")
    r = np.arange(ds.n_regions)
    s = np.arange(ds.n_samples)
    # full cartesian index: every region x every sample
    rv = ds[np.repeat(r, len(s)), np.tile(s, len(r))]
    return rv


@pytest.mark.parametrize("field", ["start", "ilen", "alt"])
def test_vcf_matches_svar(phased_vcf_gvl, phased_svar_gvl, reference, field):
    vcf_rv = _all_variants(phased_vcf_gvl, reference)
    svar_rv = _all_variants(phased_svar_gvl, reference)
    assert ak.to_list(vcf_rv[field]) == ak.to_list(svar_rv[field])


@pytest.mark.parametrize("field", ["start", "ilen", "alt"])
def test_pgen_matches_svar(phased_pgen_gvl, phased_svar_gvl, reference, field):
    pgen_rv = _all_variants(phased_pgen_gvl, reference)
    svar_rv = _all_variants(phased_svar_gvl, reference)
    assert ak.to_list(pgen_rv[field]) == ak.to_list(svar_rv[field])
```

- [ ] **Step 2: Run the parity test**

Run: `pixi run -e dev pytest tests/integration/dataset/test_vcf_pgen_svar_parity.py -v`
Expected: PASS for all parametrizations.

Troubleshooting if it fails:
- `RaggedVariants` may not carry `ilen` when `ref` is present instead — if a `field` KeyErrors, drop it from the `parametrize` list and rely on `start` + `alt` (the always-present fields per the `RaggedVariants` docstring). Confirm available fields with `print(vcf_rv.fields)` in a scratch run.
- If the index/cartesian construction returns an unexpected shape, confirm the indexing contract with `ds[np.array([0]), np.array([0])]` and adjust `_all_variants` to whatever returns one `RaggedVariants` per (region, sample) pair.
- A genuine mismatch (not an API mismatch) means the writer change is incorrect — revisit Task 2/3 rather than weakening the test.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/integration/dataset/test_vcf_pgen_svar_parity.py
rtk git commit -m "test(parity): assert VCF/PGEN sparse output matches SVAR"
```

---

## Task 6: Full suite + type check

**Files:** none (verification only)

- [ ] **Step 1: Run the Python test suite**

Run: `pixi run -e dev pytest tests -q`
Expected: all pass. Pay attention to reconstruction/dataset tests that read the regenerated ground-truth (`tests/unit/dataset/`, `tests/integration/dataset/`). A failure here that traces to changed `chromEnd`/genotype values means a test encoded the *old* over-extended behavior — update that test's expectations to the new per-haplotype-minimal truth (and note it in the commit), do not revert the writer.

- [ ] **Step 2: Type check**

Run: `pixi run -e dev typecheck`
Expected: no new errors introduced by `_write.py`. (Pre-existing unrelated errors from the uncompiled Rust extension, e.g. `count_intervals`/`intervals` in `_bigwig.py`, may appear in a fresh worktree — confirm they are unrelated to `_write.py` and pre-existing on `main`.)

- [ ] **Step 3: Lint**

Run: `pixi run -e dev ruff check python/genvarloader/_dataset/_write.py tests/unit/dataset/genotypes/test_window_to_sparse.py tests/integration/dataset/test_vcf_pgen_svar_parity.py`
Expected: clean (E501 is ignored project-wide).

- [ ] **Step 4: Commit any test-expectation fixups**

```bash
rtk git add -A
rtk git commit -m "test: update expectations for per-haplotype-minimal VCF/PGEN output"
```
(Skip if Step 1 produced no fixups.)

---

## Notes for the implementer

- **No public API change.** `gvl.write(..., extend_to_length=...)` keeps its signature and docstring meaning; the only behavior change is that `False` now actually disables extension for VCF/PGEN. No `__all__` / skill update is required by `CLAUDE.md`'s "Maintaining the genvarloader skill" rule (the parameter, defaults, and accepted values are unchanged). Double-check this judgement against `skills/genvarloader/SKILL.md` — if it documents the per-source extension behavior, update that prose.
- **Memory:** full-window assembly holds one region's dense window at once; `max_mem` still bounds each genoray read-chunk. This is the intended trade-off (genoray's own `read_ranges_with_length` materializes similarly). Do not add a guard.
- **SVAR untouched:** `_write_from_svar` is out of scope.
