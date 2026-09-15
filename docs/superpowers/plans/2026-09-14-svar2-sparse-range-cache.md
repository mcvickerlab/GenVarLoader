# Sparse `svar2_ranges` Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dense `(R, S, P, 2)` `svar2_ranges` var-key memmaps with a region-CSR sparse table holding only non-empty cells, and replace the dense `(R, S, P)` `n_variants` allocation with a zero-stride view.

**Architecture:** A new module `_svar2_ranges.py` owns both on-disk layouts behind one `_RangeLookup` protocol (`lookup` for reads, `entries_for_regions` for concat), resolved by a path-level `_ranges_reader(ranges_dir)` factory. The reader, writer and concat all go through that seam, so the layout swap never leaks into the FFI, the kernels, or `Svar2Haps`'s query code. Sparse entries are 28 bytes (`cell_id` int32 + a 24-byte `(start, len)` record) against dense's 32 bytes per cell, so sparse is strictly smaller at every fill level.

**Tech Stack:** Python 3.10+, numpy (structured arrays, memmap, vectorized binary search), polars, pytest + Hypothesis, `pixi -e dev`. No Rust changes.

**Spec:** `docs/superpowers/specs/2026-09-14-svar2-sparse-range-cache-design.md` — read it first. It carries the evidence for every decision here.

## Global Constraints

- **`DATASET_FORMAT_VERSION` stays `2.0.0`** (`python/genvarloader/_dataset/_write.py:51`). `_validate._check_format_version` matches on MAJOR only; bumping it would make new GVL refuse every existing dataset.
- **Do NOT hand-edit `version` in `pyproject.toml`.** `[tool.commitizen]` uses `version_provider = "pep621"` and `major_version_zero = true`, so a `feat:` commit bumps `0.42.1` -> `0.43.0` at release time. `Cargo.toml` (`0.2.1`) is untouched — `src/` does not change.
- **Target `main`.** Not the `streaming` branch, not the StreamingDataset project board (CLAUDE.md's carve-out: the SVAR2 stream backend builds ranges in memory and never opens `genotypes/svar2_ranges/`).
- **Raw headerless files.** `region_ptr.npy`, `cell_id.npy` and `cell_vk.npy` are raw `tofile` dumps with their shape recorded in `svar2_meta.json`, matching the existing `vk_*_range.npy` / `dense_*_range.npy` convention. Only `sample_cols.npy` is a real `.npy`.
- **`S * P < 2**31`** — asserted at write; `cell_id` is int32.
- **`genoray_core` is pinned** at `d-laub/genoray` rev `d66ec0e` (`Cargo.toml:31-32`). The whole design rests on `gather_haps_readbound_impl` never reading an empty range's `start` (`genoray:src/query/gather.rs:772`). Do not bump genoray in this PR.
- **Pure Python/numpy. No Rust, no numba.** `Cargo.toml` (`0.2.1`) is untouched; `src/` does not change. numba is not available either — gvl's own source is numba-free and `tests/parity/test_import_no_numba.py` enforces it (the `pixi.toml:94-99` pin exists only because seqpro imports it eagerly). Two measured kernel opportunities are deliberately deferred to follow-up issues; see Task 10 Step 9.
- **The Rust-migration roadmap no longer exists.** `docs/roadmaps/rust-migration.md` was retired in `8f9d3c99` and `python/genvarloader/_dispatch.py` (the backend dispatch registry) went with it. Do not cite either. The surviving convention, if a kernel ever lands, is a `#[pyfunction]` in `src/ffi/mod.rs` over a kernel in a domain module, registered in `src/lib.rs`'s `#[pymodule]`, with frozen `.npz` goldens under `tests/parity/` — note that means parity needs a hand-written numpy oracle, since there is no longer a dual-backend harness to diff against.
- **Conventional commits**, enforced by a commitizen prek hook. Prefix each commit `feat:`, `test:`, `refactor:`, `docs:` or `perf:` as the step says.
- **Every command runs under pixi:** `pixi run -e dev pytest ...`, `pixi run -e dev ruff ...`.
- **`np.multiply(..., dtype=np.int64)` for every key computation.** int32 `r_q * S` wraps silently and the later `+ arange(P)` promotes the already-wrapped value, so the dtype gives no tell.

## Deviations from the spec (measured — apply these)

Points where the spec's prose does not survive contact with the code or a stopwatch. Every number below was measured under `pixi -e dev` on an M4 Pro with numpy 1.26.4. Re-measure on the deployment machine before quoting any of them in the PR.

1. **No `argsort` in the sparse probe — settled, not deferred.** The spec's "Probe sorting" section justifies `argsort` by numpy's galloping fast path for sorted needles in `np.searchsorted`. The two-level probe never calls `searchsorted` per query, so that justification does not transfer. Measured: **0.894 ms sorted vs 0.894 ms unsorted** at All of Us chr22, and *slower* sorted at `N = 1e6`. Strike the spec's section rather than leaving a benchmark column keeping the decision open.

2. **The zero-search fast path is table-side, not query-side.** `region_ptr[r+1] - region_ptr[r] == S * P` (region fully occupied) is one vectorized comparison and captures the high-fill regime the 28-byte entry exists for. Measured 0.35 ms vs 0.89 ms. A query-side scatter for the all-samples case is future work, not built.

3. **`np.argsort(kind="stable")` on int64 is NOT radix — the central claim about the write path is false.** numpy wires radix only for integer types of 16 bits or less; numpy's own docstring says otherwise and is wrong. Measured, N = 4e6 random:

   | dtype | ns/element | sort actually used |
   |---|---|---|
   | int16 | 2.9 | radix |
   | int32 | 98.3 | comparison |
   | int64 | 136.3 | comparison |

   The per-contig sort only *looked* linear because timsort's run detection fired on the already-sorted per-chunk runs — and it degrades as chunks shrink, which is exactly the regime the sort was chosen to survive (measured 914 ms at k=30 chunks, 1369 ms at k=500). Task 5 replaces it with a genuine O(N) counting-sort scatter whose auxiliary state is O(regions), never (chunks × regions): **532 ms, flat in k, byte-identical output, 0.56× the peak RSS.**

4. **The streaming k-way merge in concat is replaced by a region-batched gather+sort.** The first draft claimed its Python loop "runs once per block boundary". Wrong: merged keys are `r*span + slot*ploidy + ploid`, so on `axis="samples"` the ownership pattern repeats *inside every region* and the loop runs once per alternation. Two shards with interleaved sample IDs — the normal case for biobank release batches — give ~N/2 iterations, i.e. **~29 minutes of pure Python at the genome projection**, against the ~3 ms the framing predicted. The replacement is simpler *and* interleaving-independent; see Task 8.

5. **No Rust kernel in this PR, on measurement rather than principle.** The probe is 93% numpy-pass-bound and 7% memory-bound, so a fused kernel would win 3–5× single-threaded — worth **0.5% of a 171 ms batch**. On the write path the real hot spot is `np.nonzero` (71% of the chunk kernel), not the four gathers, and even that sits behind genoray materializing a 128 GB dense block per chr22 contig. Both are filed as follow-ups in Task 10 Step 9, with the genoray one flagged as the larger prize.

6. **`0.43.0` is a commitizen outcome** of a `feat:` commit under `version_provider = "pep621"`, not a hand-edit.

---

## File Structure

**Create:**
- `python/genvarloader/_dataset/_svar2_ranges.py` — both layouts, the protocol, the factory, the append/merge helpers. The one place that knows what the files look like.
- `tests/unit/dataset/test_svar2_ranges.py` — pure-numpy unit + Hypothesis parity tests. No genoray store, no fixtures.
- `tests/_oracles/svar2_dense_layout.py` — test-only writer that emits the legacy dense layout (nothing in production can after Task 5).
- `tests/dataset/test_concat_svar2.py` — characterization + merge tests for `_concat_svar2_ranges`.
- `tests/benchmarks/profiling/bench_svar2_range_lookup.py` — the benchmark gate (not collected by pytest).

**Modify:**
- `python/genvarloader/_dataset/_svar2_haps.py` — `_Svar2Cache` (`:111-127`), the `n_regions`/`n_samples` docstrings (`:224-231`), `n_variants` (`:277-279`), `from_path` (`:416-441`), `_gather_inputs` (`:1546-1580`).
- `python/genvarloader/_dataset/_write.py` — `_svar2_preflight` (`:1107-1137`), `_write_from_svar2` (`:1140-1265`), the `gvl.write` docstring (`:162-166`).
- `python/genvarloader/_dataset/_concat.py` — `_concat_svar2_ranges` (`:235-311`).
- `tests/dataset/test_write_svar2.py` — fixtures (`:21-33`, `:53-71`) and six tests.
- `docs/source/format.md`, `docs/source/write.md`, `docs/source/faq.md`, `skills/genvarloader/SKILL.md`.

**Untouched (verified):** `src/` (no `svar2_ranges` reference anywhere), `src/ffi/mod.rs`, `python/genvarloader/_dataset/_open.py` (only checks `svar2_meta.json` exists), `tests/_oracles/svar2_readbound_inputs.py` and `tests/dataset/test_svar2_readbound_haps.py` (they call `svar2._find_ranges` directly and never read the cache files), `tests/unit/dataset/test_svar2_gather_memo.py` (stubs `_gather_inputs` entirely), `docs/source/api.md` and `README.md` (no `__all__` change, no cache mention).

---

## Task 1: Fixture gains empty cells

The current fixture is not actually 100% fill in the vk (sparse) channel: genoray routes each variant to the per-sample sparse channel or a per-region dense channel by carrier-call count (`choose_representation` in genoray's cost model), and only the sparse channel reaches the range cache. With 2 samples/ploidy 2, both multi-carrier indels already route dense, leaving exactly 1 of 8 cells occupied — the fixture's real gap was never fill, it was the absence of *constructed*, cost-model-independent empty structures with a documented reason for being empty. Fix the fixture before writing any code that depends on empty cells existing: add a sample that is genuinely all-reference (S2) and a region that genuinely holds no variants (`[25, 40)`), so both an empty column and an empty row exist regardless of how genoray classifies any given variant.

**Files:**
- Modify: `tests/dataset/test_write_svar2.py:21-33` (the `_VCF` constant), `:53-71` (`svar2_store`), `:84-96` (`test_write_svar2_emits_cache`'s bed)

**Interfaces:**
- Consumes: nothing.
- Produces: a `svar2_store` fixture with samples `["S0", "S1", "S2"]` where `S2` is `0|0` at every site, and a 3-region bed in `test_write_svar2_emits_cache` whose third region `[25, 40)` contains no variants. Every later task's tests rely on both.

- [ ] **Step 1: Add an all-reference third sample to the VCF**

In `tests/dataset/test_write_svar2.py`, replace the `_VCF` constant (currently `:22-33`):

```python
# 40 bp reference (chr1). VCF POS (1-based) -> 0-based: SNP@2 (A>G), INS@6 (C>CAT),
# DEL@11 (GTA>G, ilen -2). Genotypes exercise both samples and both ploids.
# S2 is 0|0 everywhere: an ENTIRELY EMPTY sample column, so the sparse cache's
# "cell not present" branch is exercised. Variants stop at 0-based 13, so a
# region past that (see test_write_svar2_emits_cache) is an entirely empty ROW.
# Mirrors tests/test_svar2_reconstruct.py's svar2_store fixture (which keeps only
# S0/S1), so the matched .svar (SVAR1) store built from the same VCF is still a
# valid parity oracle.
_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1\t0|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1\t0|0
"""
```

- [ ] **Step 2: Select all three samples in the `svar2_store` fixture**

In the `svar2_store` fixture (`:53-71`), change the sample list argument to `run_conversion_pipeline`:

```python
        ["S0", "S1", "S2"],
```

Leave `svar2_store_unsorted` (`:460-484`) at `["S1", "S0"]` — `run_conversion_pipeline` takes an explicit selection, so that fixture keeps exactly 2 samples and its `available_samples == ["S1", "S0"]` / `sample_cols == [1, 0]` assertions stay true.

- [ ] **Step 3: Add an empty region to the layout-oracle bed**

In `test_write_svar2_emits_cache` (`:87-93`), the bed becomes three regions:

```python
    bed = pl.DataFrame(
        {
            # [25, 40) holds no variants at all: an entirely empty region row,
            # which the sparse layout must round-trip as (0, 0) everywhere.
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [0, 5, 25],
            "chromEnd": [20, 15, 40],
        }
    )
```

Do **not** touch `test_write_svar2_chunked_matches_unchunked`'s bed (`:337-339`) — its `max_mem=256` budget is derived from "2 regions x ploidy 2 x 2 channels x 2 endpoints x 8 bytes = 128 bytes per sample", and the comment at `:364-369` explains why. Per-sample bytes do not change when a third sample is added, so it keeps forcing one sample per chunk.

- [ ] **Step 4: Run the svar2 write suite and confirm it still passes**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: PASS. Every assertion in the file is written generically over `sorted(svar2.available_samples)`, so three samples and three regions change nothing yet. If something fails here, the fixture change broke an assumption — fix it now rather than in a later task.

- [ ] **Step 5: Prove the fixture now contains empty cells**

Add this test at the end of `tests/dataset/test_write_svar2.py`. It documents *why* the fixture looks the way it does, and it fails loudly if someone later trims the fixture back.

```python
def test_fixture_has_empty_cells(svar2_store: Path, tmp_path: Path):
    """The fixture must NOT be 100% fill, or the sparse cache is untested.

    Empty cells are the entire point of the sparse layout (#357): if every
    (region, sample, ploid) window holds a variant, the "cell is absent" branch
    never executes and a sparse/dense divergence there is invisible. S2 is 0|0
    everywhere (empty column) and [25, 40) holds no variants (empty row).

    genoray also routes each variant to the per-sample SPARSE (vk) channel or
    the per-region DENSE channel by carrier-call count (see
    `choose_representation` in genoray's cost model), and only the sparse
    channel reaches this grid. This fixture's single-carrier SNP stays sparse
    but both three-carrier indels route dense, so the grid this test inspects
    is already 1/18 fill by construction, not 8/8 -- exactly one cell
    (region 0, S0, ploid 0) is occupied. A cost-model shift that pushed that
    last SNP dense too would empty the grid entirely and make every
    sparse/dense parity test built on this fixture pass vacuously; guard
    against that directly rather than assuming any occupancy at all.
    """
    from genoray import SparseVar2

    svar2 = SparseVar2(svar2_store)
    sorted_samples = sorted(svar2.available_samples)
    assert sorted_samples == ["S0", "S1", "S2"], "fixture lost its third sample"

    d = svar2._find_ranges(
        "chr1",
        np.array([0, 5, 25]),
        np.array([20, 15, 40]),
        samples=sorted_samples,
    )
    snp = np.asarray(d["vk_snp_range"], np.int64)  # (R*S*P, 2)
    indel = np.asarray(d["vk_indel_range"], np.int64)
    nonempty = (snp[:, 1] > snp[:, 0]) | (indel[:, 1] > indel[:, 0])

    S, P = len(sorted_samples), svar2.ploidy
    assert len(nonempty) == 3 * S * P
    # Row-major (R, S, P) -- pinned by the layout oracle in
    # test_write_svar2_emits_cache, which asserts this same reshape against the
    # cache memmaps. Assert each empty structure SEPARATELY: a single
    # `not nonempty.all()` is a disjunction that stays green when either one
    # regresses alone, which is exactly the regression this guard exists to catch.
    grid = nonempty.reshape(3, S, P)
    s2 = sorted_samples.index("S2")
    assert not grid[:, s2].any(), (
        "S2 is no longer all-reference; the empty COLUMN is gone"
    )
    assert not grid[2].any(), (
        "region [25, 40) now holds variants; the empty ROW is gone"
    )
    # Non-vacuity. genoray routes each variant to the per-sample SPARSE (vk)
    # channel or the per-region DENSE channel by carrier-call count, and only
    # the sparse channel lands in this grid: the fixture's single-carrier SNP
    # stays sparse, both three-carrier indels route dense. Exactly one cell is
    # therefore occupied -- (region 0, S0, ploid 0). If a cost-model change
    # pushed that last variant dense too, this grid would be ALL empty and
    # every sparse/dense parity test built on this fixture would pass
    # trivially against an empty table, green and meaningless. Pin it.
    assert grid.any(), (
        "vk channel is entirely empty: genoray routed every variant to the "
        "dense channel, so all sparse-cache parity tests on this fixture are "
        "now vacuous"
    )
    assert grid[0, sorted_samples.index("S0"), 0], (
        "the fixture's one sparse-channel cell (region 0, S0, ploid 0) is gone"
    )
```

- [ ] **Step 6: Run it**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_fixture_has_empty_cells -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/dataset/test_write_svar2.py
git commit -m "test(svar2): give the write fixture empty cells

The fixture was 100% fill (3 variants x 2 samples x ploidy 2, all 8 cells
non-empty), so the sparse range cache's absent-cell branch would have shipped
untested. Adds an all-0|0 sample S2 (empty column) and a [25, 40) region with
no variants (empty row), plus a guard test that fails if either is removed.

Relates to #357"
```

---

## Task 2: Characterize today's dense concat

`_concat_svar2_ranges` (`_concat.py:235-311`) has **zero test coverage** — `tests/dataset/test_concat.py` covers only the `pgen_vcf` and `svar` backends. Without a baseline, Task 8's "test the merge against the dense path" would compare new code to new code. Land the baseline first, against the code as it exists today.

**Files:**
- Create: `tests/dataset/test_concat_svar2.py`

**Interfaces:**
- Consumes: `svar2_store` from Task 1.
- Produces: two module-scoped fixtures, `svar2_shards_by_samples` and `svar2_shards_by_regions`, each returning `(list[Path], Path)` — the shard dataset dirs and the equivalent single-shot `gvl.write` dataset. Task 8 reuses both verbatim.

- [ ] **Step 1: Write the characterization test**

Create `tests/dataset/test_concat_svar2.py`:

```python
"""Characterization + merge tests for `gvl.concat` over a `.svar2` backend.

`_concat_svar2_ranges` had no coverage at all before #357. These tests pin its
OBSERVABLE contract -- a concatenated dataset reads identically to a single-shot
`gvl.write` over the union -- so the sparse-merge refactor has a real baseline
rather than a reference written in the same PR.

They deliberately assert on READS, not on file bytes: the on-disk layout is what
#357 changes, the read semantics are what must not change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl

_BED = pl.DataFrame(
    {
        "chrom": ["chr1", "chr1", "chr1"],
        "chromStart": [0, 5, 25],
        "chromEnd": [20, 15, 40],
    }
)


def _region_starts(path: Path) -> list[int]:
    """A dataset's public region row order, as `chromStart` values.

    `axis="regions"` concat's public row order is each shard's OWN input-bed
    row order concatenated in shard order (`_merged_bed` never re-sorts the
    *public* `input_regions.arrow`, only the on-disk `regions.npy`) -- it is
    NOT, in general, the same permutation as a single-shot write over the
    reassembled bed. `svar2_shards_by_regions` splits rows `[0, 2]` + `[1]`
    (not a contiguous prefix/suffix), which is exactly the non-block case
    where the two orders diverge: the merged dataset's public order comes out
    `[A, B, C]` while a single-shot write's is the original bed order
    `[A, C, B]`. `chromStart` is unique across this fixture's three regions,
    so it is a safe row-identity key -- the same idea as `_region_key_to_row`
    in `tests/dataset/test_concat.py`.
    """
    return pl.read_ipc(path / "input_regions.arrow")["chromStart"].to_list()


def _read_all(path: Path, reference: Path) -> list[np.ndarray]:
    """Every (region, sample) haplotype in the dataset, as padded byte arrays.

    Reading through the public API is the point: it exercises the range cache
    via `Svar2Haps._gather_inputs`, which is what the layout change touches.

    `reference` is the FASTA backing `svar2_store` (from `vcf_and_ref`):
    `Dataset.open` needs an explicit reference to reconstruct haplotypes --
    without one it can only return `RaggedVariants`, not `RaggedSeqs` (see
    `_open.py`'s `_build_seqs`, and every `reference=ref`-passing test in
    `test_svar2_dataset.py`).
    """
    ds = gvl.Dataset.open(path, reference=reference).with_seqs("haplotypes")
    out = []
    for r in range(ds.n_regions):
        for s in range(ds.n_samples):
            hap = ds[r, s]
            out.append(np.asarray(hap.to_padded(b"N")))
    return out


@pytest.fixture(scope="module")
def svar2_shards_by_samples(
    svar2_store: Path, tmp_path_factory
) -> tuple[list[Path], Path]:
    """Two single-sample shards over the same regions + the single-shot oracle."""
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2_concat_samples")
    shards = []
    # Disjoint, sorted sample sets: exactly what _concat_validate requires on
    # axis="samples" (no overlap, and the merged order is sorted(union)).
    for names in (["S0", "S2"], ["S1"]):
        p = d / f"shard_{'_'.join(names)}.gvl"
        gvl.write(
            p, _BED, variants=SparseVar2(svar2_store), samples=names, overwrite=True
        )
        shards.append(p)
    full = d / "full.gvl"
    gvl.write(
        full,
        _BED,
        variants=SparseVar2(svar2_store),
        samples=["S0", "S1", "S2"],
        overwrite=True,
    )
    return shards, full


@pytest.fixture(scope="module")
def svar2_shards_by_regions(
    svar2_store: Path, tmp_path_factory
) -> tuple[list[Path], Path]:
    """Two region shards over the same samples + the single-shot oracle."""
    from genoray import SparseVar2

    d = tmp_path_factory.mktemp("svar2_concat_regions")
    shards = []
    for i, rows in enumerate(([0, 2], [1])):
        p = d / f"shard_{i}.gvl"
        gvl.write(
            p,
            _BED[rows],
            variants=SparseVar2(svar2_store),
            samples=None,
            overwrite=True,
        )
        shards.append(p)
    full = d / "full.gvl"
    gvl.write(
        full, _BED, variants=SparseVar2(svar2_store), samples=None, overwrite=True
    )
    return shards, full


def test_concat_svar2_samples_reads_like_single_write(
    svar2_shards_by_samples, vcf_and_ref: tuple[Path, Path], tmp_path: Path
):
    """Concatenating sample shards must read identically to writing once."""
    shards, full = svar2_shards_by_samples
    ref = vcf_and_ref[1]
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples", overwrite=True)

    ds_m = gvl.Dataset.open(out, reference=ref)
    ds_f = gvl.Dataset.open(full, reference=ref)
    assert ds_m.samples == ds_f.samples
    assert ds_m.n_regions == ds_f.n_regions

    for a, b in zip(_read_all(out, ref), _read_all(full, ref)):
        np.testing.assert_array_equal(a, b)


def test_concat_svar2_regions_reads_like_single_write(
    svar2_shards_by_regions, vcf_and_ref: tuple[Path, Path], tmp_path: Path
):
    """Concatenating region shards must read identically to writing once.

    Matches merged rows to single-shot rows by `chromStart` (see
    `_region_starts`), not raw row position: `svar2_shards_by_regions`'s
    `[0, 2]` + `[1]` split deliberately does not reconstruct the original
    bed's row order when concatenated, so the merged dataset's public region
    order genuinely differs from the single-shot write's. The two assertions
    below pin exactly what each one is -- see their failure messages -- rather
    than leaving the permutation unverified prose. This divergence is not a
    `_concat_svar2_ranges` bug: the underlying `svar2_ranges` cache arrays
    (`vk_snp_range`/`vk_indel_range`/`dense_snp_range`/`dense_indel_range`)
    were verified byte-identical between the two datasets when compared at
    matching on-disk (sorted) rows.
    """
    shards, full = svar2_shards_by_regions
    ref = vcf_and_ref[1]
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions", overwrite=True)

    ds_m = gvl.Dataset.open(out, reference=ref)
    ds_f = gvl.Dataset.open(full, reference=ref)
    assert ds_m.n_regions == ds_f.n_regions
    assert ds_m.samples == ds_f.samples

    n_samples = ds_m.n_samples
    all_m = _read_all(out, ref)
    all_f = _read_all(full, ref)
    starts_m = _region_starts(out)
    row_f_by_start = {start: i for i, start in enumerate(_region_starts(full))}
    assert set(starts_m) == set(row_f_by_start), (
        "merged and single-shot datasets disagree on which regions exist"
    )
    assert starts_m == [0, 25, 5], (
        "axis='regions' concat's public row order is each shard's own input-bed "
        f"row order concatenated in shard order, not a re-sort of the union; got "
        f"{starts_m}, expected [0, 25, 5] (shard 0's rows [0, 25] then shard 1's "
        "row [5])"
    )
    assert _region_starts(full) == [0, 5, 25], (
        "a single-shot write's public row order is the input bed's own row "
        f"order; got {_region_starts(full)}, expected [0, 5, 25] (this fixture's "
        "_BED is already sorted by chromStart)"
    )

    for rm, start in enumerate(starts_m):
        rf = row_f_by_start[start]
        for s in range(n_samples):
            a = all_m[rm * n_samples + s]
            b = all_f[rf * n_samples + s]
            np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Make the `svar2_store` fixture reachable**

`svar2_store` lives in `tests/dataset/test_write_svar2.py`, not in a conftest. Move the three fixtures it depends on — `vcf_and_ref`, `svar2_store`, `svar1_store` (`tests/dataset/test_write_svar2.py:36-82`) — into `tests/dataset/conftest.py` verbatim, along with the `_REF` and `_VCF` constants from Task 1. Delete them from `test_write_svar2.py` and delete any now-unused imports (`subprocess`, `Path` if unused) from that file.

This is the Boy Scout fix, not a workaround: two test modules now need the same store, and a cross-module fixture import would be worse.

- [ ] **Step 3: Run both characterization tests**

Run: `pixi run -e dev pytest tests/dataset/test_concat_svar2.py -v`
Expected: PASS — this characterizes existing behaviour, so it must pass before any refactor. If it FAILS, stop: either the fixtures are wrong or `_concat_svar2_ranges` has a pre-existing bug, and the plan's baseline assumption is void. Investigate before continuing.

- [ ] **Step 4: Confirm the moved fixtures did not break the write suite**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/dataset/test_concat_svar2.py tests/dataset/conftest.py tests/dataset/test_write_svar2.py
git commit -m "test(concat): characterize svar2 range-cache concat

_concat_svar2_ranges had no coverage on either axis, so the sparse-layout
refactor would have had no baseline to test against. Pins the observable
contract -- a concatenated dataset reads identically to a single-shot write --
on both axes, and moves the shared .svar2 store fixtures into conftest.

Relates to #357"
```

---

## Task 3: The sparse layout and its probe

Pure numpy, no genoray, no fixtures. This is the load-bearing algorithm; it gets the heaviest tests.

**Files:**
- Create: `python/genvarloader/_dataset/_svar2_ranges.py`
- Create: `tests/unit/dataset/test_svar2_ranges.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ENTRY_DTYPE: np.dtype` — 24 bytes, fields `snp_start` `<i8`, `indel_start` `<i8`, `snp_len` `<i4`, `indel_len` `<i4`.
  - `class _RangeLookup(Protocol)` with attributes `n_regions: int`, `n_samples: int`, `ploidy: int` and methods `lookup(r_q, si_q, P) -> tuple[NDArray[np.int64], NDArray[np.int64]]`, `entries_for_regions(r0, r1) -> tuple[NDArray[np.int64], NDArray[np.void]]` and `iter_entries() -> Iterator[tuple[NDArray[np.int64], NDArray[np.void]]]`.
  - `class _SparseRanges` with constructor `_SparseRanges(region_ptr, cell_id, cell_vk, n_regions, n_samples, ploidy)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/dataset/test_svar2_ranges.py`. The block below is the file as
finally committed, so it already contains the `iter_entries` tests Step 5 adds and
the validation tests review round 1 asked for — write it in whatever order suits
you, but the module is not done until all of it passes.

```python
"""Unit tests for the svar2 range-cache layouts (#357).

Pure numpy: no genoray store, no dataset on disk. The layouts are the one place
where "an absent cell is an empty range" becomes load-bearing, so parity against
a dense reference is asserted exhaustively and as a Hypothesis property.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from genvarloader._dataset._svar2_ranges import (
    ENTRY_DTYPE,
    _RangeLookup,
    _SparseRanges,
)


def _dense_reference(
    dense: np.ndarray, r_q: np.ndarray, si_q: np.ndarray, P: int
) -> tuple[np.ndarray, np.ndarray]:
    """What `_gather_inputs` does today: fancy-index (R, S, P, 2) memmaps.

    `dense` is (2, R, S, P, 2) -- channel 0 snp, channel 1 indel.
    """
    snp = np.ascontiguousarray(dense[0][r_q, si_q].reshape(-1, 2), np.int64)
    indel = np.ascontiguousarray(dense[1][r_q, si_q].reshape(-1, 2), np.int64)
    return snp, indel


def _sparse_from_dense(dense: np.ndarray, R: int, S: int, P: int) -> _SparseRanges:
    """Build the CSR table the writer would emit for this dense array."""
    snp, indel = dense[0], dense[1]
    ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
    ri, sj, pj = np.nonzero(ne)  # C-order => already key-ascending
    cell_id = (sj.astype(np.int64) * P + pj).astype(np.int32)
    ent = np.empty(len(ri), ENTRY_DTYPE)
    ent["snp_start"] = snp[ri, sj, pj, 0]
    ent["snp_len"] = snp[ri, sj, pj, 1] - snp[ri, sj, pj, 0]
    ent["indel_start"] = indel[ri, sj, pj, 0]
    ent["indel_len"] = indel[ri, sj, pj, 1] - indel[ri, sj, pj, 0]
    region_ptr = np.concatenate([[0], np.bincount(ri, minlength=R).cumsum()]).astype(
        np.int64
    )
    return _SparseRanges(region_ptr, cell_id, ent, R, S, P)


def _random_dense(rng: np.random.Generator, R: int, S: int, P: int, fill: float):
    """A dense (2, R, S, P, 2) cache with `fill` of its cells non-empty.

    Empty cells get a plausible non-zero insertion point, exactly like genoray:
    `(x, x)` with x > 0. This is what makes the parity tests meaningful -- a
    sparse lookup returns (0, 0) there, so they are NOT bit-equal, only
    equal-where-it-matters. See the spec's "empty cells carry no information".
    """
    dense = np.zeros((2, R, S, P, 2), np.int64)
    for ch in (0, 1):
        starts = rng.integers(0, 1000, size=(R, S, P))
        widths = np.where(
            rng.random((R, S, P)) < fill, rng.integers(1, 5, (R, S, P)), 0
        )
        dense[ch, ..., 0] = starts
        dense[ch, ..., 1] = starts + widths
    return dense


def _assert_parity(dense, sparse, r_q, si_q, P):
    """Sparse == dense on PRESENT cells; sparse == (0, 0) on absent ones.

    Presence is a property of the CELL, not of one channel. A cell is stored when
    the SNP range OR the indel range is non-empty, so a present cell whose SNP
    range happens to be empty carries the real `snp_start` with `snp_len == 0` and
    `lookup` returns the true insertion point `(x, x)` -- NOT `(0, 0)`. Only an
    absent cell returns zeros.

    Keying this off `d[:, 1] == d[:, 0]` per channel is wrong and fails on any
    table with mixed per-channel emptiness: 400/400 randomized grids. It is
    invisible at fill 0.0 and fill 1.0, which is exactly why a full-fill test
    would pass and mask it.
    """
    d_snp, d_indel = _dense_reference(dense, r_q, si_q, P)
    s_snp, s_indel = sparse.lookup(r_q, si_q, P)

    assert s_snp.dtype == np.int64 and s_snp.flags.c_contiguous
    assert s_snp.shape == d_snp.shape

    present = (d_snp[:, 1] > d_snp[:, 0]) | (d_indel[:, 1] > d_indel[:, 0])
    for d, s in ((d_snp, s_snp), (d_indel, s_indel)):
        np.testing.assert_array_equal(s[present], d[present])
        # An absent cell must be exactly (0, 0): unconditionally in bounds for
        # the Rust slicing in gather_haps_readbound_impl.
        np.testing.assert_array_equal(s[~present], 0)


def test_lookup_parity_partial_fill():
    rng = np.random.default_rng(0)
    R, S, P = 7, 5, 2
    dense = _random_dense(rng, R, S, P, fill=0.3)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_lookup_parity_full_fill_uses_contiguous_path(monkeypatch):
    """100% fill: every region block is complete, so no binary search runs.

    Asserted, not assumed: this is the whole high-fill regime (sequence-model
    windows run at ~100% fill), measured at 0.35 ms against 0.89 ms, and a fast
    path that silently stopped firing would cost 2.5x with every test still
    green.
    """
    rng = np.random.default_rng(1)
    R, S, P = 4, 6, 2
    dense = _random_dense(rng, R, S, P, fill=1.0)
    sparse = _sparse_from_dense(dense, R, S, P)
    assert len(sparse.cell_id) == R * S * P, "fill=1.0 did not produce a full table"

    def boom(*a, **k):
        raise AssertionError("the full-block fast path did not fire")

    monkeypatch.setattr(_SparseRanges, "_lower_bound", boom)
    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_lookup_parity_mixed_full_and_partial_regions():
    """One partial region must not break the full ones sharing the query.

    The fast path is all-or-nothing per CALL -- `(hi - lo) == span` has to hold
    for every queried region -- so a mixed table sends full regions through the
    search too. They must come out identical either way; a search that assumed
    "not full" would be off by the block's own width.
    """
    rng = np.random.default_rng(12)
    R, S, P = 5, 4, 2
    dense = _random_dense(rng, R, S, P, fill=1.0)
    # Empty out one cell in region 2 only: regions 0, 1, 3, 4 stay complete.
    dense[0][2, 1, 0] = (0, 0)
    dense[1][2, 1, 0] = (0, 0)
    sparse = _sparse_from_dense(dense, R, S, P)
    assert len(sparse.cell_id) == R * S * P - 1
    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    _assert_parity(dense, sparse, r_q, si_q, P)
    # And the full regions alone must still take the fast path.
    keep = r_q != 2
    _assert_parity(dense, sparse, r_q[keep], si_q[keep], P)


def test_lookup_empty_table():
    """N == 0 must not memmap, not raise, and not probe."""
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    snp, indel = sparse.lookup(np.array([0, 2]), np.array([1, 3]), P)
    assert snp.shape == (2 * P, 2)
    np.testing.assert_array_equal(snp, 0)
    np.testing.assert_array_equal(indel, 0)


def test_lookup_zero_queries():
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    snp, indel = sparse.lookup(np.array([], np.int64), np.array([], np.int64), P)
    assert snp.shape == (0, 2) and indel.shape == (0, 2)


def test_lookup_duplicate_and_unsorted_queries():
    """Order and repetition must not change the answer, row for row."""
    rng = np.random.default_rng(2)
    R, S, P = 5, 4, 2
    dense = _random_dense(rng, R, S, P, fill=0.4)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q = np.array([4, 0, 4, 2, 0, 0])
    si_q = np.array([3, 1, 3, 0, 1, 2])
    _assert_parity(dense, sparse, r_q, si_q, P)


@pytest.mark.parametrize("bad", ["region", "sample"])
def test_lookup_out_of_bounds_raises(bad: str):
    """A miss and an empty cell are indistinguishable, so bounds must be checked.

    Today `vk_snp_range[r_q, si_q]` raises IndexError. Without this guard a bad
    index would silently return (0, 0) -- a reference-only haplotype, no error.
    """
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    r_q = np.array([R if bad == "region" else 0])
    si_q = np.array([0 if bad == "region" else S])
    with pytest.raises(IndexError, match=bad):
        sparse.lookup(r_q, si_q, P)


def test_lookup_int32_indices_are_accepted():
    """int32 r_q/si_q are accepted and agree with int64 on the same query.

    This does NOT exercise wrapping: under the spec's own ``S * P < 2**31``
    invariant, ``si_q * P`` cannot overflow int32 at these (or any legal)
    sizes, and the subsequent ``+ arange(P)`` promotes to int64 regardless of
    whether ``si_q * P`` wrapped. The ``dtype=np.int64`` on that multiply is
    defence in depth against a hypothetical caller that violates the
    invariant, not something this test can force to matter.
    """
    R, S, P = 3, 4, 2
    rng = np.random.default_rng(3)
    dense = _random_dense(rng, R, S, P, fill=0.5)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q = np.arange(R * S, dtype=np.int32) // S
    si_q = np.arange(R * S, dtype=np.int32) % S
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_entry_dtype_is_28_bytes_per_entry():
    """24-byte record + 4-byte cell_id = 28 B, against dense's 32 B per cell.

    This is what removes #357's fill threshold: sparse is strictly smaller at
    EVERY fill level, so there is no crossover and no dense-fallback writer.
    """
    assert ENTRY_DTYPE.itemsize == 24
    assert ENTRY_DTYPE.itemsize + np.dtype(np.int32).itemsize == 28
    assert ENTRY_DTYPE.names == ("snp_start", "indel_start", "snp_len", "indel_len")


@settings(deadline=None, max_examples=75)
@given(
    R=st.integers(1, 6),
    S=st.integers(1, 6),
    P=st.integers(1, 3),
    fill=st.floats(0.0, 1.0),
    seed=st.integers(0, 2**32 - 1),
    n_q=st.integers(0, 20),
)
def test_lookup_parity_property(
    R: int, S: int, P: int, fill: float, seed: int, n_q: int
):
    """Sparse and dense must agree for every grid, fill and query set."""
    rng = np.random.default_rng(seed)
    dense = _random_dense(rng, R, S, P, fill)
    sparse = _sparse_from_dense(dense, R, S, P)
    r_q = rng.integers(0, R, n_q)
    si_q = rng.integers(0, S, n_q)
    _assert_parity(dense, sparse, r_q, si_q, P)


def test_iter_entries_is_sorted_and_complete():
    """Concat's merge requires ascending, gap-free, complete key streams."""
    rng = np.random.default_rng(4)
    R, S, P = 9, 7, 2
    dense = _random_dense(rng, R, S, P, fill=0.35)
    sparse = _sparse_from_dense(dense, R, S, P)

    keys = np.concatenate(
        [k for k, _ in sparse.iter_entries()] or [np.empty(0, np.int64)]
    )
    ents = np.concatenate(
        [e for _, e in sparse.iter_entries()] or [np.empty(0, ENTRY_DTYPE)]
    )
    assert len(keys) == len(sparse.cell_id)
    assert np.all(np.diff(keys) > 0), "keys must be strictly ascending"

    snp, indel = dense[0], dense[1]
    ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
    ri, sj, pj = np.nonzero(ne)
    np.testing.assert_array_equal(keys, ri * (S * P) + sj * P + pj)
    np.testing.assert_array_equal(ents["snp_start"], snp[ri, sj, pj, 0])


def test_iter_entries_empty_table():
    R, S, P = 4, 3, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    assert list(sparse.iter_entries()) == []


def test_sparse_ranges_satisfies_range_lookup_protocol() -> None:
    """Static, not runtime: pyrefly checks this assignment against `_RangeLookup`.

    The assignment itself has no runtime effect -- annotating a local doesn't
    check anything at import time or under pytest. It exists so a `pyrefly
    check` run fails the moment `_SparseRanges`'s public methods drift from the
    Protocol Tasks 4 and 5 must also match; nothing else in this module binds
    the two together.
    """
    lookup_iface: _RangeLookup = _SparseRanges(
        np.zeros(4, np.int64), np.empty(0, np.int32), np.empty(0, ENTRY_DTYPE), 3, 4, 2
    )
    assert lookup_iface.n_regions == 3


def test_post_init_rejects_region_ptr_length_mismatch():
    R, S, P = 3, 4, 2
    with pytest.raises(ValueError, match="n_regions"):
        _SparseRanges(
            np.zeros(R, np.int64),  # one short of n_regions + 1
            np.empty(0, np.int32),
            np.empty(0, ENTRY_DTYPE),
            R,
            S,
            P,
        )


def test_post_init_rejects_cell_id_cell_vk_length_mismatch():
    R, S, P = 3, 4, 2
    with pytest.raises(ValueError, match="parallel"):
        _SparseRanges(
            np.array([0, 0, 0, 1], np.int64),
            np.zeros(1, np.int32),
            np.empty(0, ENTRY_DTYPE),
            R,
            S,
            P,
        )


def test_post_init_rejects_non_monotonic_region_ptr():
    """A writer bug that emits a decreasing region_ptr must raise construction,
    not silently mis-route `lookup` to the wrong region's block.

    Before this fix, `region_ptr=[0, 2, 1, 3]` constructed without error and
    `lookup` silently returned `(0, 0)` for affected queries -- a
    reference-only haplotype with no signal anything was wrong.
    """
    R, S, P = 3, 4, 2
    with pytest.raises(ValueError, match="non-decreasing"):
        _SparseRanges(
            np.array([0, 2, 1, 3], np.int64),
            np.zeros(3, np.int32),
            np.zeros(3, ENTRY_DTYPE),
            R,
            S,
            P,
        )


def test_post_init_rejects_truncated_region_ptr():
    """`region_ptr[-1]` must equal `len(cell_id)`.

    Before this fix, `region_ptr=[0, 1, 2]` against a 3-entry `cell_id`
    constructed without error, permanently stranding the third entry: no
    region's block could ever reach it, and `lookup` would return
    wrong-but-plausible ranges for the regions that do validate.
    """
    S, P = 4, 2
    cell_id = np.zeros(3, np.int32)
    ent = np.zeros(3, ENTRY_DTYPE)
    region_ptr = np.array([0, 1, 2], np.int64)  # claims 2 entries, cell_id has 3
    with pytest.raises(ValueError, match=r"region_ptr\[-1\]"):
        _SparseRanges(region_ptr, cell_id, ent, 2, S, P)


def test_lookup_rejects_ploidy_mismatch():
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    with pytest.raises(ValueError, match="ploidy"):
        sparse.lookup(np.array([0]), np.array([0]), P + 1)


@pytest.mark.parametrize("r0,r1", [(2, 0), (-1, 2), (0, 10)])
def test_entries_for_regions_rejects_invalid_range(r0: int, r1: int):
    """Inverted, negative, or past-n_regions bounds must raise -- not silently
    return empty, which is indistinguishable from "the range held no cells".

    Before this fix, `entries_for_regions(2, 0)` and `entries_for_regions(-1, 2)`
    both returned empty with no error (the inverted range's `b <= a` guard
    absorbed the first; negative indices simply wrapped into `region_ptr` for
    the second), while `r1 > n_regions` already raised a numpy IndexError --
    an asymmetric guard on the one primitive `concat`'s merge is built on.
    """
    R, S, P = 3, 4, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64),
        np.empty(0, np.int32),
        np.empty(0, ENTRY_DTYPE),
        R,
        S,
        P,
    )
    with pytest.raises(IndexError):
        sparse.entries_for_regions(r0, r1)


def test_lookup_needs_both_lo_and_hi_bound_checks():
    """The two-sided `lo <= pos < hi` hit test has two independently necessary
    halves, each guarding a distinct miss shape that the `cid[clamped] ==
    target` check alone does NOT catch.

    `_lower_bound`'s docstring says the `np.minimum` clamp exists only for a
    trailing empty region block; none of the other nine deterministic tests
    constructs one, and `test_lookup_parity_partial_fill` (R=7, S=5, fill=0.3)
    has roughly a 0.1% chance of doing so by chance. This table is built so the
    *coincidentally adjacent* `cid` value equals each query's target -- which
    is what makes each half of the hit test load-bearing rather than redundant
    with the `cid` equality check:

    - r=0, slot 2 is absent from region 0's block (only slots 0, 1 are
      present), but region 1's single entry has `cid == 2` right after region
      0's block ends. The search lands at `pos == hi`; only `pos < hi` rejects
      the bleed into region 1's entry.
    - r=2 (the LAST region) is empty, so `pos` clamps below `lo` to region 1's
      last entry, whose `cid` also happens to equal 2. Only `lo <= pos` rejects
      matching that unrelated, out-of-block entry.
    """
    R, S, P = 3, 3, 1
    cell_id = np.array([0, 1, 2], np.int32)  # region 0: slots 0, 1; region 1: slot 2
    ent = np.zeros(3, ENTRY_DTYPE)
    ent["snp_start"] = [10, 20, 30]
    ent["snp_len"] = [1, 1, 1]
    region_ptr = np.array([0, 2, 3, 3], np.int64)  # region 2 (LAST) is empty
    sparse = _SparseRanges(region_ptr, cell_id, ent, R, S, P)

    snp, indel = sparse.lookup(np.array([0, 2]), np.array([2, 2]), P)
    np.testing.assert_array_equal(snp, 0)
    np.testing.assert_array_equal(indel, 0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_ranges.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'genvarloader._dataset._svar2_ranges'`.

- [ ] **Step 3: Write the module**

Create `python/genvarloader/_dataset/_svar2_ranges.py`:

```python
"""On-disk layouts for the ``genotypes/svar2_ranges/`` var-key range cache.

A ``.svar2``-backed dataset caches, per ``(region, sample, ploid)``, the byte
window into its contig's var-key tables that the read-bound kernel must walk.
The obvious layout is a dense ``(R, S, P, 2)`` array -- and it is unusable at
cohort scale: 128 GB for one chromosome of All of Us, over 99% of it empty
(#357).

The **sparse** layout stores only the cells that hold a variant, in region-CSR
order. It is exact, not approximate, because an empty range carries no
information: ``gather_haps_readbound_impl`` slices ``positions[vs..ve]`` and
derives ``j = vs + k`` *inside* the loop body, so ``(0, 0)`` is byte-identical
to the true insertion point -- and strictly safer, since Rust slicing panics if
``vs > ve`` while ``(0, 0)`` is unconditionally in bounds. See the design spec
at ``docs/superpowers/specs/2026-09-14-svar2-sparse-range-cache-design.md``.

Both layouts live behind :class:`_RangeLookup`, resolved from a directory by
:func:`_ranges_reader`, so the reader, the writer and ``concat`` share one
definition of what these files mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = ["ENTRY_DTYPE", "_RangeLookup", "_SparseRanges"]

ENTRY_DTYPE = np.dtype(
    [
        ("snp_start", "<i8"),
        ("indel_start", "<i8"),
        ("snp_len", "<i4"),
        ("indel_len", "<i4"),
    ]
)
"""One sparse cache entry: 24 bytes, plus a 4-byte ``cell_id`` = 28 B per cell.

Dense costs 32 B per ``(region, sample, ploid)`` cell, so sparse is strictly
smaller at *every* fill level -- which is what lets this ship with no fill
threshold and no dense-fallback writer. Lengths rather than ends: one add on
gather, eight bytes saved per entry. Fields are ordered i8, i8, i4, i4 so the
record is naturally aligned and numpy adds no padding.
"""

ITER_BLOCK_ENTRIES = 1 << 20
"""Target entries per :meth:`_RangeLookup.iter_entries` block (~28 MB sparse).

A target, not a hard cap: a block never splits one region's cells across two
yields, so one region is the floor and a single region wider than this (only
possible if ``S * P > 2**20``) yields as one block larger than the target.
"""


class _RangeLookup(Protocol):
    """A var-key range cache, whatever its on-disk layout.

    Attributes:
        n_regions: ``R`` -- BED rows in the dataset.
        n_samples: ``S`` -- selected samples.
        ploidy: ``P``.
    """

    n_regions: int
    n_samples: int
    ploidy: int

    def lookup(
        self, r_q: NDArray[np.integer], si_q: NDArray[np.integer], P: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """The SNP and indel var-key ranges for a query block.

        Args:
            r_q: Region index per query. Must be in ``[0, n_regions)``.
            si_q: Sample slot per query, parallel to ``r_q``. In ``[0, n_samples)``.
            P: Ploidy; must equal :attr:`ploidy`.

        Returns:
            ``(vk_snp, vk_indel)``, each ``(len(r_q) * P, 2)`` C-contiguous int64
            in ``row = q * P + p`` order, which is what the Rust kernel expects.

        Raises:
            IndexError: If any region or sample index is out of bounds. A sparse
                miss is indistinguishable from an empty cell, so this cannot be
                left to fancy-indexing.
            ValueError: If ``r_q`` and ``si_q`` have different lengths, or ``P``
                does not equal :attr:`ploidy`.
        """
        ...

    def entries_for_regions(
        self, r0: int, r1: int
    ) -> tuple[NDArray[np.int64], NDArray[np.void]]:
        """Non-empty cells of regions ``[r0, r1)``, in ascending global-key order.

        The one primitive ``concat`` and :meth:`iter_entries` are both built on.
        Region-bounded rather than entry-bounded because ``concat`` merges by
        *merged region batch*: it needs "everything these regions hold", and a
        reader cannot answer that from an entry-count-bounded stream.

        Args:
            r0: First region, inclusive. Must be in ``[0, n_regions]``.
            r1: Last region, exclusive. Must be in ``[r0, n_regions]``.

        Returns:
            ``(key, entries)`` where ``key`` is int64
            ``r * (n_samples * ploidy) + slot * ploidy + ploid`` and ``entries``
            is a parallel :data:`ENTRY_DTYPE` array. Both are empty if the
            region range holds no non-empty cell.

        Raises:
            IndexError: If ``r0``/``r1`` violate ``0 <= r0 <= r1 <= n_regions``.
                An inverted or negative range must not be indistinguishable
                from a valid range that simply holds no cells.
        """
        ...

    def iter_entries(self) -> Iterator[tuple[NDArray[np.int64], NDArray[np.void]]]:
        """Non-empty cells in ascending global-key order, in blocks.

        A thin region-batched loop over :meth:`entries_for_regions`. Used by
        ``concat``'s dense-input path and by tests; never by the read path.

        Yields:
            ``(key, entries)``, as :meth:`entries_for_regions` returns them.
            Empty blocks are skipped.
        """
        ...


def _check_bounds(r_q: NDArray[np.integer], si_q: NDArray[np.integer], R: int, S: int):
    """Fail loudly on an out-of-range index.

    The dense layout got this free from fancy-indexing. Sparse does not: a probe
    that misses looks exactly like an absent (i.e. empty) cell, so an off-by-one
    would silently yield a reference-only haplotype instead of an IndexError.
    """
    if len(r_q) != len(si_q):
        raise ValueError(
            f"r_q and si_q must be parallel, got {len(r_q)} and {len(si_q)}"
        )
    if len(r_q) == 0:
        return
    if r_q.min() < 0 or r_q.max() >= R:
        raise IndexError(
            f"region index out of bounds: [{r_q.min()}, {r_q.max()}] not within [0, {R})"
        )
    if si_q.min() < 0 or si_q.max() >= S:
        raise IndexError(
            f"sample index out of bounds: [{si_q.min()}, {si_q.max()}] not within [0, {S})"
        )


@dataclass(slots=True)
class _SparseRanges:
    """Region-CSR table of non-empty ``(region, sample, ploid)`` cells.

    ``region_ptr[r]:region_ptr[r + 1]`` is region ``r``'s block; within a block,
    ``cell_id == slot * ploidy + ploid`` ascends, so a cell is found by bounded
    binary search. Search depth is fixed once from the table's WIDEST region
    block, not its average: one densely-filled region forces that many
    iterations for every query this table ever serves, including ones landing
    on sparse regions. ``region_ptr`` itself costs 1.6 MB genome-wide against a
    27 GB table.
    """

    region_ptr: NDArray[np.int64]
    cell_id: NDArray[np.int32]
    cell_vk: NDArray[np.void]
    n_regions: int
    n_samples: int
    ploidy: int
    _cell_span: int = field(init=False, repr=False, default=0)
    _depth: int = field(init=False, repr=False, default=0)

    def __post_init__(self):
        if len(self.region_ptr) != self.n_regions + 1:
            raise ValueError(
                f"region_ptr must have n_regions + 1 = {self.n_regions + 1} entries,"
                f" got {len(self.region_ptr)}"
            )
        if len(self.cell_id) != len(self.cell_vk):
            raise ValueError(
                f"cell_id ({len(self.cell_id)}) and cell_vk ({len(self.cell_vk)})"
                " must be parallel"
            )
        self._cell_span = self.n_samples * self.ploidy
        # Hoisted out of the probe. Computing this per lookup() is an O(R) pass
        # over a memmap-backed array: measured 1.5 us at R = 3,734 but 98 us at a
        # genome-scale R = 202,053, i.e. 2.9% of the whole 3.42 ms lookup budget
        # spent recomputing a constant. The same pass doubles as validation: a
        # truncated or mis-cumsum'd region_ptr is otherwise silent -- `lookup`
        # would return a wrong-but-plausible range instead of raising, because a
        # bad probe result looks exactly like a miss.
        w = self.region_ptr[1:] - self.region_ptr[:-1]
        if len(w) and w.min() < 0:
            bad = int(w.argmin())
            raise ValueError(
                "region_ptr must be non-decreasing, got region_ptr"
                f"[{bad}]={int(self.region_ptr[bad])} > region_ptr[{bad + 1}]="
                f"{int(self.region_ptr[bad + 1])}"
            )
        if len(self.region_ptr) and int(self.region_ptr[-1]) != len(self.cell_id):
            raise ValueError(
                f"region_ptr[-1] ({int(self.region_ptr[-1])}) must equal"
                f" len(cell_id) ({len(self.cell_id)}); the table is truncated"
            )
        widest = int(w.max()) if len(w) else 0
        # partition_point halves `size` to 1 in exactly ceil(log2(widest)) steps.
        self._depth = max(1, (widest - 1).bit_length())

    def lookup(
        self, r_q: NDArray[np.integer], si_q: NDArray[np.integer], P: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        if P != self.ploidy:
            raise ValueError(f"query ploidy {P} != cache ploidy {self.ploidy}")
        r_q = np.atleast_1d(np.asarray(r_q))
        si_q = np.atleast_1d(np.asarray(si_q))
        _check_bounds(r_q, si_q, self.n_regions, self.n_samples)

        n = len(r_q)
        vk_snp = np.zeros((n * P, 2), np.int64)
        vk_indel = np.zeros((n * P, 2), np.int64)
        if n == 0 or len(self.cell_id) == 0:
            return vk_snp, vk_indel

        cid = self.cell_id
        last = len(cid) - 1

        # int64 throughout: int32 si_q * P wraps silently and the later promotion
        # hides it, because the *wrapped* value is what gets promoted.
        # One broadcast rather than repeat + tile + add.
        target = (
            np.multiply(si_q, P, dtype=np.int64)[:, None] + np.arange(P, dtype=np.int64)
        ).reshape(-1)
        # region_ptr is gathered n times, not n * P times, then broadcast.
        r64 = np.asarray(r_q, np.int64)
        lo1 = self.region_ptr[r64]
        hi1 = self.region_ptr[r64 + 1]
        lo = np.repeat(lo1, P)

        if ((hi1 - lo1) == self._cell_span).all():
            # Every queried region block holds every cell, so cell_id is exactly
            # arange(S * P) there and position = lo + target with no search and no
            # hit test. This is the whole high-fill regime -- sequence-model
            # windows run at ~100% fill -- measured at 0.35 ms against 0.89 ms.
            pos = lo + target
            e = self.cell_vk[pos]
            vk_snp[:, 0] = e["snp_start"]
            np.add(e["snp_start"], e["snp_len"], out=vk_snp[:, 1])
            vk_indel[:, 0] = e["indel_start"]
            np.add(e["indel_start"], e["indel_len"], out=vk_indel[:, 1])
            return vk_snp, vk_indel

        hi = np.repeat(hi1, P)
        pos = self._lower_bound(lo, hi, target)
        # `lo <= pos` is what makes an empty trailing block (lo == hi == N, where
        # `base` had to be clamped) report a miss rather than matching the
        # previous region's last entry.
        hit = (lo <= pos) & (pos < hi)
        clamped = np.minimum(pos, last)
        np.logical_and(hit, cid[clamped] == target, out=hit)

        e = self.cell_vk[clamped]
        for out, start, length in (
            (vk_snp, e["snp_start"], e["snp_len"]),
            (vk_indel, e["indel_start"], e["indel_len"]),
        ):
            s = np.where(hit, start, 0)  # already int64; no .astype
            out[:, 0] = s
            np.add(s, np.where(hit, length, 0), out=out[:, 1])
        return vk_snp, vk_indel

    def _lower_bound(
        self, lo: NDArray[np.int64], hi: NDArray[np.int64], target: NDArray[np.int64]
    ) -> NDArray[np.int64]:
        """Per-element ``lower_bound`` of ``target`` in ``cell_id[lo:hi]``.

        ``np.searchsorted`` cannot express per-element bounds, so this is a
        manually vectorized search: the loop is over bit-depth, never over
        queries. It is the branchless ``std::partition_point`` form rather than
        the textbook ``(lo, hi)`` form for one reason -- it is *self-stabilizing*.
        Once a block's ``size`` reaches 1, ``half == 0``, ``mid == base`` and the
        update is a no-op, so extra iterations are free and no ``active = lo < hi``
        guard is needed. That drops the loop from twelve O(n) temporaries per
        iteration to two, and the whole probe by 1.13%.

        (The textbook form genuinely *needs* that guard: once ``lo == hi`` you get
        ``mid == lo``, and a true ``go`` sets ``lo = mid + 1 > hi``, so ``lo``
        drifts upward by one per remaining iteration. Do not delete the guard from
        that formulation -- this one removes the need for it instead.)

        The single ``np.minimum`` is needed only when the table's *last* region
        block is empty, which puts ``lo == len(cell_id)``; for every other block
        ``base`` stays within ``[lo, hi - 1]`` by construction. Padding ``cell_id``
        with a sentinel instead would force a full RAM copy of a memmap -- 27 GB
        genome-wide -- to avoid one clamp.
        """
        size = hi - lo
        base = np.minimum(lo, len(self.cell_id) - 1)
        cid = self.cell_id
        half = np.empty_like(size)
        mid = np.empty_like(size)
        go = np.empty(len(size), bool)
        for _ in range(self._depth):
            np.right_shift(size, 1, out=half)
            np.add(base, half, out=mid)
            np.less(cid[mid], target, out=go)
            base = np.where(go, mid, base)
            np.subtract(size, half, out=size)
        np.less(cid[base], target, out=go)
        base += go
        return base

    def entries_for_regions(
        self, r0: int, r1: int
    ) -> tuple[NDArray[np.int64], NDArray[np.void]]:
        """Non-empty cells of regions ``[r0, r1)``, key-ascending.

        The primitive both :meth:`iter_entries` and ``concat``'s region-batched
        merge are built on. Keys are dataset-global
        ``r * (n_samples * ploidy) + slot * ploidy + ploid``.

        Raises:
            IndexError: If ``r0``/``r1`` violate ``0 <= r0 <= r1 <= n_regions``.
        """
        if not 0 <= r0 <= r1 <= self.n_regions:
            raise IndexError(
                f"region range out of bounds: got r0={r0}, r1={r1}, expected"
                f" 0 <= r0 <= r1 <= n_regions={self.n_regions}"
            )
        a, b = int(self.region_ptr[r0]), int(self.region_ptr[r1])
        if b <= a:
            return np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE)
        rows = np.repeat(
            np.arange(r0, r1, dtype=np.int64), np.diff(self.region_ptr[r0 : r1 + 1])
        )
        return (
            rows * self._cell_span + self.cell_id[a:b].astype(np.int64),
            np.asarray(self.cell_vk[a:b]),
        )

    def iter_entries(self) -> Iterator[tuple[NDArray[np.int64], NDArray[np.void]]]:
        rows = max(1, ITER_BLOCK_ENTRIES // max(self._cell_span, 1))
        for r0 in range(0, self.n_regions, rows):
            key, ent = self.entries_for_regions(r0, min(r0 + rows, self.n_regions))
            if len(key):
                yield key, ent
```

- [ ] **Step 4: Run the tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_ranges.py -q`
Expected: PASS, all cases including the Hypothesis property.

- [ ] **Step 5: Add and run an `iter_entries` test**

Append to `tests/unit/dataset/test_svar2_ranges.py`:

```python
def test_iter_entries_is_sorted_and_complete():
    """Concat's merge requires ascending, gap-free, complete key streams."""
    rng = np.random.default_rng(4)
    R, S, P = 9, 7, 2
    dense = _random_dense(rng, R, S, P, fill=0.35)
    sparse = _sparse_from_dense(dense, R, S, P)

    keys = np.concatenate([k for k, _ in sparse.iter_entries()] or [np.empty(0, np.int64)])
    ents = np.concatenate(
        [e for _, e in sparse.iter_entries()] or [np.empty(0, ENTRY_DTYPE)]
    )
    assert len(keys) == len(sparse.cell_id)
    assert np.all(np.diff(keys) > 0), "keys must be strictly ascending"

    snp, indel = dense[0], dense[1]
    ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
    ri, sj, pj = np.nonzero(ne)
    np.testing.assert_array_equal(keys, ri * (S * P) + sj * P + pj)
    np.testing.assert_array_equal(ents["snp_start"], snp[ri, sj, pj, 0])


def test_iter_entries_empty_table():
    R, S, P = 4, 3, 2
    sparse = _SparseRanges(
        np.zeros(R + 1, np.int64), np.empty(0, np.int32), np.empty(0, ENTRY_DTYPE), R, S, P
    )
    assert list(sparse.iter_entries()) == []
```

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_ranges.py -q`
Expected: PASS.

- [ ] **Step 6: Lint and typecheck**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck`
Expected: clean. If pyrefly reports zero files checked, see `reference_pyrefly_worktree_vacuous_check` — a worktree under gitignored `.claude/worktrees/` silently checks nothing; un-exclude it in `.git/info/exclude` locally.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_svar2_ranges.py tests/unit/dataset/test_svar2_ranges.py
git commit -m "feat(svar2): add the sparse region-CSR range layout

A 28-byte CSR entry (int32 cell_id + a 24-byte start/len record) against
dense's 32 bytes per (region, sample, ploid) cell, so sparse is strictly
smaller at every fill level. Lookup is a bounded, manually vectorized binary
search inside one region's block -- log2(N/R) rather than log2(N) -- with a
zero-search path for fully occupied regions.

Bounds, int64-key and empty-table guards are included: once an absent cell and
an empty cell are indistinguishable, a bad index would silently yield a
reference-only haplotype instead of raising.

Relates to #357"
```

---

## Task 4: The dense layout, the factory, and the reader seam

Wire `Svar2Haps` through `_RangeLookup` **without changing the on-disk format**. The tree stays green and every existing dataset keeps working; only the indirection lands.

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_ranges.py`
- Modify: `python/genvarloader/_dataset/_svar2_haps.py:111-127, 224-231, 416-441, 1546-1580`
- Modify: `tests/unit/dataset/test_svar2_ranges.py`

**Interfaces:**
- Consumes: `ENTRY_DTYPE`, `_RangeLookup`, `_SparseRanges`, `_check_bounds` from Task 3.
- Produces:
  - `class _DenseRanges` with constructor `_DenseRanges(vk_snp_range, vk_indel_range, n_regions, n_samples, ploidy)`.
  - `def _ranges_reader(ranges_dir: Path) -> _RangeLookup` — reads `svar2_meta.json`, dispatches on `meta.get("layout", "dense")`, cross-checks the grid, and returns a ready lookup. Task 5's writer and Task 8's concat both call it.
  - `_Svar2Cache` loses `vk_snp_range`/`vk_indel_range` and gains `ranges: _RangeLookup`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/dataset/test_svar2_ranges.py`:

```python
def _write_dense_layout(d, dense: np.ndarray, R: int, S: int, P: int):
    """Emit the legacy dense layout at `d`, exactly as gvl <= 0.42.1 did."""
    import json

    d.mkdir(parents=True, exist_ok=True)
    for name, arr in (("vk_snp_range", dense[0]), ("vk_indel_range", dense[1])):
        np.asarray(arr, np.int64).tofile(d / f"{name}.npy")
    for name in ("dense_snp_range", "dense_indel_range"):
        np.zeros((R, 2), np.int64).tofile(d / f"{name}.npy")
    np.save(d / "sample_cols.npy", np.arange(S, dtype=np.int64))
    (d / "svar2_meta.json").write_text(
        json.dumps(
            {
                "vk_snp_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
                "vk_indel_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
                "dense_snp_range": {"shape": [R, 2], "dtype": "<i8"},
                "dense_indel_range": {"shape": [R, 2], "dtype": "<i8"},
                "sample_cols": {"shape": [S], "dtype": "<i8"},
                "ploidy": P,
            }
        )
    )


def test_dense_ranges_matches_fancy_indexing(tmp_path):
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    rng = np.random.default_rng(5)
    R, S, P = 5, 4, 2
    dense = _random_dense(rng, R, S, P, fill=0.5)
    _write_dense_layout(tmp_path, dense, R, S, P)

    reader = _ranges_reader(tmp_path)
    assert (reader.n_regions, reader.n_samples, reader.ploidy) == (R, S, P)

    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    got_snp, got_indel = reader.lookup(r_q, si_q, P)
    exp_snp, exp_indel = _dense_reference(dense, r_q, si_q, P)
    # The dense reader is the status quo: bit-equal, insertion points included.
    np.testing.assert_array_equal(got_snp, exp_snp)
    np.testing.assert_array_equal(got_indel, exp_indel)


def test_dense_and_sparse_iter_entries_agree(tmp_path):
    """Dense -> sparse concat feeds off _DenseRanges.iter_entries; it must match."""
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    rng = np.random.default_rng(6)
    R, S, P = 6, 5, 2
    dense = _random_dense(rng, R, S, P, fill=0.4)
    _write_dense_layout(tmp_path, dense, R, S, P)

    d_keys = np.concatenate(
        [k for k, _ in _ranges_reader(tmp_path).iter_entries()] or [np.empty(0, np.int64)]
    )
    sparse = _sparse_from_dense(dense, R, S, P)
    s_keys = np.concatenate(
        [k for k, _ in sparse.iter_entries()] or [np.empty(0, np.int64)]
    )
    np.testing.assert_array_equal(d_keys, s_keys)


def test_entries_for_regions_agree_on_subranges(tmp_path):
    """concat asks for arbitrary [r0, r1); both layouts must answer identically.

    iter_entries only ever exercises the block boundaries the reader picks for
    itself. concat picks its own, so a dense/sparse disagreement on a partial
    region range -- an off-by-one in the region offset, say -- would slip past the
    test above and corrupt only merged datasets.
    """
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    rng = np.random.default_rng(8)
    R, S, P = 7, 3, 2
    dense = _random_dense(rng, R, S, P, fill=0.4)
    _write_dense_layout(tmp_path, dense, R, S, P)
    d_reader = _ranges_reader(tmp_path)
    sparse = _sparse_from_dense(dense, R, S, P)

    for r0, r1 in ((0, 0), (0, 1), (2, 5), (3, 3), (0, R), (R, R)):
        d_key, d_ent = d_reader.entries_for_regions(r0, r1)
        s_key, s_ent = sparse.entries_for_regions(r0, r1)
        assert d_key.dtype == np.int64 and s_key.dtype == np.int64
        assert d_ent.dtype == ENTRY_DTYPE and s_ent.dtype == ENTRY_DTYPE
        np.testing.assert_array_equal(d_key, s_key)
        np.testing.assert_array_equal(d_ent, s_ent)
        # Keys must ascend; concat's merge and _SparseWriter both assume it.
        assert np.all(np.diff(d_key) > 0)


def test_dense_reader_bounds_check(tmp_path):
    """Both layouts must raise on a bad index, so callers behave identically."""
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    rng = np.random.default_rng(7)
    R, S, P = 3, 3, 2
    _write_dense_layout(tmp_path, _random_dense(rng, R, S, P, 0.5), R, S, P)
    with pytest.raises(IndexError, match="region"):
        _ranges_reader(tmp_path).lookup(np.array([R]), np.array([0]), P)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_ranges.py -q -k "dense"`
Expected: FAIL with `ImportError: cannot import name '_ranges_reader'`.

- [ ] **Step 3: Add `_DenseRanges` and the factory**

Append to `python/genvarloader/_dataset/_svar2_ranges.py` (and add `import json` / `from pathlib import Path` at the top, and `"_DenseRanges", "_ranges_reader"` to `__all__`):

```python
@dataclass(slots=True)
class _DenseRanges:
    """The legacy ``(R, S, P, 2)`` layout, kept so old datasets still open.

    Written by GVL <= 0.42.1 and by nothing since; ``gvl.write`` emits only the
    sparse layout. Re-running ``gvl.write`` is how an existing dataset is shrunk
    -- there is no migration tool (``gvl.migrate`` handles only the 1.x -> 2.0
    AoS-to-SoA track, ``_migrate.py:65``).
    """

    vk_snp_range: NDArray[np.int64]
    vk_indel_range: NDArray[np.int64]
    n_regions: int
    n_samples: int
    ploidy: int

    def lookup(self, r_q, si_q, P):
        if P != self.ploidy:
            raise ValueError(f"query ploidy {P} != cache ploidy {self.ploidy}")
        r_q = np.asarray(r_q)
        si_q = np.asarray(si_q)
        # Fancy-indexing would raise on its own, but only for the region axis in
        # some shapes; check both so the two layouts fail identically.
        _check_bounds(r_q, si_q, self.n_regions, self.n_samples)
        snp = np.ascontiguousarray(
            np.asarray(self.vk_snp_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        indel = np.ascontiguousarray(
            np.asarray(self.vk_indel_range[r_q, si_q]).reshape(-1, 2), np.int64
        )
        return snp, indel

    def entries_for_regions(self, r0: int, r1: int):
        """Scan the dense arrays over ``[r0, r1)``, emitting non-empty cells.

        This has no analogue in the old code and is not free: a full pass reads
        the entire dense array once, which at All of Us chr22 is 128 GB. It
        exists solely so ``concat`` can merge a legacy dense shard into a sparse
        output, and it is region-bounded so ``concat`` can ask for exactly the
        merged batch it is assembling rather than driving a stream.
        """
        span = self.n_samples * self.ploidy
        if r1 <= r0:
            return np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE)
        snp = np.asarray(self.vk_snp_range[r0:r1])
        indel = np.asarray(self.vk_indel_range[r0:r1])
        ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
        # C-order nonzero => already ascending in (r, slot, ploid) = key order.
        ri, sj, pj = np.nonzero(ne)
        if len(ri) == 0:
            return np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE)
        key = (
            (r0 + ri).astype(np.int64) * span
            + sj.astype(np.int64) * self.ploidy
            + pj
        )
        ent = np.empty(len(ri), ENTRY_DTYPE)
        ent["snp_start"] = snp[ri, sj, pj, 0]
        ent["snp_len"] = snp[ri, sj, pj, 1] - snp[ri, sj, pj, 0]
        ent["indel_start"] = indel[ri, sj, pj, 0]
        ent["indel_len"] = indel[ri, sj, pj, 1] - indel[ri, sj, pj, 0]
        return key, ent

    def iter_entries(self):
        span = self.n_samples * self.ploidy
        rows = max(1, ITER_BLOCK_ENTRIES // max(span, 1))
        for r0 in range(0, self.n_regions, rows):
            key, ent = self.entries_for_regions(r0, min(r0 + rows, self.n_regions))
            if len(key):
                yield key, ent


def _raw(path: Path, dtype, shape) -> NDArray:
    """Memmap a raw headerless cache file, or an empty array if it holds nothing.

    ``np.memmap`` raises ``ValueError: cannot mmap an empty file`` on a 0-byte
    file, which a variant-free dataset or a per-contig shard can legitimately
    produce.
    """
    n = int(np.prod(shape)) if len(shape) else 0
    if n == 0:
        return np.empty(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))


def _ranges_reader(ranges_dir: Path) -> _RangeLookup:
    """Open whichever range-cache layout is at ``ranges_dir``.

    A path-level factory rather than a method on ``Svar2Haps``: ``concat`` needs
    the same reader, and ``Svar2Haps.from_path`` additionally resolves and
    fingerprints the external ``.svar2`` store, which ``concat`` must not do.

    Args:
        ranges_dir: The dataset's ``genotypes/svar2_ranges/`` directory.

    Returns:
        A :class:`_SparseRanges` or :class:`_DenseRanges`.

    Raises:
        ValueError: If the meta's grid disagrees with the files beside it. A
            wrong ``n_samples`` makes every sparse probe miss, which reads as a
            silently variant-free dataset rather than an error.
    """
    ranges_dir = Path(ranges_dir)
    with open(ranges_dir / "svar2_meta.json") as f:
        meta = json.load(f)

    P = int(meta["ploidy"])
    R = int(meta["dense_snp_range"]["shape"][0])
    layout = meta.get("layout", "dense")

    if layout == "dense":
        # Pre-0.43.0 datasets carry S only in the vk array's shape.
        S = int(meta["vk_snp_range"]["shape"][1])
    elif layout == "sparse":
        S = int(meta["n_samples"])
        if int(meta["n_regions"]) != R:
            raise ValueError(
                f"svar2 cache meta is inconsistent: n_regions={meta['n_regions']} but"
                f" dense_snp_range has {R} rows."
            )
    else:
        raise ValueError(
            f"Unknown svar2 range cache layout {layout!r} at {ranges_dir}. This"
            " dataset was written by a newer GenVarLoader."
        )

    n_cols = len(np.load(ranges_dir / "sample_cols.npy"))
    if n_cols != S:
        raise ValueError(
            f"svar2 cache meta claims {S} samples but sample_cols.npy holds {n_cols}."
        )

    if layout == "dense":
        return _DenseRanges(
            vk_snp_range=_raw(ranges_dir / "vk_snp_range.npy", np.int64, (R, S, P, 2)),
            vk_indel_range=_raw(ranges_dir / "vk_indel_range.npy", np.int64, (R, S, P, 2)),
            n_regions=R,
            n_samples=S,
            ploidy=P,
        )

    n = int(meta["n_entries"])
    return _SparseRanges(
        # np.array, not np.asarray: np.asarray(memmap, np.int64) returns a VIEW
        # still backed by the mmap (np.shares_memory is True), so every lookup
        # would fancy-index through page faults and __post_init__ would scan the
        # file. region_ptr is 1.6 MB genome-wide against a 27 GB table -- read it
        # into RAM once.
        region_ptr=np.array(
            _raw(ranges_dir / "region_ptr.npy", np.int64, (R + 1,)), np.int64
        ),
        cell_id=_raw(ranges_dir / "cell_id.npy", np.int32, (n,)),
        cell_vk=_raw(ranges_dir / "cell_vk.npy", ENTRY_DTYPE, (n,)),
        n_regions=R,
        n_samples=S,
        ploidy=P,
    )
```

- [ ] **Step 4: Run the layout tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_ranges.py -q`
Expected: PASS.

- [ ] **Step 5: Route `Svar2Haps` through the protocol**

In `python/genvarloader/_dataset/_svar2_haps.py`:

(a) Replace the `_Svar2Cache` dataclass (`:111-127`):

```python
@dataclass(slots=True)
class _Svar2Cache:
    """The ``svar2_ranges/`` cache, sliced per query.

    ``ranges`` answers the per-``(region, sample, ploid)`` var-key question and
    owns its own on-disk layout (dense or sparse -- see ``_svar2_ranges``).
    ``dense_*_range`` are ``(R, 2)`` memmaps (per-region, sample-independent, and
    NOT sparsifiable: ``genoray``'s ``dense_abs_row`` uses ``.start`` as an index
    base). ``sample_cols`` is ``(S,)``: selected slot -> original store sample
    index. Per-query starts are recomputed post-jitter at read time, so they are
    not cached here.
    """

    ranges: "_RangeLookup"
    dense_snp_range: NDArray[np.int64]
    dense_indel_range: NDArray[np.int64]
    sample_cols: NDArray[np.int64]
```

Add the import at the top of the file: `from ._svar2_ranges import _RangeLookup, _ranges_reader`.

(b) Fix the `n_samples` docstring (`:226-231`) — it names a file that the sparse layout does not have:

```python
    n_samples: int
    """The dataset's sample count, from ``svar2_meta.json``.

    Together with :attr:`n_regions` this is the ``(R, S)`` grid a flat dataset
    index unravels into. SVAR1 reads the same two numbers off its ``genotypes``
    array's leading shape; there is no such array here, so the meta is the only
    source -- and a wrong value makes every sparse probe miss, which reads as a
    silently variant-free dataset. ``_ranges_reader`` cross-checks it against
    ``sample_cols.npy``.
    """
```

(c) Replace the cache construction in `from_path` (`:420-441`) — delete the local `_mm` helper and the `R`/`S` lines:

```python
        ranges = _ranges_reader(ranges_dir)
        R, S, P = ranges.n_regions, ranges.n_samples, ranges.ploidy
        if P != ploidy:
            raise ValueError(f"svar2 cache ploidy ({P}) != dataset ploidy ({ploidy}).")

        def _mm(name: str, shape: list[int]) -> NDArray[np.int64]:
            return np.memmap(
                ranges_dir / name, dtype=np.int64, mode="r", shape=tuple(shape)
            )

        cache = _Svar2Cache(
            ranges=ranges,
            dense_snp_range=_mm("dense_snp_range.npy", meta["dense_snp_range"]["shape"]),
            dense_indel_range=_mm(
                "dense_indel_range.npy", meta["dense_indel_range"]["shape"]
            ),
            sample_cols=np.load(ranges_dir / "sample_cols.npy"),
        )
```

(d) Replace the two `vk_*` lines in `_gather_inputs` (`:1562-1567`):

```python
        vk_snp, vk_indel = c.ranges.lookup(r_q, si_q, P)
```

and update that method's docstring paragraph (`:1553-1557`) to:

```python
        """Cache-slice a per-contig query block into the read-bound FFI inputs.

        The ``vk_*`` rows come back ``(n * P, 2)`` in ``row = q * P + p`` order,
        which is exactly what the kernel expects; how they are found is the range
        layout's business (see ``_svar2_ranges``).
        """
```

- [ ] **Step 6: Run the svar2 read + write suites**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q -k "svar2"`
Expected: PASS. The on-disk format has not changed yet, so every existing test must still pass unmodified. Any failure here is an indirection bug, not a format issue.

- [ ] **Step 7: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS. This touches shared reader code, so per CLAUDE.md the scoped run is not enough.

- [ ] **Step 8: Lint, typecheck, commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_svar2_ranges.py python/genvarloader/_dataset/_svar2_haps.py tests/unit/dataset/test_svar2_ranges.py
git commit -m "refactor(svar2): read ranges through a layout protocol

Svar2Haps now asks a _RangeLookup for its var-key ranges instead of
fancy-indexing two memmaps directly, and _ranges_reader resolves the layout
from svar2_meta.json. The dense layout is unchanged on disk; this is the seam
the sparse layout lands behind.

_ranges_reader also cross-checks the meta's grid against sample_cols.npy: with
no genotypes array to read (R, S) off, a wrong n_samples would otherwise make
every probe miss and read as a variant-free dataset.

Relates to #357"
```

---

## Task 5: Write the sparse layout

The format flip. After this task, `gvl.write` emits sparse and nothing produces dense any more — so the dense-layout test helper lands here too, or Task 4's `_DenseRanges` becomes untested and the backward-compat path ships unverified.

**Files:**
- Create: `tests/_oracles/svar2_dense_layout.py`
- Modify: `python/genvarloader/_dataset/_svar2_ranges.py` (append helper)
- Modify: `python/genvarloader/_dataset/_write.py:1140-1265`
- Modify: `tests/dataset/test_write_svar2.py` (four tests)

**Interfaces:**
- Consumes: `ENTRY_DTYPE`, `_ranges_reader`, `_SparseRanges` from Tasks 3-4.
- Produces:
  - `class _SparseWriter` in `_svar2_ranges.py` with `append(region_local_key, entries, lo, rc)`, `close() -> int` (returns `N`), and attribute `n_entries`. Task 8's concat merge reuses it.
  - `def write_dense_layout(ranges_dir, snp, indel, dense_snp, dense_indel, sample_cols, ploidy)` in `tests/_oracles/svar2_dense_layout.py`. Tasks 7 and 8 use it.
  - `svar2_meta.json` gains `"layout": "sparse"`, `"n_regions"`, `"n_samples"`, `"n_entries"`, `"fill"`, `"region_ptr"`, `"cell_id"`, `"cell_vk"`; loses `"vk_snp_range"`, `"vk_indel_range"`.

- [ ] **Step 1: Write the failing test**

Replace `test_write_svar2_emits_cache`'s meta-key block and layout oracle (`tests/dataset/test_write_svar2.py:98-168`) with:

```python
    rd = out / "genotypes" / "svar2_ranges"
    meta = json.loads((rd / "svar2_meta.json").read_text())
    assert meta["layout"] == "sparse"
    assert set(meta) >= {
        "layout",
        "n_regions",
        "n_samples",
        "n_entries",
        "fill",
        "region_ptr",
        "cell_id",
        "cell_vk",
        "dense_snp_range",
        "dense_indel_range",
        "sample_cols",
    }
    assert "vk_snp_range" not in meta, "the dense layout must no longer be written"
    assert meta["ploidy"] == svar2.ploidy

    md = json.loads((out / "metadata.json").read_text())
    assert md["svar2_link"] is not None
    assert md["ploidy"] == svar2.ploidy
    Svar2Link.model_validate(md["svar2_link"])  # shape check

    # ---- The layout oracle. Replays _find_ranges over the same regions and the
    # sorted sample list gvl.write wrote, then compares through _ranges_reader.
    # This LOCKS the region-major (R, S, P) ordering: a scrambled or
    # mis-transposed cache fails loudly here.
    #
    # It compares WIDTHS and NON-EMPTY entries, not raw bytes: the sparse layout
    # deliberately discards an empty cell's insertion point, which is exactly the
    # semantic #357 buys. Asserting byte-equality would assert the bug back in.
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    sorted_samples = sorted(svar2.available_samples)
    S, P = len(sorted_samples), svar2.ploidy
    reader = _ranges_reader(rd)
    assert (reader.n_regions, reader.n_samples, reader.ploidy) == (bed.height, S, P)

    sample_cols = np.load(rd / "sample_cols.npy")
    assert sample_cols.tolist() == [
        svar2.available_samples.index(s) for s in sorted_samples
    ]

    def mm(name: str) -> np.ndarray:
        shape = tuple(meta[name]["shape"])
        return np.array(
            np.memmap(rd / f"{name}.npy", dtype=np.int64, mode="r", shape=shape)
        )

    dense_snp = mm("dense_snp_range")  # (R, 2), still dense: dense_abs_row
    dense_indel = mm("dense_indel_range")  # uses .start as an index base.

    contig_offset = 0
    n_empty_seen = 0
    for (c,), df in bed.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        rc = df.height
        lo, hi = contig_offset, contig_offset + rc
        d = svar2._find_ranges(
            c,
            df["chromStart"].to_numpy(),
            df["chromEnd"].to_numpy(),
            samples=sorted_samples,
        )
        exp_snp = np.asarray(d["vk_snp_range"], np.int64)  # (rc*S*P, 2)
        exp_indel = np.asarray(d["vk_indel_range"], np.int64)

        r_q, si_q = np.unravel_index(np.arange(rc * S), (rc, S))
        got_snp, got_indel = reader.lookup(r_q + lo, si_q, P)

        for got, exp in ((got_snp, exp_snp), (got_indel, exp_indel)):
            widths = exp[:, 1] - exp[:, 0]
            np.testing.assert_array_equal(got[:, 1] - got[:, 0], widths)
            ne = widths > 0
            np.testing.assert_array_equal(got[ne], exp[ne])
            np.testing.assert_array_equal(got[~ne], 0)
            n_empty_seen += int((~ne).sum())

        np.testing.assert_array_equal(
            dense_snp[lo:hi], np.asarray(d["dense_snp_range"], np.int64)
        )
        np.testing.assert_array_equal(
            dense_indel[lo:hi], np.asarray(d["dense_indel_range"], np.int64)
        )
        contig_offset += rc

    assert n_empty_seen > 0, "fixture lost its empty cells (see Task 1)"

    # Sparse must be smaller than the dense layout would have been, at this fill.
    n = meta["n_entries"]
    assert n * 28 < bed.height * S * P * 32
    assert meta["fill"] == pytest.approx(n / (bed.height * S * P))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_write_svar2_emits_cache -q`
Expected: FAIL with `KeyError: 'layout'` — the writer still emits dense.

- [ ] **Step 3: Add the sparse writer helper**

Append to `python/genvarloader/_dataset/_svar2_ranges.py` (add `"_SparseWriter"` and `"nonempty_entries"` to `__all__`, and `IO` to the `typing` import — `_SparseWriter` declares its two file handles as fields, which `slots=True` requires):

```python
@dataclass(slots=True)
class _SparseWriter:
    """Streams a region-CSR table to ``region_ptr``/``cell_id``/``cell_vk``.

    Files are raw and headerless -- the existing convention for everything in
    ``svar2_ranges/`` except ``sample_cols.npy`` -- so they can simply be
    appended to. A real ``.npy`` would need a placeholder header pre-written and
    patched at the end, because ``N`` is unknown until the last contig is done.

    The same consequence applies to ``svar2_meta.json``: it can only be written
    *after* the loop, so an aborted write leaves data files with no meta. That is
    safe only because ``write`` builds into an ``atomic_dir`` tmp that is
    discarded on failure.

    Use it as a context manager. ``region_ptr`` is published only on clean exit,
    because a ``region_ptr`` written from a ``finally`` would index a truncated
    ``cell_id``/``cell_vk`` -- a dataset that opens and silently returns garbage
    rather than one that fails.
    """

    ranges_dir: "Path"
    n_samples: int
    ploidy: int
    n_entries: int = 0
    # Every attribute must be declared: `slots=True` gives the class no __dict__,
    # so an undeclared `self._span = ...` in __post_init__ raises AttributeError.
    _span: int = field(init=False, repr=False, default=0)
    _ptr: "list[NDArray[np.int64]]" = field(init=False, repr=False, default_factory=list)
    _regions_done: int = field(init=False, repr=False, default=0)
    _f_cell: "IO[bytes]" = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    _f_vk: "IO[bytes]" = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    def __post_init__(self):
        span = self.n_samples * self.ploidy
        if span >= 2**31:
            raise ValueError(
                f"n_samples * ploidy = {span} does not fit int32, so the sparse"
                " range cache cannot address a cell. Shard the samples."
            )
        self._span = span
        self._ptr = [np.zeros(1, np.int64)]
        self._f_cell = open(self.ranges_dir / "cell_id.npy", "wb")
        self._f_vk = open(self.ranges_dir / "cell_vk.npy", "wb")

    def __enter__(self) -> "_SparseWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Close the handles unconditionally: `atomic_dir` discards the tmp tree on
        # failure, but the handles are pinned by the propagating traceback's frame
        # until GC, which on a long write is an open-fd leak per shard.
        if exc_type is None:
            self.close()
        else:
            self._close_files()
        return False

    def _counts(self, r: NDArray[np.integer], rc: int) -> NDArray[np.int64]:
        """Per-region entry counts, with the overshoot guard.

        ``np.bincount(x, minlength=rc)`` silently returns a **longer** array when
        an index exceeds ``rc`` (index 9 with ``minlength=5`` gives 10 bins).
        Unchecked, that lengthens ``region_ptr`` past ``R + 1`` while the meta
        still declares ``[R + 1]``, and the reader memmaps a truncated prefix --
        a silently corrupt dataset rather than a crash. Keys must be contig-local.
        """
        cnt = np.bincount(r, minlength=rc)
        if len(cnt) != rc:
            raise ValueError(
                f"svar2 range cache got region index {len(cnt) - 1} for a contig"
                f" of {rc} regions; indices must be contig-local."
            )
        return cnt.astype(np.int64, copy=False)

    def append(self, key: NDArray[np.int64], ent: NDArray[np.void], lo: int, rc: int):
        """Append one already-ordered block of entries.

        The key-based entry point, used by ``concat``'s merge. The write path
        calls :meth:`append_contig` instead, which does the ordering itself.

        Args:
            key: Strictly ascending **region-local** keys,
                ``(r - lo) * n_samples * ploidy + slot * ploidy + ploid``.
            ent: Parallel :data:`ENTRY_DTYPE` entries.
            lo: The contig's first region index in the dataset.
            rc: The contig's region count.

        Raises:
            ValueError: If the caller's regions are not contiguous and in order,
                if the keys are not strictly ascending, or if a key is out of
                range for this block. Contiguity is the one load-bearing
                invariant of the write path: blocks from ``bed.partition_by``
                must partition ``[0, R)`` in the same order as the running
                ``contig_offset``. Asserting it directly covers every way a
                future bed could break it.
        """
        self._check_lo(lo)
        if len(key) and not np.all(np.diff(key) > 0):
            raise ValueError("svar2 range cache entries are not strictly ascending")

        ri = (key // self._span).astype(np.int64)
        cnt = self._counts(ri, rc)
        np.asarray(key % self._span, np.int32).tofile(self._f_cell)
        np.asarray(ent, ENTRY_DTYPE).tofile(self._f_vk)

        self._ptr.append(self.n_entries + cnt.cumsum())
        self.n_entries += len(key)
        self._regions_done += rc

    def append_contig(
        self,
        regions: "list[NDArray[np.int32]]",
        cells: "list[NDArray[np.int32]]",
        ents: "list[NDArray[np.void]]",
        lo: int,
        rc: int,
    ) -> None:
        """Merge one contig's per-chunk blocks into region-major order and append.

        Each block from :func:`nonempty_entries` is already region-major, and
        chunk ``i``'s sample slots lie entirely below chunk ``i + 1``'s, so the
        merged order is fixed by region alone. That makes this a stable counting
        sort with ``O(rc)`` of auxiliary state, not a comparison sort.

        This is not a micro-optimization over ``np.argsort(key, kind="stable")``:
        numpy maps ``kind="stable"`` to radix **only** for integer types of 16
        bits or fewer, whatever its docstring says. int32 and int64 get timsort,
        ``O(N log N)`` -- measured 98 and 136 ns/element on random input against
        2.9 ns for int16. An int64 key sort only *looks* linear here because
        timsort's run detection fires on the per-chunk runs, and that degrades as
        chunks get smaller, which is exactly the regime the sort was chosen to
        survive. Measured at ``N = 18e6``: 914 ms (k=30) / 1369 ms (k=500) for the
        sort against 532 / 528 ms here, byte-identical output.

        Args:
            regions: Per-chunk contig-local region indices, chunks in ascending
                sample order.
            cells: Parallel ``slot * ploidy + ploid`` values.
            ents: Parallel :data:`ENTRY_DTYPE` entries.
            lo: The contig's first region index in the dataset.
            rc: The contig's region count.

        Raises:
            ValueError: If the caller's regions are not contiguous and in order,
                or if a region index is out of range for this contig.
        """
        self._check_lo(lo)

        # Pass 1: per-region totals. O(rc) of state -- never (n_chunks x rc),
        # which is what makes this safe at samples_per_chunk == 1 (535k chunks at
        # cohort scale).
        total = np.zeros(rc, np.int64)
        for r in regions:
            total += self._counts(r, rc)

        n = int(total.sum())
        cursor = np.empty(rc, np.int64)
        cursor[0] = 0
        np.cumsum(total[:-1], out=cursor[1:])

        # Pass 2: scatter each chunk to its final offsets. `arange - start[r]` is
        # the within-region rank, valid because each block is region-grouped.
        out_cell = np.empty(n, np.int32)
        out_ent = np.empty(n, ENTRY_DTYPE)
        for r, c, e in zip(regions, cells, ents):
            cnt = self._counts(r, rc)
            start = np.empty(rc, np.int64)
            start[0] = 0
            np.cumsum(cnt[:-1], out=start[1:])
            dst = cursor[r]
            dst += np.arange(len(r), dtype=np.int64)
            dst -= start[r]
            out_cell[dst] = c
            out_ent[dst] = e
            cursor += cnt

        out_cell.tofile(self._f_cell)
        out_ent.tofile(self._f_vk)
        self._ptr.append(self.n_entries + total.cumsum())
        self.n_entries += n
        self._regions_done += rc

    def _check_lo(self, lo: int) -> None:
        if lo != self._regions_done:
            raise ValueError(
                f"svar2 range cache requires contiguous region blocks in order:"
                f" got a block starting at region {lo} after {self._regions_done}"
                f" regions. Is the bed still contig-grouped (sp.bed.sort)?"
            )

    def _close_files(self) -> None:
        try:
            self._f_cell.close()
        finally:
            self._f_vk.close()

    def close(self) -> int:
        """Close the data files, publish ``region_ptr``, and return ``N``."""
        self._close_files()
        np.concatenate(self._ptr).astype(np.int64).tofile(
            self.ranges_dir / "region_ptr.npy"
        )
        return self.n_entries


def nonempty_entries(
    snp: NDArray[np.int64], indel: NDArray[np.int64], slot0: int, ploidy: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.void]]:
    """Extract non-empty cells from a ``(rc, ns, P, 2)`` pair of range blocks.

    Args:
        snp: SNP ranges, ``(rc, ns, P, 2)`` -- normally a ``transpose(2, 0, 1, 3)``
            view of a hap-major genoray chunk.
        indel: Indel ranges, same shape.
        slot0: Dataset sample slot of this block's first column.
        ploidy: ``P``.

    Returns:
        ``(region, cell, entries)``, region-major: ``region`` is **contig-local**
        and non-decreasing, ``cell`` is ``slot * ploidy + ploid`` and ascends
        within each region. Split rather than combined into one key because
        :meth:`_SparseWriter.append_contig` needs the region axis on its own to
        count, and ``cell`` is what lands on disk -- combining them would only be
        undone again.
    """
    ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
    # np.nonzero walks the LOGICAL shape in C order, so (r, slot, ploid) comes
    # out ascending even though `ne` is NOT C-contiguous: the `>` above inherits
    # the transposed view's stride permutation, because numpy allocates ufunc
    # output with NPY_KEEPORDER. Do not "fix" that with ascontiguousarray --
    # materializing (rc, ns, P) in C order is a strided scatter costing ~11x the
    # comparison itself (97.7 ms vs 8.8 ms on a 15e6-cell chunk).
    ri, sj, pj = np.nonzero(ne)
    ent = np.empty(len(ri), ENTRY_DTYPE)
    ent["snp_start"] = snp[ri, sj, pj, 0]
    ent["snp_len"] = snp[ri, sj, pj, 1] - snp[ri, sj, pj, 0]
    ent["indel_start"] = indel[ri, sj, pj, 0]
    ent["indel_len"] = indel[ri, sj, pj, 1] - indel[ri, sj, pj, 0]
    return ri.astype(np.int32), ((slot0 + sj) * ploidy + pj).astype(np.int32), ent
```

- [ ] **Step 4: Rewrite `_write_from_svar2`'s cache emission**

In `python/genvarloader/_dataset/_write.py`, add `from ._svar2_ranges import ENTRY_DTYPE, _SparseWriter, nonempty_entries` at the top.

(a) Replace the `vk_snp` / `vk_indel` memmap creation (`:1163-1167`) with a `with` block wrapping the contig loop — the writer owns two open file handles, so it must close them on the failure path too:

```python
    with _SparseWriter(out_dir, n_samples=S, ploidy=P) as writer:
        ...  # the existing `for (c,), df in bed.partition_by(...)` loop, re-indented
```

(b) Delete the `svar2_meta.json` block at `:1199-1211` entirely — `N` and `fill` are unknown until the loop ends, so the meta moves after it.

(c) Replace the per-chunk scatter loop (`:1241-1250`) with per-contig accumulation:

```python
            acc_r: list[NDArray[np.int32]] = []
            acc_c: list[NDArray[np.int32]] = []
            acc_e: list[NDArray[np.void]] = []
            for ch in stream.chunks:
                # Chunks are hap-major (samples, ploidy, regions, 2); transpose to
                # region-major (regions, samples, ploidy, 2). transpose() is a
                # view, and nonempty_entries relies on that -- see its comment on
                # np.nonzero and NPY_KEEPORDER.
                r, cell, ent = nonempty_entries(
                    ch.vk_snp_range.transpose(2, 0, 1, 3),
                    ch.vk_indel_range.transpose(2, 0, 1, 3),
                    slot0=ch.sample_start,
                    ploidy=P,
                )
                acc_r.append(r)
                acc_c.append(cell)
                acc_e.append(ent)
                np.maximum(keys, ch.max_end_keys, out=keys)
                pbar.update(rc * ch.n_samples / S)

            # Merge the contig's chunks into region-major order and append. Each
            # chunk is already region-major over a contiguous, ascending slot
            # block, so the merge is a counting sort keyed on region alone -- see
            # _SparseWriter.append_contig for why this is not an argsort.
            writer.append_contig(acc_r, acc_c, acc_e, lo=lo, rc=rc)
            del acc_r, acc_c, acc_e
```

(d) Replace the trailing flush block (`:1260-1262`) with:

```python
    # Outside the `with`: the writer has closed its handles and published
    # region_ptr, so n_entries is final.
    pbar.close()
    n_entries = writer.n_entries
    # dense_snp/dense_indel are still memmaps (dense_abs_row uses .start as an
    # index base, so those two stay dense); flush them before the meta claims
    # they exist.
    for mm in (dense_snp, dense_indel):
        mm.flush()

    with open(out_dir / "svar2_meta.json", "w") as f:
        json.dump(
            {
                "layout": "sparse",
                "n_regions": R,
                "n_samples": S,
                "n_entries": n_entries,
                "fill": (n_entries / (R * S * P)) if R * S * P else 0.0,
                "region_ptr": {"shape": [R + 1], "dtype": "<i8"},
                "cell_id": {"shape": [n_entries], "dtype": "<i4"},
                "cell_vk": {"shape": [n_entries], "dtype": ENTRY_DTYPE.descr},
                "dense_snp_range": {"shape": [R, 2], "dtype": "<i8"},
                "dense_indel_range": {"shape": [R, 2], "dtype": "<i8"},
                "sample_cols": {"shape": [S], "dtype": "<i8"},
                "ploidy": P,
            },
            f,
        )
```

- [ ] **Step 5: Run the layout oracle**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_write_svar2_emits_cache -v`
Expected: PASS.

- [ ] **Step 6: Retarget the remaining three affected write tests**

(a) `test_write_svar2_chunked_matches_unchunked` (`:361-368`) — the file list becomes the sparse files. This is the test that catches a chunk-merge ordering bug, so keep it byte-comparing:

```python
    for name in (
        "region_ptr.npy",
        "cell_id.npy",
        "cell_vk.npy",
        "dense_snp_range.npy",
        "dense_indel_range.npy",
        "sample_cols.npy",
        "svar2_meta.json",
    ):
        a = (big / "genotypes" / "svar2_ranges" / name).read_bytes()
        b = (small / "genotypes" / "svar2_ranges" / name).read_bytes()
        assert a == b, name
```

(b) `test_write_svar2_sample_cols_permutes_unsorted_store` (`:519-531`) — replace the `meta["vk_snp_range"]["shape"]` memmap and its comparison with a reader lookup:

```python
    # The cache must be laid out in the SORTED slot order, i.e. match a direct
    # _find_ranges over the sorted names -- the `samples=None` fast path must not
    # have fired and silently written the store's own order.
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    reader = _ranges_reader(rd)
    S, P = len(sorted_samples), svar2.ploidy
    r_q, si_q = np.unravel_index(np.arange(bed.height * S), (bed.height, S))
    got_snp, _ = reader.lookup(r_q, si_q, P)

    d = svar2._find_ranges(
        "chr1",
        bed["chromStart"].to_numpy(),
        bed["chromEnd"].to_numpy(),
        samples=sorted_samples,
    )
    exp = np.asarray(d["vk_snp_range"], np.int64).reshape(-1, 2)
    ne = exp[:, 1] > exp[:, 0]
    np.testing.assert_array_equal(got_snp[ne], exp[ne])
    np.testing.assert_array_equal(got_snp[~ne], 0)
```

(c) Add a guard for the int32 `cell_id` bound. Append to `tests/dataset/test_write_svar2.py`:

```python
def test_sparse_writer_rejects_oversized_grid(tmp_path):
    """cell_id is int32, so S * P must stay under 2**31."""
    from genvarloader._dataset._svar2_ranges import _SparseWriter

    with pytest.raises(ValueError, match="int32"):
        _SparseWriter(tmp_path, n_samples=2**30, ploidy=2)


def test_sparse_writer_rejects_noncontiguous_regions(tmp_path):
    """The write path's one load-bearing invariant, asserted directly."""
    import numpy as np

    from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE, _SparseWriter

    w = _SparseWriter(tmp_path, n_samples=2, ploidy=2)
    w.append(np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE), lo=0, rc=3)
    with pytest.raises(ValueError, match="contiguous region blocks"):
        w.append(np.empty(0, np.int64), np.empty(0, ENTRY_DTYPE), lo=7, rc=2)
    w.close()


def test_sparse_writer_rejects_global_region_indices(tmp_path):
    """bincount overshoot would silently lengthen region_ptr past R + 1.

    `np.bincount(x, minlength=rc)` returns MORE than `rc` bins when an index
    exceeds `rc` rather than raising, so a caller that passed dataset-global
    region indices would write a longer `region_ptr` than the meta declares and
    the reader would memmap a truncated prefix -- garbage lookups, no error.
    """
    import numpy as np

    from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE, _SparseWriter

    w = _SparseWriter(tmp_path, n_samples=2, ploidy=2)
    with pytest.raises(ValueError, match="contig-local"):
        w.append_contig(
            [np.array([7], np.int32)],
            [np.zeros(1, np.int32)],
            [np.zeros(1, ENTRY_DTYPE)],
            lo=0,
            rc=3,
        )
    w.close()
```

(d) Add the counting-sort equivalence test. This one goes in **`tests/unit/dataset/test_svar2_ranges.py`** (Task 3's file), not in `test_write_svar2.py`, because it reuses that file's `_random_dense` helper and needs no dataset fixture:

```python
def test_sparse_writer_counting_sort_matches_argsort(tmp_path):
    """append_contig's counting sort must equal the stable key sort it replaced.

    The chunks deliberately interleave: chunk 0 holds slots [0, 2) and chunk 1
    slots [2, 4), so every region draws from both and the merge is a real
    interleave rather than a concatenation.
    """
    from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE, _SparseWriter

    rng = np.random.default_rng(11)
    rc, S, P = 6, 4, 2
    dense = _random_dense(rng, rc, S, P, fill=0.5)  # (2, rc, S, P, 2)

    regions, cells, ents, keys = [], [], [], []
    for s0, s1 in ((0, 2), (2, 4)):
        snp = dense[0][:, s0:s1]
        indel = dense[1][:, s0:s1]
        ne = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
        ri, sj, pj = np.nonzero(ne)
        ent = np.empty(len(ri), ENTRY_DTYPE)
        ent["snp_start"] = snp[ri, sj, pj, 0]
        ent["snp_len"] = snp[ri, sj, pj, 1] - snp[ri, sj, pj, 0]
        ent["indel_start"] = indel[ri, sj, pj, 0]
        ent["indel_len"] = indel[ri, sj, pj, 1] - indel[ri, sj, pj, 0]
        cell = ((s0 + sj) * P + pj).astype(np.int32)
        regions.append(ri.astype(np.int32))
        cells.append(cell)
        ents.append(ent)
        keys.append(ri.astype(np.int64) * (S * P) + cell)

    w = _SparseWriter(tmp_path, n_samples=S, ploidy=P)
    w.append_contig(regions, cells, ents, lo=0, rc=rc)
    n = w.close()

    key = np.concatenate(keys)
    ent = np.concatenate(ents)
    perm = np.argsort(key, kind="stable")
    assert n == len(key)
    got_cell = np.fromfile(tmp_path / "cell_id.npy", np.int32)
    got_ent = np.fromfile(tmp_path / "cell_vk.npy", ENTRY_DTYPE)
    np.testing.assert_array_equal(got_cell, (key[perm] % (S * P)).astype(np.int32))
    np.testing.assert_array_equal(got_ent, ent[perm])
    ptr = np.fromfile(tmp_path / "region_ptr.npy", np.int64)
    assert len(ptr) == rc + 1 and ptr[0] == 0 and ptr[-1] == n
```

- [ ] **Step 7: Add the test-only dense-layout writer**

Nothing in production emits the dense layout after Step 4, so `_DenseRanges` and every backward-compat path would go untested. Create `tests/_oracles/svar2_dense_layout.py`:

```python
"""Emit the pre-0.43.0 dense `svar2_ranges` layout, for backward-compat tests.

`gvl.write` emits only the sparse layout (#357), so without this helper the dense
reader and the dense-input concat path would ship with no coverage at all.

Deliberately a test helper and NOT a `layout=` kwarg on `gvl.write`: a deprecated
on-disk format should not be reachable from the public API. Deliberately not a
checked-in binary fixture either: that would rot silently as the meta evolves.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def rewrite_as_dense(dataset: Path, out: Path) -> Path:
    """Copy a sparse `.gvl` dataset, re-emitting its range cache as dense.

    Args:
        dataset: A dataset written by `gvl.write` (sparse layout).
        out: Destination directory; must not exist.

    Returns:
        `out`, a byte-for-byte copy except that `genotypes/svar2_ranges/` holds
        the legacy dense layout.
    """
    from genvarloader._dataset._svar2_ranges import _ranges_reader

    shutil.copytree(dataset, out)
    rd = out / "genotypes" / "svar2_ranges"
    reader = _ranges_reader(rd)
    R, S, P = reader.n_regions, reader.n_samples, reader.ploidy

    r_q, si_q = np.unravel_index(np.arange(R * S), (R, S))
    snp, indel = reader.lookup(r_q, si_q, P)
    snp.reshape(R, S, P, 2).tofile(rd / "vk_snp_range.npy")
    indel.reshape(R, S, P, 2).tofile(rd / "vk_indel_range.npy")

    for name in ("region_ptr.npy", "cell_id.npy", "cell_vk.npy"):
        (rd / name).unlink()

    meta = json.loads((rd / "svar2_meta.json").read_text())
    for k in ("layout", "n_regions", "n_samples", "n_entries", "fill",
              "region_ptr", "cell_id", "cell_vk"):
        meta.pop(k, None)
    meta["vk_snp_range"] = {"shape": [R, S, P, 2], "dtype": "<i8"}
    meta["vk_indel_range"] = {"shape": [R, S, P, 2], "dtype": "<i8"}
    (rd / "svar2_meta.json").write_text(json.dumps(meta))
    return out
```

Note: the dense array this produces holds `(0, 0)` where the sparse cache had no entry, rather than genoray's insertion point. That is fine and is the point — the spec's core claim is that the two are interchangeable. Do not assert byte-equality against a genoray-written dense array anywhere.

- [ ] **Step 8: Add the backward-compat read test**

Append to `tests/dataset/test_write_svar2.py`:

```python
def test_dense_layout_dataset_still_opens_and_reads(svar2_store: Path, tmp_path: Path):
    """A pre-0.43.0 dataset must read identically under the new reader.

    #357 bumps the on-disk layout but NOT DATASET_FORMAT_VERSION (which matches
    on MAJOR only, so a bump would make new GVL refuse every old dataset). The
    dense reader is what keeps old datasets openable.
    """
    from genoray import SparseVar2

    from tests._oracles.svar2_dense_layout import rewrite_as_dense

    bed = pl.DataFrame(
        {"chrom": ["chr1"] * 3, "chromStart": [0, 5, 25], "chromEnd": [20, 15, 40]}
    )
    sparse_ds = tmp_path / "sparse.gvl"
    gvl.write(sparse_ds, bed, variants=SparseVar2(svar2_store), samples=None, overwrite=True)
    dense_ds = rewrite_as_dense(sparse_ds, tmp_path / "dense.gvl")

    a = gvl.Dataset.open(sparse_ds).with_seqs("haplotypes")
    b = gvl.Dataset.open(dense_ds).with_seqs("haplotypes")
    for r in range(a.n_regions):
        for s in range(a.n_samples):
            np.testing.assert_array_equal(
                np.asarray(a[r, s].to_padded(b"N")), np.asarray(b[r, s].to_padded(b"N"))
            )
```

- [ ] **Step 9: Run the whole svar2 write suite**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -v`
Expected: PASS, including `test_dense_layout_dataset_still_opens_and_reads`.

- [ ] **Step 10: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS except possibly `tests/dataset/test_concat_svar2.py`, which Task 8 fixes. If concat fails here, that is expected — `_concat_svar2_ranges` still `gather_fixed`s `vk_*_range.npy`, which no longer exists. Note the failure and continue; do NOT patch concat ad hoc.

- [ ] **Step 11: Lint, typecheck, commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_svar2_ranges.py python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py tests/_oracles/svar2_dense_layout.py
git commit -m "feat(svar2)!: write the range cache sparsely

gvl.write now emits region_ptr/cell_id/cell_vk instead of two dense
(R, S, P, 2) memmaps. At All of Us chr22 that is 504 MB instead of 128 GB;
genome-wide, 27 GB instead of 6.93 TB.

Entries accumulate per contig and are radix-sorted once before append -- each
genoray chunk is already key-sorted, but samples_per_chunk can be 1, which at
cohort scale would make an interleave a 535k-row counts matrix. svar2_meta.json
gains layout/n_regions/n_samples/n_entries/fill and moves after the contig loop,
since N is unknown until then.

BREAKING CHANGE: datasets written by 0.43.0+ cannot be opened by earlier
GenVarLoader, which fails with KeyError: 'vk_snp_range'. Reading older datasets
is unaffected. Re-run gvl.write to shrink an existing dataset; there is no
migration tool.

Closes #357"
```

---

## Task 6: Preflight against realized fill

`_svar2_ranges_cache_bytes` cannot be evaluated before the ranges are known, and a worst-case bound would be actively misleading: `28 * R * S * P` genome-wide is 6.06 TB against the 6.93 TB dense cache this change eliminates, for a real answer of 27 GB. That is a 220x overstatement firing on essentially every cohort build.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:1089-1137` and the preflight call site at `:1162`
- Modify: `tests/dataset/test_write_svar2.py:427-457`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_svar2_ranges_cache_bytes(n_regions, n_samples, ploidy) -> int` keeps its dense formula and its signature; `_svar2_preflight(out_dir, n_regions, n_samples, ploidy) -> int` keeps its signature and now returns the *dense-equivalent* figure it logs as context. New: `_svar2_fill_projection(out_dir, n_entries_so_far, regions_done, n_regions, n_samples, ploidy) -> int`.

- [ ] **Step 1: Write the failing test**

Replace `test_svar2_ranges_cache_bytes` and `test_svar2_preflight_warns_when_disk_is_short` (`:427-457`) with:

```python
def test_svar2_ranges_cache_bytes():
    """The DENSE formula: 2 * R * S * P * 2 endpoints * 8 bytes.

    Retained after #357 as the "what the old layout would have cost" reference
    the preflight logs for context. It is no longer a projection.
    """
    from genvarloader._dataset._write import _svar2_ranges_cache_bytes

    assert _svar2_ranges_cache_bytes(1, 1, 2) == 2 * 1 * 1 * 2 * 2 * 8
    # The scale from gvl#333: ~98 GiB for one chromosome/panel.
    big = _svar2_ranges_cache_bytes(3964, 414830, 2)
    assert 90 * 1024**3 < big < 110 * 1024**3


def test_svar2_preflight_logs_dense_equivalent_without_warning(tmp_path, monkeypatch):
    """Preflight must NOT warn on the dense figure: the sparse cache is ~200x smaller.

    Warning on `28 * R * S * P` would fire on every cohort build for a cache that
    actually fits, which is exactly the false alarm #357 removes.
    """
    from collections import namedtuple

    from loguru import logger

    from genvarloader._dataset import _write

    Usage = namedtuple("Usage", "total used free")
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        monkeypatch.setattr(
            _write.shutil, "disk_usage", lambda p: Usage(total=1000, used=999, free=1)
        )
        n = _write._svar2_preflight(tmp_path, 3964, 414830, 2)
    finally:
        logger.remove(sink)

    assert n == _write._svar2_ranges_cache_bytes(3964, 414830, 2)
    assert not msgs, f"preflight must not warn on the dense-equivalent figure: {msgs}"


def test_svar2_fill_projection_warns_when_disk_is_short(tmp_path, monkeypatch):
    """After the first contig, the realized fill drives the free-space check."""
    from collections import namedtuple

    from loguru import logger

    from genvarloader._dataset import _write

    Usage = namedtuple("Usage", "total used free")
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        monkeypatch.setattr(
            _write.shutil, "disk_usage", lambda p: Usage(total=1000, used=999, free=1)
        )
        # 1 of 100 regions done, 1e6 entries => ~1e8 entries * 28 B projected.
        n = _write._svar2_fill_projection(tmp_path, 1_000_000, 1, 100, 500, 2)
    finally:
        logger.remove(sink)

    assert n == 28 * 100 * 1_000_000
    assert any("free" in m for m in msgs), msgs


def test_svar2_fill_projection_zero_entries_projects_nothing(tmp_path, monkeypatch):
    """A variant-free first contig must not project 0 bytes and call it a day.

    This is the failure mode the `projected` latch at the call site exists for: a
    leading contig that happens to hold no variant extrapolates to 0, which would
    pass any free-space check and, without the latch, suppress the projection for
    the whole build. The helper is honest about having nothing to say; the caller
    is responsible for asking again.
    """
    from collections import namedtuple

    from loguru import logger

    from genvarloader._dataset import _write

    Usage = namedtuple("Usage", "total used free")
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        monkeypatch.setattr(
            _write.shutil, "disk_usage", lambda p: Usage(total=1000, used=999, free=1)
        )
        assert _write._svar2_fill_projection(tmp_path, 0, 0, 100, 500, 2) == 0
        assert _write._svar2_fill_projection(tmp_path, 0, 10, 100, 500, 2) == 0
    finally:
        logger.remove(sink)

    assert not msgs, f"an empty projection must not warn about free space: {msgs}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q -k "preflight or cache_bytes or fill_projection"`
Expected: FAIL — `AttributeError: module ... has no attribute '_svar2_fill_projection'`, and the no-warning test fails because the current preflight warns.

- [ ] **Step 3: Rework the preflight**

In `python/genvarloader/_dataset/_write.py`:

(a) Update `_svar2_ranges_cache_bytes`'s docstring (`:1090-1103`) — the formula is unchanged, its meaning is not:

```python
def _svar2_ranges_cache_bytes(n_regions: int, n_samples: int, ploidy: int) -> int:
    """What the pre-0.43.0 DENSE var-key cache would have occupied.

    Two ``(regions, samples, ploidy, 2)`` int64 arrays: 32 bytes for every
    ``(region, sample, ploid)`` cell, over 99% of them empty at cohort scale.
    One chromosome of a 414k-sample cohort over ~4k regions is ~98 GiB.

    The writer no longer emits this layout (#357) -- this is kept as the
    reference figure the preflight logs for context, so the log says what the
    change is worth. For what will actually be written, see
    :func:`_svar2_fill_projection`.

    Args:
        n_regions: Number of BED rows in the dataset.
        n_samples: Number of selected samples.
        ploidy: Ploidy of the variant source.

    Returns:
        Total bytes the two dense channels would have occupied.
    """
    return 2 * n_regions * n_samples * ploidy * 2 * 8
```

(b) Replace `_svar2_preflight`'s body (`:1121-1137`) so it logs but does not warn, and add the projection function after it:

```python
def _svar2_preflight(out_dir: Path, n_regions: int, n_samples: int, ploidy: int) -> int:
    """Log what the old dense range cache would have cost. Does not warn.

    A worst-case sparse bound is useless here: ``28 * R * S * P`` genome-wide is
    6.06 TB against a 6.93 TB dense cache, for a realized ~27 GB. Warning on that
    would fire on every cohort build. The real check runs after the first contig
    (:func:`_svar2_fill_projection`), once there is a fill to extrapolate from.

    Args:
        out_dir: Directory the cache will be written to.
        n_regions: Number of BED rows in the dataset.
        n_samples: Number of selected samples.
        ploidy: Ploidy of the variant source.

    Returns:
        The dense-equivalent byte count that was logged.
    """
    n_bytes = _svar2_ranges_cache_bytes(n_regions, n_samples, ploidy)
    logger.info(
        f"svar2 range cache at {out_dir}: the pre-0.43.0 dense layout would have "
        f"needed {format_memory(n_bytes)} for {n_regions} regions x {n_samples} "
        f"samples x ploidy {ploidy}. The sparse layout stores only non-empty "
        f"windows; the projected size is logged after the first contig."
    )
    return n_bytes


def _svar2_fill_projection(
    out_dir: Path,
    n_entries: int,
    regions_done: int,
    n_regions: int,
    n_samples: int,
    ploidy: int,
) -> int:
    """Project the sparse cache size from realized fill, and check free space.

    Warns rather than raising: free-space reporting is unreliable on some network
    filesystems, and a false refusal would block a valid large build.

    Args:
        out_dir: Directory the cache is being written to.
        n_entries: Non-empty cells written so far.
        regions_done: Region rows completed so far.
        n_regions: Total BED rows in the dataset.
        n_samples: Number of selected samples.
        ploidy: Ploidy of the variant source.

    Returns:
        Projected total bytes, or ``0`` if there is nothing to extrapolate from
        yet -- no finished contig, or a finished contig that held no variant.
    """
    if regions_done <= 0 or n_entries <= 0:
        # Nothing to extrapolate from. Returning 0 rather than warning about a
        # 0-byte projection is what lets the caller's latch ask again after the
        # next contig; see _write_from_svar2.
        return 0
    per_region = n_entries / regions_done
    n_bytes = int(28 * per_region * n_regions)
    fill = n_entries / max(regions_done * n_samples * ploidy, 1)
    logger.info(
        f"svar2 range cache: {fill:.4%} of windows hold a variant so far; "
        f"projecting {format_memory(n_bytes)} for {n_regions} regions."
    )
    try:
        free = shutil.disk_usage(out_dir).free
    except OSError:
        return n_bytes
    if n_bytes > free:
        logger.warning(
            f"svar2 range cache projects {format_memory(n_bytes)} but only "
            f"{format_memory(free)} is free at {out_dir}. The write will likely "
            f"fail with ENOSPC."
        )
    return n_bytes
```

(c) Call it once, after the first contig that actually produced entries. Immediately after `writer.append_contig(...)` in the contig loop (Task 5, Step 4c), add:

```python
            # Project from the first contig that produced anything, not from the
            # first contig full stop. A variant-free leading contig -- a small or
            # unplaced one, or a region set whose first contig happens to miss
            # every variant -- would otherwise project 0 bytes, log "0.0000% of
            # windows hold a variant", and permanently suppress the free-space
            # check for the rest of the build. `projected` latches, so this still
            # runs exactly once.
            if not projected and writer.n_entries:
                _svar2_fill_projection(out_dir, writer.n_entries, hi, R, S, P)
                projected = True
```

and initialize the latch just above the contig loop:

```python
    projected = False
```

- [ ] **Step 4: Run the preflight tests**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q -k "preflight or cache_bytes or fill_projection"`
Expected: PASS.

- [ ] **Step 5: Run the write suite and commit**

```bash
pixi run -e dev pytest tests/dataset/test_write_svar2.py -q
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py
git commit -m "feat(svar2): project the range cache from realized fill

A worst-case sparse bound is useless: 28 * R * S * P genome-wide is 6.06 TB
against the 6.93 TB dense cache this replaces, for a realized ~27 GB -- a 220x
overstatement that would warn on every cohort build. Preflight now logs the
dense-equivalent figure as context only, and the free-space check runs after
the first contig against fill extrapolated from what was actually written.

Relates to #357"
```

---

## Task 7: `n_variants` without the allocation

The #355 remainder: a dense `(R, S, P)` int32 of zeros allocated at open — 50.7 GB at All of Us chr19 — that is only ever read for its shape and for the documented zeros `Dataset.n_variants()` returns on SVAR2 (#363).

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py:270-279`
- Create test in: `tests/unit/dataset/test_svar2_ranges.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Svar2Haps.n_variants` is a read-only zero-stride `np.broadcast_to` view rather than a real array. Its `.shape` and `.dtype` are unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/dataset/test_svar2_ranges.py`:

```python
def test_svar2_n_variants_is_a_zero_stride_view():
    """#355: a dense (R, S, P) int32 of zeros is 50.7 GB at All of Us chr19.

    Deliberately asserted on a SMALL shape. Asserting via a cohort-scale
    allocation is O(1) only while the change holds -- the moment someone reverts
    to np.zeros, the test attempts 50 GB and the CI runner is killed rather than
    the test failing.
    """
    n_variants = np.broadcast_to(np.zeros((), np.int32), (11, 7, 2))
    assert n_variants.strides == (0, 0, 0)
    assert not n_variants.flags.writeable
    assert n_variants.dtype == np.int32
    assert n_variants.shape == (11, 7, 2)
    # Consumers fancy-index it (_impl.py:1290), which always copies, so the
    # array a user receives is writable and independent.
    taken = n_variants[np.array([0, 1])]
    assert taken.flags.writeable and taken.shape == (2, 7, 2)
```

- [ ] **Step 2: Run it**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_ranges.py -q -k n_variants`
Expected: PASS — it characterizes numpy, not GVL. It exists so the contract is written down next to the change.

- [ ] **Step 3: Replace the allocation**

In `python/genvarloader/_dataset/_svar2_haps.py`, replace the `n_variants` block in `__post_init__` (`:270-279`):

```python
        # n_variants is all zeros, and wrong: the read-bound decode never counts
        # a query's variants without also decoding them, so there is nothing to
        # fill this with at open time. Tracked as #363 -- Dataset.n_variants()
        # reports zeros on SVAR2. Kept zero-valued rather than absent because the
        # shape (R, S, P) is what callers read it for.
        #
        # A real np.zeros here is 50.7 GB at All of Us chr19 (#355), allocated at
        # open, for an array nothing writes to. A zero-stride broadcast has the
        # same shape and dtype for 0.8 KiB. Two caveats: `.nbytes` still reports
        # the full 50 GB, and np.broadcast_to pickles by materializing, so spawn
        # workers (to_dataloader(num_workers>0)) serialize it in full -- exactly
        # as they did with np.zeros, so no regression, but not free either.
        self.n_variants = np.broadcast_to(
            np.zeros((), np.int32), (self.n_regions, self.n_samples, self.ploidy)
        )
```

- [ ] **Step 4: Note the contract asymmetry**

Update the `n_variants` annotation's docstring at `python/genvarloader/_dataset/_haps.py:294` to record that the SVAR2 view is read-only:

```python
    n_variants: NDArray[np.int32]
    """Per ``(region, sample, ploid)`` variant counts.

    SVAR1 fills this with real counts and it is writable. SVAR2 cannot count
    without decoding (#363), so it is a read-only zero-stride view of zeros
    there -- ``+=`` on it raises. Read ``.shape`` and index it; do not mutate it.
    """
```

- [ ] **Step 5: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS. If anything mutates `n_variants` in place it fails here with `ValueError: assignment destination is read-only` — that is a real bug the old writable array was hiding, so fix the mutation, do not revert the view.

- [ ] **Step 6: Commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_svar2_haps.py python/genvarloader/_dataset/_haps.py tests/unit/dataset/test_svar2_ranges.py
git commit -m "perf(svar2): stop allocating a dense n_variants at open

Svar2Haps allocated a real (R, S, P) int32 of zeros -- 50.7 GB at All of Us
chr19 -- for an array nothing ever writes to and that only #363's documented
zeros are read from. A zero-stride broadcast_to view has the same shape and
dtype for 0.8 KiB.

Closes #355"
```

---

## Task 8: Streaming concat merge

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_ranges.py` (the merge helper)
- Modify: `python/genvarloader/_dataset/_concat.py:235-311`
- Modify: `tests/dataset/test_concat_svar2.py`

**Interfaces:**
- Consumes: `_ranges_reader`, `_SparseWriter`, `ENTRY_DTYPE` (Tasks 3-5); `rewrite_as_dense` (Task 5); the two shard fixtures (Task 2).
- Produces: `def merge_region_blocks(readers, r_maps, s_maps, writer, n_regions, span, ploidy)` in `_svar2_ranges.py`, and a `_concat_svar2_ranges` with an unchanged signature that emits the sparse layout.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_concat_svar2.py`:

```python
def test_concat_svar2_emits_sparse_layout(svar2_shards_by_samples, tmp_path: Path):
    """The merged output must be sparse, whatever the inputs were."""
    import json

    shards, _ = svar2_shards_by_samples
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples", overwrite=True)

    meta = json.loads(
        (out / "genotypes" / "svar2_ranges" / "svar2_meta.json").read_text()
    )
    assert meta["layout"] == "sparse"
    # The output meta is built FRESH, not patched from input #0 -- a patched meta
    # would carry stale vk_* keys and claim two layouts at once.
    assert "vk_snp_range" not in meta
    assert meta["n_samples"] == 3 and meta["n_regions"] == 3


def test_concat_svar2_dense_input_reads_like_single_write(
    svar2_shards_by_samples, tmp_path: Path
):
    """A legacy dense shard must merge into a sparse output correctly."""
    from tests._oracles.svar2_dense_layout import rewrite_as_dense

    shards, full = svar2_shards_by_samples
    mixed = [rewrite_as_dense(shards[0], tmp_path / "dense_shard.gvl"), shards[1]]
    out = tmp_path / "merged_mixed.gvl"
    gvl.concat(out, mixed, axis="samples", overwrite=True)

    for a, b in zip(_read_all(out), _read_all(full)):
        np.testing.assert_array_equal(a, b)


def test_concat_svar2_regions_fast_path_matches_general_merge(
    svar2_shards_by_regions, tmp_path: Path
):
    """The all-sparse axis="regions" copy_runs path must equal the general merge.

    Rewriting one shard as dense is what forces the general path (the fast path
    requires every reader to be sparse), so the two outputs are produced by two
    genuinely different code paths from the same data and must agree byte for
    byte -- region_ptr included, since that is what the fast path computes
    itself rather than accumulating.
    """
    from tests._oracles.svar2_dense_layout import rewrite_as_dense

    shards, _ = svar2_shards_by_regions
    fast = tmp_path / "fast.gvl"
    gvl.concat(fast, shards, axis="regions", overwrite=True)

    mixed = [rewrite_as_dense(shards[0], tmp_path / "dense_shard.gvl"), shards[1]]
    slow = tmp_path / "slow.gvl"
    gvl.concat(slow, mixed, axis="regions", overwrite=True)

    for name in ("region_ptr.npy", "cell_id.npy", "cell_vk.npy"):
        a = (fast / "genotypes" / "svar2_ranges" / name).read_bytes()
        b = (slow / "genotypes" / "svar2_ranges" / name).read_bytes()
        assert a == b, name


def test_concat_svar2_rejects_grid_disagreement(
    svar2_shards_by_samples, tmp_path: Path
):
    """svar2_meta.json and metadata.json must agree about the grid.

    They are decoded against each other -- the reader's n_samples sets the key
    stride, `shapes` places the result -- so a disagreement scrambles keys rather
    than producing a wrong-sized output, which nothing downstream would catch.
    """
    import json
    import shutil

    shards, _ = svar2_shards_by_samples
    bad = tmp_path / "bad_shard.gvl"
    shutil.copytree(shards[0], bad)
    mp = bad / "genotypes" / "svar2_ranges" / "svar2_meta.json"
    meta = json.loads(mp.read_text())
    meta["n_samples"] += 1
    mp.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="disagree about the dataset's shape"):
        gvl.concat(tmp_path / "out.gvl", [bad, shards[1]], axis="samples", overwrite=True)


def test_concat_svar2_rejects_unsorted_input_samples(
    svar2_shards_by_samples, tmp_path: Path
):
    """An unsorted input breaks the ascending-remap the merge depends on."""
    import json
    import shutil

    shards, _ = svar2_shards_by_samples
    bad = tmp_path / "unsorted_shard.gvl"
    shutil.copytree(shards[0], bad)
    mp = bad / "metadata.json"
    md = json.loads(mp.read_text())
    md["samples"] = list(reversed(md["samples"]))
    mp.write_text(json.dumps(md))

    with pytest.raises(ValueError, match="samples are not sorted"):
        gvl.concat(tmp_path / "out2.gvl", [bad, shards[1]], axis="samples", overwrite=True)


def test_merge_region_blocks_interleaves(tmp_path: Path):
    """The merge must interleave inputs INSIDE a region, not concatenate them.

    Two shards whose sample slots interleave in the merged order (A owns merged
    slots 0 and 2, B owns slot 1) with entries in the same regions. Region 0 must
    come out A, B, A. A merge that appended one input's block after the other's
    would still be ascending *per input* and would still write the right count,
    so only the interleaved key order catches it.
    """
    import numpy as np

    from genvarloader._dataset._svar2_ranges import (
        ENTRY_DTYPE,
        _SparseRanges,
        _SparseWriter,
        merge_region_blocks,
    )

    R, P = 3, 2

    def mk(region_ptr, cell_id, n_samples):
        cid = np.asarray(cell_id, np.int32)
        ent = np.zeros(len(cid), ENTRY_DTYPE)
        # Tag each entry so a scrambled merge is visible in the payload too.
        ent["snp_start"] = np.arange(len(cid)) + 100 * n_samples
        ent["snp_len"] = 1
        return _SparseRanges(
            region_ptr=np.asarray(region_ptr, np.int64),
            cell_id=cid,
            cell_vk=ent,
            n_regions=R,
            n_samples=n_samples,
            ploidy=P,
        )

    # A: 2 samples. region 0 holds cells {0, 3}, region 1 none, region 2 {1}.
    a = mk([0, 2, 2, 3], [0, 3, 1], n_samples=2)
    # B: 1 sample. region 0 holds cell {1}, region 1 {0}, region 2 none.
    b = mk([0, 1, 2, 2], [1, 0], n_samples=1)

    n_samples = 3
    span = n_samples * P
    r_maps = [np.arange(R, dtype=np.int64)] * 2
    s_maps = [np.array([0, 2], np.int64), np.array([1], np.int64)]

    w = _SparseWriter(tmp_path, n_samples=n_samples, ploidy=P)
    merge_region_blocks([a, b], r_maps, s_maps, w, n_regions=R, span=span, ploidy=P)
    n = w.close()

    # region 0: A slot0/p0 -> cell 0; B slot1/p1 -> cell 3; A slot2/p1 -> cell 5.
    # region 1: B slot1/p0 -> cell 2.   region 2: A slot0/p1 -> cell 1.
    assert n == 5
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "region_ptr.npy", np.int64), [0, 3, 4, 5]
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "cell_id.npy", np.int32), [0, 3, 5, 2, 1]
    )
    got = np.fromfile(tmp_path / "cell_vk.npy", ENTRY_DTYPE)
    np.testing.assert_array_equal(
        got["snp_start"], [200, 100, 201, 101, 202]
    )


def test_merge_region_blocks_handles_empty_regions(tmp_path: Path):
    """A merged region no input contributes to still needs a region_ptr entry.

    Trailing and leading empty regions are the case where an off-by-one in
    `lo`/`rc` produces a region_ptr of the wrong LENGTH, which the reader then
    memmaps as a truncated prefix.
    """
    import numpy as np

    from genvarloader._dataset._svar2_ranges import (
        ENTRY_DTYPE,
        _SparseRanges,
        _SparseWriter,
        merge_region_blocks,
    )

    R, P, S = 4, 2, 1
    empty = _SparseRanges(
        region_ptr=np.zeros(R + 1, np.int64),
        cell_id=np.empty(0, np.int32),
        cell_vk=np.empty(0, ENTRY_DTYPE),
        n_regions=R,
        n_samples=S,
        ploidy=P,
    )
    w = _SparseWriter(tmp_path, n_samples=S, ploidy=P)
    merge_region_blocks(
        [empty],
        [np.arange(R, dtype=np.int64)],
        [np.arange(S, dtype=np.int64)],
        w,
        n_regions=R,
        span=S * P,
        ploidy=P,
    )
    assert w.close() == 0
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "region_ptr.npy", np.int64), np.zeros(R + 1, np.int64)
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_concat_svar2.py -q`
Expected: FAIL — `_concat_svar2_ranges` still calls `gather_fixed` on `vk_snp_range.npy`, which Task 5 stopped writing, so all four tests fail.

- [ ] **Step 3: Add the merge helper**

Append to `python/genvarloader/_dataset/_svar2_ranges.py` (add `"merge_region_blocks"` to `__all__`, and `from ._concat_plan import CONCAT_CHUNK_BYTES` to the module's imports):

```python
def merge_region_blocks(
    readers: "list[_RangeLookup]",
    r_maps: "list[NDArray[np.int64]]",
    s_maps: "list[NDArray[np.int64]]",
    writer: "_SparseWriter",
    n_regions: int,
    span: int,
    ploidy: int,
) -> None:
    """Merge k range caches into ``writer``, one merged-region batch at a time.

    Merged keys are ``r * span + slot * ploidy + ploid`` and both index maps are
    strictly increasing *within one input*, so a merged region's entries are
    exactly the union of that region's per-input blocks. There is no cross-region
    state: each batch is gathered, remapped, ordered with one ``argsort``, and
    written. Ties are impossible -- ``_concat_validate`` makes the inputs disjoint
    on whichever axis is being merged -- so the sort's stability is irrelevant and
    ``_SparseWriter``'s strictly-ascending check is a real assertion rather than a
    formality.

    This replaces a block-wise k-way merge over ``iter_entries``. That merge ran
    its Python loop once per *alternation* in the merged key sequence, not once
    per block: on ``axis="samples"`` the ownership pattern repeats inside every
    region, so two shards with interleaved sample IDs cost ~N/2 iterations --
    4.9e8 iterations, ~29 minutes, at the All of Us genome projection. This costs
    ``ceil(n_regions / rows)`` iterations no matter how the inputs interleave, and
    is less code.

    Args:
        readers: One per input dataset, in input order.
        r_maps: Per input, source region index -> merged region index. The
            **scatter-inverse** of ``provenance``'s ``order`` (which is merged ->
            source); ``order[:, 1]`` is the inverse permutation and would silently
            scramble regions. Strictly increasing.
        s_maps: Per input, source sample slot -> merged sample slot. Likewise
            strictly increasing -- merged samples are ``sorted(union)`` of
            non-overlapping sorted inputs.
        writer: Destination; receives one ``append`` per region batch.
        n_regions: Merged region count.
        span: ``n_samples * ploidy`` in the merged keyspace.
        ploidy: ``P``, identical across inputs (``_concat_validate`` enforces it).
    """
    rows = max(1, CONCAT_CHUNK_BYTES // ((ENTRY_DTYPE.itemsize + 8) * max(span, 1)))
    for r0 in range(0, n_regions, rows):
        r1 = min(r0 + rows, n_regions)
        keys: "list[NDArray[np.int64]]" = []
        ents: "list[NDArray[np.void]]" = []
        for rd, r_map, s_map in zip(readers, r_maps, s_maps):
            # r_map is strictly increasing, so merged [r0, r1) is a contiguous
            # slice of this input's own region axis -- two searchsorteds, no scan.
            w0 = int(np.searchsorted(r_map, r0, "left"))
            w1 = int(np.searchsorted(r_map, r1, "left"))
            if w1 <= w0:
                continue
            src_key, ent = rd.entries_for_regions(w0, w1)
            if not len(src_key):
                continue
            r_src, rest = np.divmod(src_key, rd.n_samples * ploidy)
            s_src, p = np.divmod(rest, ploidy)
            # Region-LOCAL keys: _SparseWriter.append takes lo/rc separately.
            keys.append((r_map[r_src] - r0) * span + s_map[s_src] * ploidy + p)
            ents.append(ent)

        if not keys:
            key = np.empty(0, np.int64)
            ent = np.empty(0, ENTRY_DTYPE)
        elif len(keys) == 1:
            key, ent = keys[0], ents[0]
        else:
            key = np.concatenate(keys)
            ent = np.concatenate(ents)
            perm = np.argsort(key, kind="stable")
            key, ent = key[perm], ent[perm]
        writer.append(key, ent, lo=r0, rc=r1 - r0)
```

- [ ] **Step 4: Rewrite `_concat_svar2_ranges`**

In `python/genvarloader/_dataset/_concat.py`, replace `_concat_svar2_ranges` (`:235-311`):

```python
def _concat_svar2_ranges(
    paths: list[Path],
    out_dir: Path,
    axis: str,
    shapes: list[tuple[int, int]],
    ploidy: int,
    n_regions: int,
    n_samples: int,
    order: "NDArray[np.int64]",
) -> None:
    """Merge a .svar2 dataset's cached range arrays.

    The per-``(region, sample, ploid)`` var-key ranges are sparse (#357), so they
    merge rather than gather: each merged region batch is gathered from every
    input, remapped into the merged keyspace, ordered, and appended. Legacy dense
    inputs feed the same merge through ``_DenseRanges.entries_for_regions``.

    After this change an svar2 ``concat`` with no per-sample tracks holds nothing
    ``R x S``-sized: the ``(R*S*P, 2)`` ``provenance`` array (64 GB at All of Us
    chr22) and the ``list[Run]`` ``coalesce`` builds from it (~204 bytes per run,
    which on an interleaved sample merge degenerates to one run per slot) are both
    gone from this path. Per-sample tracks still plan in core at
    ``_concat.py:465`` -- 32 GB plus a ~424 GB run list at the same projection --
    so ``concat`` is bounded only for tracks-free datasets. Tracked separately.

    ``dense_snp_range``/``dense_indel_range`` are per-region only (sample- and
    ploidy-independent), and cannot be sparsified -- genoray's ``dense_abs_row``
    uses ``.start`` as an index base. On ``axis="samples"`` they're identical
    across inputs and just linked from input #0; on ``axis="regions"`` they need
    the region ordering, so they gather through ``gather_fixed`` using region-only
    runs (``n_samples=1, ploidy=1``) built from the merged region ``order``.

    ``sample_cols`` maps merged sample slot -> index into the linked svar2 store's
    ``available_samples``; each input's own ``sample_cols.npy`` is already indexed
    by that input's own (sorted) sample list, so the merged array is a direct
    per-merged-sample lookup through ``order``.

    Raises:
        ValueError: If an input's ``svar2_meta.json`` disagrees with its
            ``metadata.json`` about the grid, or if an input's own sample list is
            not sorted on an ``axis="samples"`` merge.
    """
    readers = [_ranges_reader(p / "genotypes" / "svar2_ranges") for p in paths]

    # The merge trusts each reader's own (n_regions, n_samples, ploidy) to decode
    # its keys, and `shapes` to place them. If the two files disagree, every key
    # is decoded against the wrong stride and the output is silently scrambled
    # rather than wrong-sized -- so check, rather than let it through.
    for d, (rd, (R_d, S_d)) in enumerate(zip(readers, shapes)):
        if (rd.n_regions, rd.n_samples, rd.ploidy) != (R_d, S_d, ploidy):
            raise ValueError(
                f"input #{d}'s svar2_meta.json describes an "
                f"({rd.n_regions}, {rd.n_samples}, {rd.ploidy}) grid but its "
                f"metadata.json describes ({R_d}, {S_d}, {ploidy}); the two files "
                "disagree about the dataset's shape."
            )

    # `order` is merged -> source. The remap needs source -> merged, i.e. the
    # scatter-inverse; `order[:, 1]` is the inverse permutation and would produce
    # a silently scrambled dataset.
    if axis == "regions":
        # _concat_validate requires identical samples in identical order here.
        s_maps = [np.arange(n_samples, dtype=np.int64) for _ in paths]
        r_maps = [np.empty(r, np.int64) for r, _ in shapes]
        for i, (d, w) in enumerate(order):
            r_maps[d][w] = i
    else:
        # _concat_validate requires inp.bed.equals(ref.bed) here.
        r_maps = [np.arange(n_regions, dtype=np.int64) for _ in paths]
        s_maps = [np.empty(s, np.int64) for _, s in shapes]
        for i, (d, w) in enumerate(order):
            s_maps[d][w] = i
        # merge_region_blocks relies on each s_map being strictly INCREASING, so
        # that a merged region's entries come out ascending after one sort of the
        # concatenated blocks. That holds iff each input's own sample list is
        # sorted, because the merged order is sorted(union). gvl.write sorts
        # unconditionally (_write.py:284), so this only fires on a hand-built or
        # externally-produced store -- where it would otherwise scramble samples.
        for d, p in enumerate(paths):
            inp_samples = json.loads((p / "metadata.json").read_text())["samples"]
            if list(inp_samples) != sorted(inp_samples):
                raise ValueError(
                    f"input #{d}'s samples are not sorted. concat merges the "
                    "sparse range caches by remapping each input's keys into the "
                    "merged keyspace and relying on that remap to stay ascending, "
                    "which requires each input's own sample list to be sorted "
                    "(gvl.write sorts unconditionally)."
                )

    out_span = n_samples * ploidy
    if axis == "regions" and all(isinstance(rd, _SparseRanges) for rd in readers):
        # Every merged region draws its whole CSR block from exactly one input,
        # with cell ids unchanged (s_map is the identity and S is equal across
        # inputs), so this is a pure reorder of ragged blocks -- the same shape of
        # problem copy_runs already solves for tracks, with region_ptr as the
        # offsets array. No decode, no remap, no sort: byte ranges only.
        region_runs = coalesce(
            provenance("regions", [(r, 1) for r, _ in shapes], 1, order=order)
        )
        src_ptr = [np.asarray(rd.region_ptr, np.int64) for rd in readers]
        merged_ptr = None
        for fname, itemsize in (
            ("cell_id.npy", 4),
            ("cell_vk.npy", ENTRY_DTYPE.itemsize),
        ):
            # Both calls return the same offsets -- the two files are parallel --
            # so keeping the last is keeping any of them.
            merged_ptr = copy_runs(
                [p / "genotypes" / "svar2_ranges" / fname for p in paths],
                out_dir / fname,
                region_runs,
                src_ptr,
                itemsize=itemsize,
            )
        assert merged_ptr is not None
        merged_ptr.astype(np.int64).tofile(out_dir / "region_ptr.npy")
        n_entries = int(merged_ptr[-1])
    else:
        with _SparseWriter(out_dir, n_samples=n_samples, ploidy=ploidy) as writer:
            merge_region_blocks(
                readers,
                r_maps,
                s_maps,
                writer,
                n_regions=n_regions,
                span=out_span,
                ploidy=ploidy,
            )
        n_entries = writer.n_entries

    if axis == "samples":
        for name in ("dense_snp_range", "dense_indel_range"):
            link_or_copy_buffered(
                paths[0] / "genotypes" / "svar2_ranges" / f"{name}.npy",
                out_dir / f"{name}.npy",
            )
    else:
        region_shapes = [(r, 1) for r, _ in shapes]
        region_runs = coalesce(provenance("regions", region_shapes, 1, order=order))
        for name in ("dense_snp_range", "dense_indel_range"):
            gather_fixed(
                [p / "genotypes" / "svar2_ranges" / f"{name}.npy" for p in paths],
                out_dir / f"{name}.npy",
                region_runs,
                record_bytes=16,
            )

    cols = [
        np.load(p / "genotypes" / "svar2_ranges" / "sample_cols.npy") for p in paths
    ]
    if axis == "samples":
        merged_cols = np.array([cols[d][w] for d, w in order], np.int64)
    else:
        merged_cols = cols[0]
    np.save(out_dir / "sample_cols.npy", merged_cols)

    # Built FRESH, not patched from input #0: a dense input has no region_ptr key
    # and a patched meta would keep stale vk_* keys, leaving a file that claims
    # both layouts -- and from which the reader would take a stale `n_samples`.
    R, S, P = n_regions, n_samples, ploidy
    (out_dir / "svar2_meta.json").write_text(
        json.dumps(
            {
                "layout": "sparse",
                "n_regions": R,
                "n_samples": S,
                "n_entries": n_entries,
                "fill": (n_entries / (R * S * P)) if R * S * P else 0.0,
                "region_ptr": {"shape": [R + 1], "dtype": "<i8"},
                "cell_id": {"shape": [n_entries], "dtype": "<i4"},
                "cell_vk": {"shape": [n_entries], "dtype": ENTRY_DTYPE.descr},
                "dense_snp_range": {"shape": [R, 2], "dtype": "<i8"},
                "dense_indel_range": {"shape": [R, 2], "dtype": "<i8"},
                "sample_cols": {"shape": [S], "dtype": "<i8"},
                "ploidy": P,
            }
        )
    )
```

Add to `_concat.py`'s imports:

```python
from ._svar2_ranges import (
    ENTRY_DTYPE,
    _ranges_reader,
    _SparseRanges,
    _SparseWriter,
    merge_region_blocks,
)
```

`copy_runs` is already imported at `_concat.py:15-22`; `provenance`/`coalesce` likewise.

- [ ] **Step 5: Run the concat tests**

Run: `pixi run -e dev pytest tests/dataset/test_concat_svar2.py -v`
Expected: PASS, all nine — including the two characterization tests from Task 2, which are the real gate: the merged dataset must read exactly like a single-shot `gvl.write`.

- [ ] **Step 6: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS.

- [ ] **Step 7: Lint, typecheck, commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_svar2_ranges.py python/genvarloader/_dataset/_concat.py tests/dataset/test_concat_svar2.py
git commit -m "feat(concat): merge svar2 range caches by region batch

The var-key ranges are no longer fixed-size per slot, so gather_fixed cannot
move them. Each merged region batch is gathered from every input, remapped into
the merged keyspace, ordered and appended; legacy dense inputs feed the same
merge through _DenseRanges.entries_for_regions.

Region-batched rather than a k-way merge over an entry stream. A block-wise
k-way merge iterates once per ALTERNATION in the merged key sequence, not once
per block, and on axis=samples the ownership pattern repeats inside every
region: two shards with interleaved sample IDs cost ~N/2 iterations, ~29 min at
the All of Us genome projection. This costs ceil(R / rows) iterations however
the inputs interleave, and is less code.

axis=regions with all-sparse inputs skips the merge entirely: every merged
region draws its whole CSR block from one input with cell ids unchanged, which
is a ragged reorder copy_runs already does, with region_ptr as the offsets.

The remap uses the scatter-INVERSE of provenance's order, which is merged ->
source; order[:, 1] is the inverse permutation and would silently scramble
regions. The output meta is built fresh rather than patched from input #0,
which would otherwise carry stale vk_* keys and a stale n_samples.

No R x S allocation is left on this path -- provenance's (R*S*P, 2) array and
the run list coalesce built from it are both gone. Per-sample tracks still plan
in core, so concat is bounded only for tracks-free datasets.

Relates to #357"
```

---

## Task 9: Benchmark gate

The fixture dataset cannot gate this: its dense cache is ~6 KB, a key array that lives in L1, where the measured delta is 9.5 microseconds — 0.0095% of a 100 ms batch. As a ratio it blocks the change over nothing; as wall-clock it can never fire.

**Files:**
- Create: `tests/benchmarks/profiling/bench_svar2_range_lookup.py`

**Interfaces:**
- Consumes: `_SparseRanges`, `ENTRY_DTYPE` from Task 3.
- Produces: a standalone script, not collected by pytest. Its printed table goes in the PR description.

- [ ] **Step 1: Write the harness**

Create `tests/benchmarks/profiling/bench_svar2_range_lookup.py`:

```python
"""Benchmark gate for the sparse svar2 range-cache probe (#357).

Synthetic: needs no All of Us access and runs in ~30 s. Run it directly, not
under pytest:

    pixi run -e dev python tests/benchmarks/profiling/bench_svar2_range_lookup.py

THE GATE: sparse lookup <= 2% of batch wall on the GLOBALLY SHUFFLED
distribution, against the 171 ms single-threaded spliced batch recorded in
`Svar2Haps._readbound_gather`'s docstring.

The shuffled row is the training case, not a pathological one:
`to_dataloader(shuffle=True)` (_impl.py:1825) shuffles the flat (R*S) index.
The clustered row is what `ds[:, :]` produces.

The baseline is a RESIDENT dense array at a size that fits (1000 x 2000 x 2 =
128 MB) -- not the 6 KB fixture, and not the 128 GB array the real comparison
would need, which nobody can run.

If the two-level probe misses the gate, do NOT take the spec's flat-int64-key
fallback. It was written before the depth was measured and it makes the probe
slower, not faster: a flat key searches the whole table, raising depth from
log2(N/R) = 13 to log2(N) = 24 at All of Us chr22, on the term measured at 91%
of lookup cost. Its only real effect is +4 bytes per entry.

The actual fallback is to fuse the search loop, which is where the time is: the
probe is ~93% numpy-pass-bound (measured -- the gather itself is the minority),
so a single-pass kernel over (region_ptr, cell_id) removes the per-iteration
temporaries the vectorized form cannot. That is a Rust #[pyfunction], deferred to
a follow-up issue (Task 10, Step 9) precisely because it is only worth doing if
this gate fails: at the measured numbers the whole probe is 0.5% of batch wall,
so a 3-5x relative win is worth ~0.35% of wall.
"""

from __future__ import annotations

import time

import numpy as np

from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE, _SparseRanges

BATCH_MS = 171.0
"""Single-threaded spliced 8192-cell batch wall, from _readbound_gather's docstring.

This is a recorded figure from another machine, so the percentages below are only
as good as it is. Re-measure it on the machine running the gate before trusting a
borderline result -- a batch that is actually 60 ms here turns a 1.9% PASS into a
5.4% FAIL. The spec's companion figure of 18.4 ms for a shuffled 8192-cell probe
does NOT reproduce (measured ~13x lower); the table this script prints supersedes
it.
"""
GATE = 0.02


def build(R: int, S: int, P: int, N: int, seed: int = 0) -> _SparseRanges:
    """A CSR table with N entries spread over R regions."""
    rng = np.random.default_rng(seed)
    span = S * P
    per = np.bincount(rng.integers(0, R, N), minlength=R)
    per = np.minimum(per, span)
    cell = np.concatenate(
        [np.sort(rng.choice(span, size=c, replace=False)) for c in per]
    ).astype(np.int32)
    ent = np.zeros(len(cell), ENTRY_DTYPE)
    ent["snp_len"] = 1
    ptr = np.concatenate([[0], per.cumsum()]).astype(np.int64)
    return _SparseRanges(ptr, cell, ent, R, S, P)


def probes(R: int, S: int, n_q: int, how: str, seed: int = 1):
    rng = np.random.default_rng(seed)
    if how == "clustered":
        # What ds[:, :] produces: few regions, many sorted slots.
        n_r = max(1, n_q // 512)
        r_q = np.repeat(rng.integers(0, R, n_r), n_q // n_r)[:n_q]
        si_q = np.tile(np.sort(rng.choice(S, n_q // n_r, replace=False)), n_r)[:n_q]
    else:
        # What to_dataloader(shuffle=True) produces.
        r_q = rng.integers(0, R, n_q)
        si_q = rng.integers(0, S, n_q)
    return r_q.astype(np.int64), si_q.astype(np.int64)


def timeit(fn, reps: int = 20) -> float:
    fn()  # warm
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1000


def main():
    R, S, P = 3734, 535662, 2  # All of Us chr22
    print(f"{'N':>10} {'n_q':>6} {'dist':>10} {'ms':>8} {'% batch':>8}")
    worst = 0.0
    for N in (10**5, 10**6, 10**7, 10**8):
        table = build(R, S, P, N)
        for n_q in (512, 2048, 8192):
            for how in ("clustered", "shuffled"):
                r_q, si_q = probes(R, S, n_q, how)
                ms = timeit(lambda: table.lookup(r_q, si_q, P))
                pct = ms / BATCH_MS * 100
                if how == "shuffled":
                    worst = max(worst, pct)
                print(f"{N:>10} {n_q:>6} {how:>10} {ms:>8.2f} {pct:>7.2f}%")

    # Baseline: a dense array at a size that actually fits.
    d = np.zeros((1000, 2000, 2, 2), np.int64)  # 128 MB
    rng = np.random.default_rng(2)
    r_q = rng.integers(0, 1000, 8192)
    si_q = rng.integers(0, 2000, 8192)
    ms = timeit(lambda: d[r_q, si_q].reshape(-1, 2))
    print(f"\nresident dense baseline (128 MB, 8192 cells): {ms:.2f} ms")

    print(f"\nworst shuffled: {worst:.2f}% of a {BATCH_MS} ms batch (gate: {GATE:.0%})")
    print("PASS" if worst <= GATE * 100 else "FAIL -- take the spec's flat-key fallback")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `pixi run -e dev python tests/benchmarks/profiling/bench_svar2_range_lookup.py`
Expected: a table, ending `PASS`. Capture the output for the PR description.

First, sanity-check `BATCH_MS` before acting on a borderline number: it is a figure recorded on another machine. Re-measure a spliced 8192-cell batch here (`Svar2Haps._readbound_gather`'s docstring says how it was produced) and substitute it. A gate expressed as a fraction of a stale denominator is not a measurement.

If it still prints `FAIL`: do not tune ad hoc, and do **not** take the spec's flat-int64-key fallback — it raises search depth from 13 to 24 at chr22 (`log2(N)` instead of `log2(N/R)`) on the term measured at 91% of lookup cost, and its only other effect is +4 bytes per entry. Stop, file the fused-kernel issue from Step 9 as blocking, and bring the measurement back for a decision.

- [ ] **Step 3: Commit**

```bash
git add tests/benchmarks/profiling/bench_svar2_range_lookup.py python/genvarloader/_dataset/_svar2_ranges.py
git commit -m "perf(svar2): add the sparse range-lookup benchmark gate

Synthetic, no All of Us access, ~30 s. Sweeps N over 1e5..1e8 at All of Us
chr22's grid, on both the clustered (ds[:, :]) and globally shuffled
(to_dataloader(shuffle=True)) probe distributions, against a resident 128 MB
dense baseline. Gate: sparse lookup <= 2% of the 171 ms spliced batch wall on
the shuffled distribution.

Relates to #357"
```

---

## Task 10: Docs, skill, and the PR

CLAUDE.md's docs gate applies — this is an on-disk format change — and it says explicitly that the auto-generated changelog does not count.

**Files:**
- Modify: `docs/source/format.md` (lines 19, 27-28, 80-93, 95-107, 109-111, 188)
- Modify: `docs/source/write.md:110`
- Modify: `docs/source/faq.md:99`
- Modify: `python/genvarloader/_dataset/_write.py:162-166`
- Modify: `skills/genvarloader/SKILL.md` (lines 86, 135, 440, 503-508)

**Interfaces:**
- Consumes: the final layout from Tasks 3-8.
- Produces: no code interfaces. The PR.

- [ ] **Step 1: Rewrite `format.md`'s layout and size sections**

Open `docs/source/format.md` and fix, in order:

- `:19` and `:27-28` — "caches per-`(region, sample, ploidy)` range arrays" becomes "caches the var-key window for each `(region, sample, ploidy)` that holds a variant".
- `:80-93` — the layout table: replace the `vk_snp_range.npy` / `vk_indel_range.npy` rows with `region_ptr.npy` (`(R+1,)` int64), `cell_id.npy` (`(N,)` int32) and `cell_vk.npy` (`(N,)`, a 24-byte record of `snp_start` int64, `indel_start` int64, `snp_len` int32, `indel_len` int32). Add `layout`, `n_regions`, `n_samples`, `n_entries` and `fill` to the `svar2_meta.json` description. State that all three new files are raw and headerless, like their neighbours, and that only `sample_cols.npy` is a real `.npy`.
- `:95-107` — the size block. All four claims die: the `2 x regions x samples x ploidy x 2 x 8` formula, "grows linearly in **both**", "approximately **98 GiB**", and "logs the projected size before allocating". Replace with: 28 bytes per non-empty `(region, sample, ploid)` window, so size scales with variants observed inside regions rather than with `regions x samples`; a 128 GB dense cache at All of Us chr22 becomes 504 MB at 0.45% fill; the writer logs realized fill after the first contig and projects from it.
- `:109-111` — "slices these memmaps (numpy fancy-indexing; no interval search)" becomes a bounded binary search within one region's block, with no search at all for fully occupied regions.
- `:188` — add a `0.43.0` row, and an upgrade note: **re-run `gvl.write`** to shrink an existing dataset. `gvl.migrate` does *not* do this (it handles only the 1.x -> 2.0 track AoS-to-SoA migration). Also state that GVL <= 0.42.1 opening a 0.43.0 dataset fails with a bare `KeyError: 'vk_snp_range'` — loud, not silently wrong, but unhelpful.

- [ ] **Step 2: Fix `write.md` and `faq.md`**

- `docs/source/write.md:110` — "not small at cohort scale" is now false, **and** the `max_mem` RAM-bounding promise is wrong. Replace with: the range cache is small at cohort scale (~504 MB for All of Us chr22); `max_mem` bounds the genoray chunk stream but **not** the per-contig entry accumulator, which peaks at roughly `60 bytes x entries on the largest contig` — the per-chunk `(region int32, cell int32, 24-byte entry)` blocks plus the merged `(cell, entry)` output, live at the same time. That is ~0.8 GB at All of Us chr22 and ~2.6 GB at chr19. Do not quote a figure from the argsort formulation this plan replaced: `append_contig` measured 0.56x its peak RSS.
- `docs/source/faq.md:99` — "not small at cohort scale, see the size formula" becomes the new figure and a pointer to `format.md`'s layout section.

- [ ] **Step 3: Fix `gvl.write`'s own docstring**

`python/genvarloader/_dataset/_write.py:162-166` currently says the cache "is two `(n_regions, n_samples, ploidy, 2)` int64 memmaps ... 60.7 GiB for 1,901 regions x 535,662 diploid samples". Every clause is now false. Replace with: 28 bytes per `(region, sample, ploid)` window that holds a variant, so the cache scales with observed variants rather than with `regions x samples`; the same All of Us chr22 grid is ~504 MB. Say explicitly that `max_mem` does not bound the per-contig accumulator (~60 bytes per entry on the largest contig), since that is the one memory surprise left. CLAUDE.md's skill-maintenance rule names this docstring explicitly, so it is not optional.

- [ ] **Step 4: Update the skill**

`skills/genvarloader/SKILL.md`:
- `:86` and `:135` — the size formula and "scales with `regions x samples x ploidy`".
- `:440` — the `svar2_ranges/` layout tree.
- `:503-508` — the "Common gotchas" bullet about cache size; it should now warn about the *write-time accumulator* instead (~60 bytes per entry on the largest contig, not bounded by `max_mem`), since that is the remaining memory surprise.

Then re-check the "Where to look next" pointer table per CLAUDE.md.

- [ ] **Step 5: Verify `api.md` is still in sync**

Run: `pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none`. No public symbol changed, so this is a confirmation, not a fix. `README.md` needs no change either — it does not mention the cache.

- [ ] **Step 6: Build the docs**

Run: `pixi run -e docs doc`
Expected: builds without warnings from the edited files.

- [ ] **Step 7: Full test run and lint**

Run: `pixi run -e dev test`
Expected: PASS (pytest + cargo). Then: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck && pixi run -e dev python scripts/docstring_style.py --check python/genvarloader`

- [ ] **Step 8: Commit and push**

```bash
git add docs/source/format.md docs/source/write.md docs/source/faq.md skills/genvarloader/SKILL.md python/genvarloader/_dataset/_write.py
git commit -m "docs(svar2): describe the sparse range cache

Twelve sites still described the dense (R, S, P, 2) layout, its size formula
and its ~98 GiB figure, including gvl.write's own docstring and the agent
skill. Also corrects write.md's claim that max_mem bounds the range cache's
memory -- it bounds the genoray chunk stream, not the per-contig accumulator.

Relates to #355, #357"
git push -u origin worktree-feat+svar2-sparse-range-cache
```

- [ ] **Step 9: File the three follow-up issues**

Three measured findings are deliberately out of this PR's scope. File them before opening the PR so the PR body can reference them by number, and do not fold any of them in — each one is its own change with its own gate.

```bash
gh issue create --title "concat still plans per-sample tracks in core (R x S)" --body "After #357, an svar2 \`concat\` with no per-sample tracks holds nothing \`R x S\`-sized: the range caches merge by region batch and \`provenance\` is no longer called on that path.

Per-sample tracks still are. \`_concat.py:465\` calls \`provenance(...)\` for each per-sample track, which allocates an \`(R*S*P, 2)\` int64 array -- 32 GB at All of Us chr22 -- and \`coalesce\` then builds a \`list[Run]\` from it at ~204 bytes per run. On an interleaved sample merge (the common case: two cohorts whose sample IDs interleave in sorted order) that degenerates to one run per slot, i.e. ~424 GB of Python objects.

So \`concat\` is bounded only for tracks-free datasets. The fix is the same shape as the range-cache one: plan per region batch rather than materializing the whole provenance map, or emit runs as an iterator instead of a list.

Found while implementing #357."

gh issue create --title "svar2 range probe: fuse the search loop into a Rust kernel" --body "The sparse range probe (#357) is ~93% numpy-pass-bound: the branchless \`partition_point\` loop runs \`_depth\` iterations over the whole query block, each one a handful of full-array passes, and the \`cell_vk\` gather is the minority of the time. A single-pass kernel over \`(region_ptr, cell_id)\` would do the whole search per query in registers -- measured headroom is 3-5x on the probe itself.

Deliberately NOT done in #357, on measurement: the whole probe is ~0.5% of batch wall, so 3-5x on it is worth ~0.35% of wall. It becomes worth doing if the benchmark gate in \`tests/benchmarks/profiling/bench_svar2_range_lookup.py\` ever fails, or if the read path gets fast enough elsewhere that 0.5% matters.

Shape, if taken: a \`#[pyfunction]\` in \`src/ffi/mod.rs\` over a kernel in a domain module, registered in \`src/lib.rs\`'s \`#[pymodule]\`. Note there is no dual-backend parity harness any more (\`_dispatch.py\` and \`docs/roadmaps/rust-migration.md\` were retired in 8f9d3c99), so parity needs a hand-written numpy oracle plus frozen \`.npz\` goldens under \`tests/parity/\`."

gh issue create --title "genoray: emit sparse ranges from find_ranges_chunk" --body "GVL's sparse range cache (#357) is built by materializing genoray's dense \`(samples, ploidy, regions, 2)\` chunk and then throwing >99% of it away. \`np.nonzero\` on that dense block is 71% of GVL's per-chunk kernel time, and the dense intermediate is ~128 GB per All of Us chr22 contig at default \`max_mem\`.

The larger prize is sparsifying at the source: have \`find_ranges_chunk\` emit only the non-empty windows, so the dense intermediate never exists. That removes both the allocation and the scan, and GVL's \`nonempty_entries\` collapses into a passthrough.

This is a genoray change, filed here for tracking; it is the bigger win of the two kernel opportunities found while implementing #357."
```

Record the three issue numbers; Step 10's PR body references them.

- [ ] **Step 10: Open the PR**

```bash
gh pr create --title "feat(svar2)!: sparse range cache" --body "$(cat <<'EOF'
Closes #357, closes #355.

`genotypes/svar2_ranges/` stored two dense `(R, S, P, 2)` int64 memmaps -- 32
bytes for every `(region, sample, ploid)` cell, over 99% of them empty at cohort
scale. At All of Us chr22 that is 128 GB on disk for 0.58 GB of useful windows;
genome-wide, 6.93 TB. `Svar2Haps` additionally allocated a dense `(R, S, P)`
int32 `n_variants` of zeros at open -- 50.7 GB at chr19 -- that nothing writes to.

This replaces both.

**Layout.** A region-CSR table: `region_ptr` int64 `[R+1]`, `cell_id` int32
`[N]`, and a 24-byte `cell_vk` record of `(snp_start, indel_start, snp_len,
indel_len)`. 28 bytes per non-empty cell against dense's 32 per cell, so it is
strictly smaller at every fill level -- no threshold, no dense-fallback writer.
chr22: 504 MB. Genome: 27 GB.

**Why it is exact.** genoray's `gather_haps_readbound_impl` derives `j = vs + k`
*inside* the var-key loop, so it never reads an empty range's start. `(0, 0)` is
byte-identical to the true insertion point -- and strictly safer, since Rust
slicing panics if `vs > ve` while `(0, 0)` is unconditionally in bounds.

**Lookup.** A bounded, manually vectorized binary search inside one region's
block: 13 iterations at chr22 (`ceil(log2(widest block))`) rather than the 24 a
flat key would need, and no search at all for a fully occupied region -- the
whole high-fill regime that sequence-model windows live in, measured at 0.35 ms
against 0.89 ms. `np.searchsorted` cannot express per-element bounds, hence the
manual loop -- over bit-depth, never over queries, in the branchless
`std::partition_point` form so no `active` mask is needed.

**Write path.** The per-contig merge is a counting sort with `O(regions)`
auxiliary state, not `np.argsort(kind="stable")`: numpy maps that to radix only
for integers of 16 bits or fewer, so an int64 key sort is timsort. Measured at
N = 18e6: 914 ms (k=30) / 1369 ms (k=500) for the sort against 532 / 528 ms for
the counting sort, byte-identical, at 0.56x peak RSS -- and flat in chunk count,
which matters because `samples_per_chunk` can be 1.

**Concat.** Merges one merged-region batch at a time rather than k-way merging an
entry stream, which iterated once per *alternation* in the merged key sequence
(~29 min at the genome projection on an interleaved sample merge) rather than
once per block. `axis="regions"` with all-sparse inputs skips the merge entirely
and moves byte ranges through `copy_runs`. No `R x S` allocation is left on this
path.

**Guards.** Once an absent cell and an empty cell are indistinguishable, several
errors stop being loud: explicit bounds checks (a miss would otherwise be a
silent reference-only haplotype), an explicit `n_samples` in the meta
cross-checked against `sample_cols.npy` (a wrong `S` makes every probe miss),
int64 key arithmetic, and no memmap when `N == 0`.

**Compatibility.** `DATASET_FORMAT_VERSION` stays `2.0.0` -- it matches on MAJOR
only, so a bump would make new GVL refuse every existing dataset. A dense reader
is kept behind the same interface, so old datasets open unchanged. Re-run
`gvl.write` to shrink one; there is no migration tool.

**Breaking:** GVL <= 0.42.1 cannot open a dataset written by this version. It
fails with `KeyError: 'vk_snp_range'` -- loud, not silently wrong.

**Deliberately out of scope**, each filed in Step 9 with its measurement:
concat's remaining `R x S` planning for per-sample tracks; a fused Rust kernel
for the probe (worth ~0.35% of batch wall at current numbers); and sparsifying
genoray's `find_ranges_chunk` at the source, which is the larger prize -- the
dense chunk this PR scans with `np.nonzero` is 71% of the write kernel and
~128 GB per chr22 contig.

Design: `docs/superpowers/specs/2026-09-14-svar2-sparse-range-cache-design.md`
Plan: `docs/superpowers/plans/2026-09-14-svar2-sparse-range-cache.md`

Benchmark output (`tests/benchmarks/profiling/bench_svar2_range_lookup.py`):

```
<paste the table from Task 9>
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015VxRqNngU7Eg1wdb1aEgD6
EOF
)"
```

---

## Self-review

**Spec coverage.** Every spec section maps to a task: Layout -> 3; Lookup incl. the guards and both fast paths -> 3; Write path incl. the append convention, the contiguity assert and the `S*P` bound -> 5; Preflight -> 6; `n_variants` -> 7; Concat incl. the scatter-inverse and fresh meta -> 8; Meta schema and backward compat -> 4 (reader) + 5 (writer + dense helper); Testing incl. all six rewrites, the fixture, the characterization test and the Hypothesis property -> 1, 2, 3, 5; Docs -> 10; benchmark gate -> 9. Process notes (target `main`, no Rust gate, no version hand-edit) are in Global Constraints.

**Deliberately not built**, and recorded in the spec's "Out of scope": the `svar2_ranges="lazy"` mode, moving the probe into Rust, making `concat` scale, and genoray's O(S²) `_sample_idxs`. Three of those are now filed as issues in Task 10 Step 9 with the measurement that justifies deferring them, and their scope is narrower than the spec assumed: `concat` is `R x S`-bound *only* for per-sample tracks after Task 8, not for the range caches. One new deferral is added by this plan: the query-side all-samples scatter (see Deviation 2).

**Measurement-driven revisions.** Six of the plan's original choices were replaced after measurement rather than review; they are listed with their numbers in "Deviations from the spec" at the top, and each one's reasoning is carried in the docstring of the code that implements it, not only here. The load-bearing ones: `np.argsort(kind="stable")` is not radix above 16-bit (so Task 5 counting-sorts), the k-way entry merge iterates per alternation rather than per block (so Task 8 batches by region), and neither a numba nor a Rust kernel is worth taking in this PR (Global Constraints, with the two measured opportunities filed in Task 10 Step 9).

**Placeholders:** none. Every code step carries the actual code; the one `<paste the table>` is a benchmark result that cannot exist before Task 9 runs, and the three issue numbers in Step 10's PR body come from Step 9.

**Type consistency:** `_RangeLookup` exposes `n_regions` / `n_samples` / `ploidy` and `lookup` / `entries_for_regions` / `iter_entries`; `_SparseRanges` and `_DenseRanges` both implement all six, and `iter_entries` is a loop over `entries_for_regions` in both. Readers are constructed only in `_ranges_reader` (production) or directly in tests. `ENTRY_DTYPE`'s field names are used identically in `nonempty_entries`, `_DenseRanges.entries_for_regions`, `_SparseRanges.lookup` and the tests. `_SparseWriter` has two append entry points with distinct signatures and no overlap in callers: `append(key, ent, lo, rc)` takes ordered region-local keys and is called only from `merge_region_blocks`; `append_contig(regions, cells, ents, lo, rc)` takes the three parallel arrays `nonempty_entries` returns and is called only from `_write_from_svar2`. `nonempty_entries(snp, indel, slot0, ploidy)` returns `(int32 region, int32 cell, ENTRY_DTYPE)` — no `span` parameter and no combined key, which is what `append_contig` consumes. `_svar2_preflight` and `_svar2_ranges_cache_bytes` keep their existing signatures; `_svar2_fill_projection` is new and called once, behind a latch.
