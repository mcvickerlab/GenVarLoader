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
    svar2_shards_by_samples, vcf_and_ref: tuple[Path, Path], tmp_path: Path
):
    """A legacy dense shard must merge into a sparse output correctly."""
    from tests._oracles.svar2_dense_layout import rewrite_as_dense

    shards, full = svar2_shards_by_samples
    ref = vcf_and_ref[1]
    mixed = [rewrite_as_dense(shards[0], tmp_path / "dense_shard.gvl"), shards[1]]
    out = tmp_path / "merged_mixed.gvl"
    gvl.concat(out, mixed, axis="samples", overwrite=True)

    for a, b in zip(_read_all(out, ref), _read_all(full, ref)):
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

    The mutation is on `metadata.json`'s sample list (which is what `shapes`,
    read via `ConcatInput.n_samples = len(meta.samples)`, is built from) rather
    than on `svar2_meta.json` itself: bumping `svar2_meta.json`'s own `n_samples`
    trips `_ranges_reader`'s own internal `sample_cols.npy`-length check first
    (a real but different error), never reaching the cross-file check this test
    targets.

    The appended name is chosen to sort LAST (`"Z_FAKE"`, after `"S0"`/`"S2"`)
    so the mutated sample list stays sorted. An earlier version of this test
    appended `"FAKE"`, which also happens to sort before `"S0"`/`"S2"` and so
    made the input's own sample list unsorted -- a second, independent
    violation. That version passed for the wrong reason: it never proved the
    grid check specifically, only that *some* check fires on a broken input.
    """
    import json
    import shutil

    shards, _ = svar2_shards_by_samples
    bad = tmp_path / "bad_shard.gvl"
    shutil.copytree(shards[0], bad)
    mp = bad / "metadata.json"
    meta = json.loads(mp.read_text())
    meta["samples"] = [*meta["samples"], "Z_FAKE"]
    assert meta["samples"] == sorted(meta["samples"]), (
        "mutation must stay sorted so this test isolates the grid check from "
        "the separate sortedness check"
    )
    mp.write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="disagree about the dataset's shape"):
        gvl.concat(
            tmp_path / "out.gvl", [bad, shards[1]], axis="samples", overwrite=True
        )


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
        gvl.concat(
            tmp_path / "out2.gvl", [bad, shards[1]], axis="samples", overwrite=True
        )


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
    np.testing.assert_array_equal(got["snp_start"], [200, 100, 201, 101, 202])


def test_merge_region_blocks_handles_empty_regions(tmp_path: Path):
    """A merge where every region is empty still needs a full-length region_ptr.

    A grid with no entries anywhere is the case where an off-by-one in `lo`/`rc`
    most easily produces a `region_ptr` of the wrong LENGTH (rather than wrong
    content) -- `close()` reporting 0 entries could otherwise mask a `region_ptr`
    that stops short of `n_regions + 1`, which the reader would then memmap as a
    truncated prefix. This test's grid is uniformly empty; it does not exercise
    a *mix* of populated and empty regions (that is covered by
    `test_merge_region_blocks_interleaves`, whose region 1 is empty).
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


def test_concat_svar2_samples_axis_remaps_nonzero_local_slot(tmp_path: Path):
    """The samples-axis s_map (scatter-inverse of `order`) must be exercised at a
    non-trivial local index, not just index 0.

    The shared `svar2_store` fixture (used by `svar2_shards_by_samples`) has only
    ONE occupied sparse cell across its whole (3 region, 3 sample, ploidy 2) grid,
    and that cell sits at local sample slot 0 in both its owning shard and the
    merged order -- so it stays in place under a completely broken s_map that
    forgets to invert `order` at all (`s_maps[d][w] = w`). Verified directly: that
    exact one-line mutation in `_concat_svar2_ranges` left the whole
    `test_concat_svar2.py` suite (all 9 tests, including both characterization
    tests) passing.

    This test builds two minimal, from-scratch `svar2_ranges/` directories (no
    `gvl.write`, no VCF) whose one occupied cell each sits at a local index that
    is NOT its merged index, and drives `_concat_svar2_ranges` directly so the
    scatter-inverse itself is what's under test, not diluted by an otherwise-empty
    grid.
    """
    import json

    from genvarloader._dataset._concat import _concat_svar2_ranges
    from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE

    R, P = 1, 2

    def mk_shard(
        root: Path, sample_names: list[str], occupied_w: int, tag: int
    ) -> Path:
        """A minimal on-disk svar2_ranges/ shard with one occupied cell."""
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps({"samples": sample_names}))
        rd = root / "genotypes" / "svar2_ranges"
        rd.mkdir(parents=True)
        S = len(sample_names)
        cell = occupied_w * P + 0  # ploid 0
        np.array([0, 1], np.int64).tofile(rd / "region_ptr.npy")
        np.array([cell], np.int32).tofile(rd / "cell_id.npy")
        ent = np.zeros(1, ENTRY_DTYPE)
        ent["snp_start"] = tag
        ent["snp_len"] = 1
        ent.tofile(rd / "cell_vk.npy")
        np.save(rd / "sample_cols.npy", np.arange(S, dtype=np.int64))
        np.zeros((R, 2), np.int64).tofile(rd / "dense_snp_range.npy")
        np.zeros((R, 2), np.int64).tofile(rd / "dense_indel_range.npy")
        (rd / "svar2_meta.json").write_text(
            json.dumps(
                {
                    "layout": "sparse",
                    "n_regions": R,
                    "n_samples": S,
                    "n_entries": 1,
                    "fill": 1 / (R * S * P),
                    "region_ptr": {"shape": [R + 1], "dtype": "<i8"},
                    "cell_id": {"shape": [1], "dtype": "<i4"},
                    "cell_vk": {"shape": [1], "dtype": ENTRY_DTYPE.descr},
                    "dense_snp_range": {"shape": [R, 2], "dtype": "<i8"},
                    "dense_indel_range": {"shape": [R, 2], "dtype": "<i8"},
                    "sample_cols": {"shape": [S], "dtype": "<i8"},
                    "ploidy": P,
                }
            )
        )
        return root

    # Shard A: samples S0 (local w=0, empty), S2 (local w=1, occupied, tag=222).
    # Shard B: sample S1 (local w=0, occupied, tag=111).
    # Merged sorted order is S0, S1, S2 -> slots 0, 1, 2. Both occupied cells
    # (A's S2 at local 1, B's S1 at local 0) move to a DIFFERENT merged slot than
    # their local one -- exactly what an un-inverted (identity) s_map gets wrong.
    a = mk_shard(tmp_path / "a", ["S0", "S2"], occupied_w=1, tag=222)
    b = mk_shard(tmp_path / "b", ["S1"], occupied_w=0, tag=111)

    order = np.array([[0, 0], [1, 0], [0, 1]], np.int64)  # S0, S1, S2
    out_dir = tmp_path / "out" / "genotypes" / "svar2_ranges"
    out_dir.mkdir(parents=True)
    _concat_svar2_ranges([a, b], out_dir, "samples", [(R, 2), (R, 1)], P, R, 3, order)

    region_ptr = np.fromfile(out_dir / "region_ptr.npy", np.int64)
    cell_id = np.fromfile(out_dir / "cell_id.npy", np.int32)
    vk = np.fromfile(out_dir / "cell_vk.npy", ENTRY_DTYPE)

    np.testing.assert_array_equal(region_ptr, [0, 2])
    # S1 -> merged slot 1 -> cell id 1*2+0=2; S2 -> merged slot 2 -> cell id
    # 2*2+0=4. An identity (un-inverted) s_map instead produces [0, 2].
    np.testing.assert_array_equal(cell_id, [2, 4])
    np.testing.assert_array_equal(vk["snp_start"], [111, 222])


def test_concat_svar2_regions_axis_remaps_and_orders_correctly(tmp_path: Path):
    """Regions-axis twin of `..._samples_axis_remaps_nonzero_local_slot`.

    `axis="regions"` has TWO independent merge implementations, each with its
    own way to get the region order wrong, and the shared fixture's regions
    shards (`svar2_shards_by_regions`) only ever put their one occupied cell in
    merged region 0 -- where every ordering agrees -- so neither bug is visible
    to `..._regions_reads_like_single_write` or
    `..._regions_fast_path_matches_general_merge`. Verified directly, by
    injecting each bug into `_concat_svar2_ranges` and re-running the whole
    `test_concat_svar2.py` suite (10 tests, both characterization tests
    included):

    - the general merge path's `r_maps` (used whenever any input is
      `_DenseRanges`, so the fast path is unavailable) must be the
      scatter-inverse of `order` (source -> merged). Setting
      `r_maps[d][w] = w` (un-inverted / identity) left all 10 tests passing.
    - the fast path (all-sparse inputs) reorders region-CSR blocks via
      `provenance("regions", ..., order=order)`; passing `order=None` (block
      concat: all of input #0's regions, then all of input #1's) also left
      all 10 tests passing.

    Both bugs preserve every per-region entry COUNT and never collide two
    entries onto the same `(region, cell_id)` pair, so `region_ptr`'s shape is
    unaffected, nothing raises, and the dataset opens cleanly -- they only
    move entries to the wrong region or reorder them within one. This test
    hand-builds two from-scratch shards (no `gvl.write`) with entries placed
    so the correct interleaved order visibly disagrees with both bugs' output,
    and drives `_concat_svar2_ranges` directly against both shards sparse
    (fast path) and with one rewritten dense (general path via
    `rewrite_as_dense`, the existing dense-input test oracle), so both merge
    implementations are actually exercised.
    """
    import json

    from genvarloader._dataset._concat import _concat_svar2_ranges
    from genvarloader._dataset._svar2_ranges import ENTRY_DTYPE
    from tests._oracles.svar2_dense_layout import rewrite_as_dense

    S, P = 2, 1  # 2 samples so co-located regions still get distinct cell ids.

    def mk_region_shard(
        root: Path, n_regions: int, occupied: list[tuple[int, int, int]]
    ) -> Path:
        """A minimal on-disk svar2_ranges/ shard; `occupied` is (region, slot, tag)."""
        root.mkdir(parents=True)
        (root / "metadata.json").write_text(json.dumps({"samples": ["S0", "S1"]}))
        rd = root / "genotypes" / "svar2_ranges"
        rd.mkdir(parents=True)
        by_region: dict[int, list[tuple[int, int]]] = {}
        for r, slot, tag in occupied:
            by_region.setdefault(r, []).append((slot, tag))
        ptr = [0]
        cell_ids: list[int] = []
        tags: list[int] = []
        for r in range(n_regions):
            for slot, tag in sorted(by_region.get(r, [])):
                cell_ids.append(slot)  # ploidy=1, so cell id == slot.
                tags.append(tag)
            ptr.append(len(cell_ids))
        n = len(cell_ids)
        np.array(ptr, np.int64).tofile(rd / "region_ptr.npy")
        np.array(cell_ids, np.int32).tofile(rd / "cell_id.npy")
        ent = np.zeros(n, ENTRY_DTYPE)
        ent["snp_start"] = tags
        ent["snp_len"] = 1
        ent.tofile(rd / "cell_vk.npy")
        np.save(rd / "sample_cols.npy", np.arange(S, dtype=np.int64))
        np.zeros((n_regions, 2), np.int64).tofile(rd / "dense_snp_range.npy")
        np.zeros((n_regions, 2), np.int64).tofile(rd / "dense_indel_range.npy")
        (rd / "svar2_meta.json").write_text(
            json.dumps(
                {
                    "layout": "sparse",
                    "n_regions": n_regions,
                    "n_samples": S,
                    "n_entries": n,
                    "fill": (n / (n_regions * S * P)) if n_regions else 0.0,
                    "region_ptr": {"shape": [n_regions + 1], "dtype": "<i8"},
                    "cell_id": {"shape": [n], "dtype": "<i4"},
                    "cell_vk": {"shape": [n], "dtype": ENTRY_DTYPE.descr},
                    "dense_snp_range": {"shape": [n_regions, 2], "dtype": "<i8"},
                    "dense_indel_range": {"shape": [n_regions, 2], "dtype": "<i8"},
                    "sample_cols": {"shape": [S], "dtype": "<i8"},
                    "ploidy": P,
                }
            )
        )
        return root

    # Shard A: 1 region (local 0), occupied at slot 0, tag 100.
    # Shard B: 2 regions; local 0 occupied at slot 1 (tag 201), local 1 at
    # slot 0 (tag 202). Correct merged order interleaves the two shards:
    # merged = [B.local0, A.local0, B.local1].
    order = np.array([[1, 0], [0, 0], [1, 1]], np.int64)
    n_regions = 3
    shapes = [(1, S), (2, S)]

    def run(paths: list[Path], name: str):
        out_dir = tmp_path / name / "genotypes" / "svar2_ranges"
        out_dir.mkdir(parents=True)
        _concat_svar2_ranges(paths, out_dir, "regions", shapes, P, n_regions, S, order)
        region_ptr = np.fromfile(out_dir / "region_ptr.npy", np.int64)
        cell_id = np.fromfile(out_dir / "cell_id.npy", np.int32)
        vk = np.fromfile(out_dir / "cell_vk.npy", ENTRY_DTYPE)
        return region_ptr, cell_id, vk["snp_start"]

    a = mk_region_shard(tmp_path / "a_sparse", 1, [(0, 0, 100)])
    b = mk_region_shard(tmp_path / "b_sparse", 2, [(0, 1, 201), (1, 0, 202)])

    # merged region0 = B.local0 (cell 1, tag 201); region1 = A.local0 (cell 0,
    # tag 100); region2 = B.local1 (cell 0, tag 202). Block concat (order=None)
    # instead produces cell_id=[0, 1, 0], tags=[100, 201, 202]; an un-inverted
    # r_map produces region_ptr=[0, 2, 3, 3], cell_id=[0, 1, 0],
    # tags=[100, 201, 202] (both verified directly against the mutated code).
    expected = ([0, 1, 2, 3], [1, 0, 0], [201, 100, 202])

    # Fast path: both inputs sparse. Catches a broken `order=` in the
    # copy_runs-based reorder.
    region_ptr, cell_id, tags = run([a, b], "out_fast")
    np.testing.assert_array_equal(region_ptr, expected[0])
    np.testing.assert_array_equal(cell_id, expected[1])
    np.testing.assert_array_equal(tags, expected[2])

    # General path: rewrite shard A as dense so not every input is sparse.
    # Catches an un-inverted `r_maps` in `merge_region_blocks`.
    a_dense = rewrite_as_dense(a, tmp_path / "a_dense")
    region_ptr, cell_id, tags = run([a_dense, b], "out_general")
    np.testing.assert_array_equal(region_ptr, expected[0])
    np.testing.assert_array_equal(cell_id, expected[1])
    np.testing.assert_array_equal(tags, expected[2])


def test_merge_region_blocks_multi_batch_matches_single_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Batching must not change the result.

    `merge_region_blocks` sizes its batches from `CONCAT_CHUNK_BYTES`; every
    other test in this file has small enough inputs to run in a single batch,
    so a batch-boundary bug (state leaking across batches, an off-by-one in
    `r0`/`r1`) would be invisible everywhere else. Monkeypatching
    `CONCAT_CHUNK_BYTES` down to force one region per batch is cheap; building
    a fixture large enough to batch for real is not.
    """
    from genvarloader._dataset import _svar2_ranges
    from genvarloader._dataset._svar2_ranges import (
        ENTRY_DTYPE,
        _SparseRanges,
        _SparseWriter,
        merge_region_blocks,
    )

    R, P, S = 6, 2, 2
    span = S * P
    n_cells = R * span
    # Every (region, sample, ploid) cell occupied, with a distinguishable
    # payload, so a batch boundary landing anywhere in [1, R) would scramble
    # some cell's content or region assignment if it were handled wrong.
    cell_id = np.tile(np.arange(span, dtype=np.int32), R)
    region_ptr = np.arange(0, n_cells + span, span, dtype=np.int64)
    ent = np.zeros(n_cells, ENTRY_DTYPE)
    ent["snp_start"] = np.arange(n_cells)
    ent["snp_len"] = 1
    reader = _SparseRanges(
        region_ptr=region_ptr,
        cell_id=cell_id,
        cell_vk=ent,
        n_regions=R,
        n_samples=S,
        ploidy=P,
    )
    r_map = np.arange(R, dtype=np.int64)
    s_map = np.arange(S, dtype=np.int64)

    single_dir = tmp_path / "single"
    single_dir.mkdir()
    w1 = _SparseWriter(single_dir, n_samples=S, ploidy=P)
    merge_region_blocks(
        [reader], [r_map], [s_map], w1, n_regions=R, span=span, ploidy=P
    )
    n1 = w1.close()

    # Default CONCAT_CHUNK_BYTES (16 MiB) comfortably fits all 6 regions in one
    # batch; forcing it down to 1 byte forces `rows = max(1, ...) == 1`, i.e.
    # one region per batch -- 6 batches instead of 1.
    monkeypatch.setattr(_svar2_ranges, "CONCAT_CHUNK_BYTES", 1)
    multi_dir = tmp_path / "multi"
    multi_dir.mkdir()
    w2 = _SparseWriter(multi_dir, n_samples=S, ploidy=P)
    merge_region_blocks(
        [reader], [r_map], [s_map], w2, n_regions=R, span=span, ploidy=P
    )
    n2 = w2.close()

    assert n1 == n2 == n_cells
    for name in ("region_ptr.npy", "cell_id.npy", "cell_vk.npy"):
        a = (single_dir / name).read_bytes()
        b = (multi_dir / name).read_bytes()
        assert a == b, name
